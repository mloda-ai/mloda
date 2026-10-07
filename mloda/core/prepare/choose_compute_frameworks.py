"""Chooses one compute framework per block of features (one block is one plan step group) before links resolve.
Arc consistency prunes the domains, then branch and bound with forward checking minimises conversions,
splitting into independent parts once assigned blocks separate them.
"""

import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial
from typing import Any, NamedTuple
from uuid import UUID

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import ComputeFrameworkTransformer
from mloda.core.abstract_plugins.components.options import Options, _str_option_dict
from mloda.core.abstract_plugins.components.link import JoinType, Link
from mloda.core.abstract_plugins.components.validators.feature_validator import FeatureValidator
from mloda.core.abstract_plugins.compute_framework import ComputeFramework, framework_rank_key
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.graph.graph import Graph

Framework = type[ComputeFramework]
Values = tuple[Framework, ...]


Cost = tuple[int, int]


class _Block(NamedTuple):
    fg: type[FeatureGroup]
    features: tuple[Feature, ...]
    domain: Values


class _Rule(NamedTuple):
    blocks: tuple[int, ...]
    allows: Callable[[Values], bool]
    why: str


class _Frame(NamedTuple):
    block: int
    at: int
    before: Cost
    values: Iterator[Framework]
    current: dict[int, list[Framework]]
    cost: Cost
    parts: list[list[int]]
    fixed: list[int]


_UNFLIPPABLE = (JoinType.RIGHT, JoinType.APPEND, JoinType.UNION)
_RIGHT_ONE_SIDE = (
    "consumers of one RIGHT link join on one side. "
    "Request them separately or restrict their compute frameworks to one side of the link"
)

PINNED = "pinned"
ONLY_ALLOWED = "only allowed framework"
RULES = "rules exclude preferred frameworks"
LIST_ORDER = "your list order"
DEFAULT_ORDER = "default order"
CONVERTS_LATER = "converts later"


def saves_conversions(count: int) -> str:
    return f"saves {count} conversion{'' if count == 1 else 's'}"


def conversion_cost(from_framework: Framework, to_framework: Framework) -> int:
    """Cost of one transform step between two frameworks."""
    return 1


_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def _names(features: tuple[Feature, ...]) -> list[str]:
    return sorted(str(feature.name) for feature in features)


def stable_text(value: object) -> str:
    """Text of a value with addresses removed and set elements sorted, so it is equal across interpreters."""
    if isinstance(value, Options):
        parts = f"group={stable_text(_str_option_dict(value.group))}, context={stable_text(_str_option_dict(value.context))}"
        if value.propagate_context_keys:
            parts += f", propagate_context_keys={stable_text(value.propagate_context_keys)}"
        return f"Options({parts})"
    if type(value) is dict:
        items = sorted((stable_text(k), stable_text(v)) for k, v in value.items())
        return "{" + ", ".join(f"{k}: {v}" for k, v in items) + "}"
    if type(value) is list:
        return "[" + ", ".join(stable_text(v) for v in value) + "]"
    if type(value) is tuple:
        return "(" + ", ".join(stable_text(v) for v in value) + ("," if len(value) == 1 else "") + ")"
    if type(value) is set or type(value) is frozenset:
        return "{" + ", ".join(sorted(stable_text(v) for v in value)) + "}"
    return _ADDRESS.sub("", repr(value))


def _block_key(fg: type[FeatureGroup], features: tuple[Feature, ...]) -> tuple[object, ...]:
    allowed = sorted(f.get_class_name() for f in features[0].compute_frameworks or ())
    data_types = sorted(f.data_type.name if f.data_type else "" for f in features)
    options = sorted(stable_text(f.options) for f in features)
    return (fg.__module__, fg.__qualname__, tuple(_names(features)), tuple(data_types), tuple(allowed), tuple(options))


def _refine(
    keys: Sequence[tuple[object, ...]], neighbors: list[set[int]], waiters: Sequence[set[int]] | None = None
) -> list[int]:
    """Colour refinement by neighbour classes; given waiters, neighbors are producers and direction counts."""
    key_rank = {key: rank for rank, key in enumerate(sorted(set(keys)))}
    classes = [key_rank[key] for key in keys]
    while True:
        labels: list[tuple[object, ...]] = [
            (classes[i], tuple(sorted(classes[n] for n in neighbors[i])))
            + (() if waiters is None else (tuple(sorted(classes[n] for n in waiters[i])),))
            for i in range(len(keys))
        ]
        label_rank = {label: rank for rank, label in enumerate(sorted(set(labels)))}
        refined = [label_rank[label] for label in labels]
        if len(set(refined)) == len(set(classes)):
            return classes
        classes = refined


def _join_allows(jointype: JoinType, values: Values, right_off_child: bool) -> bool:
    left, right, child = values
    if jointype is JoinType.RIGHT:
        return child is right or (right_off_child and child is left)
    if jointype in (JoinType.APPEND, JoinType.UNION):
        return child is left
    return child is left or child is right


def _any_join_allows(occurrences: list[tuple[JoinType, bool, bool]], values: Values) -> bool:
    """One link must be satisfied; a same-framework cross-group link may not be skipped (it would self-merge)."""
    satisfied = [_join_allows(jt, values[3 * i : 3 * i + 3], off) for i, (jt, _, off) in enumerate(occurrences)]
    for i, (_, strict, _) in enumerate(occurrences):
        if strict and values[3 * i] is values[3 * i + 1] and not satisfied[i]:
            return False
    return any(satisfied)


def _swapped_pairs_differ(values: Values) -> bool:
    """Children whose parents sit on swapped frameworks join apart, or their flipped keys would merge."""
    left_a, right_a, child_a, left_b, right_b, child_b = values
    if left_a is right_a or left_a is not right_b or right_a is not left_b:
        return True
    return child_a is not child_b


def _runs_after_parent(framework: Framework, values: Values) -> bool:
    parent, child = values
    return child is not framework or parent is framework


def _same_side(values: Values) -> bool:
    left_a, right_a, child_a, left_b, right_b, child_b = values
    if (left_a, right_a) != (left_b, right_b) or left_a is right_a:
        return True
    return (child_a is left_a) == (child_b is left_b)


def _no_side_mix(values: Values) -> bool:
    left_a, right_a, child_a, left_b, right_b, child_b = values
    if (left_a, right_a) != (left_b, right_b) or left_a is right_a:
        return True
    return not ((child_a is left_a and child_b is right_b) or (child_a is right_a and child_b is left_b))


def _left_with_off_sibling(values: Values) -> bool:
    """A consumer sits on the left of a RIGHT link only beside an off sibling on the same sides that does too."""
    left, right, child = values[:3]
    if child is not left or left is right:
        return True
    siblings = [values[i : i + 3] for i in range(3, len(values), 3)]
    return any(sl is left and sr is right and sc is sl for sl, sr, sc in siblings)


class ChooseComputeFrameworks:
    """Sets chosen_compute_framework on every feature, cheapest consistent assignment first by preference."""

    def __init__(
        self,
        graph: Graph,
        nodes_per_feature_group: dict[type[FeatureGroup], set[Feature]],
        link_occurrences: list[tuple[Link, UUID, UUID, UUID]],
        filter_ties: list[tuple[UUID, UUID]],
        positions: Mapping[Framework, int],
        output_framework: Framework | None = None,
        dropped: Mapping[UUID, frozenset[Framework]] | None = None,
        connected: Mapping[Framework, Any] | None = None,
    ) -> None:
        self.output_framework = output_framework
        self.dropped = dropped or {}
        self.connected: Mapping[Framework, Any] = connected or {}
        self.graph = graph
        self.nodes = nodes_per_feature_group
        self.occurrences = link_occurrences
        self.ties = filter_ties
        self.positions = positions
        self.rank = framework_rank_key(positions)
        self.transformer = ComputeFrameworkTransformer()
        self.paths: dict[tuple[Framework, Framework], bool] = {}
        self._step_depth: dict[tuple[int, type[FeatureGroup]], int] = {}
        self._final_depth: list[int] = []
        self._horizon = 1

    def choose(self) -> None:
        """Solve each connected component exactly and write the result onto the features."""
        blocks = self._blocks()
        owner = {f.uuid: i for i, block in enumerate(blocks) for f in block.features}
        blocks, borrowed = self._regain_dropped(blocks, owner)
        rules = self._rules(blocks, owner) + borrowed
        groups = self._cost_groups(blocks, owner)
        depth = self._feature_depths()
        self._horizon = max(depth.values(), default=0) + 1
        self._step_depth = self._step_depths(blocks, owner, depth)
        self._final_depth = [
            max((depth.get(f.uuid, 0) for f in block.features if f.initial_requested_data), default=0)
            for block in blocks
        ]
        domains = self._prune(blocks, rules)
        for order in self._search_orders(len(blocks), rules, groups):
            members = set(order)
            local_rules = [r for r in rules if r.blocks[0] in members]
            touching, steps = self._component_indexes(order, local_rules, groups)
            assignment, _ = self._solve(blocks, order, local_rules, groups, domains, touching, steps)
            for index, framework in assignment.items():
                reason = self._reason(blocks, touching, steps, assignment, index)
                for feature in blocks[index].features:
                    feature.chosen_compute_framework = framework
                    feature.chosen_compute_framework_reason = reason
        self._require_all_chosen()

    def _component_indexes(
        self, order: list[int], rules: list[_Rule], groups: dict[tuple[int, type[FeatureGroup]], set[int]]
    ) -> tuple[dict[int, list[_Rule]], dict[int, list[tuple[int, set[int], int]]]]:
        """Per block of one component: the rules touching it and the transform steps it takes part in."""
        touching: dict[int, list[_Rule]] = {b: [] for b in order}
        for rule in rules:
            for b in self._rule_blocks(rule):
                touching[b].append(rule)
        steps: dict[int, list[tuple[int, set[int], int]]] = {b: [] for b in order}
        for key, children in groups.items():
            if key[0] in touching:
                for b in (key[0], *children):
                    steps[b].append((key[0], children, self._step_depth[key]))
        return touching, steps

    def _require_all_chosen(self) -> None:
        for node in self.graph.nodes:
            feature = self.graph.nodes[node].feature
            if feature.chosen_compute_framework is None:
                raise ValueError(f"Feature {feature.name} ended without a chosen compute framework.")

    def _blocks(self) -> list[_Block]:
        keyed: list[tuple[tuple[object, ...], _Block]] = []
        for fg, features in self.nodes.items():
            for group in ExecutionPlan.group_features_by_compute_framework_and_options(set(features)).values():
                members = tuple(sorted(group, key=lambda f: (str(f.name), stable_text(f.options))))
                domain = tuple(sorted(members[0].compute_frameworks or (), key=self.rank))
                if not domain:
                    FeatureValidator.validate_compute_frameworks_resolved(
                        members[0].compute_frameworks, str(members[0].name)
                    )
                keyed.append((_block_key(fg, members), _Block(fg, members, domain)))
        keys = [key for key, _ in keyed]
        classes = _refine(keys, self._block_neighbors(keyed))
        order = sorted(range(len(keyed)), key=lambda i: (keys[i], classes[i], _names(keyed[i][1].features)))
        return [keyed[i][1] for i in order]

    def _regain_dropped(self, blocks: list[_Block], owner: dict[UUID, int]) -> tuple[list[_Block], list[_Rule]]:
        """Frameworks dropped for lack of a connection return when parents share one connection or the DAC has one."""
        if not self.dropped:
            return blocks, []
        parents: list[set[int]] = [set() for _ in blocks]
        for parent, child in self.graph.edges:
            if parent in owner and child in owner and owner[parent] != owner[child]:
                parents[owner[child]].add(owner[parent])
        carried = self._carried_connections(blocks, parents)
        domains, gained, via_dac = self._regain_fixpoint(blocks, owner, parents, carried)
        widened = [
            block._replace(domain=tuple(sorted(domains[i], key=self.rank))) if gained[i] else block
            for i, block in enumerate(blocks)
        ]
        rules = [
            _Rule(
                (parent, child),
                partial(_runs_after_parent, framework),
                f"{blocks[child].fg.__name__} has no connection for {framework.get_class_name()}, "
                "so it runs there only after its parents",
            )
            for child in range(len(blocks))
            for framework in sorted(gained[child] - via_dac[child], key=self.rank)
            for parent in sorted(parents[child])
        ]
        return widened, rules

    def _carried_connections(self, blocks: list[_Block], parents: list[set[int]]) -> list[dict[Framework, list[Any]]]:
        """Per block and framework, the distinct connections its data would carry, compared by identity."""
        frameworks = set(self.connected).union(*self.dropped.values())
        carried: list[dict[Framework, list[Any]]] = [{} for _ in blocks]
        for index, block in enumerate(blocks):
            key = block.fg.get_class_name()
            for framework in frameworks:
                found = [c for c in (f.options.get(key) for f in block.features) if framework._connection_matches(c)]
                carried[index][framework] = [c for i, c in enumerate(found) if not any(c is d for d in found[:i])]
        own = [{f for f in frameworks if carried[i][f]} for i in range(len(blocks))]
        changed = True
        while changed:
            changed = False
            for index, framework in ((i, f) for i in range(len(blocks)) for f in frameworks):
                if framework in own[index]:
                    continue
                incoming = [c for p in parents[index] for c in carried[p][framework]]
                if framework in self.connected and any(framework not in blocks[p].domain for p in parents[index]):
                    incoming.append(self.connected[framework])
                for conn in incoming:
                    if not any(conn is known for known in carried[index][framework]):
                        carried[index][framework].append(conn)
                        changed = True
        return carried

    def _regain_fixpoint(
        self,
        blocks: list[_Block],
        owner: dict[UUID, int],
        parents: list[set[int]],
        carried: list[dict[Framework, list[Any]]],
    ) -> tuple[list[set[Framework]], list[set[Framework]], list[set[Framework]]]:
        """Domains with regained frameworks, what each block gained, and which gains came from the DAC."""
        lost = [frozenset().union(*(self.dropped.get(f.uuid, frozenset()) for f in b.features)) for b in blocks]
        pinned = [any(f.framework_pinned for f in b.features) for b in blocks]
        domains = [set(block.domain) for block in blocks]
        gained: list[set[Framework]] = [set() for _ in blocks]
        via_dac: list[set[Framework]] = [set() for _ in blocks]
        changed = True
        while changed:
            changed = False
            for index in range(len(blocks)):
                if pinned[index] or not parents[index]:
                    continue
                for framework in lost[index] - domains[index]:
                    if len(carried[index][framework]) > 1:
                        continue
                    from_dac = framework in self.connected and all(
                        framework not in blocks[p].domain for p in parents[index]
                    )
                    if from_dac or all(framework in domains[p] for p in parents[index]):
                        domains[index].add(framework)
                        gained[index].add(framework)
                        if from_dac:
                            via_dac[index].add(framework)
                        changed = True
            for host, tied in self.ties:
                if host in owner and tied in owner and not pinned[owner[tied]]:
                    new = gained[owner[host]] - domains[owner[tied]]
                    if new:
                        domains[owner[tied]] |= new
                        gained[owner[tied]] |= new
                        changed = True
        return domains, gained, via_dac

    def _block_neighbors(self, keyed: list[tuple[tuple[object, ...], _Block]]) -> list[set[int]]:
        """Blocks adjacent through graph edges, links and filter ties, for telling equal-keyed blocks apart."""
        owner = {f.uuid: i for i, (_, block) in enumerate(keyed) for f in block.features}
        pairs = list(self.graph.edges) + list(self.ties)
        pairs += [(left, child) for _, left, _, child in self.occurrences]
        pairs += [(right, child) for _, _, right, child in self.occurrences]
        neighbors: list[set[int]] = [set() for _ in keyed]
        for one, other in pairs:
            if one in owner and other in owner and owner[one] != owner[other]:
                neighbors[owner[one]].add(owner[other])
                neighbors[owner[other]].add(owner[one])
        return neighbors

    def _convertible(self, source: Framework, target: Framework) -> bool:
        if source is target:
            return True
        pair = (source, target)
        if pair not in self.paths:
            src, dst = source.expected_data_framework(), target.expected_data_framework()
            self.paths[pair] = src is dst or self.transformer.get_transformation_chain(src, dst) is not None
        return self.paths[pair]

    @staticmethod
    def _requested(block: _Block) -> bool:
        return any(feature.initial_requested_data for feature in block.features)

    def _transform_cost(self, depth: int, costs: list[int]) -> Cost:
        """Conversions of one transform step, then how early they happen."""
        return (sum(costs), sum(costs) * (self._horizon - depth))

    def _final_cost(self, block: _Block, framework: Framework, index: int) -> Cost:
        output = self.output_framework
        if output is None or not self._requested(block) or framework is output:
            return (0, 0)
        if framework.expected_data_framework() is output.expected_data_framework():
            return (0, 0)
        cost = conversion_cost(framework, output)
        return (cost, cost * (self._horizon - self._final_depth[index] - 1))

    def _step_depths(
        self, blocks: list[_Block], owner: dict[UUID, int], depth: dict[UUID, int]
    ) -> dict[tuple[int, type[FeatureGroup]], int]:
        """Deepest parent feature with an edge into each transform step."""
        result: dict[tuple[int, type[FeatureGroup]], int] = {}
        for parent, child in self.graph.edges:
            if parent in owner and child in owner and owner[parent] != owner[child]:
                key = (owner[parent], blocks[owner[child]].fg)
                result[key] = max(result.get(key, 0), depth.get(parent, 0))
        return result

    def _feature_depths(self) -> dict[UUID, int]:
        """Longest path from a root per feature."""
        children: dict[UUID, list[UUID]] = {}
        waiting: Counter[UUID] = Counter()
        for parent, child in self.graph.edges:
            children.setdefault(parent, []).append(child)
            waiting[child] += 1
        depth: dict[UUID, int] = {}
        ready = [node for node in self.graph.nodes if not waiting[node]]
        while ready:
            node = ready.pop()
            for child in children.get(node, ()):
                depth[child] = max(depth.get(child, 0), depth.get(node, 0) + 1)
                waiting[child] -= 1
                if not waiting[child]:
                    ready.append(child)
        return depth

    def _rules(self, blocks: list[_Block], owner: dict[UUID, int]) -> list[_Rule]:
        rules: list[_Rule] = []
        for parent, child in {(owner[p], owner[c]) for p, c in self.graph.edges if p in owner and c in owner}:
            if parent != child:
                rules.append(
                    _Rule(
                        (parent, child),
                        lambda v: self._convertible(v[0], v[1]),
                        f"transform path {blocks[parent].fg.__name__} -> {blocks[child].fg.__name__}",
                    )
                )
        output = self.output_framework
        if output is not None:
            for index, block in enumerate(blocks):
                if self._requested(block):
                    rules.append(
                        _Rule(
                            (index,),
                            lambda v: self._convertible(v[0], output),
                            f"transform path to output framework {output.get_class_name()}",
                        )
                    )
        for host, tied in self.ties:
            if host in owner and tied in owner:
                rules.append(_Rule((owner[host], owner[tied]), lambda v: v[0] is v[1], "filter tied to its host"))
        rules += self._side_path_rules(blocks, owner)
        by_link: dict[UUID, list[tuple[int, int, int]]] = {}
        right_by_link: dict[UUID, list[tuple[UUID, tuple[int, int, int], bool]]] = {}
        by_child: dict[UUID, list[tuple[Link, tuple[int, int, int]]]] = {}
        for link, left_uuid, right_uuid, child_uuid in self.occurrences:
            joined = (owner[left_uuid], owner[right_uuid], owner[child_uuid])
            by_child.setdefault(child_uuid, []).append((link, joined))
            if link.jointype not in _UNFLIPPABLE:
                by_link.setdefault(link.uuid, []).append(joined)
            elif link.jointype is JoinType.RIGHT:
                off = link.left_feature_group != link.right_feature_group and set(blocks[joined[2]].domain).isdisjoint(
                    blocks[joined[1]].domain
                )
                right_by_link.setdefault(link.uuid, []).append((child_uuid, joined, off))
        widened: dict[tuple[UUID, UUID, tuple[int, int, int]], bool] = {}
        for link_uuid, entries in right_by_link.items():
            for child_uuid, joined, off in entries:
                siblings = [j for c, j, o in entries if o and c != child_uuid]
                widened[(link_uuid, child_uuid, joined)] = off
                if siblings and not off:
                    widened[(link_uuid, child_uuid, joined)] = True
                    rules.append(
                        _Rule(
                            joined + tuple(b for j in siblings for b in j),
                            _left_with_off_sibling,
                            _RIGHT_ONE_SIDE,
                        )
                    )
        for child_uuid, occurrences in by_child.items():
            jointypes = [link.jointype for link, _ in occurrences]
            kinds = [
                (
                    link.jointype,
                    link.left_feature_group != link.right_feature_group
                    and link.jointype not in (JoinType.APPEND, JoinType.UNION),
                    widened.get((link.uuid, child_uuid, joined), False),
                )
                for link, joined in occurrences
            ]
            rules.append(
                _Rule(
                    tuple(b for _, joined in occurrences for b in joined),
                    partial(_any_join_allows, kinds),
                    "join " + "/".join(sorted({j.value for j in jointypes})),
                )
            )
        for joins in by_link.values():
            for i, first in enumerate(joins):
                for second in joins[i + 1 :]:
                    if first[2] != second[2]:
                        rules.append(_Rule(first + second, _same_side, "one side per link"))
                        rules.append(_Rule(first + second, _swapped_pairs_differ, "swapped pairs join apart"))
        for entries in right_by_link.values():
            for i, (_, first, _) in enumerate(entries):
                for _, second, _ in entries[i + 1 :]:
                    if first[2] != second[2]:
                        rules.append(_Rule(first + second, _no_side_mix, _RIGHT_ONE_SIDE))
        return rules

    def _reach(self, start: UUID, neighbours: Mapping[UUID, set[UUID]]) -> set[UUID]:
        """Nodes reachable from `start` over the given edge map, iteratively."""
        found: set[UUID] = set()
        stack = [start]
        while stack:
            for node in neighbours.get(stack.pop(), ()):
                if node not in found:
                    found.add(node)
                    stack.append(node)
        return found

    def _side_path_rules(self, blocks: list[_Block], owner: dict[UUID, int]) -> list[_Rule]:
        """A feature between a link side and its consumer shares the side's framework, unless it is a hoppable carrier."""
        if not self.occurrences:
            return []
        parents: dict[UUID, set[UUID]] = {}
        children: dict[UUID, set[UUID]] = {}
        for parent, child in self.graph.edges:
            parents.setdefault(child, set()).add(parent)
            children.setdefault(parent, set()).add(child)
        consumers = Counter(child for _, _, _, child in self.occurrences)
        pairs: set[tuple[int, int]] = set()
        peer_pairs: set[tuple[int, int]] = set()
        for link, left_uuid, right_uuid, child_uuid in self.occurrences:
            above_child = self._reach(child_uuid, parents)
            hoppable = link.jointype not in (JoinType.APPEND, JoinType.UNION) and consumers[child_uuid] == 1
            side_domains: set[Framework] = set()
            for side in (left_uuid, right_uuid):
                if side in owner:
                    side_domains.update(blocks[owner[side]].domain)
            for side in (left_uuid, right_uuid):
                mids = (above_child & self._reach(side, children)) - {left_uuid, right_uuid}
                carriers = {
                    mid
                    for mid in mids
                    if hoppable
                    and mid in owner
                    and mid in parents.get(child_uuid, set())
                    and not self._reach(mid, children) & above_child
                    and side_domains.isdisjoint(blocks[owner[mid]].domain)
                }
                for mid in mids - carriers:
                    if side in owner and mid in owner and owner[side] != owner[mid]:
                        pairs.add((owner[side], owner[mid]))
                for carrier in carriers:
                    for peer in mids & parents.get(child_uuid, set()) - {carrier}:
                        if peer in owner and owner[peer] != owner[carrier]:
                            peer_pairs.add((owner[carrier], owner[peer]))
        same = "so they must run on the same framework"
        path = "{} and {} lie on one path from a link side to its join consumer, " + same
        parallel = "{} and {} feed one join consumer on parallel paths from a link side, " + same
        return [
            _Rule((a, b), lambda v: v[0] is v[1], template.format(blocks[b].fg.__name__, blocks[a].fg.__name__))
            for template, found in ((path, pairs), (parallel, peer_pairs))
            for a, b in sorted(found)
        ]

    def _cost_groups(
        self, blocks: list[_Block], owner: dict[UUID, int]
    ) -> dict[tuple[int, type[FeatureGroup]], set[int]]:
        """Child blocks per (parent block, child feature group): the identity of one transform step."""
        groups: dict[tuple[int, type[FeatureGroup]], set[int]] = {}
        for parent, child in self.graph.edges:
            if parent in owner and child in owner and owner[parent] != owner[child]:
                groups.setdefault((owner[parent], blocks[owner[child]].fg), set()).add(owner[child])
        return groups

    @staticmethod
    def _search_orders(
        count: int, rules: list[_Rule], groups: dict[tuple[int, type[FeatureGroup]], set[int]]
    ) -> list[list[int]]:
        """Per connected component, blocks breadth-first from the first in content order (neighbours in content order)."""
        neighbors = ChooseComputeFrameworks._neighbor_map(count, rules, groups)
        orders: list[list[int]] = []
        seen: set[int] = set()
        for start in range(count):
            if start in seen:
                continue
            order = [start]
            seen.add(start)
            for index in order:
                for other in sorted(neighbors[index] - seen):
                    seen.add(other)
                    order.append(other)
            orders.append(order)
        return orders

    @staticmethod
    def _neighbor_map(
        count: int, rules: list[_Rule], groups: dict[tuple[int, type[FeatureGroup]], set[int]]
    ) -> list[set[int]]:
        """Blocks sharing a rule or a transform step; a step couples its children, so it is a hyperedge."""
        neighbors: list[set[int]] = [set() for _ in range(count)]
        for members in [r.blocks for r in rules] + [(parent, *kids) for (parent, _), kids in groups.items()]:
            for one in members:
                neighbors[one].update(members)
        return neighbors

    @staticmethod
    def _rule_blocks(rule: _Rule) -> list[int]:
        return list(dict.fromkeys(rule.blocks))

    @staticmethod
    def _allows(rule: _Rule, values: Mapping[int, Framework]) -> bool:
        return rule.allows(tuple(values[b] for b in rule.blocks))

    def _prune(self, blocks: list[_Block], rules: list[_Rule]) -> list[list[Framework]]:
        """Unary rules filter domains, then arc consistency over the two-block rules; an emptied domain raises."""
        domains = [list(block.domain) for block in blocks]
        for rule in (r for r in rules if len(self._rule_blocks(r)) == 1):
            target = rule.blocks[0]
            domains[target] = [v for v in domains[target] if self._allows(rule, {target: v})]
            if not domains[target]:
                raise ValueError(self._emptied(blocks[target], rule, None, []))
        pairwise = [r for r in rules if len(self._rule_blocks(r)) == 2]
        changed = True
        while changed:
            changed = False
            for rule in pairwise:
                for target in reversed(self._rule_blocks(rule)):
                    other = next(b for b in self._rule_blocks(rule) if b != target)
                    kept = [
                        v
                        for v in domains[target]
                        if any(self._allows(rule, {target: v, other: w}) for w in domains[other])
                    ]
                    if not kept:
                        raise ValueError(self._emptied(blocks[target], rule, blocks[other], domains[other]))
                    if len(kept) != len(domains[target]):
                        domains[target] = kept
                        changed = True
        return domains

    @staticmethod
    def _emptied(block: _Block, rule: _Rule, other: _Block | None, other_domain: list[Framework]) -> str:
        """Names the emptied block and, when its partner is fixed to one framework, the partner too."""
        message = f"No compute framework is left for features {', '.join(_names(block.features))}"
        message += f" (allowed: {sorted(fw.get_class_name() for fw in block.domain)}): {rule.why}"
        if other is not None and len(other_domain) == 1:
            message += (
                f", with features {', '.join(_names(other.features))} fixed to {other_domain[0].get_class_name()}"
            )
        return message + "."

    def _order_reason(self, chosen: Framework, other: Framework) -> str:
        unlisted = max(self.positions.values(), default=-1) + 1
        return (
            LIST_ORDER if self.positions.get(chosen, unlisted) != self.positions.get(other, unlisted) else DEFAULT_ORDER
        )

    def _reason(
        self,
        blocks: list[_Block],
        touching_by_block: dict[int, list[_Rule]],
        steps_by_block: dict[int, list[tuple[int, set[int], int]]],
        assignment: dict[int, Framework],
        index: int,
    ) -> str:
        """Why one block runs on its framework: pin, sole option, rules, order, or conversions saved."""
        block = blocks[index]
        chosen = assignment[index]
        if any(feature.framework_pinned for feature in block.features):
            return PINNED
        if len(block.domain) == 1:
            return ONLY_ALLOWED
        touching = touching_by_block[index]
        steps = steps_by_block[index]

        def feasible(framework: Framework) -> bool:
            return all(
                rule.allows(tuple(framework if b == index else assignment[b] for b in rule.blocks)) for rule in touching
            )

        def step_cost(swap: Framework | None) -> Cost:
            def at(b: int) -> Framework:
                return swap if swap is not None and b == index else assignment[b]

            costs = [
                self._transform_cost(
                    depth,
                    [conversion_cost(at(parent), t) for t in {at(c) for c in kids} - {at(parent)}],
                )
                for parent, kids, depth in steps
            ]
            final = self._final_cost(block, at(index), index)
            return (sum(c[0] for c in costs) + final[0], sum(c[1] for c in costs) + final[1])

        alternatives = [fw for fw in block.domain if fw is not chosen and feasible(fw)]
        if not alternatives:
            return RULES
        chosen_rank = self.rank(chosen)
        earlier = [fw for fw in alternatives if self.rank(fw) < chosen_rank]
        if not earlier:
            return self._order_reason(chosen, alternatives[0])
        base = step_cost(None)
        costs = {fw: step_cost(fw) for fw in earlier}
        delta = min(c[0] for c in costs.values()) - base[0]
        if delta > 0:
            return saves_conversions(delta)
        if all(c[1] > base[1] for c in costs.values() if c[0] == base[0]):
            return CONVERTS_LATER
        return self._order_reason(chosen, earlier[0])

    def _solve(
        self,
        blocks: list[_Block],
        order: list[int],
        rules: list[_Rule],
        groups: dict[tuple[int, type[FeatureGroup]], set[int]],
        domains: list[list[Framework]],
        touching: dict[int, list[_Rule]],
        steps: dict[int, list[tuple[int, set[int], int]]],
    ) -> tuple[dict[int, Framework], Cost]:
        """Branch and bound; returns the best assignment and its cost, raising when no assignment is feasible."""
        linked = self._neighbor_map(
            len(blocks), rules, {key: kids for key, kids in groups.items() if key[0] in touching}
        )
        search_order = [b for b in order if len(domains[b]) == 1] + [b for b in order if len(domains[b]) != 1]
        position = {b: i for i, b in enumerate(search_order)}
        assigned: dict[int, Framework] = {}

        def step_cost(parent: int, children: set[int], depth: int) -> Cost:
            if parent not in assigned:
                return (0, 0)
            moved = {assigned[c] for c in children if c in assigned and assigned[c] is not assigned[parent]}
            return self._transform_cost(depth, [conversion_cost(assigned[parent], target) for target in moved])

        def total(indices: Sequence[tuple[int, set[int], int]]) -> Cost:
            costs = [step_cost(*step) for step in indices]
            return (sum(c[0] for c in costs), sum(c[1] for c in costs))

        def narrow(index: int, current: dict[int, list[Framework]]) -> dict[int, list[Framework]] | None:
            """Forward check: values of the one block still open in a rule must be supported by the assignment."""
            narrowed = current  # copy on write: frames hold current, never mutate it
            for rule in touching[index]:
                pending = [b for b in self._rule_blocks(rule) if b not in assigned]
                if not pending:
                    if not self._allows(rule, assigned):
                        return None
                elif len(pending) == 1:
                    open_block = pending[0]
                    kept = [
                        v
                        for v in narrowed[open_block]
                        if rule.allows(tuple(v if b == open_block else assigned[b] for b in rule.blocks))
                    ]
                    if not kept:
                        return None
                    if len(kept) != len(narrowed[open_block]):
                        if narrowed is current:
                            narrowed = dict(current)
                        narrowed[open_block] = kept
            return narrowed

        cuts: dict[int, list[list[int]]] = {}

        def separated(index: int) -> list[list[int]]:
            """Open parts cut off by assigning index, except the largest, which continues in place."""
            starts = sorted(b for b in linked[index] if b != index and b not in assigned)
            if len(starts) < 2:
                return []
            owner = {start: start for start in starts}  # start -> id (smallest start) of the part holding it
            found = {start: [start] for start in starts}
            stacks = {start: [start] for start in starts}
            while sum(1 for stack in stacks.values() if stack) > 1:  # lockstep: the last growing part is not walked
                for pid in [p for p in stacks if stacks[p]]:
                    if pid not in stacks or not stacks[pid]:
                        continue
                    current = stacks[pid].pop()
                    for other in linked[current]:
                        if other == index or other in assigned:
                            continue
                        held = owner.get(other)
                        if held is None:
                            owner[other] = pid
                            found[pid].append(other)
                            stacks[pid].append(other)
                        elif held != pid:
                            keep, drop = min(pid, held), max(pid, held)
                            for node in found[drop]:
                                owner[node] = keep
                            found[keep].extend(found.pop(drop))
                            stacks[keep].extend(stacks.pop(drop))
                            stacks[keep].append(current)
                            break
            if len(found) < 2:
                return []
            parts = [found[pid] for pid in sorted(found)]
            growing = [pid for pid in sorted(found) if stacks[pid]]
            if growing:
                rest = max(len(found[pid]) for pid in found if pid != growing[0])
                if len(found[growing[0]]) > rest:
                    parts.remove(found[growing[0]])
                    return [sorted(part, key=position.__getitem__) for part in parts]
                stack = stacks[growing[0]]
                while stack:
                    for other in linked[stack.pop()]:
                        if other not in owner and other != index and other not in assigned:
                            owner[other] = growing[0]
                            found[growing[0]].append(other)
                            stack.append(other)
            parts.remove(max(parts, key=len))
            return [sorted(part, key=position.__getitem__) for part in parts]

        def search(part: list[int], start: dict[int, list[Framework]]) -> tuple[dict[int, Framework], Cost] | None:
            """Best assignment and cost of one part given the blocks assigned around it, or None when infeasible."""
            best: dict[int, Framework] = {}
            best_cost: list[Cost | None] = [None]

            def open_frame(at: int, current: dict[int, list[Framework]], cost: Cost) -> _Frame | None:
                while at < len(part) and part[at] in assigned:
                    at += 1
                if at == len(part):
                    best.clear()
                    best.update({b: assigned[b] for b in part})
                    best_cost[0] = cost
                    return None
                index = part[at]
                if index not in cuts:  # the assigned set at a block's opening depends only on the decomposition
                    cuts[index] = separated(index)
                return _Frame(index, at, total(steps[index]), iter(current[index]), current, cost, cuts[index], [])

            frames: list[_Frame] = []
            root = open_frame(0, start, (0, 0))
            if root is not None:
                frames.append(root)
            while frames:
                index, at, before, values, current, cost, parts, fixed = frames[-1]
                assigned.pop(index, None)
                for b in fixed:
                    assigned.pop(b)
                fixed.clear()
                value = next(values, None)
                if value is None:
                    frames.pop()
                    continue
                assigned[index] = value
                narrowed = narrow(index, current)
                if narrowed is None:
                    continue
                after = total(steps[index])
                final = self._final_cost(blocks[index], value, index)
                new_cost = (cost[0] + after[0] - before[0] + final[0], cost[1] + after[1] - before[1] + final[1])
                incumbent = best_cost[0]
                if incumbent is not None and not new_cost < incumbent:
                    continue
                feasible = True
                for other in parts:
                    solved = search(other, narrowed)
                    if solved is None:
                        feasible = False
                        break
                    sub, sub_cost = solved
                    assigned.update(sub)
                    fixed.extend(sub)
                    new_cost = (new_cost[0] + sub_cost[0], new_cost[1] + sub_cost[1])
                if not feasible or (incumbent is not None and not new_cost < incumbent):
                    continue
                frame = open_frame(at + 1, narrowed, new_cost)
                if frame is not None:
                    frames.append(frame)
            found = best_cost[0]
            if found is None:
                return None
            return dict(best), found

        result = search(search_order, {b: domains[b] for b in order})
        if result is None:
            raise ValueError(self._infeasible(blocks, order, rules))
        return result

    @staticmethod
    def _infeasible(blocks: list[_Block], order: list[int], rules: list[_Rule]) -> str:
        features = [
            f"{feature.name} (allowed: {sorted(fw.get_class_name() for fw in blocks[i].domain)})"
            for i in order
            for feature in blocks[i].features
        ]
        return (
            "No compute framework assignment satisfies the hard rules for features "
            f"{', '.join(features)}. Rules involved: {sorted({r.why for r in rules})}."
        )
