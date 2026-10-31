"""Chooses one compute framework per block of features (one block is one plan step group) before links resolve.
Arc consistency prunes the domains, then branch and bound with forward checking minimises conversions,
splitting into independent parts once assigned blocks separate them.
"""

from itertools import combinations
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import partial
from typing import Any, NamedTuple, get_origin
from uuid import UUID

from mloda.core.abstract_plugins.components.connection_requirement import ConnectionRequirement
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import ComputeFrameworkTransformer
from mloda.core.abstract_plugins.components.input_data.source_feature_group import SourceFeatureGroup
from mloda.core.abstract_plugins.components.link import JoinType, Link
from mloda.core.abstract_plugins.components.utils import is_nominal_subclass
from mloda.core.abstract_plugins.components.validators.feature_validator import FeatureValidator
from mloda.core.abstract_plugins.compute_framework import ComputeFramework, framework_rank_key
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.implementation import Implementation, serving_implementation
from mloda.core.abstract_plugins.operation_feature_group import OperationFeatureGroup
from mloda.core.prepare.declared_sides import (
    narrow_by_right_index,
    one_framework_members_reason,
    star_keeping_two_groups_reason,
)
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.stable_text import stable_text as stable_text

Framework = type[ComputeFramework]
Values = tuple[Framework, ...]


Cost = tuple[int, int, int]


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
READS_IN_PLACE = "reads in place"
LOADS_NATIVELY = "loads natively"
PREFERS_MATCHING = "prefers a matching version"
DIFFERS = "runs a version that differs"


def saves_conversions(count: int) -> str:
    return f"saves {count} conversion{'' if count == 1 else 's'}"


def conversion_cost(from_framework: Framework, to_framework: Framework) -> int:
    """Cost of one transform step between two frameworks."""
    return 1


def _add_connection(found: list[Any], conn: Any) -> bool:
    if conn in found:
        return False
    found.append(conn)
    return True


_SELF_PROVIDED = object()


def _label(conn: Any) -> str:
    if conn is _SELF_PROVIDED:
        return "a connection its parent brings itself"
    return repr(conn) if isinstance(conn, str) else f"a {type(conn).__name__} connection"


def _names(features: tuple[Feature, ...]) -> list[str]:
    return sorted(str(feature.name) for feature in features)


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


def _side_member_allows(relaxed: bool, values: Values) -> bool:
    """Members share a framework, the other side runs on one of theirs, or (relaxed) on the consumer's."""
    first, second, consumer, *others = values
    if first is second or any(o is first or o is second for o in others):
        return True
    return relaxed and all(o is consumer for o in others)


def _members_apart(values: Values) -> bool:
    return values[0] is not values[1]


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
        pin_frameworks: Mapping[UUID, frozenset[Framework]] | None = None,
        pinned: set[UUID] | None = None,
        implementations: Iterable[type[Implementation[Any]]] | None = None,
        handle_of: Callable[[Framework, Any], tuple[str | None, bool]] | None = None,
        dropped_reasons: Mapping[UUID, Mapping[Framework, str]] | None = None,
    ) -> None:
        self.implementations = list(implementations or ())
        self._differs: list[dict[Framework, tuple[str, ...]]] = []
        self._penalty: list[dict[Framework, int]] = []
        self._free: dict[int, Framework] = {}
        self._home: list[frozenset[Framework] | None] = []
        self._home_reason: list[str | None] = []
        self._homed = True
        self._unhomed: dict[int, Framework] = {}
        self.pin_frameworks = pin_frameworks or {}
        self.pinned = pinned or set()
        self.output_framework = output_framework
        self.dropped = dropped or {}
        self.handle_of = handle_of
        self.dropped_reasons = dropped_reasons or {}
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
        self._differs = [self._block_differs(block) for block in blocks]
        self._penalty = [
            {} if self._is_pinned(block) else {fw: 1 for fw, found in differs.items() if found}
            for block, differs in zip(blocks, self._differs)
        ]
        homes = [self._block_home(block) for block in blocks]
        self._home = [home for home, _ in homes]
        self._home_reason = [why for _, why in homes]
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
        linked = self._neighbor_map(len(blocks), rules, groups)
        touching, steps = self._component_indexes(len(blocks), rules, groups)
        for order in self._search_orders(len(blocks), linked):
            assignment, _ = self._solve(blocks, order, rules, linked, domains, touching, steps)
            self._free = {}
            self._unhomed = {}
            if any(self._home[i] is not None for i in assignment):
                self._homed = False
                self._unhomed, _ = self._solve(blocks, order, rules, linked, domains, touching, steps)
                self._homed = True
            if any(self._penalty[i] for i in assignment):
                penalty, self._penalty = self._penalty, [{} for _ in self._penalty]
                self._free, _ = self._solve(blocks, order, rules, linked, domains, touching, steps)
                self._penalty = penalty
            self._homed = False  # reasons never count the home charge as a saved conversion
            for index, framework in assignment.items():
                reason = self._reason(blocks, touching, steps, assignment, index)
                lost_why = self._lost_reasons(blocks[index], framework)
                if lost_why:
                    reason = f"{reason}; {lost_why}"
                for feature in blocks[index].features:
                    feature.chosen_compute_framework = framework
                    feature.chosen_compute_framework_reason = reason
            self._homed = True
        self._require_all_chosen()

    def _split_by_pin(self, group: Iterable[Feature]) -> list[list[Feature]]:
        """Features pinned to different framework sets cannot share a block."""
        parts: dict[frozenset[Framework] | None, list[Feature]] = {}
        for feature in group:
            parts.setdefault(self.pin_frameworks.get(feature.uuid), []).append(feature)
        return list(parts.values())

    def _block_differs(self, block: _Block) -> dict[Framework, tuple[str, ...]]:
        """Per framework, the declared divergences of the implementations serving the block's features."""
        if not self.implementations or not is_nominal_subclass(block.fg, OperationFeatureGroup):
            return {}
        result: dict[Framework, tuple[str, ...]] = {}
        for framework in block.domain:
            impl = serving_implementation(block.fg, framework, self.implementations)
            if impl is None:
                continue
            found = {impl.divergence(block.fg, str(f.name), f.options) for f in block.features} - {None}
            if found:
                result[framework] = tuple(sorted(r for r in found if r is not None))
        return result

    def _block_home(self, block: _Block) -> tuple[frozenset[Framework] | None, str | None]:
        """Frameworks that read the source in place, or on a default run take its neutral type, with the reason."""
        match = block.features[0].input_data_match
        if match is None or not is_nominal_subclass(block.fg, SourceFeatureGroup):
            return None, None
        conn = block.fg.match_connection(match[1])
        if conn is not None:
            home = frozenset(fw for fw in block.domain if fw._connection_matches(conn) and self._serves(block, fw))
            if home:
                return home, READS_IN_PLACE
        neutral = None if self.positions else block.fg.neutral_type(match[1])
        neutral = get_origin(neutral) or neutral
        if not isinstance(neutral, type):
            return None, None
        loads = frozenset(fw for fw in block.domain if fw._takes_as_is(neutral) or self._serves(block, fw))
        return (loads, LOADS_NATIVELY) if loads else (None, None)

    def _serves(self, block: _Block, framework: Framework) -> bool:
        return bool(serving_implementation(block.fg, framework, self.implementations, stop_after="load_neutral"))

    def _is_pinned(self, block: _Block) -> bool:
        return any(f.framework_pinned or f.uuid in self.pinned for f in block.features)

    def _lost_reasons(self, block: _Block, chosen: Framework) -> str:
        """Why other frameworks lost their connection for this block, for the reason of the chosen one."""
        found = {
            why
            for feature in block.features
            for framework, why in self.dropped_reasons.get(feature.uuid, {}).items()
            if framework is not chosen
        }
        return "; ".join(sorted(found))

    def _component_indexes(
        self, count: int, rules: list[_Rule], groups: dict[tuple[int, type[FeatureGroup]], set[int]]
    ) -> tuple[dict[int, list[_Rule]], dict[int, list[tuple[int, set[int], int]]]]:
        """Per block: the rules touching it and the transform steps it takes part in."""
        touching: dict[int, list[_Rule]] = {b: [] for b in range(count)}
        for rule in rules:
            for b in self._rule_blocks(rule):
                touching[b].append(rule)
        steps: dict[int, list[tuple[int, set[int], int]]] = {b: [] for b in range(count)}
        for key, children in groups.items():
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
            grouped = ExecutionPlan.group_features_by_compute_framework_and_options(set(features)).values()
            for group in (part for whole in grouped for part in self._split_by_pin(whole)):
                members = tuple(sorted(group, key=lambda f: (str(f.name), stable_text(f.options))))
                allowed = set(members[0].compute_frameworks or ())
                if members[0].uuid in self.pin_frameworks:
                    allowed &= self.pin_frameworks[members[0].uuid]
                domain = tuple(sorted(allowed, key=self.rank))
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
        """Dropped frameworks return when the parents share one connection; parents on two handles block a child."""
        frameworks = set().union(*self.dropped.values())
        for block in blocks:
            frameworks.update(
                fw for fw in block.domain if fw.connection_requirement() is ConnectionRequirement.REQUIRED
            )
        if not frameworks:
            return blocks, []
        parents: list[set[int]] = [set() for _ in blocks]
        for parent, child in self.graph.edges:
            if parent in owner and child in owner and owner[parent] != owner[child]:
                parents[owner[child]].add(owner[parent])
        carried, incoming = self._carried_connections(blocks, owner, parents, frameworks)
        domains = self._drop_conflicting(blocks, parents, incoming)
        gained = self._regain_fixpoint(blocks, owner, parents, carried, incoming, domains)
        widened = [
            block._replace(domain=tuple(sorted(domains[i], key=self.rank)))
            if domains[i] != set(block.domain)
            else block
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
            for framework in sorted(gained[child], key=self.rank)
            for parent in sorted(parents[child])
        ]
        return widened, rules

    def _carried_connections(
        self, blocks: list[_Block], owner: dict[UUID, int], parents: list[set[int]], frameworks: set[Framework]
    ) -> tuple[list[dict[Framework, list[Any]]], list[dict[Framework, list[Any]]]]:
        """Per block and framework, the connections its data carries and those its parents deliver."""
        host_of = {owner[tied]: owner[host] for host, tied in self.ties if host in owner and tied in owner}
        carried: list[dict[Framework, list[Any]]] = [{} for _ in blocks]
        incoming: list[dict[Framework, list[Any]]] = [{} for _ in blocks]
        implicit: list[dict[Framework, str | None]] = [{} for _ in blocks]
        own: set[tuple[int, Framework]] = set()
        roots: set[tuple[int, Framework]] = set()
        for index, block in enumerate(blocks):
            for framework in frameworks:
                keys: list[Any] = []
                handle, explicit = None, False
                if self.handle_of is not None and framework in block.domain:
                    handle, explicit = self.handle_of(framework, block.features[0].options)
                if explicit and handle is not None:
                    _add_connection(keys, handle)
                if keys:
                    own.add((index, framework))
                elif handle is not None and not parents[index]:
                    keys.append(handle)
                    roots.add((index, framework))
                carried[index][framework] = keys
                incoming[index][framework] = []
                implicit[index][framework] = None if explicit else handle
        changed = True
        while changed:
            changed = False
            for index, framework in ((i, f) for i in range(len(blocks)) for f in frameworks):
                sources = [carried[p][framework] for p in parents[index]]
                if index in host_of:
                    sources.append(carried[host_of[index]][framework])
                if not sources:
                    continue
                found = incoming[index][framework]
                handle = implicit[index][framework]
                bound = handle is not None and any(framework not in blocks[p].domain for p in parents[index])
                if bound:
                    sources.append([handle])
                for conn in (c for conns in sources for c in conns):
                    if _add_connection(found, conn) and (index, framework) not in own:
                        _add_connection(carried[index][framework], conn)
                        changed = True
                if bound and any((p, framework) in roots for p in parents[index]):
                    _add_connection(found, _SELF_PROVIDED)
        return carried, incoming

    def _drop_conflicting(
        self, blocks: list[_Block], parents: list[set[int]], incoming: list[dict[Framework, list[Any]]]
    ) -> list[set[Framework]]:
        """Domains without frameworks whose parents deliver two connections; a pinned block fails at planning."""
        domains = [set(block.domain) for block in blocks]
        for index, block in enumerate(blocks):
            if not parents[index]:
                continue
            pinned = any(f.framework_pinned for f in block.features)
            for framework in block.domain:
                found = incoming[index].get(framework, [])
                if len(found) < 2:
                    continue
                if pinned or len(block.domain) == 1:
                    labels = " and ".join(sorted(_label(c) for c in found))
                    raise ValueError(
                        f"Features {', '.join(_names(block.features))} can only run on {framework.get_class_name()}, "
                        f"but their parents deliver different connections: {labels}. "
                        "Select one with 'connection_handle' or choose another framework."
                    )
                domains[index].discard(framework)
                if not domains[index]:
                    names = " and ".join(sorted(f.get_class_name() for f in block.domain))
                    raise ValueError(
                        f"Features {', '.join(_names(block.features))} can run on none of {names}, "
                        "because their parents deliver different connections to each. "
                        "Select one with 'connection_handle' or choose another framework."
                    )
        return domains

    def _regain_fixpoint(
        self,
        blocks: list[_Block],
        owner: dict[UUID, int],
        parents: list[set[int]],
        carried: list[dict[Framework, list[Any]]],
        incoming: list[dict[Framework, list[Any]]],
        domains: list[set[Framework]],
    ) -> list[set[Framework]]:
        """Adds regained frameworks to the domains in place and returns what each block gained."""
        lost = [frozenset().union(*(self.dropped.get(f.uuid, frozenset()) for f in b.features)) for b in blocks]
        pinned = [self._is_pinned(b) for b in blocks]
        gained: list[set[Framework]] = [set() for _ in blocks]
        changed = True
        while changed:
            changed = False
            for index in range(len(blocks)):
                if pinned[index] or not parents[index]:
                    continue
                for framework in lost[index] - domains[index]:
                    if len(carried[index][framework]) > 1 or len(incoming[index][framework]) > 1:
                        continue
                    if all(framework in domains[p] for p in parents[index]):
                        domains[index].add(framework)
                        gained[index].add(framework)
                        changed = True
            for host, tied in self.ties:
                if host in owner and tied in owner and not pinned[owner[tied]]:
                    one, other = owner[host], owner[tied]
                    admitted = {f for f in lost[other] & domains[one] if carried[one][f]}
                    new = (gained[one] | admitted) - domains[other]
                    if new:
                        domains[other] |= new
                        gained[other] |= new
                        changed = True
        return gained

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
        return (0, sum(costs), sum(costs) * (self._horizon - depth))

    def _final_cost(self, block: _Block, framework: Framework, index: int) -> Cost:
        output = self.output_framework
        divergent = self._penalty[index].get(framework, 0)
        home = self._home[index]
        away = (0, 1, self._horizon + 1) if self._homed and home is not None and framework not in home else (0, 0, 0)
        if output is None or not self._requested(block) or framework is output:
            return (divergent, away[1], away[2])
        if framework.expected_data_framework() is output.expected_data_framework():
            return (divergent, away[1], away[2])
        cost = conversion_cost(framework, output)
        return (divergent, cost + away[1], cost * (self._horizon - self._final_depth[index] - 1) + away[2])

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
        parents, children = self._edge_maps()
        rules += self._side_path_rules(blocks, owner, parents, children)
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
        rules += self._side_member_rules(blocks, by_child, parents, children)
        return rules

    def _side_member_rules(
        self,
        blocks: list[_Block],
        by_child: dict[UUID, list[tuple[Link, tuple[int, int, int]]]],
        parents: Mapping[UUID, set[UUID]],
        children: Mapping[UUID, set[UUID]],
    ) -> list[_Rule]:
        """Members of one link side differ in framework only if both joins merge into the other side's frame."""
        direct: dict[tuple[UUID, UUID, int], bool] = {}

        def reads_directly(child_uuid: UUID, link_uuid: UUID, pos: int) -> bool:
            key = (child_uuid, link_uuid, pos)
            if key not in direct:
                direct[key] = True
                for link, left_uuid, right_uuid, occ_child in self.occurrences:
                    if occ_child == child_uuid and link.uuid == link_uuid:
                        side = (left_uuid, right_uuid)[pos]
                        if self._reach(side, children) & self._reach(child_uuid, parents):
                            direct[key] = False
            return direct[key]

        rules: dict[tuple[object, ...], _Rule] = {}
        for occurrences in by_child.values():
            joinable = [(k, j) for k, j in occurrences if k.jointype not in (JoinType.APPEND, JoinType.UNION)]
            for link_uuid in {k.uuid for k, _ in joinable}:
                found = [j for k, j in joinable if k.uuid == link_uuid]
                jointype = next(k.jointype for k, _ in joinable if k.uuid == link_uuid)
                for pos in (0, 1):
                    members = sorted({j[pos] for j in found}, key=lambda b: (blocks[b].fg.__name__, b))
                    others = tuple(sorted({j[1 - pos] for j in found}))
                    consumer = found[0][2]
                    offers = (
                        jointype in (JoinType.INNER, JoinType.OUTER)
                        or jointype
                        is (
                            JoinType.LEFT,
                            JoinType.RIGHT,
                        )[1 - pos]
                    )
                    relaxed: bool | None = None
                    link_of = next(k for k, _ in joinable if k.uuid == link_uuid)
                    unresolved = (
                        narrow_by_right_index(link_of, members, lambda m: blocks[m].features) if pos == 1 else []
                    ) or members
                    for a, b in combinations(members, 2):
                        if blocks[a].fg is blocks[b].fg:
                            continue
                        if a in unresolved and b in unresolved:
                            rules.setdefault(
                                (a, b, consumer, "apart"),
                                _Rule(
                                    (a, b),
                                    _members_apart,
                                    one_framework_members_reason(
                                        blocks[consumer].fg.__name__,
                                        link_of,
                                        blocks[a].fg,
                                        blocks[b].fg,
                                    ),
                                ),
                            )
                        if relaxed is None:
                            relaxed = (
                                offers
                                and len(others) == 1
                                and all(reads_directly(f.uuid, link_uuid, pos) for f in blocks[consumer].features)
                            )
                        name = blocks[consumer].fg.__name__
                        why = (
                            f"{name} joins {blocks[a].fg.__name__} and {blocks[b].fg.__name__}, members of one link "
                            "side, which can run on different frameworks only when both joins merge into the other "
                            "link side's frame: the other link side runs on one member's framework"
                        )
                        if not offers and jointype in (JoinType.LEFT, JoinType.RIGHT):
                            why = star_keeping_two_groups_reason(
                                name,
                                blocks[a].fg.__name__,
                                blocks[b].fg.__name__,
                                " and ".join(sorted({blocks[o].fg.__name__ for o in others})),
                                (jointype,),
                            )
                        else:
                            if offers:
                                why += (
                                    f", or {name} reads both members directly and runs on the other link side's "
                                    "framework"
                                )
                            why += ". Otherwise read one member of the link side per consumer"
                        blocks_of = (a, b, consumer, *others)
                        rules.setdefault(
                            (*blocks_of, relaxed), _Rule(blocks_of, partial(_side_member_allows, relaxed), why)
                        )
        return list(rules.values())

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

    def _edge_maps(self) -> tuple[dict[UUID, set[UUID]], dict[UUID, set[UUID]]]:
        parents: dict[UUID, set[UUID]] = {}
        children: dict[UUID, set[UUID]] = {}
        for parent, child in self.graph.edges:
            parents.setdefault(child, set()).add(parent)
            children.setdefault(parent, set()).add(child)
        return parents, children

    def _side_path_rules(
        self,
        blocks: list[_Block],
        owner: dict[UUID, int],
        parents: Mapping[UUID, set[UUID]],
        children: Mapping[UUID, set[UUID]],
    ) -> list[_Rule]:
        """A feature between a link side and its consumer shares the side's framework, unless it is a hoppable carrier."""
        if not self.occurrences:
            return []
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
    def _search_orders(count: int, neighbors: list[set[int]]) -> list[list[int]]:
        """Per connected component, blocks breadth-first from the first in content order (neighbours in content order)."""
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
        found = self._differs[index].get(chosen, ())
        suffix = f"; {DIFFERS}: {', '.join(found)}" if found else ""
        if self._is_pinned(block):
            return PINNED + suffix
        if len(block.domain) == 1:
            return ONLY_ALLOWED + suffix
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
            return (
                sum(c[0] for c in costs) + final[0],
                sum(c[1] for c in costs) + final[1],
                sum(c[2] for c in costs) + final[2],
            )

        alternatives = [fw for fw in block.domain if fw is not chosen and feasible(fw)]
        if not alternatives:
            return RULES + suffix
        chosen_rank = self.rank(chosen)
        earlier = [fw for fw in alternatives if self.rank(fw) < chosen_rank]
        base = step_cost(None)
        if self._unhomed.get(index, chosen) is not chosen:
            moved = [
                b
                for b in assignment
                if self._home_reason[b] == READS_IN_PLACE and self._unhomed[b] is not assignment[b]
            ]
            return (self._home_reason[index] or (READS_IN_PLACE if moved else LOADS_NATIVELY)) + suffix
        free = self._free.get(index)
        if free is not None and free is not chosen and not found and self._differs[index].get(free):
            return PREFERS_MATCHING
        if not earlier:
            return self._order_reason(chosen, alternatives[0]) + suffix
        costs = {fw: step_cost(fw) for fw in earlier}
        delta = min(c[1] for c in costs.values()) - base[1]
        if delta > 0:
            return saves_conversions(delta) + suffix
        if all(c[2] > base[2] for c in costs.values() if c[1] == base[1]):
            return CONVERTS_LATER + suffix
        return self._order_reason(chosen, earlier[0]) + suffix

    def _solve(
        self,
        blocks: list[_Block],
        order: list[int],
        rules: list[_Rule],
        linked: list[set[int]],
        domains: list[list[Framework]],
        touching: dict[int, list[_Rule]],
        steps: dict[int, list[tuple[int, set[int], int]]],
    ) -> tuple[dict[int, Framework], Cost]:
        """Branch and bound; returns the best assignment and its cost, raising when no assignment is feasible."""
        search_order = [b for b in order if len(domains[b]) == 1] + [b for b in order if len(domains[b]) != 1]
        position = {b: i for i, b in enumerate(search_order)}
        assigned: dict[int, Framework] = {}

        def step_cost(parent: int, children: set[int], depth: int) -> Cost:
            if parent not in assigned:
                return (0, 0, 0)
            moved = {assigned[c] for c in children if c in assigned and assigned[c] is not assigned[parent]}
            return self._transform_cost(depth, [conversion_cost(assigned[parent], target) for target in moved])

        def total(indices: Sequence[tuple[int, set[int], int]]) -> Cost:
            costs = [step_cost(*step) for step in indices]
            return (sum(c[0] for c in costs), sum(c[1] for c in costs), sum(c[2] for c in costs))

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
            root = open_frame(0, start, (0, 0, 0))
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
                new_cost = (
                    cost[0] + after[0] - before[0] + final[0],
                    cost[1] + after[1] - before[1] + final[1],
                    cost[2] + after[2] - before[2] + final[2],
                )
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
                    new_cost = (new_cost[0] + sub_cost[0], new_cost[1] + sub_cost[1], new_cost[2] + sub_cost[2])
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
            members = set(order)
            raise ValueError(self._infeasible(blocks, order, [r for r in rules if r.blocks[0] in members]))
        return result

    @staticmethod
    def _infeasible(blocks: list[_Block], order: list[int], rules: list[_Rule]) -> str:
        features = [
            f"{feature.name} (allowed: {sorted(fw.get_class_name() for fw in blocks[i].domain)})"
            for i in order
            for feature in blocks[i].features
        ]
        involved = [
            r
            for r in rules
            if not (
                all(len(blocks[b].domain) == 1 for b in r.blocks)
                and r.allows(tuple(blocks[b].domain[0] for b in r.blocks))
            )
        ]
        return (
            "No compute framework assignment satisfies the hard rules for features "
            f"{', '.join(features)}. Rules involved: {sorted({r.why for r in involved})}."
        )
