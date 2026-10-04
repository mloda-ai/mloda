"""Chooses one compute framework per block of features (one block is one plan step group) before links resolve.
Arc consistency prunes the domains, then branch and bound with forward checking minimises conversions.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import NamedTuple
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


class _Block(NamedTuple):
    fg: type[FeatureGroup]
    features: tuple[Feature, ...]
    domain: Values


class _Rule(NamedTuple):
    blocks: tuple[int, ...]
    allows: Callable[[Values], bool]
    why: str


_UNFLIPPABLE = (JoinType.RIGHT, JoinType.APPEND, JoinType.UNION)

PINNED = "pinned"
ONLY_ALLOWED = "only allowed framework"
RULES = "rules exclude preferred frameworks"
LIST_ORDER = "your list order"
DEFAULT_ORDER = "default order"


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


def _join_allows(jointype: JoinType, values: Values) -> bool:
    left, right, child = values
    if jointype is JoinType.RIGHT:
        return child is right
    if jointype in (JoinType.APPEND, JoinType.UNION):
        return child is left
    return child is left or child is right


def _any_join_allows(occurrences: list[tuple[JoinType, bool]], values: Values) -> bool:
    """One link must be satisfied; a same-framework cross-group link may not be skipped (it would self-merge)."""
    satisfied = [_join_allows(jt, values[3 * i : 3 * i + 3]) for i, (jt, _) in enumerate(occurrences)]
    for i, (_, strict) in enumerate(occurrences):
        if strict and values[3 * i] is values[3 * i + 1] and not satisfied[i]:
            return False
    return any(satisfied)


def _swapped_pairs_differ(values: Values) -> bool:
    """Children whose parents sit on swapped frameworks join apart, or their flipped keys would merge."""
    left_a, right_a, child_a, left_b, right_b, child_b = values
    if left_a is right_a or left_a is not right_b or right_a is not left_b:
        return True
    return child_a is not child_b


def _same_side(values: Values) -> bool:
    left_a, right_a, child_a, left_b, right_b, child_b = values
    if (left_a, right_a) != (left_b, right_b) or left_a is right_a:
        return True
    return (child_a is left_a) == (child_b is left_b)


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
    ) -> None:
        self.output_framework = output_framework
        self.graph = graph
        self.nodes = nodes_per_feature_group
        self.occurrences = link_occurrences
        self.ties = filter_ties
        self.positions = positions
        self.rank = framework_rank_key(positions)
        self.transformer = ComputeFrameworkTransformer()
        self.paths: dict[tuple[Framework, Framework], bool] = {}

    def choose(self) -> None:
        """Solve each connected component exactly and write the result onto the features."""
        blocks = self._blocks()
        owner = {f.uuid: i for i, block in enumerate(blocks) for f in block.features}
        rules = self._rules(blocks, owner)
        groups = self._cost_groups(blocks, owner)
        domains = self._prune(blocks, rules)
        for order in self._search_orders(len(blocks), rules, groups):
            members = set(order)
            local_rules = [r for r in rules if r.blocks[0] in members]
            assignment, _ = self._solve(blocks, order, local_rules, groups, domains)
            for index, framework in assignment.items():
                reason = self._reason(blocks, local_rules, groups, assignment, index)
                for feature in blocks[index].features:
                    feature.chosen_compute_framework = framework
                    feature.chosen_compute_framework_reason = reason
        self._require_all_chosen()

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

    def _final_cost(self, block: _Block, framework: Framework) -> int:
        output = self.output_framework
        if output is None or not self._requested(block) or framework is output:
            return 0
        if framework.expected_data_framework() is output.expected_data_framework():
            return 0
        return conversion_cost(framework, output)

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
        by_link: dict[UUID, list[tuple[int, int, int]]] = {}
        by_child: dict[UUID, list[tuple[Link, tuple[int, int, int]]]] = {}
        for link, left_uuid, right_uuid, child_uuid in self.occurrences:
            joined = (owner[left_uuid], owner[right_uuid], owner[child_uuid])
            jointype = link.jointype
            by_child.setdefault(child_uuid, []).append((link, joined))
            if jointype not in _UNFLIPPABLE:
                by_link.setdefault(link.uuid, []).append(joined)
        for occurrences in by_child.values():
            jointypes = [link.jointype for link, _ in occurrences]
            kinds = [
                (
                    link.jointype,
                    link.left_feature_group != link.right_feature_group
                    and link.jointype not in (JoinType.APPEND, JoinType.UNION),
                )
                for link, _ in occurrences
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
        return rules

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
        neighbors: list[set[int]] = [set() for _ in range(count)]
        for members in [r.blocks for r in rules] + [(parent, *kids) for (parent, _), kids in groups.items()]:
            for one in members:
                neighbors[one].update(members)
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
        rules: list[_Rule],
        groups: dict[tuple[int, type[FeatureGroup]], set[int]],
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
        touching = [r for r in rules if index in r.blocks]
        steps = [(parent, kids) for (parent, _), kids in groups.items() if index == parent or index in kids]

        def feasible(framework: Framework) -> bool:
            switched = {**assignment, index: framework}
            return all(self._allows(rule, switched) for rule in touching)

        def step_cost(values: Mapping[int, Framework]) -> int:
            return sum(
                conversion_cost(values[parent], target)
                for parent, kids in steps
                for target in {values[c] for c in kids if values[c] is not values[parent]}
            ) + self._final_cost(block, values[index])

        alternatives = [fw for fw in block.domain if fw is not chosen and feasible(fw)]
        if not alternatives:
            return RULES
        chosen_rank = self.rank(chosen)
        earlier = [fw for fw in alternatives if self.rank(fw) < chosen_rank]
        if not earlier:
            return self._order_reason(chosen, alternatives[0])
        base = step_cost(assignment)
        delta = min(step_cost({**assignment, index: fw}) for fw in earlier) - base
        if delta > 0:
            return saves_conversions(delta)
        return self._order_reason(chosen, earlier[0])

    def _solve(
        self,
        blocks: list[_Block],
        order: list[int],
        rules: list[_Rule],
        groups: dict[tuple[int, type[FeatureGroup]], set[int]],
        domains: list[list[Framework]],
    ) -> tuple[dict[int, Framework], int]:
        """Branch and bound; returns the best assignment and its cost, raising when no assignment is feasible."""
        touching: dict[int, list[_Rule]] = {b: [] for b in order}
        for rule in rules:
            for b in self._rule_blocks(rule):
                touching[b].append(rule)
        local = [(parent, children) for (parent, _), children in groups.items() if parent in touching]
        steps: dict[int, list[tuple[int, set[int]]]] = {b: [] for b in order}
        for parent, children in local:
            for b in (parent, *children):
                steps[b].append((parent, children))
        assigned: dict[int, Framework] = {}
        best: dict[int, Framework] = {}
        best_cost = [-1]

        def step_cost(parent: int, children: set[int]) -> int:
            if parent not in assigned:
                return 0
            moved = {assigned[c] for c in children if c in assigned and assigned[c] is not assigned[parent]}
            return sum(conversion_cost(assigned[parent], target) for target in moved)

        def narrow(index: int, current: dict[int, list[Framework]]) -> dict[int, list[Framework]] | None:
            """Forward check: values of the one block still open in a rule must be supported by the assignment."""
            narrowed = dict(current)
            for rule in touching[index]:
                pending = [b for b in self._rule_blocks(rule) if b not in assigned]
                if not pending:
                    if not self._allows(rule, assigned):
                        return None
                elif len(pending) == 1:
                    open_block = pending[0]
                    kept = [v for v in narrowed[open_block] if self._allows(rule, {**assigned, open_block: v})]
                    if not kept:
                        return None
                    narrowed[open_block] = kept
            return narrowed

        def descend(depth: int, current: dict[int, list[Framework]], cost: int) -> None:
            if depth == len(order):
                best.update(assigned)
                best_cost[0] = cost
                return
            index = order[depth]
            before = sum(step_cost(*step) for step in steps[index])
            for value in current[index]:
                assigned[index] = value
                narrowed = narrow(index, current)
                new_cost = cost + sum(step_cost(*step) for step in steps[index]) - before
                new_cost += self._final_cost(blocks[index], value)
                if narrowed is not None and (best_cost[0] < 0 or new_cost < best_cost[0]):
                    descend(depth + 1, narrowed, new_cost)
                del assigned[index]

        descend(0, {b: domains[b] for b in order}, 0)
        if best_cost[0] < 0:
            raise ValueError(self._infeasible(blocks, order, rules))
        return best, best_cost[0]

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
