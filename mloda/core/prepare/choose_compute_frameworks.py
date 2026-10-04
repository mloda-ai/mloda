"""Chooses one compute framework per block of features (one block is one plan step group) before links resolve.
Exact branch and bound over hard rules (domain, join, side, filter, transform path), minimising conversions.
"""

from collections.abc import Callable, Mapping
from functools import partial
from typing import NamedTuple
from uuid import UUID

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import ComputeFrameworkTransformer
from mloda.core.abstract_plugins.components.link import JoinType, Link
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


def conversion_cost(from_framework: Framework, to_framework: Framework) -> int:
    """Cost of one transform step between two frameworks."""
    return 1


def _names(features: tuple[Feature, ...]) -> list[str]:
    return sorted(str(feature.name) for feature in features)


def _block_key(fg: type[FeatureGroup], features: tuple[Feature, ...]) -> tuple[object, ...]:
    allowed = sorted(f.get_class_name() for f in features[0].compute_frameworks or ())
    return (fg.__module__, fg.__qualname__, _names(features), allowed, sorted(str(f.options) for f in features))


def _join_allows(jointype: JoinType, values: Values) -> bool:
    left, right, child = values
    if jointype is JoinType.RIGHT:
        return child is right
    if jointype in (JoinType.APPEND, JoinType.UNION):
        return child is left
    return child is left or child is right


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
    ) -> None:
        self.graph = graph
        self.nodes = nodes_per_feature_group
        self.occurrences = link_occurrences
        self.ties = filter_ties
        self.rank = framework_rank_key(positions)
        self.transformer = ComputeFrameworkTransformer()
        self.paths: dict[tuple[Framework, Framework], bool] = {}

    def choose(self) -> None:
        """Solve each connected component exactly and write the result onto the features."""
        blocks = self._blocks()
        owner = {f.uuid: i for i, block in enumerate(blocks) for f in block.features}
        rules = self._rules(blocks, owner)
        groups = self._cost_groups(blocks, owner)
        for order in self._search_orders(len(blocks), rules, groups):
            members = set(order)
            assignment = self._solve(blocks, order, [r for r in rules if r.blocks[0] in members], groups)
            for index, framework in assignment.items():
                for feature in blocks[index].features:
                    feature.chosen_compute_framework = framework

    def _blocks(self) -> list[_Block]:
        keyed: list[tuple[tuple[object, ...], _Block]] = []
        for fg, features in self.nodes.items():
            for group in ExecutionPlan.group_features_by_compute_framework_and_options(set(features)).values():
                members = tuple(sorted(group, key=lambda f: (str(f.name), str(f.options))))
                domain = tuple(sorted(members[0].compute_frameworks or (), key=self.rank))
                keyed.append((_block_key(fg, members), _Block(fg, members, domain)))
        return [block for _, block in sorted(keyed, key=lambda pair: pair[0])]

    def _convertible(self, source: Framework, target: Framework) -> bool:
        if source is target:
            return True
        pair = (source, target)
        if pair not in self.paths:
            src, dst = source.expected_data_framework(), target.expected_data_framework()
            self.paths[pair] = src is dst or self.transformer.get_transformation_chain(src, dst) is not None
        return self.paths[pair]

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
        for host, tied in self.ties:
            if host in owner and tied in owner:
                rules.append(_Rule((owner[host], owner[tied]), lambda v: v[0] is v[1], "filter tied to its host"))
        by_link: dict[UUID, list[tuple[int, int, int]]] = {}
        for link, left_uuid, right_uuid, child_uuid in self.occurrences:
            joined = (owner[left_uuid], owner[right_uuid], owner[child_uuid])
            jointype = link.jointype
            rules.append(_Rule(joined, partial(_join_allows, jointype), f"join {jointype.value}"))
            if jointype not in (JoinType.RIGHT, JoinType.APPEND, JoinType.UNION):
                by_link.setdefault(link.uuid, []).append(joined)
        for joins in by_link.values():
            for i, first in enumerate(joins):
                for second in joins[i + 1 :]:
                    if first[2] != second[2]:
                        rules.append(_Rule(first + second, _same_side, "one side per link"))
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

    def _solve(
        self,
        blocks: list[_Block],
        order: list[int],
        rules: list[_Rule],
        groups: dict[tuple[int, type[FeatureGroup]], set[int]],
    ) -> dict[int, Framework]:
        position = {b: k for k, b in enumerate(order)}
        triggers: list[list[_Rule]] = [[] for _ in order]
        for rule in rules:
            triggers[max(position[b] for b in rule.blocks)].append(rule)
        local = [(parent, children) for (parent, _), children in groups.items() if parent in position]
        assigned: dict[int, Framework] = {}
        best: dict[int, Framework] = {}
        best_cost = [-1]

        def cost() -> int:
            total = 0
            for parent, children in local:
                if parent in assigned:
                    moved = {assigned[c] for c in children if c in assigned and assigned[c] is not assigned[parent]}
                    total += sum(conversion_cost(assigned[parent], target) for target in moved)
            return total

        def descend(depth: int) -> None:
            if depth == len(order):
                best.update(assigned)
                best_cost[0] = cost()
                return
            index = order[depth]
            for value in blocks[index].domain:
                assigned[index] = value
                holds = all(r.allows(tuple(assigned[b] for b in r.blocks)) for r in triggers[depth])
                if holds and (best_cost[0] < 0 or cost() < best_cost[0]):
                    descend(depth + 1)
                del assigned[index]

        descend(0)
        if best_cost[0] < 0:
            raise ValueError(self._infeasible(blocks, order, rules))
        return best

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
