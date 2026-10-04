"""ChooseComputeFrameworks as a unit: hard rules, conversion cost, tie order, determinism.
Graphs are built by hand; no run or Engine is involved.
"""

import itertools
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare import choose_compute_frameworks
from mloda.core.prepare.choose_compute_frameworks import ChooseComputeFrameworks
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import EdgeProperties, NodeProperties
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.helpers.probe_runner import run_probes

P = PandasDataFrame
A = PyArrowTable
D = PythonDictFramework

# Reason texts are a plan-lock contract, so the tests spell them out instead of importing the constants.
PINNED = "pinned"
ONLY_ALLOWED = "only allowed framework"
RULES = "rules exclude preferred frameworks"
LIST_ORDER = "your list order"
DEFAULT_ORDER = "default order"
SAVES_ONE = "saves 1 conversion"
SAVES_TWO = "saves 2 conversions"

_PROBE = Path(__file__).with_name("choose_probe.py")
_PROBE_EXPECTED = {
    "x": "PandasDataFrame",
    "y": "PandasDataFrame",
    "kp": "PandasDataFrame",
    "ka": "PyArrowTable",
    "reason_x": "default order",
    "reason_y": "default order",
    "reason_kp": "only allowed framework",
    "reason_ka": "only allowed framework",
}


class ChooserRootFG(FeatureGroup):
    pass


class ChooserLeafFG(FeatureGroup):
    pass


class ChooserOtherLeafFG(FeatureGroup):
    pass


class ChooserSinkFG(FeatureGroup):
    pass


class ChooserOtherSinkFG(FeatureGroup):
    pass


class ChooserLeftFG(FeatureGroup):
    pass


class ChooserRightFG(FeatureGroup):
    pass


class ChooserChildFG(FeatureGroup):
    pass


class ChooserHostFG(FeatureGroup):
    pass


class ChooserFilterFG(FeatureGroup):
    pass


class ChooserLayerFG(FeatureGroup):
    pass


class _Net:
    """Hand-built plan input: graph, FG buckets, link occurrences and filter ties."""

    def __init__(self) -> None:
        self.graph = Graph()
        self.nodes: dict[type[FeatureGroup], set[Feature]] = {}
        self.fg_of: dict[UUID, type[FeatureGroup]] = {}
        self.features: list[Feature] = []
        self.occurrences: list[tuple[Link, UUID, UUID, UUID]] = []
        self.ties: list[tuple[UUID, UUID]] = []

    def add(
        self,
        fg: type[FeatureGroup],
        name: str,
        allowed: set[type[ComputeFramework]] | None,
        options: dict[str, Any] | None = None,
        data_type: DataType | None = None,
        pinned: bool = False,
        requested: bool = False,
    ) -> Feature:
        feature = Feature(name, options=options, data_type=data_type)
        feature.initial_requested_data = requested
        feature.framework_pinned = pinned
        feature.compute_frameworks = None if allowed is None else set(allowed)
        self.graph.add_node(feature.uuid, NodeProperties(feature, fg))
        self.nodes.setdefault(fg, set()).add(feature)
        self.fg_of[feature.uuid] = fg
        self.features.append(feature)
        return feature

    def edge(self, parent: Feature, child: Feature) -> None:
        props = EdgeProperties(self.fg_of[parent.uuid], self.fg_of[child.uuid])
        self.graph.add_edge(parent.uuid, child.uuid, props)

    def join(self, link: Link, left: Feature, right: Feature, child: Feature) -> None:
        self.edge(left, child)
        if right is not left:
            self.edge(right, child)
        self.occurrences.append((link, left.uuid, right.uuid, child.uuid))

    def tie(self, host: Feature, filter_feature: Feature) -> None:
        self.ties.append((host.uuid, filter_feature.uuid))

    def orphan(self, fg: type[FeatureGroup], name: str, allowed: set[type[ComputeFramework]]) -> Feature:
        """A graph node the FG buckets do not list."""
        feature = Feature(name)
        feature.compute_frameworks = set(allowed)
        self.graph.add_node(feature.uuid, NodeProperties(feature, fg))
        return feature

    def chooser(
        self,
        positions: Mapping[type[ComputeFramework], int] | None = None,
        chooser_class: type[ChooseComputeFrameworks] = ChooseComputeFrameworks,
        output_framework: type[ComputeFramework] | None = None,
    ) -> ChooseComputeFrameworks:
        return chooser_class(
            self.graph,
            self.nodes,
            self.occurrences,
            self.ties,
            dict(positions) if positions else {},
            output_framework=output_framework,
        )

    def choose(
        self,
        positions: Mapping[type[ComputeFramework], int] | None = None,
        chooser_class: type[ChooseComputeFrameworks] = ChooseComputeFrameworks,
        output_framework: type[ComputeFramework] | None = None,
    ) -> None:
        self.chooser(positions, chooser_class, output_framework).choose()

    def conversions(self) -> int:
        """Graph edges whose ends ended up on different frameworks."""
        by_uuid = {feature.uuid: feature for feature in self.features}
        return sum(
            1
            for parent, child in self.graph.edges
            if by_uuid[parent].chosen_compute_framework is not by_uuid[child].chosen_compute_framework
        )


def _link(factory: Callable[[JoinSpec, JoinSpec], Link], left: type[FeatureGroup], right: type[FeatureGroup]) -> Link:
    return factory(JoinSpec(left, "idx"), JoinSpec(right, "idx"))


def _throwaway_pair(shared_expected: bool) -> tuple[type[ComputeFramework], type[ComputeFramework]]:
    """Unavailable frameworks with no transformer path unless they share expected_data_framework."""

    class _ExpectedOne:
        pass

    class _ExpectedTwo:
        pass

    class ZzPathOneThrowawayFramework(ComputeFramework):
        @staticmethod
        def is_available() -> bool:
            return False

        @classmethod
        def expected_data_framework(cls) -> type:
            return _ExpectedOne

    class ZzPathTwoThrowawayFramework(ComputeFramework):
        @staticmethod
        def is_available() -> bool:
            return False

        @classmethod
        def expected_data_framework(cls) -> type:
            return _ExpectedOne if shared_expected else _ExpectedTwo

    return ZzPathOneThrowawayFramework, ZzPathTwoThrowawayFramework


# --- output contract and tie order ------------------------------------------------------------------------


def test_choose_sets_a_framework_on_every_feature() -> None:
    net = _Net()
    root = net.add(ChooserRootFG, "choose_every_root", {P, A})
    leaf = net.add(ChooserLeafFG, "choose_every_leaf", {P, A})
    net.edge(root, leaf)

    net.choose()

    assert root.chosen_compute_framework is not None
    assert leaf.chosen_compute_framework is not None


@pytest.mark.parametrize(
    ("positions", "expected", "reason"),
    [
        (None, P, DEFAULT_ORDER),
        ({A: 0}, A, LIST_ORDER),
        ({D: 0, A: 1}, D, LIST_ORDER),
        ({A: 0, D: 1}, A, LIST_ORDER),
    ],
    ids=["default_order", "list_first", "list_first_of_three", "listed_before_unlisted"],
)
def test_a_free_feature_follows_positions_then_default_order(
    positions: dict[type[ComputeFramework], int] | None, expected: type[ComputeFramework], reason: str
) -> None:
    net = _Net()
    feature = net.add(ChooserRootFG, "tie_order_feature", {P, A, D})

    net.choose(positions)

    assert feature.chosen_compute_framework is expected
    assert feature.chosen_compute_framework_reason == reason


def test_connected_free_blocks_share_one_framework() -> None:
    net = _Net()
    root = net.add(ChooserRootFG, "connected_root", {P, A, D})
    leaf = net.add(ChooserLeafFG, "connected_leaf", {P, A, D})
    net.edge(root, leaf)

    net.choose({D: 0})

    assert root.chosen_compute_framework is D
    assert leaf.chosen_compute_framework is D
    assert net.conversions() == 0


# --- rule a and the P1 / P3 shapes -----------------------------------------------------------------------


def test_value_stays_inside_the_allowed_set() -> None:
    net = _Net()
    feature = net.add(ChooserRootFG, "domain_feature", {A, D})

    net.choose({P: 0})

    assert feature.chosen_compute_framework in {A, D}


def test_consumer_pulls_an_unrestricted_source_onto_its_framework() -> None:
    net = _Net()
    root = net.add(ChooserRootFG, "p1_root", {P, A})
    consumer = net.add(ChooserLeafFG, "p1_consumer", {A})
    net.edge(root, consumer)

    net.choose()

    assert root.chosen_compute_framework is A
    assert consumer.chosen_compute_framework is A
    assert net.conversions() == 0
    assert root.chosen_compute_framework_reason == SAVES_ONE
    assert consumer.chosen_compute_framework_reason == ONLY_ALLOWED


def test_restricted_source_pulls_an_unrestricted_consumer_onto_its_framework() -> None:
    net = _Net()
    source = net.add(ChooserRootFG, "p3_source", {A})
    consumer = net.add(ChooserLeafFG, "p3_consumer", {P, A})
    net.edge(source, consumer)

    net.choose()

    assert source.chosen_compute_framework is A
    assert consumer.chosen_compute_framework is A
    assert net.conversions() == 0
    assert source.chosen_compute_framework_reason == ONLY_ALLOWED
    assert consumer.chosen_compute_framework_reason == SAVES_ONE


def test_a_restricted_consumer_of_two_free_roots_pulls_a_link_onto_its_framework() -> None:
    net = _Net()
    left = net.add(ChooserLeftFG, "p2_left", {P, A})
    right = net.add(ChooserRightFG, "p2_right", {P, A})
    child = net.add(ChooserChildFG, "p2_child", {A})
    net.join(_link(Link.inner, ChooserLeftFG, ChooserRightFG), left, right, child)

    net.choose()

    assert {left.chosen_compute_framework, right.chosen_compute_framework, child.chosen_compute_framework} == {A}


# --- rule b: join agreement ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory", "left_fw", "right_fw", "positions", "expected"),
    [
        (Link.inner, P, A, None, P),
        (Link.inner, P, A, {A: 0}, A),
        (Link.inner, A, P, None, P),
        (Link.left, P, A, None, P),
        (Link.outer, P, A, {A: 0}, A),
        (Link.right, P, A, None, A),
        (Link.right, A, P, {A: 0}, P),
        (Link.append, P, A, {A: 0}, P),
        (Link.append, A, P, None, A),
        (Link.union, P, A, {A: 0}, P),
        (Link.union, A, P, None, A),
    ],
    ids=[
        "inner_tie_default",
        "inner_tie_positions",
        "inner_swapped_tie_default",
        "left_tie_default",
        "outer_tie_positions",
        "right_is_right_side",
        "right_beats_positions",
        "append_is_left_side",
        "append_swapped",
        "union_is_left_side",
        "union_swapped",
    ],
)
def test_child_framework_follows_the_join_rule(
    factory: Callable[[JoinSpec, JoinSpec], Link],
    left_fw: type[ComputeFramework],
    right_fw: type[ComputeFramework],
    positions: dict[type[ComputeFramework], int] | None,
    expected: type[ComputeFramework],
) -> None:
    net = _Net()
    left = net.add(ChooserLeftFG, "join_rule_left", {left_fw})
    right = net.add(ChooserRightFG, "join_rule_right", {right_fw})
    child = net.add(ChooserChildFG, "join_rule_child", {P, A, D})
    net.join(_link(factory, ChooserLeftFG, ChooserRightFG), left, right, child)

    net.choose(positions)

    assert left.chosen_compute_framework is left_fw
    assert right.chosen_compute_framework is right_fw
    assert child.chosen_compute_framework is expected


def test_a_child_outside_both_join_sides_is_infeasible() -> None:
    net = _Net()
    left = net.add(ChooserLeftFG, "outside_left", {P})
    right = net.add(ChooserRightFG, "outside_right", {A})
    child = net.add(ChooserChildFG, "outside_child", {D})
    net.join(_link(Link.inner, ChooserLeftFG, ChooserRightFG), left, right, child)

    with pytest.raises(ValueError, match="outside_child"):
        net.choose()


@pytest.mark.parametrize(
    ("factory", "child_allowed"),
    [(Link.right, {P}), (Link.append, {A}), (Link.union, {A})],
    ids=["right_needs_right_side", "append_needs_left_side", "union_needs_left_side"],
)
def test_a_non_flippable_join_rejects_the_other_side(
    factory: Callable[[JoinSpec, JoinSpec], Link], child_allowed: set[type[ComputeFramework]]
) -> None:
    net = _Net()
    left = net.add(ChooserLeftFG, "rigid_left", {P})
    right = net.add(ChooserRightFG, "rigid_right", {A})
    child = net.add(ChooserChildFG, "rigid_child", child_allowed)
    net.join(_link(factory, ChooserLeftFG, ChooserRightFG), left, right, child)

    with pytest.raises(ValueError, match="rigid_child"):
        net.choose()


@pytest.mark.parametrize(
    ("factory", "expected"),
    [(Link.inner, P), (Link.right, A), (Link.append, P)],
    ids=["inner", "right", "append"],
)
def test_a_self_join_follows_the_same_rule(
    factory: Callable[[JoinSpec, JoinSpec], Link], expected: type[ComputeFramework]
) -> None:
    net = _Net()
    left = net.add(ChooserLeftFG, "self_join_left", {P}, {"side": 1})
    right = net.add(ChooserLeftFG, "self_join_right", {A}, {"side": 2})
    child = net.add(ChooserChildFG, "self_join_child", {P, A})
    net.join(_link(factory, ChooserLeftFG, ChooserLeftFG), left, right, child)

    net.choose()

    assert child.chosen_compute_framework is expected


# --- rule c: side agreement ------------------------------------------------------------------------------


def test_children_of_one_link_with_the_same_side_pair_pick_the_same_side() -> None:
    link = _link(Link.inner, ChooserLeftFG, ChooserRightFG)
    net = _Net()
    left_one = net.add(ChooserLeftFG, "side_left", {P}, {"n": 1})
    left_two = net.add(ChooserLeftFG, "side_left", {P}, {"n": 2})
    right_one = net.add(ChooserRightFG, "side_right", {A}, {"n": 1})
    right_two = net.add(ChooserRightFG, "side_right", {A}, {"n": 2})
    child_one = net.add(ChooserChildFG, "side_child", {P, A}, {"n": 1})
    child_two = net.add(ChooserChildFG, "side_child", {P, A}, {"n": 2})
    sink_one = net.add(ChooserSinkFG, "side_sink", {A})
    sink_two = net.add(ChooserOtherSinkFG, "side_other_sink", {P})
    net.join(link, left_one, right_one, child_one)
    net.join(link, left_two, right_two, child_two)
    net.edge(child_one, sink_one)
    net.edge(child_two, sink_two)

    net.choose()

    assert child_one.chosen_compute_framework is child_two.chosen_compute_framework


# --- rule d: filter ties ---------------------------------------------------------------------------------


def test_a_pinned_filter_pulls_its_host_onto_the_pin() -> None:
    net = _Net()
    host = net.add(ChooserHostFG, "filter_pull_host", {P, A})
    pinned = net.add(ChooserFilterFG, "filter_pull_filter", {A}, pinned=True)
    net.tie(host, pinned)

    net.choose()

    assert host.chosen_compute_framework is A
    assert pinned.chosen_compute_framework is A
    assert pinned.chosen_compute_framework_reason == PINNED
    assert host.chosen_compute_framework_reason == RULES


@pytest.mark.parametrize(
    ("host_allowed", "filter_allowed", "positions", "expected"),
    [
        ({P, A}, {P, A, D}, {A: 0}, A),
        ({P, A}, {A, D}, None, A),
        ({P, A, D}, {D, A}, {D: 0}, D),
    ],
    ids=["wider_filter", "narrower_overlap", "positions_inside_overlap"],
)
def test_an_unpinned_filter_stays_with_its_host(
    host_allowed: set[type[ComputeFramework]],
    filter_allowed: set[type[ComputeFramework]],
    positions: dict[type[ComputeFramework], int] | None,
    expected: type[ComputeFramework],
) -> None:
    net = _Net()
    host = net.add(ChooserHostFG, "filter_tie_host", host_allowed)
    filter_feature = net.add(ChooserFilterFG, "filter_tie_filter", filter_allowed)
    net.tie(host, filter_feature)

    net.choose(positions)

    assert host.chosen_compute_framework is expected
    assert filter_feature.chosen_compute_framework is expected


def test_a_filter_pin_outside_the_host_domain_is_infeasible() -> None:
    net = _Net()
    host = net.add(ChooserHostFG, "filter_outside_host", {P})
    pinned = net.add(ChooserFilterFG, "filter_outside_filter", {A})
    net.tie(host, pinned)

    with pytest.raises(ValueError, match="filter_outside_host"):
        net.choose()


def test_two_hosts_of_one_block_tied_to_differently_pinned_filters_are_infeasible() -> None:
    net = _Net()
    host_one = net.add(ChooserHostFG, "one_block_host_one", {P, A})
    host_two = net.add(ChooserHostFG, "one_block_host_two", {P, A})
    pinned_p = net.add(ChooserFilterFG, "one_block_filter_p", {P})
    pinned_a = net.add(ChooserFilterFG, "one_block_filter_a", {A})
    net.tie(host_one, pinned_p)
    net.tie(host_two, pinned_a)

    with pytest.raises(ValueError, match="one_block_host_one"):
        net.choose()


# --- rule e: transformer paths ---------------------------------------------------------------------------


def test_an_edge_without_a_transformer_path_is_infeasible() -> None:
    one, two = _throwaway_pair(shared_expected=False)
    net = _Net()
    root = net.add(ChooserRootFG, "no_path_root", {one})
    leaf = net.add(ChooserLeafFG, "no_path_leaf", {two})
    net.edge(root, leaf)

    with pytest.raises(ValueError, match="no_path_root"):
        net.choose()


def test_a_free_child_avoids_the_missing_path() -> None:
    one, two = _throwaway_pair(shared_expected=False)
    net = _Net()
    root = net.add(ChooserRootFG, "avoid_path_root", {one})
    leaf = net.add(ChooserLeafFG, "avoid_path_leaf", {two, one})
    net.edge(root, leaf)

    net.choose()

    assert leaf.chosen_compute_framework is one


def test_the_same_expected_data_framework_needs_no_path() -> None:
    one, two = _throwaway_pair(shared_expected=True)
    net = _Net()
    root = net.add(ChooserRootFG, "twin_root", {one})
    leaf = net.add(ChooserLeafFG, "twin_leaf", {two})
    net.edge(root, leaf)

    net.choose()

    assert root.chosen_compute_framework is one
    assert leaf.chosen_compute_framework is two


# --- cost ------------------------------------------------------------------------------------------------


def test_one_parent_feeding_two_consumer_groups_on_another_framework_costs_two_conversions() -> None:
    net = _Net()
    root = net.add(ChooserRootFG, "cost_two_root", {P, A})
    leaf_one = net.add(ChooserLeafFG, "cost_two_leaf", {P})
    leaf_two = net.add(ChooserOtherLeafFG, "cost_two_other_leaf", {P})
    sink = net.add(ChooserSinkFG, "cost_two_sink", {A})
    for child in (leaf_one, leaf_two, sink):
        net.edge(root, child)

    net.choose({A: 0})

    assert root.chosen_compute_framework is P


def test_one_parent_feeding_two_blocks_of_one_group_on_another_framework_costs_one_conversion() -> None:
    net = _Net()
    root = net.add(ChooserRootFG, "cost_one_root", {P, A})
    leaf_one = net.add(ChooserLeafFG, "cost_one_leaf", {P}, {"n": 1})
    leaf_two = net.add(ChooserLeafFG, "cost_one_leaf", {P}, {"n": 2})
    sink = net.add(ChooserSinkFG, "cost_one_sink", {A})
    for child in (leaf_one, leaf_two, sink):
        net.edge(root, child)

    net.choose({A: 0})

    assert root.chosen_compute_framework is A


# --- blocks ----------------------------------------------------------------------------------------------


def test_features_of_one_group_with_equal_options_and_allowed_set_get_one_framework() -> None:
    net = _Net()
    first = net.add(ChooserRootFG, "block_first", {P, A, D})
    second = net.add(ChooserRootFG, "block_second", {P, A, D})
    sink_a = net.add(ChooserSinkFG, "block_sink_a", {A})
    sink_d = net.add(ChooserOtherSinkFG, "block_sink_d", {D})
    net.edge(first, sink_a)
    net.edge(second, sink_d)

    net.choose()

    assert first.chosen_compute_framework is second.chosen_compute_framework


# --- output framework ------------------------------------------------------------------------------------


def test_a_free_requested_block_moves_to_the_output_framework() -> None:
    net = _Net()
    feature = net.add(ChooserRootFG, "output_pull_feature", {P, A}, requested=True)

    net.choose({P: 0, A: 1}, output_framework=A)

    assert feature.chosen_compute_framework is A
    assert feature.chosen_compute_framework_reason == SAVES_ONE


def test_an_unrequested_block_is_not_pulled_to_the_output_framework() -> None:
    net = _Net()
    feature = net.add(ChooserRootFG, "output_unrequested_feature", {P, A})

    net.choose({P: 0, A: 1}, output_framework=A)

    assert feature.chosen_compute_framework is P


def test_a_requested_block_without_a_path_to_the_output_is_pruned_to_a_framework_with_one() -> None:
    one, two = _throwaway_pair(shared_expected=False)
    net = _Net()
    feature = net.add(ChooserRootFG, "output_pruned_feature", {one, two}, requested=True)

    net.choose({one: 0, two: 1}, output_framework=two)

    assert feature.chosen_compute_framework is two


def test_a_requested_block_with_no_path_to_the_output_is_infeasible() -> None:
    one, two = _throwaway_pair(shared_expected=False)
    net = _Net()
    net.add(ChooserRootFG, "output_infeasible_feature", {one}, requested=True)

    with pytest.raises(ValueError, match="output_infeasible_feature"):
        net.choose(output_framework=two)


def test_a_pinned_requested_block_with_no_path_to_the_output_is_infeasible() -> None:
    one, two = _throwaway_pair(shared_expected=False)
    net = _Net()
    net.add(ChooserRootFG, "output_pinned_feature", {one}, pinned=True, requested=True)

    with pytest.raises(ValueError, match="output_pinned_feature"):
        net.choose(output_framework=two)


def test_a_framework_sharing_the_outputs_expected_data_framework_costs_nothing() -> None:
    one, two = _throwaway_pair(shared_expected=True)
    net = _Net()
    feature = net.add(ChooserRootFG, "output_twin_feature", {one, two}, requested=True)

    net.choose({one: 0, two: 1}, output_framework=two)

    assert feature.chosen_compute_framework is one
    assert feature.chosen_compute_framework_reason == LIST_ORDER


# --- reasons ---------------------------------------------------------------------------------------------


def test_the_reason_texts_are_the_pinned_lock_contract() -> None:
    assert choose_compute_frameworks.PINNED == PINNED
    assert choose_compute_frameworks.ONLY_ALLOWED == ONLY_ALLOWED
    assert choose_compute_frameworks.RULES == RULES
    assert choose_compute_frameworks.LIST_ORDER == LIST_ORDER
    assert choose_compute_frameworks.DEFAULT_ORDER == DEFAULT_ORDER
    assert choose_compute_frameworks.saves_conversions(1) == SAVES_ONE
    assert choose_compute_frameworks.saves_conversions(2) == SAVES_TWO
    assert choose_compute_frameworks.saves_conversions(7) == "saves 7 conversions"


def _reason_pinned(net: _Net) -> list[Feature]:
    return [net.add(ChooserRootFG, "reason_pinned", {P, A}, pinned=True)]


def _reason_saves_two(net: _Net) -> list[Feature]:
    root = net.add(ChooserRootFG, "reason_two_root", {P, A})
    for fg, name in [(ChooserLeafFG, "reason_two_leaf"), (ChooserOtherLeafFG, "reason_two_other_leaf")]:
        net.edge(root, net.add(fg, name, {A}))
    return [root]


def _reason_rules_via_join(net: _Net) -> list[Feature]:
    left = net.add(ChooserLeftFG, "reason_join_left", {P})
    right = net.add(ChooserRightFG, "reason_join_right", {A})
    child = net.add(ChooserChildFG, "reason_join_child", {P, A, D})
    net.join(_link(Link.right, ChooserLeftFG, ChooserRightFG), left, right, child)
    return [child]


def _reason_equal_cost_tie(net: _Net) -> list[Feature]:
    root = net.add(ChooserRootFG, "reason_tie_root", {P, A})
    net.edge(root, net.add(ChooserLeafFG, "reason_tie_leaf_p", {P}))
    net.edge(root, net.add(ChooserOtherLeafFG, "reason_tie_leaf_a", {A}))
    return [root]


def _reason_shared_list_position(net: _Net) -> list[Feature]:
    return [net.add(ChooserRootFG, "reason_shared_position", {P, A})]


@pytest.mark.parametrize(
    ("build", "positions", "expected"),
    [
        (_reason_pinned, None, PINNED),
        (_reason_saves_two, None, SAVES_TWO),
        (_reason_rules_via_join, None, RULES),
        (_reason_equal_cost_tie, None, DEFAULT_ORDER),
        (_reason_equal_cost_tie, {A: 0}, LIST_ORDER),
        (_reason_shared_list_position, {D: 0}, DEFAULT_ORDER),
    ],
    ids=["pinned", "saves_two", "rules_via_join", "tie_default_order", "tie_list_order", "shared_list_position"],
)
def test_the_reason_names_why_the_framework_was_chosen(
    build: Callable[[_Net], list[Feature]], positions: dict[type[ComputeFramework], int] | None, expected: str
) -> None:
    net = _Net()
    judged = build(net)

    net.choose(positions)

    assert [f.chosen_compute_framework_reason for f in judged] == [expected]


def test_a_child_on_the_swapped_side_of_one_inner_link_reports_rules() -> None:
    link = _link(Link.inner, ChooserLeftFG, ChooserRightFG)
    net = _Net()
    children = []
    for index, (left_fw, right_fw) in enumerate([(P, A), (A, P)]):
        left = net.add(ChooserLeftFG, "swap_left", {left_fw}, {"n": index})
        right = net.add(ChooserRightFG, "swap_right", {right_fw}, {"n": index})
        children.append(net.add(ChooserChildFG, "swap_child", {P, A}, {"n": index}))
        net.join(link, left, right, children[-1])

    net.choose()

    on_a = [child for child in children if child.chosen_compute_framework is A]
    assert len(on_a) == 1
    assert on_a[0].chosen_compute_framework_reason == RULES


def test_a_host_tied_to_a_top_ranked_pinned_filter_reports_rules() -> None:
    net = _Net()
    host = net.add(ChooserHostFG, "top_pin_host", {P, A})
    pinned = net.add(ChooserFilterFG, "top_pin_filter", {P}, pinned=True)
    net.tie(host, pinned)

    net.choose()

    assert host.chosen_compute_framework is P
    assert host.chosen_compute_framework_reason == RULES


def test_every_member_of_a_block_gets_the_blocks_reason() -> None:
    net = _Net()
    first = net.add(ChooserRootFG, "member_first", {P, A}, pinned=True)
    second = net.add(ChooserRootFG, "member_second", {P, A})

    net.choose()

    assert first.chosen_compute_framework_reason == PINNED
    assert second.chosen_compute_framework_reason == PINNED


# --- determinism and scale -------------------------------------------------------------------------------


# Fresh interpreters cost roughly a second each, so this one needs more than the suite-wide per-test budget.
@pytest.mark.timeout(60)
def test_fresh_interpreters_choose_the_same_frameworks() -> None:
    seeds = [1, 2]
    outputs = run_probes(_PROBE, len(seeds), seeds=seeds)

    assert len(outputs) == len(seeds)
    for position, output in enumerate(outputs):
        plain = {key: value for key, value in output.items() if not key.startswith("addr_")}
        assert plain == _PROBE_EXPECTED, f"probe {position} chose {plain}, expected {_PROBE_EXPECTED}"
    assert outputs[0] == outputs[1], "blocks equal but for an option object's address chose differently per process"


@pytest.mark.parametrize("positions", [{A: 0, D: 1}, None], ids=["list_order", "no_positions"])
def test_a_thirty_block_component_with_a_tied_optimum_solves_quickly(
    positions: dict[type[ComputeFramework], int] | None,
) -> None:
    net = _Net()
    pins: dict[int, type[ComputeFramework]] = {0: A, 10: D, 20: A, 29: D}
    layers = [
        net.add(ChooserLayerFG, "layer", {pins[index]} if index in pins else {P, A, D}, {"layer": index})
        for index in range(30)
    ]
    for parent, child in zip(layers, layers[1:]):
        net.edge(parent, child)

    chooser = net.chooser(positions)  # built before the timer: construction carries one-time cold setup
    started = time.perf_counter()
    chooser.choose()
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0, f"choosing took {elapsed:.1f}s"
    assert net.conversions() == 3
    for index, framework in pins.items():
        assert layers[index].chosen_compute_framework is framework


# --- search scale (F2) -----------------------------------------------------------------------------------


def test_an_infeasible_chain_raises_quickly_and_names_the_failing_block() -> None:
    one, _two = _throwaway_pair(shared_expected=False)
    net = _Net()
    layers = [net.add(ChooserLayerFG, "blowup_layer", {P, A, D}, {"layer": index}) for index in range(16)]
    sink = net.add(ChooserSinkFG, "blowup_sink", {one})
    for parent, child in zip(layers, layers[1:]):
        net.edge(parent, child)
    net.edge(layers[-1], sink)

    chooser = net.chooser()
    started = time.perf_counter()
    with pytest.raises(ValueError, match="blowup_sink") as raised:
        chooser.choose()
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0, f"raising took {elapsed:.1f}s"
    assert "blowup_layer" not in str(raised.value), "the message must name the emptied block, not the whole component"
    assert "transform path" in str(raised.value)


class _AllConvertibleChooser(ChooseComputeFrameworks):
    """Every framework pair converts, so throwaway frameworks need no registered transformers."""

    def _convertible(self, source: type[ComputeFramework], target: type[ComputeFramework]) -> bool:
        return True


def _six_frameworks() -> list[type[ComputeFramework]]:
    def make(name: str) -> type[ComputeFramework]:
        return type(name, (ComputeFramework,), {"is_available": staticmethod(lambda: False)})

    return [make(f"ZzScale{letter}ThrowawayFramework") for letter in "ABCDEF"]


def test_a_twenty_five_block_six_framework_component_with_a_forced_conversion_solves_quickly() -> None:
    fws = _six_frameworks()
    everything = set(fws)
    pins = [fws[5]] * 5 + [fws[4]] * 2 + fws[:2]
    net = _Net()
    hub = net.add(ChooserHostFG, "scale_hub", everything)
    leaves = []
    for index, pin in enumerate(pins):
        leaf_fg = type(f"ChooserScaleLeaf{index}FG", (FeatureGroup,), {})
        leaves.append(net.add(leaf_fg, f"scale_leaf_{index}", {pin}))
        net.edge(hub, leaves[-1])
    fillers = [net.add(ChooserLayerFG, "scale_filler", everything, {"layer": index}) for index in range(15)]
    for parent, child in zip([hub, *fillers], fillers):
        net.edge(parent, child)
    assert len(net.features) == 25

    chooser = net.chooser(chooser_class=_AllConvertibleChooser)
    started = time.perf_counter()
    chooser.choose()
    elapsed = time.perf_counter() - started

    assert elapsed < 0.2, f"choosing took {elapsed:.1f}s"
    assert hub.chosen_compute_framework is fws[5]
    assert {f.chosen_compute_framework for f in fillers} == {fws[5]}
    assert [leaf.chosen_compute_framework for leaf in leaves] == pins
    assert net.conversions() == 4


# --- RIGHT self-merge guard (F4) -------------------------------------------------------------------------


def test_a_right_link_whose_sides_share_a_framework_the_child_is_not_on_is_infeasible() -> None:
    net = _Net()
    left_one = net.add(ChooserLeftFG, "guard_left_one", {P})
    right_one = net.add(ChooserRightFG, "guard_right_one", {A})
    left_two = net.add(ChooserLeafFG, "guard_left_two", {D})
    right_two = net.add(ChooserOtherLeafFG, "guard_right_two", {D})
    child = net.add(ChooserChildFG, "guard_child", {P, A})
    net.join(_link(Link.inner, ChooserLeftFG, ChooserRightFG), left_one, right_one, child)
    net.join(_link(Link.right, ChooserLeafFG, ChooserOtherLeafFG), left_two, right_two, child)

    with pytest.raises(ValueError, match="guard_child"):
        net.choose()


# --- block order determinism (F6) ------------------------------------------------------------------------


class _AddressRepr:
    """Option value whose repr carries a chosen address, as a default object repr does."""

    def __init__(self, address: int) -> None:
        self.address = address

    def __repr__(self) -> str:
        return f"<_AddressRepr object at 0x{self.address:x}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _AddressRepr) and other.address == self.address

    def __hash__(self) -> int:
        return hash(self.address)


def _choose_with_addresses(addresses: tuple[int, ...]) -> list[str]:
    everything = {P, A, D}
    net = _Net()
    objs = [net.add(ChooserLayerFG, "addr_o", everything, {"o": _AddressRepr(a)}) for a in addresses]
    src = net.add(ChooserRootFG, "addr_src", {P})
    snk = net.add(ChooserSinkFG, "addr_snk", {D})
    leaf = net.add(ChooserLeafFG, "addr_leaf", {A})
    for parent, child in [
        (objs[0], objs[1]),
        (objs[0], src),
        (objs[0], leaf),
        (objs[1], objs[2]),
        (objs[1], snk),
        (objs[2], snk),
        (objs[2], leaf),
    ]:
        net.edge(parent, child)

    net.choose()

    return [f.get_class_name() for f in (c.chosen_compute_framework for c in net.features) if f is not None]


def test_blocks_equal_but_for_an_address_in_an_option_repr_choose_the_same_whatever_the_addresses() -> None:
    outcomes = {tuple(_choose_with_addresses(p)) for p in itertools.permutations((0x1000, 0x2000, 0x3000))}

    assert len(outcomes) == 1, f"the address order changed the assignment: {outcomes}"


def _block_data_types(types: list[DataType]) -> list[str]:
    net = _Net()
    for data_type in types:
        net.add(ChooserRootFG, "typed", {P, A}, None, data_type)
    return [block.features[0].data_type.value for block in net.chooser()._blocks() if block.features[0].data_type]


def test_blocks_differing_only_in_data_type_are_ordered_by_data_type_name() -> None:
    types = [DataType.STRING, DataType.INT32, DataType.DOUBLE, DataType.BOOLEAN, DataType.DATE, DataType.BINARY]

    assert _block_data_types(types) == sorted(t.value for t in types)
    assert _block_data_types(types[::-1]) == sorted(t.value for t in types)


# --- empty domains (F7) ----------------------------------------------------------------------------------


@pytest.mark.parametrize("allowed", [set(), None], ids=["empty_set", "none"])
def test_a_block_without_any_allowed_framework_raises_the_unresolved_frameworks_error(
    allowed: set[type[ComputeFramework]] | None,
) -> None:
    net = _Net()
    net.add(ChooserRootFG, "no_framework_feature", allowed)

    with pytest.raises(ValueError, match="no_framework_feature does not have any compute framework"):
        net.choose()


# --- completeness guard (F8) -----------------------------------------------------------------------------


def test_every_graph_node_ends_with_a_chosen_framework() -> None:
    """Regression guard: holds today because every graph node is bucketed."""
    net = _Net()
    root = net.add(ChooserRootFG, "complete_root", {P, A})
    leaf = net.add(ChooserLeafFG, "complete_leaf", {A})
    net.edge(root, leaf)

    net.choose()

    for node in net.graph.nodes:
        assert net.graph.nodes[node].feature.chosen_compute_framework is not None


def test_a_graph_node_the_chooser_never_sees_raises() -> None:
    net = _Net()
    net.add(ChooserRootFG, "seen_root", {P})
    net.orphan(ChooserLeafFG, "unseen_orphan", {P})

    with pytest.raises(ValueError, match="unseen_orphan"):
        net.choose()
