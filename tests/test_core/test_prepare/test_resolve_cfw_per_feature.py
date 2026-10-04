"""Group agreement per trekked link in ResolveComputeFrameworks.links."""

from collections.abc import Callable
from typing import Any, NamedTuple
from uuid import UUID

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.choose_compute_frameworks import ChooseComputeFrameworks
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import EdgeProperties, NodeProperties
from mloda.core.prepare.resolve_compute_frameworks import ResolveComputeFrameworks
from mloda.core.prepare.resolve_links import LinkFrameworkTrekker, LinkTrekker
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_plugins.compute_framework.test_tooling.shared_compute_frameworks import SecondCfw


class CfwPerFeatureLeftFG(FeatureGroup):
    pass


class CfwPerFeatureRightFG(FeatureGroup):
    pass


def _make_trekker(trekked_uuids: set[UUID]) -> tuple[LinkTrekker, Any]:
    link = Link.inner(JoinSpec(CfwPerFeatureLeftFG, "idx"), JoinSpec(CfwPerFeatureRightFG, "idx"))
    trekker = (link, PandasDataFrame, PyArrowTable)

    link_trekker = LinkTrekker()
    link_trekker.data[trekker] = set(trekked_uuids)
    link_trekker.data_ordered[trekker] = set(trekked_uuids)
    return link_trekker, trekker


class _TrekkerSpec(NamedTuple):
    """One trekker: its join index, framework pair, trekked uuids and link kind."""

    join_index: str
    frameworks: tuple[type[ComputeFramework], type[ComputeFramework]]
    uuids: set[UUID]
    link_factory: Callable[[JoinSpec, JoinSpec], Link] = Link.inner


def _make_trekkers(*specs: _TrekkerSpec) -> tuple[LinkTrekker, list[LinkFrameworkTrekker]]:
    """Register one trekker per spec over distinct links of the same feature group pair."""
    link_trekker = LinkTrekker()
    trekkers: list[LinkFrameworkTrekker] = []
    for spec in specs:
        left = JoinSpec(CfwPerFeatureLeftFG, spec.join_index)
        right = JoinSpec(CfwPerFeatureRightFG, spec.join_index)
        link = spec.link_factory(left, right)
        trekker: LinkFrameworkTrekker = (link, spec.frameworks[0], spec.frameworks[1])
        # data and data_ordered share one set object, as production does.
        shared_uuids = set(spec.uuids)
        link_trekker.data[trekker] = shared_uuids
        link_trekker.data_ordered[trekker] = shared_uuids
        trekkers.append(trekker)
    return link_trekker, trekkers


def _choose(feature: Feature, framework: type[ComputeFramework]) -> Feature:
    feature.chosen_compute_framework = framework
    return feature


def _resolve_two_feature_scenario() -> tuple[Feature, Feature, LinkTrekker, Any]:
    """Both uuids trekked; feature_both allows both frameworks, feature_right only the right one."""
    feature_both = Feature("feature_both")
    feature_right = Feature("feature_right")
    feature_both.compute_frameworks = {PandasDataFrame, PyArrowTable}
    feature_right.compute_frameworks = {PyArrowTable}
    _choose(feature_both, PyArrowTable)
    _choose(feature_right, PyArrowTable)

    link_trekker, trekker = _make_trekker({feature_both.uuid, feature_right.uuid})
    planned_queue: list[Any] = [(CfwPerFeatureLeftFG, {feature_both, feature_right})]

    ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)
    return feature_both, feature_right, link_trekker, trekker


def test_group_agrees_on_right_when_left_is_not_shared_by_all() -> None:
    feature_both, feature_right, _, _ = _resolve_two_feature_scenario()

    assert feature_both.compute_frameworks == {PandasDataFrame, PyArrowTable}
    assert feature_both.get_compute_framework() is PyArrowTable
    assert feature_right.get_compute_framework() is PyArrowTable


def test_inversion_is_all_or_nothing() -> None:
    feature_both, feature_right, link_trekker, trekker = _resolve_two_feature_scenario()
    link = trekker[0]
    inverted = (link, PyArrowTable, PandasDataFrame)

    assert trekker not in link_trekker.data_ordered
    assert inverted in link_trekker.data_ordered
    assert feature_both.uuid in link_trekker.data_ordered[inverted]
    assert feature_right.uuid in link_trekker.data_ordered[inverted]


def test_group_agrees_on_left_when_all_support_left() -> None:
    feature_both = Feature("feature_both")
    feature_left = Feature("feature_left")
    feature_both.compute_frameworks = {PandasDataFrame, PyArrowTable}
    feature_left.compute_frameworks = {PandasDataFrame}
    _choose(feature_both, PandasDataFrame)
    _choose(feature_left, PandasDataFrame)

    link_trekker, trekker = _make_trekker({feature_both.uuid, feature_left.uuid})
    planned_queue: list[Any] = [(CfwPerFeatureLeftFG, {feature_both, feature_left})]

    ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    assert feature_both.compute_frameworks == {PandasDataFrame, PyArrowTable}
    assert feature_both.get_compute_framework() is PandasDataFrame
    assert feature_left.get_compute_framework() is PandasDataFrame

    link = trekker[0]
    inverted = (link, PyArrowTable, PandasDataFrame)
    assert inverted not in link_trekker.data_ordered
    assert link_trekker.data_ordered[trekker] == {feature_both.uuid, feature_left.uuid}


def test_untrekked_feature_is_left_alone() -> None:
    feature_trekked = _choose(Feature("feature_trekked"), PandasDataFrame)
    feature_free = _choose(Feature("feature_free"), PandasDataFrame)
    feature_trekked.compute_frameworks = {PandasDataFrame}
    feature_free.compute_frameworks = {PandasDataFrame, PyArrowTable}

    link_trekker, _ = _make_trekker({feature_trekked.uuid})
    planned_queue: list[Any] = [(CfwPerFeatureLeftFG, {feature_trekked, feature_free})]

    ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    assert feature_trekked.compute_frameworks == {PandasDataFrame}
    assert feature_free.compute_frameworks == {PandasDataFrame, PyArrowTable}
    assert feature_free.get_compute_framework() is PandasDataFrame


def test_incompatible_members_raise_at_planning_time() -> None:
    feature_left_only = Feature("feature_left_only")
    feature_right_only = Feature("feature_right_only")
    feature_left_only.compute_frameworks = {PandasDataFrame}
    feature_right_only.compute_frameworks = {PyArrowTable}
    _choose(feature_left_only, PandasDataFrame)
    _choose(feature_right_only, PyArrowTable)

    link_trekker, _ = _make_trekker({feature_left_only.uuid, feature_right_only.uuid})
    planned_queue: list[Any] = [(CfwPerFeatureLeftFG, {feature_left_only, feature_right_only})]

    with pytest.raises(ValueError):
        ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)


def test_dropped_link_joining_in_one_unsupported_framework_is_infeasible_for_the_chooser() -> None:
    """A same-framework link no child runs on would send both join sides to the same input."""
    one = Feature("feature_one")
    two = Feature("feature_two")
    one.compute_frameworks = {PandasDataFrame}
    two.compute_frameworks = {PandasDataFrame}
    left = Feature("feature_left")
    right = Feature("feature_right")
    left.compute_frameworks = {PandasDataFrame}
    right.compute_frameworks = {PyArrowTable}
    other_left = Feature("feature_other_left")
    other_right = Feature("feature_other_right")
    other_left.compute_frameworks = {PyArrowTable}
    other_right.compute_frameworks = {PyArrowTable}

    graph = Graph()
    for feature, fg in (
        (left, CfwPerFeatureLeftFG),
        (right, CfwPerFeatureRightFG),
        (other_left, CfwPerFeatureLeftFG),
        (other_right, CfwPerFeatureRightFG),
        (one, CfwPerFeatureLeftFG),
        (two, CfwPerFeatureRightFG),
    ):
        graph.add_node(feature.uuid, NodeProperties(feature, fg))
    for parent in (left, right, other_left, other_right):
        for child in (one, two):
            graph.add_edge(parent.uuid, child.uuid, EdgeProperties(CfwPerFeatureLeftFG, CfwPerFeatureRightFG))

    first = Link.inner(JoinSpec(CfwPerFeatureLeftFG, "idx"), JoinSpec(CfwPerFeatureRightFG, "idx"))
    second = Link.inner(JoinSpec(CfwPerFeatureLeftFG, "second_idx"), JoinSpec(CfwPerFeatureRightFG, "second_idx"))
    occurrences = [
        (first, left.uuid, right.uuid, one.uuid),
        (first, left.uuid, right.uuid, two.uuid),
        (second, other_left.uuid, other_right.uuid, one.uuid),
        (second, other_left.uuid, other_right.uuid, two.uuid),
    ]
    nodes: dict[type[FeatureGroup], set[Feature]] = {
        CfwPerFeatureLeftFG: {left, other_left, one},
        CfwPerFeatureRightFG: {right, other_right, two},
    }

    with pytest.raises(ValueError, match="No compute framework assignment"):
        ChooseComputeFrameworks(graph, nodes, occurrences, [], {}).choose()


def test_dropped_link_is_kept_when_a_child_ends_on_its_framework() -> None:
    """A child ending on the dropped link's framework keeps its two join sides distinguishable."""
    feature_one = Feature("feature_one")
    feature_two = Feature("feature_two")
    feature_one.compute_frameworks = {PandasDataFrame}
    feature_two.compute_frameworks = {PandasDataFrame, PyArrowTable}
    _choose(feature_one, PandasDataFrame)
    _choose(feature_two, PyArrowTable)

    link_trekker, (_, dropped, _) = _make_trekkers(
        _TrekkerSpec("idx", (PandasDataFrame, PyArrowTable), {feature_one.uuid}),
        _TrekkerSpec("second_idx", (PyArrowTable, PyArrowTable), {feature_one.uuid, feature_two.uuid}),
        _TrekkerSpec("third_idx", (PyArrowTable, PandasDataFrame), {feature_two.uuid}),
    )
    planned_queue: list[Any] = [(CfwPerFeatureLeftFG, {feature_one}), (CfwPerFeatureRightFG, {feature_two})]

    ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    assert feature_one.get_compute_framework() is PandasDataFrame
    assert feature_two.get_compute_framework() is PyArrowTable
    assert feature_two.compute_frameworks == {PandasDataFrame, PyArrowTable}
    assert link_trekker.data[dropped] == {feature_one.uuid, feature_two.uuid}


@pytest.mark.parametrize("link_factory", [Link.append, Link.union], ids=["append", "union"])
def test_dropped_append_or_union_link_with_equal_frameworks_is_kept(
    link_factory: Callable[[JoinSpec, JoinSpec], Link],
) -> None:
    """APPEND and UNION build their join step by index and feature group, never by framework."""
    feature_one = Feature("feature_one")
    feature_two = Feature("feature_two")
    feature_one.compute_frameworks = {PandasDataFrame}
    feature_two.compute_frameworks = {PandasDataFrame}
    _choose(feature_one, PandasDataFrame)
    _choose(feature_two, PandasDataFrame)

    trekked = {feature_one.uuid, feature_two.uuid}
    link_trekker, (_, dropped) = _make_trekkers(
        _TrekkerSpec("idx", (PandasDataFrame, PyArrowTable), trekked),
        _TrekkerSpec("second_idx", (PyArrowTable, PyArrowTable), trekked, link_factory),
    )
    planned_queue: list[Any] = [(CfwPerFeatureLeftFG, {feature_one, feature_two})]

    ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    assert feature_one.compute_frameworks == {PandasDataFrame}
    assert feature_two.compute_frameworks == {PandasDataFrame}
    assert link_trekker.data[dropped] == trekked


def test_dropped_link_joining_across_distinct_frameworks_is_skipped_silently() -> None:
    feature_one = Feature("feature_one")
    feature_two = Feature("feature_two")
    feature_one.compute_frameworks = {PandasDataFrame}
    feature_two.compute_frameworks = {PandasDataFrame}
    _choose(feature_one, PandasDataFrame)
    _choose(feature_two, PandasDataFrame)

    trekked = {feature_one.uuid, feature_two.uuid}
    link_trekker, (_, dropped) = _make_trekkers(
        _TrekkerSpec("idx", (PandasDataFrame, PyArrowTable), trekked),
        _TrekkerSpec("second_idx", (PyArrowTable, SecondCfw), trekked),
    )
    planned_queue: list[Any] = [(CfwPerFeatureLeftFG, {feature_one, feature_two})]

    ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    assert feature_one.compute_frameworks == {PandasDataFrame}
    assert feature_two.compute_frameworks == {PandasDataFrame}
    assert link_trekker.data_ordered[dropped] == trekked


def test_resolution_is_deterministic_across_fresh_inputs() -> None:
    for _ in range(20):
        feature_both, feature_right, link_trekker, trekker = _resolve_two_feature_scenario()
        assert feature_both.get_compute_framework() is PyArrowTable
        assert feature_right.get_compute_framework() is PyArrowTable
        assert trekker not in link_trekker.data_ordered
