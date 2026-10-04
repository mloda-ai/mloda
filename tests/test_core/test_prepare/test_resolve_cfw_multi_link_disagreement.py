"""One feature on two opposite-oriented cross-group links runs on one chosen framework and flips the other link."""

from typing import Any
from uuid import UUID

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.resolve_compute_frameworks import ResolveComputeFrameworks
from mloda.core.prepare.resolve_links import LinkTrekker
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class MultiLinkLeftFG(FeatureGroup):
    pass


class MultiLinkRightFG(FeatureGroup):
    pass


class MultiLinkOtherLeftFG(FeatureGroup):
    pass


class MultiLinkOtherRightFG(FeatureGroup):
    pass


class MultiLinkChildFG(FeatureGroup):
    pass


def _trek(link_trekker: LinkTrekker, trekker: Any, uuid: UUID) -> None:
    # Production shares one set object between data and data_ordered, and invert_link relies on that.
    shared_uuids = {uuid}
    link_trekker.data[trekker] = shared_uuids
    link_trekker.data_ordered[trekker] = shared_uuids


def _two_link_scenario() -> tuple[Feature, list[Any], LinkTrekker]:
    """Opposite orientations; the child runs on pandas, the left of link_a and the right of link_b."""
    feature = Feature("multi_link_feature")
    feature.compute_frameworks = {PandasDataFrame, PyArrowTable}
    feature.chosen_compute_framework = PandasDataFrame

    link_a = Link.inner(JoinSpec(MultiLinkLeftFG, "idx"), JoinSpec(MultiLinkRightFG, "idx"))
    link_b = Link.inner(JoinSpec(MultiLinkOtherLeftFG, "idx"), JoinSpec(MultiLinkOtherRightFG, "idx"))

    link_trekker = LinkTrekker()
    _trek(link_trekker, (link_a, PandasDataFrame, PyArrowTable), feature.uuid)
    _trek(link_trekker, (link_b, PyArrowTable, PandasDataFrame), feature.uuid)

    planned_queue: list[Any] = [(MultiLinkChildFG, {feature})]
    return feature, planned_queue, link_trekker


def test_two_opposite_links_plan_on_the_childs_framework() -> None:
    feature, planned_queue, link_trekker = _two_link_scenario()

    ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    assert feature.get_compute_framework() is PandasDataFrame
    assert feature.compute_frameworks == {PandasDataFrame, PyArrowTable}


def test_the_second_link_is_flipped_onto_the_childs_side() -> None:
    """Each link ends oriented so the child's framework is its left side."""
    feature, planned_queue, link_trekker = _two_link_scenario()

    ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    pairs = sorted(
        (key[0].left_feature_group.get_class_name(), key[1].get_class_name(), key[2].get_class_name())
        for key, uuids in link_trekker.data.items()
        if feature.uuid in uuids
    )
    assert pairs == [
        (MultiLinkLeftFG.get_class_name(), PandasDataFrame.get_class_name(), PyArrowTable.get_class_name()),
        (MultiLinkOtherLeftFG.get_class_name(), PandasDataFrame.get_class_name(), PyArrowTable.get_class_name()),
    ]


def test_a_self_join_keeps_both_orientations_on_the_chosen_framework() -> None:
    """A self-join child records every ordered parent pair, so the two orientations are not a disagreement."""
    feature = Feature("multi_link_self_join_feature")
    feature.compute_frameworks = {PandasDataFrame, PyArrowTable}
    feature.chosen_compute_framework = PandasDataFrame

    link = Link.inner(JoinSpec(MultiLinkLeftFG, "idx"), JoinSpec(MultiLinkLeftFG, "idx"))

    link_trekker = LinkTrekker()
    _trek(link_trekker, (link, PandasDataFrame, PyArrowTable), feature.uuid)
    _trek(link_trekker, (link, PyArrowTable, PandasDataFrame), feature.uuid)

    planned_queue: list[Any] = [(MultiLinkChildFG, {feature})]
    ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    assert feature.compute_frameworks == {PandasDataFrame, PyArrowTable}


def test_a_single_link_keeps_the_chosen_framework() -> None:
    """Control: one trekker resolves on the chosen framework and leaves the allowed set alone."""
    feature = Feature("multi_link_control_feature")
    feature.compute_frameworks = {PandasDataFrame, PyArrowTable}
    feature.chosen_compute_framework = PandasDataFrame

    link = Link.inner(JoinSpec(MultiLinkLeftFG, "idx"), JoinSpec(MultiLinkRightFG, "idx"))
    link_trekker = LinkTrekker()
    _trek(link_trekker, (link, PandasDataFrame, PyArrowTable), feature.uuid)

    planned_queue: list[Any] = [(MultiLinkChildFG, {feature})]
    ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    assert feature.compute_frameworks == {PandasDataFrame, PyArrowTable}
    assert feature.get_compute_framework() is PandasDataFrame


def test_two_links_agreeing_on_one_framework_do_not_raise() -> None:
    """Control: same orientation twice resolves to a single framework."""
    feature = Feature("multi_link_agreeing_feature")
    feature.compute_frameworks = {PandasDataFrame, PyArrowTable}
    feature.chosen_compute_framework = PandasDataFrame

    link_a = Link.inner(JoinSpec(MultiLinkLeftFG, "idx"), JoinSpec(MultiLinkRightFG, "idx"))
    link_b = Link.inner(JoinSpec(MultiLinkOtherLeftFG, "idx"), JoinSpec(MultiLinkOtherRightFG, "idx"))

    link_trekker = LinkTrekker()
    _trek(link_trekker, (link_a, PandasDataFrame, PyArrowTable), feature.uuid)
    _trek(link_trekker, (link_b, PandasDataFrame, PyArrowTable), feature.uuid)

    planned_queue: list[Any] = [(MultiLinkChildFG, {feature})]
    ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    assert feature.get_compute_framework() is PandasDataFrame
