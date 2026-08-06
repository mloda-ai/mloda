"""One feature on two cross-group links that resolve to different frameworks must fail planning.

A self-join records every ordered parent pair, so it is exempt.
"""

from typing import Any
from uuid import UUID

import pytest

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
    """Opposite orientations, so link_a resolves to pandas and link_b to pyarrow."""
    feature = Feature("multi_link_feature")
    feature.compute_frameworks = {PandasDataFrame, PyArrowTable}

    link_a = Link.inner(JoinSpec(MultiLinkLeftFG, "idx"), JoinSpec(MultiLinkRightFG, "idx"))
    link_b = Link.inner(JoinSpec(MultiLinkOtherLeftFG, "idx"), JoinSpec(MultiLinkOtherRightFG, "idx"))

    link_trekker = LinkTrekker()
    _trek(link_trekker, (link_a, PandasDataFrame, PyArrowTable), feature.uuid)
    _trek(link_trekker, (link_b, PyArrowTable, PandasDataFrame), feature.uuid)

    planned_queue: list[Any] = [(MultiLinkChildFG, {feature})]
    return feature, planned_queue, link_trekker


def test_two_links_resolving_to_different_frameworks_raise_at_planning_time() -> None:
    _, planned_queue, link_trekker = _two_link_scenario()

    with pytest.raises(ValueError, match="more than one compute framework"):
        ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)


def test_the_disagreement_message_names_the_feature_and_both_frameworks() -> None:
    feature, planned_queue, link_trekker = _two_link_scenario()

    with pytest.raises(ValueError) as excinfo:
        ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    message = str(excinfo.value)
    assert str(feature.name) in message, message
    assert PandasDataFrame.get_class_name() in message, message
    assert PyArrowTable.get_class_name() in message, message


def test_the_disagreement_message_reports_each_link_once_with_its_pair_and_verdict() -> None:
    """The pair and the framework it resolved to is what tells the two links apart."""
    _, planned_queue, link_trekker = _two_link_scenario()

    with pytest.raises(ValueError) as excinfo:
        ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    message = str(excinfo.value)
    assert (
        f"({PandasDataFrame.get_class_name()}, {PyArrowTable.get_class_name()}) -> {PandasDataFrame.get_class_name()}"
        in message
    ), message
    assert (
        f"({PyArrowTable.get_class_name()}, {PandasDataFrame.get_class_name()}) -> {PyArrowTable.get_class_name()}"
        in message
    ), message
    assert message.count(str(MultiLinkLeftFG.get_class_name())) == 1, message


def test_a_self_join_keeps_both_frameworks_instead_of_raising() -> None:
    """A self-join child records every ordered parent pair, so the two orientations are not a disagreement."""
    feature = Feature("multi_link_self_join_feature")
    feature.compute_frameworks = {PandasDataFrame, PyArrowTable}

    link = Link.inner(JoinSpec(MultiLinkLeftFG, "idx"), JoinSpec(MultiLinkLeftFG, "idx"))

    link_trekker = LinkTrekker()
    _trek(link_trekker, (link, PandasDataFrame, PyArrowTable), feature.uuid)
    _trek(link_trekker, (link, PyArrowTable, PandasDataFrame), feature.uuid)

    planned_queue: list[Any] = [(MultiLinkChildFG, {feature})]
    ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    assert feature.compute_frameworks == {PandasDataFrame, PyArrowTable}


def test_a_single_link_still_narrows_the_feature() -> None:
    """Control: one trekker keeps rewriting instead of raising."""
    feature = Feature("multi_link_control_feature")
    feature.compute_frameworks = {PandasDataFrame, PyArrowTable}

    link = Link.inner(JoinSpec(MultiLinkLeftFG, "idx"), JoinSpec(MultiLinkRightFG, "idx"))
    link_trekker = LinkTrekker()
    _trek(link_trekker, (link, PandasDataFrame, PyArrowTable), feature.uuid)

    planned_queue: list[Any] = [(MultiLinkChildFG, {feature})]
    ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    assert feature.compute_frameworks == {PandasDataFrame}


def test_two_links_agreeing_on_one_framework_do_not_raise() -> None:
    """Control: same orientation twice resolves to a single framework."""
    feature = Feature("multi_link_agreeing_feature")
    feature.compute_frameworks = {PandasDataFrame, PyArrowTable}

    link_a = Link.inner(JoinSpec(MultiLinkLeftFG, "idx"), JoinSpec(MultiLinkRightFG, "idx"))
    link_b = Link.inner(JoinSpec(MultiLinkOtherLeftFG, "idx"), JoinSpec(MultiLinkOtherRightFG, "idx"))

    link_trekker = LinkTrekker()
    _trek(link_trekker, (link_a, PandasDataFrame, PyArrowTable), feature.uuid)
    _trek(link_trekker, (link_b, PandasDataFrame, PyArrowTable), feature.uuid)

    planned_queue: list[Any] = [(MultiLinkChildFG, {feature})]
    ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    assert feature.compute_frameworks == {PandasDataFrame}
