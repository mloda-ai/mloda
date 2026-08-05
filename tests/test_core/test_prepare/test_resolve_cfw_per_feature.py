"""Group agreement per trekked link in ResolveComputeFrameworks.links."""

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


def _resolve_two_feature_scenario() -> tuple[Feature, Feature, LinkTrekker, Any]:
    """Both uuids trekked; feature_both holds both frameworks, feature_right only the right one."""
    feature_both = Feature("feature_both")
    feature_right = Feature("feature_right")
    feature_both.compute_frameworks = {PandasDataFrame, PyArrowTable}
    feature_right.compute_frameworks = {PyArrowTable}

    link_trekker, trekker = _make_trekker({feature_both.uuid, feature_right.uuid})
    planned_queue: list[Any] = [(CfwPerFeatureLeftFG, {feature_both, feature_right})]

    ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)
    return feature_both, feature_right, link_trekker, trekker


def test_group_agrees_on_right_when_left_is_not_shared_by_all() -> None:
    feature_both, feature_right, _, _ = _resolve_two_feature_scenario()

    assert feature_both.compute_frameworks == {PyArrowTable}
    assert feature_right.compute_frameworks == {PyArrowTable}


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

    link_trekker, trekker = _make_trekker({feature_both.uuid, feature_left.uuid})
    planned_queue: list[Any] = [(CfwPerFeatureLeftFG, {feature_both, feature_left})]

    ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    assert feature_both.compute_frameworks == {PandasDataFrame}
    assert feature_left.compute_frameworks == {PandasDataFrame}

    link = trekker[0]
    inverted = (link, PyArrowTable, PandasDataFrame)
    assert inverted not in link_trekker.data_ordered
    assert link_trekker.data_ordered[trekker] == {feature_both.uuid, feature_left.uuid}


def test_untrekked_feature_follows_group_rewrite() -> None:
    feature_trekked = Feature("feature_trekked")
    feature_free = Feature("feature_free")
    feature_trekked.compute_frameworks = {PandasDataFrame}
    feature_free.compute_frameworks = {PandasDataFrame, PyArrowTable}

    link_trekker, _ = _make_trekker({feature_trekked.uuid})
    planned_queue: list[Any] = [(CfwPerFeatureLeftFG, {feature_trekked, feature_free})]

    ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    assert feature_trekked.compute_frameworks == {PandasDataFrame}
    assert feature_free.compute_frameworks == {PandasDataFrame}


def test_incompatible_members_raise_at_planning_time() -> None:
    feature_left_only = Feature("feature_left_only")
    feature_right_only = Feature("feature_right_only")
    feature_left_only.compute_frameworks = {PandasDataFrame}
    feature_right_only.compute_frameworks = {PyArrowTable}

    link_trekker, _ = _make_trekker({feature_left_only.uuid, feature_right_only.uuid})
    planned_queue: list[Any] = [(CfwPerFeatureLeftFG, {feature_left_only, feature_right_only})]

    with pytest.raises(ValueError):
        ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)


def test_resolution_is_deterministic_across_fresh_inputs() -> None:
    for _ in range(20):
        feature_both, feature_right, link_trekker, trekker = _resolve_two_feature_scenario()
        assert feature_both.compute_frameworks == {PyArrowTable}
        assert feature_right.compute_frameworks == {PyArrowTable}
        assert trekker not in link_trekker.data_ordered
