"""Set membership of planned-queue features after ResolveComputeFrameworks.links rewrites compute_frameworks."""

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.resolve_compute_frameworks import ResolveComputeFrameworks
from mloda.core.prepare.resolve_links import LinkTrekker
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class CfwRehashLeftFG(FeatureGroup):
    pass


class CfwRehashRightFG(FeatureGroup):
    pass


def _links_call(planned_queue: list[Any], features: list[Feature]) -> Any:
    link = Link.inner(JoinSpec(CfwRehashLeftFG, "idx"), JoinSpec(CfwRehashRightFG, "idx"))
    trekker = (link, PandasDataFrame, PyArrowTable)

    link_trekker = LinkTrekker()
    link_trekker.data_ordered[trekker] = {feature.uuid for feature in features}

    return ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)


def test_features_remain_set_members_after_links_rewrites_compute_frameworks() -> None:
    """Rewriting compute_frameworks in links() must not strand features in their queue set."""
    feature_a = Feature("feature_a")
    feature_b = Feature("feature_b")
    feature_a.compute_frameworks = {PandasDataFrame, PyArrowTable}
    feature_b.compute_frameworks = {PandasDataFrame, PyArrowTable}

    feature_set = {feature_a, feature_b}
    planned_queue: list[Any] = [(CfwRehashLeftFG, feature_set)]

    result = _links_call(planned_queue, [feature_a, feature_b])

    returned_set = result[0][1]
    # The trekked path narrows the frameworks for every member.
    for feature in returned_set:
        assert feature.compute_frameworks == {PandasDataFrame}

    for feature in returned_set:
        assert feature in returned_set, f"{feature.name} stranded in returned queue set after rewrite"

    assert feature_set is returned_set


def test_links_raises_when_rewrite_collapses_cfw_distinguished_twins() -> None:
    """Twins distinct only by compute_frameworks collapse after the rewrite; links() must raise, not drop one."""
    feature_a = Feature("twin_feature")
    feature_b = Feature("twin_feature")
    feature_a.compute_frameworks = {PandasDataFrame, PyArrowTable}
    feature_b.compute_frameworks = {PandasDataFrame}

    feature_set = {feature_a, feature_b}
    # Differing compute_frameworks keep the twins unequal at insertion.
    assert len(feature_set) == 2

    planned_queue: list[Any] = [(CfwRehashLeftFG, feature_set)]

    with pytest.raises(ValueError):
        _links_call(planned_queue, [feature_a, feature_b])
