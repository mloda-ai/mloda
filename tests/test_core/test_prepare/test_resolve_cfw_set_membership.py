"""Set membership of planned-queue features after ResolveComputeFrameworks.links rewrites compute_frameworks."""

from typing import Any

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.resolve_compute_frameworks import ResolveComputeFrameworks
from mloda.core.prepare.resolve_links import LinkTrekker
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class LeftGroup(FeatureGroup):
    pass


class RightGroup(FeatureGroup):
    pass


def test_features_remain_set_members_after_links_rewrites_compute_frameworks() -> None:
    """Rewriting compute_frameworks in links() must not strand features in their queue set."""
    feature_a = Feature("feature_a")
    feature_b = Feature("feature_b")
    feature_a.compute_frameworks = {PandasDataFrame, PyArrowTable}
    feature_b.compute_frameworks = {PandasDataFrame, PyArrowTable}

    feature_set = {feature_a, feature_b}
    planned_queue: list[Any] = [(LeftGroup, feature_set)]

    link = Link.inner(JoinSpec(LeftGroup, "idx"), JoinSpec(RightGroup, "idx"))
    trekker = (link, PandasDataFrame, PyArrowTable)

    link_trekker = LinkTrekker()
    link_trekker.data_ordered[trekker] = {feature_a.uuid, feature_b.uuid}

    result = ResolveComputeFrameworks(Graph()).links(planned_queue, link_trekker)

    returned_set = result[0][1]
    # The trekked path narrows the frameworks.
    assert next(iter(returned_set)).compute_frameworks == {PandasDataFrame}

    for feature in returned_set:
        assert feature in returned_set, f"{feature.name} stranded in returned queue set after rewrite"

    assert feature_set is returned_set
    for feature in feature_set:
        assert feature in feature_set, f"{feature.name} stranded in aliased source set after rewrite"
