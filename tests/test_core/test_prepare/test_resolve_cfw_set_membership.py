"""Set membership of planned-queue features after ResolveComputeFrameworks.links, which never rewrites compute_frameworks."""

from typing import Any

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


def test_features_remain_set_members_after_links() -> None:
    """links() must not strand features in their queue set, nor narrow their allowed frameworks."""
    feature_a = Feature("feature_a")
    feature_b = Feature("feature_b")
    feature_a.compute_frameworks = {PandasDataFrame, PyArrowTable}
    feature_b.compute_frameworks = {PandasDataFrame, PyArrowTable}
    feature_a.chosen_compute_framework = PandasDataFrame
    feature_b.chosen_compute_framework = PandasDataFrame

    feature_set = {feature_a, feature_b}
    planned_queue: list[Any] = [(CfwRehashLeftFG, feature_set)]

    result = _links_call(planned_queue, [feature_a, feature_b])

    returned_set = result[0][1]
    for feature in returned_set:
        assert feature.compute_frameworks == {PandasDataFrame, PyArrowTable}
        assert feature.get_compute_framework() is PandasDataFrame

    for feature in returned_set:
        assert feature in returned_set, f"{feature.name} stranded in returned queue set after links"

    assert feature_set is returned_set


def test_links_keeps_cfw_distinguished_twins_apart() -> None:
    """Twins distinct only by compute_frameworks stay two members: links() never narrows them into one."""
    feature_a = Feature("twin_feature")
    feature_b = Feature("twin_feature")
    feature_a.compute_frameworks = {PandasDataFrame, PyArrowTable}
    feature_b.compute_frameworks = {PandasDataFrame}
    feature_a.chosen_compute_framework = PandasDataFrame
    feature_b.chosen_compute_framework = PandasDataFrame

    feature_set = {feature_a, feature_b}
    assert len(feature_set) == 2

    planned_queue: list[Any] = [(CfwRehashLeftFG, feature_set)]

    result = _links_call(planned_queue, [feature_a, feature_b])

    assert len(result[0][1]) == 2
