"""White-box pin for add_tfs's same-compute-framework, non-APPEND/UNION join branch: it must set
``inner_ep.tfs_ids`` to the destination-side parent uuid, not leave it empty.
"""

from uuid import UUID, uuid4

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.graph.graph import Graph
from mloda.provider import ComputeFramework
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import JoinSpec
from mloda.user import Link
from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class SameFwBaseFG(FeatureGroup):
    """Unmatchable base so a leaked subclass stays invisible to feature resolution."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class SameFwLeftFG(SameFwBaseFG):
    pass


class SameFwRightFG(SameFwBaseFG):
    pass


def _feature(name: str, cfw: type[ComputeFramework]) -> Feature:
    feature = Feature(name)
    feature.compute_frameworks = {cfw}
    return feature


def _same_framework_join_scenario() -> tuple[JoinStep, FeatureGroupStep, UUID]:
    """One same-cfw, INNER JoinStep and its destination-side FeatureGroupStep, wired so the
    step's own uuid is the join's destination-side uuid and its required_uuids already carry
    both sides, exactly what add_tfs's ``else: inner_ep.tfs_ids = {store_val}`` branch checks for."""
    dest_feature = _feature("same_fw_dest", PandasDataFrame)
    dest_uuid = dest_feature.uuid
    src_uuid = uuid4()

    link = Link.inner(JoinSpec(SameFwLeftFG, "id"), JoinSpec(SameFwRightFG, "id"))
    join_step = JoinStep(
        link=link,
        destination_framework=PandasDataFrame,
        source_framework=PandasDataFrame,
        required_uuids=set(),
        destination_framework_uuids={dest_uuid},
        source_framework_uuids={src_uuid},
    )

    feature_set = FeatureSet()
    feature_set.add(dest_feature)
    inner_ep = FeatureGroupStep(
        SameFwLeftFG,
        feature_set,
        {dest_uuid, src_uuid},
        PandasDataFrame,
    )

    return join_step, inner_ep, dest_uuid


def test_same_framework_non_append_union_join_sets_tfs_ids_to_destination_side_uuid() -> None:
    join_step, inner_ep, dest_uuid = _same_framework_join_scenario()

    ExecutionPlan().add_tfs([join_step, inner_ep], Graph())

    assert inner_ep.tfs_ids == {dest_uuid}, (
        f"expected tfs_ids to pin the destination-side uuid {dest_uuid}, got: {inner_ep.tfs_ids}"
    )
