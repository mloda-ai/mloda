"""Upload marking must key on any step uuid in the collector, not on one arbitrary FeatureSet representative."""

from typing import NamedTuple

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.link import Link
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import NodeProperties
from mloda.provider import ComputeFramework
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import JoinSpec
from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


UPLOAD_MARK_KEY = "upload_mark_key"
UPLOAD_MARK_INDEX = Index((UPLOAD_MARK_KEY,))


class UploadMarkBaseFG(FeatureGroup):
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


class UploadMarkSourceFG(UploadMarkBaseFG):
    pass


class UploadMarkDestFG(UploadMarkBaseFG):
    pass


class UploadMarkUnrelatedFG(UploadMarkBaseFG):
    pass


class Scenario(NamedTuple):
    source_step: FeatureGroupStep
    unrelated_step: FeatureGroupStep
    steps: list[JoinStep | FeatureGroupStep]
    graph: Graph
    link: Link


def _feature(name: str, cfw: type[ComputeFramework], index: Index | None = None) -> Feature:
    feature = Feature(name, index=index)
    feature.compute_frameworks = {cfw}
    return feature


def _fg_step(
    fg: type[FeatureGroup], features: list[Feature], cfw: type[ComputeFramework], graph: Graph
) -> FeatureGroupStep:
    feature_set = FeatureSet()
    for feature in features:
        feature_set.add(feature)
        graph.add_node(feature.uuid, NodeProperties(feature, fg))
        graph.parent_to_children_mapping[feature.uuid] = set()
    return FeatureGroupStep(fg, feature_set, set(), cfw)


def _scenario() -> Scenario:
    """Cross-framework join whose source-side set carries the link's index feature added FIRST."""
    graph = Graph()

    index_feature = _feature(UPLOAD_MARK_KEY, PyArrowTable, UPLOAD_MARK_INDEX)
    payload_one = _feature("upload_mark_payload_one", PyArrowTable)
    payload_two = _feature("upload_mark_payload_two", PyArrowTable)
    source_step = _fg_step(UploadMarkSourceFG, [index_feature, payload_one, payload_two], PyArrowTable, graph)
    assert source_step.features.any_uuid == index_feature.uuid

    dest_feature = _feature("upload_mark_dest_payload", PandasDataFrame)
    dest_step = _fg_step(UploadMarkDestFG, [dest_feature], PandasDataFrame, graph)

    unrelated_feature = _feature("upload_mark_unrelated_payload", PyArrowTable)
    unrelated_step = _fg_step(UploadMarkUnrelatedFG, [unrelated_feature], PyArrowTable, graph)

    link = Link.inner(JoinSpec(UploadMarkDestFG, UPLOAD_MARK_INDEX), JoinSpec(UploadMarkSourceFG, UPLOAD_MARK_INDEX))
    join_step = JoinStep(
        link=link,
        destination_framework=PandasDataFrame,
        source_framework=PyArrowTable,
        required_uuids={dest_feature.uuid, payload_one.uuid, payload_two.uuid},
        destination_framework_uuids={dest_feature.uuid},
        source_framework_uuids={payload_one.uuid, payload_two.uuid},
    )

    steps: list[JoinStep | FeatureGroupStep] = [source_step, dest_step, unrelated_step, join_step]
    return Scenario(source_step, unrelated_step, steps, graph, link)


def test_source_step_whose_representative_is_the_index_feature_is_marked_for_upload() -> None:
    scenario = _scenario()

    ExecutionPlan().add_tfs(scenario.steps, scenario.graph)

    assert scenario.source_step.need_to_upload is True


def test_step_sharing_no_uuid_with_the_collector_stays_unmarked() -> None:
    scenario = _scenario()

    ExecutionPlan().add_tfs(scenario.steps, scenario.graph)

    assert scenario.unrelated_step.need_to_upload is False


def test_the_join_hop_names_the_declared_right_group_as_its_source() -> None:
    """swap_merge_sides is False, so the destination side is the declared left group."""
    scenario = _scenario()
    link = scenario.link

    plan = ExecutionPlan()
    new_plan = plan.add_tfs(scenario.steps, scenario.graph)

    tfs_steps = [step for step in new_plan if isinstance(step, TransformFrameworkStep)]
    assert len(tfs_steps) == 1
    tfs = tfs_steps[0]

    assert tfs.from_feature_group is link.right_feature_group
    assert tfs.to_feature_group is link.left_feature_group
    assert tfs.link_id == link.uuid
