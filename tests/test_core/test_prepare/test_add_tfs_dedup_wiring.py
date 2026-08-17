"""Regression coverage for add_tfs: a deduped TransformFrameworkStep must still wire its uuid
into every consuming step, not just the one that first created it.
"""

from typing import NamedTuple
from uuid import UUID

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
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
from mloda.user import JoinSpec
from mloda.user import Link
from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class DedupBaseFG(FeatureGroup):
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


class DedupLeftFG(DedupBaseFG):
    pass


class DedupRightFG(DedupBaseFG):
    pass


class DedupUpstreamFG(DedupBaseFG):
    pass


class DedupDestFG(DedupBaseFG):
    pass


def _feature(name: str, cfw: type[ComputeFramework]) -> Feature:
    feature = Feature(name)
    feature.compute_frameworks = {cfw}
    return feature


# ---------------------------------------------------------------------------
# Scenario A: add_tfs's JoinStep branch
# ---------------------------------------------------------------------------


class JoinStepDedupScenario(NamedTuple):
    js1: JoinStep
    js2: JoinStep
    graph: Graph


def _join_step(link: Link) -> JoinStep:
    return JoinStep(
        link=link,
        destination_framework=PandasDataFrame,
        source_framework=PyArrowTable,
        required_uuids=set(),
        destination_framework_uuids=set(),
        source_framework_uuids=set(),
    )


def _join_step_dedup_scenario() -> JoinStepDedupScenario:
    """Two JoinSteps over the same link, frameworks, and orientation, so ``fill_tfs_by_joinstep``
    builds two equal ``TransformFrameworkStep``s."""
    link = Link.inner(JoinSpec(DedupLeftFG, "id"), JoinSpec(DedupRightFG, "id"))
    return JoinStepDedupScenario(_join_step(link), _join_step(link), Graph())


def test_both_joinsteps_of_a_deduped_hop_depend_on_the_surviving_transform_step() -> None:
    scenario = _join_step_dedup_scenario()

    new_plan = ExecutionPlan().add_tfs([scenario.js1, scenario.js2], scenario.graph)

    tfs_steps = [step for step in new_plan if isinstance(step, TransformFrameworkStep)]
    assert len(tfs_steps) == 1, f"expected the two equal hops to dedup into one, got: {tfs_steps}"
    tfs_uuid = tfs_steps[0].uuid

    assert tfs_uuid in scenario.js1.required_uuids
    assert tfs_uuid in scenario.js2.required_uuids


# ---------------------------------------------------------------------------
# Scenario B: add_tfs's FeatureGroupStep branch
# ---------------------------------------------------------------------------


class FeatureGroupStepDedupScenario(NamedTuple):
    step_a: FeatureGroupStep
    step_b: FeatureGroupStep
    graph: Graph


def _root_node(graph: Graph, feature: Feature, fg: type[FeatureGroup]) -> None:
    graph.add_node(feature.uuid, NodeProperties(feature, fg))
    graph.parent_to_children_mapping[feature.uuid] = set()


def _dest_step(
    fg: type[FeatureGroup],
    dest_feature: Feature,
    cfw: type[ComputeFramework],
    parent_uuid: UUID,
    graph: Graph,
) -> FeatureGroupStep:
    feature_set = FeatureSet()
    feature_set.add(dest_feature)
    graph.parent_to_children_mapping[dest_feature.uuid] = {parent_uuid}
    return FeatureGroupStep(fg, feature_set, set(), cfw)


def _feature_group_step_dedup_scenario() -> FeatureGroupStepDedupScenario:
    """Two FeatureGroupSteps with a root parent from the same upstream feature group and compute
    framework, so their built ``TransformFrameworkStep``s compare equal."""
    graph = Graph()

    parent_a = _feature("dedup_upstream_a", PyArrowTable)
    parent_b = _feature("dedup_upstream_b", PyArrowTable)
    _root_node(graph, parent_a, DedupUpstreamFG)
    _root_node(graph, parent_b, DedupUpstreamFG)

    dest_feature_a = _feature("dedup_dest_a", PandasDataFrame)
    dest_feature_b = _feature("dedup_dest_b", PandasDataFrame)
    step_a = _dest_step(DedupDestFG, dest_feature_a, PandasDataFrame, parent_a.uuid, graph)
    step_b = _dest_step(DedupDestFG, dest_feature_b, PandasDataFrame, parent_b.uuid, graph)

    return FeatureGroupStepDedupScenario(step_a, step_b, graph)


def test_both_feature_group_steps_of_a_deduped_hop_reference_the_surviving_transform_step() -> None:
    scenario = _feature_group_step_dedup_scenario()

    new_plan = ExecutionPlan().add_tfs([scenario.step_a, scenario.step_b], scenario.graph)

    tfs_steps = [step for step in new_plan if isinstance(step, TransformFrameworkStep)]
    assert len(tfs_steps) == 1, f"expected the two equal hops to dedup into one, got: {tfs_steps}"
    tfs_uuid = tfs_steps[0].uuid

    assert scenario.step_a.tfs_ids == {tfs_uuid}
    assert scenario.step_b.tfs_ids == {tfs_uuid}
