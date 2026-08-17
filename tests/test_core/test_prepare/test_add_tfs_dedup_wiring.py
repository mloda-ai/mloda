"""Regression coverage for add_tfs: a deduped TransformFrameworkStep must still wire its uuid
into every consuming step, not just the one that first created it (Scenario A). Scenario B also
pins that two same-shaped hops from genuinely DIFFERENT parents must NOT dedup (issue #1141).
"""

from typing import NamedTuple
from uuid import UUID, uuid4

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


def _join_step(link: Link, source_framework_uuid: UUID, destination_framework_uuid: UUID) -> JoinStep:
    return JoinStep(
        link=link,
        destination_framework=PandasDataFrame,
        source_framework=PyArrowTable,
        required_uuids=set(),
        destination_framework_uuids={destination_framework_uuid},
        source_framework_uuids={source_framework_uuid},
    )


def _join_step_dedup_scenario() -> JoinStepDedupScenario:
    """Two JoinSteps over the same link, frameworks, and orientation, so ``fill_tfs_by_joinstep``
    builds two equal ``TransformFrameworkStep``s. Each carries its own distinct, non-empty
    source/destination framework uuids, as real JoinSteps do."""
    link = Link.inner(JoinSpec(DedupLeftFG, "id"), JoinSpec(DedupRightFG, "id"))
    js1 = _join_step(link, uuid4(), uuid4())
    js2 = _join_step(link, uuid4(), uuid4())
    return JoinStepDedupScenario(js1, js2, Graph())


def test_both_joinsteps_of_a_deduped_hop_depend_on_the_surviving_transform_step() -> None:
    scenario = _join_step_dedup_scenario()

    new_plan = ExecutionPlan().add_tfs([scenario.js1, scenario.js2], scenario.graph)

    tfs_steps = [step for step in new_plan if isinstance(step, TransformFrameworkStep)]
    assert len(tfs_steps) == 1, f"expected the two equal hops to dedup into one, got: {tfs_steps}"
    tfs_uuid = tfs_steps[0].uuid

    assert tfs_uuid in scenario.js1.required_uuids
    assert tfs_uuid in scenario.js2.required_uuids

    # Pins current dedup behavior: the survivor is js1's hop verbatim (first-inserted wins),
    # not a blend of both JoinSteps' source_framework_uuids.
    assert tfs_steps[0].source_framework_uuid == next(iter(scenario.js1.source_framework_uuids))


# ---------------------------------------------------------------------------
# Scenario B: add_tfs's FeatureGroupStep branch
# ---------------------------------------------------------------------------


class FeatureGroupStepDedupScenario(NamedTuple):
    step_a: FeatureGroupStep
    step_b: FeatureGroupStep
    parent_a: Feature
    parent_b: Feature
    graph: Graph


def _root_node(graph: Graph, feature: Feature, fg: type[FeatureGroup]) -> None:
    if feature.uuid in graph.get_nodes():
        return
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


def _feature_group_step_dedup_scenario(same_parent: bool) -> FeatureGroupStepDedupScenario:
    """Two FeatureGroupSteps whose built ``TransformFrameworkStep``s share the same from/to
    framework and from/to feature-group shape. ``same_parent=True`` models a legitimate dedup (both
    steps actually pull from the one upstream feature); ``same_parent=False`` models two genuinely
    different upstream features (dedup must NOT collapse these)."""
    graph = Graph()

    parent_a = _feature("dedup_upstream_a", PyArrowTable)
    parent_b = parent_a if same_parent else _feature("dedup_upstream_b", PyArrowTable)
    _root_node(graph, parent_a, DedupUpstreamFG)
    _root_node(graph, parent_b, DedupUpstreamFG)

    dest_feature_a = _feature("dedup_dest_a", PandasDataFrame)
    dest_feature_b = _feature("dedup_dest_b", PandasDataFrame)
    step_a = _dest_step(DedupDestFG, dest_feature_a, PandasDataFrame, parent_a.uuid, graph)
    step_b = _dest_step(DedupDestFG, dest_feature_b, PandasDataFrame, parent_b.uuid, graph)

    return FeatureGroupStepDedupScenario(step_a, step_b, parent_a, parent_b, graph)


def test_two_feature_group_steps_with_the_same_parent_and_shape_dedup_into_one_transform_step() -> None:
    """Guard against an overly-broad fix: genuinely shared parents must still dedup to one hop."""
    scenario = _feature_group_step_dedup_scenario(same_parent=True)

    new_plan = ExecutionPlan().add_tfs([scenario.step_a, scenario.step_b], scenario.graph)

    tfs_steps = [step for step in new_plan if isinstance(step, TransformFrameworkStep)]
    assert len(tfs_steps) == 1, f"expected the two same-parent hops to dedup into one, got: {tfs_steps}"
    tfs_uuid = tfs_steps[0].uuid

    assert scenario.step_a.tfs_ids == {tfs_uuid}
    assert scenario.step_b.tfs_ids == {tfs_uuid}

    assert tfs_uuid in scenario.step_a.required_uuids
    assert tfs_uuid in scenario.step_b.required_uuids


def test_two_feature_group_steps_with_different_parents_get_separate_transform_hops() -> None:
    """Two FeatureGroupSteps that share a from/to-framework + from/to-feature-group shape but pull
    from genuinely DIFFERENT parent features must each get their own TransformFrameworkStep. A hop
    only ever moves one physical source's data (TransformFrameworkStep.execute takes a single
    from_cfw), so collapsing two different-parent hops into one starves whichever step's uuid loses
    the dedup of its own source data."""
    scenario = _feature_group_step_dedup_scenario(same_parent=False)

    new_plan = ExecutionPlan().add_tfs([scenario.step_a, scenario.step_b], scenario.graph)

    tfs_steps = [step for step in new_plan if isinstance(step, TransformFrameworkStep)]
    assert len(tfs_steps) == 2, (
        f"two feature-group steps with genuinely different parents must NOT collapse into one "
        f"transform hop, got: {[(s.uuid, s.required_uuids) for s in tfs_steps]}"
    )

    hop_a = next(step for step in tfs_steps if scenario.parent_a.uuid in step.required_uuids)
    hop_b = next(step for step in tfs_steps if scenario.parent_b.uuid in step.required_uuids)
    assert hop_a.uuid != hop_b.uuid
    assert hop_a.required_uuids == {scenario.parent_a.uuid}
    assert hop_b.required_uuids == {scenario.parent_b.uuid}

    assert scenario.step_a.tfs_ids == {hop_a.uuid}
    assert scenario.step_b.tfs_ids == {hop_b.uuid}
    assert scenario.step_a.required_uuids == {hop_a.uuid}
    assert scenario.step_b.required_uuids == {hop_b.uuid}
