"""Regression coverage for add_tfs's FeatureGroupStep branch: it must plan a transform hop for
every member feature's parents, not only for the member that happens to be FeatureSet.any_uuid
(set to whichever feature was ``.add()``-ed first).
"""

from typing import NamedTuple

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import NodeProperties
from mloda.provider import ComputeFramework
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class MultiMemberBaseFG(FeatureGroup):
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


class MultiMemberUpstreamFG(MultiMemberBaseFG):
    pass


class MultiMemberDestFG(MultiMemberBaseFG):
    pass


def _feature(name: str, cfw: type[ComputeFramework]) -> Feature:
    feature = Feature(name)
    feature.compute_frameworks = {cfw}
    return feature


def _root_node(graph: Graph, feature: Feature, fg: type[FeatureGroup]) -> None:
    graph.add_node(feature.uuid, NodeProperties(feature, fg))
    graph.parent_to_children_mapping[feature.uuid] = set()


def _producer_step(fg: type[FeatureGroup], feature: Feature, cfw: type[ComputeFramework]) -> FeatureGroupStep:
    feature_set = FeatureSet()
    feature_set.add(feature)
    return FeatureGroupStep(fg, feature_set, set(), cfw)


class MultiMemberScenario(NamedTuple):
    step: FeatureGroupStep
    producer: FeatureGroupStep
    member_with_parent: Feature
    member_without_parent: Feature
    parent: Feature
    graph: Graph


def _multi_member_scenario(parent_bearing_member_added_first: bool) -> MultiMemberScenario:
    """One FeatureGroupStep whose FeatureSet holds two members: one with a parent living on a
    different compute framework (needs a transform hop), and one parentless, like an
    auto-added index companion feature. Add order decides which member becomes any_uuid."""
    graph = Graph()

    parent = _feature("multi_member_upstream", PyArrowTable)
    _root_node(graph, parent, MultiMemberUpstreamFG)
    producer = _producer_step(MultiMemberUpstreamFG, parent, PyArrowTable)

    member_with_parent = _feature("multi_member_with_parent", PandasDataFrame)
    member_without_parent = _feature("multi_member_without_parent", PandasDataFrame)

    graph.parent_to_children_mapping[member_with_parent.uuid] = {parent.uuid}
    graph.parent_to_children_mapping[member_without_parent.uuid] = set()

    feature_set = FeatureSet()
    if parent_bearing_member_added_first:
        feature_set.add(member_with_parent)
        feature_set.add(member_without_parent)
    else:
        feature_set.add(member_without_parent)
        feature_set.add(member_with_parent)

    step = FeatureGroupStep(MultiMemberDestFG, feature_set, set(), PandasDataFrame)

    return MultiMemberScenario(step, producer, member_with_parent, member_without_parent, parent, graph)


def test_transform_hop_created_when_parent_bearing_member_is_any_uuid() -> None:
    """any_uuid resolves to the parent-bearing member: the existing loop already finds the hop."""
    scenario = _multi_member_scenario(parent_bearing_member_added_first=True)

    new_plan = ExecutionPlan().add_tfs([scenario.producer, scenario.step], scenario.graph)

    tfs_steps = [step for step in new_plan if isinstance(step, TransformFrameworkStep)]
    assert len(tfs_steps) == 1, f"expected one transform hop for member_with_parent's parent, got: {tfs_steps}"
    tfs_uuid = tfs_steps[0].uuid

    assert scenario.parent.uuid in tfs_steps[0].required_uuids
    assert tfs_uuid in scenario.step.required_uuids
    assert tfs_uuid in scenario.step.tfs_ids


def test_transform_hop_created_when_parentless_member_is_any_uuid() -> None:
    """any_uuid resolves to the parentless member: add_tfs must still find member_with_parent's
    parent by scanning all of the step's member features (FeatureGroupStep.get_uuids()), not just
    any_uuid, or the hop for member_with_parent's parent is silently skipped.
    """
    scenario = _multi_member_scenario(parent_bearing_member_added_first=False)

    new_plan = ExecutionPlan().add_tfs([scenario.producer, scenario.step], scenario.graph)

    tfs_steps = [step for step in new_plan if isinstance(step, TransformFrameworkStep)]
    assert len(tfs_steps) == 1, (
        f"expected one transform hop for member_with_parent's parent regardless of which member "
        f"is any_uuid, got: {tfs_steps}"
    )
    tfs_uuid = tfs_steps[0].uuid

    assert scenario.parent.uuid in tfs_steps[0].required_uuids
    assert tfs_uuid in scenario.step.required_uuids
    assert tfs_uuid in scenario.step.tfs_ids


# ---------------------------------------------------------------------------
# Overlapping ancestor chains across members
# ---------------------------------------------------------------------------


class OverlappingAncestorsScenario(NamedTuple):
    step: FeatureGroupStep
    producer_p: FeatureGroupStep
    producer_q: FeatureGroupStep
    member_a: Feature
    member_b: Feature
    p: Feature
    q: Feature
    graph: Graph


def _overlapping_ancestors_scenario() -> OverlappingAncestorsScenario:
    """One FeatureGroupStep with two members whose direct parents chain into each other: member_a's
    direct parent is p, member_b's direct parent is q, and q's own direct parent is p. Pruning
    member_a's direct parent p just because p is also an indirect ancestor (via q) of member_b
    would wrongly drop member_a's hop."""
    graph = Graph()

    p = _feature("multi_member_overlap_p", PyArrowTable)
    _root_node(graph, p, MultiMemberUpstreamFG)
    producer_p = _producer_step(MultiMemberUpstreamFG, p, PyArrowTable)

    q = _feature("multi_member_overlap_q", PyArrowTable)
    graph.add_node(q.uuid, NodeProperties(q, MultiMemberUpstreamFG))
    graph.parent_to_children_mapping[q.uuid] = {p.uuid}
    producer_q = _producer_step(MultiMemberUpstreamFG, q, PyArrowTable)

    member_a = _feature("multi_member_overlap_a", PandasDataFrame)
    member_b = _feature("multi_member_overlap_b", PandasDataFrame)

    graph.parent_to_children_mapping[member_a.uuid] = {p.uuid}
    graph.parent_to_children_mapping[member_b.uuid] = {p.uuid, q.uuid}

    feature_set = FeatureSet()
    feature_set.add(member_a)
    feature_set.add(member_b)

    step = FeatureGroupStep(MultiMemberDestFG, feature_set, set(), PandasDataFrame)

    return OverlappingAncestorsScenario(step, producer_p, producer_q, member_a, member_b, p, q, graph)


def test_transform_hop_created_for_each_members_own_direct_parent_with_overlapping_ancestor_chains() -> None:
    """member_a's direct parent p must not be pruned away just because it is also an indirect
    ancestor of member_b's direct parent q."""
    scenario = _overlapping_ancestors_scenario()

    new_plan = ExecutionPlan().add_tfs([scenario.producer_p, scenario.producer_q, scenario.step], scenario.graph)

    tfs_steps = [step for step in new_plan if isinstance(step, TransformFrameworkStep)]
    assert len(tfs_steps) == 2, (
        f"expected one transform hop for member_a's direct parent p and one for member_b's "
        f"direct parent q, got: {[(s.uuid, s.required_uuids) for s in tfs_steps]}"
    )

    hop_p = next(step for step in tfs_steps if scenario.p.uuid in step.required_uuids)
    hop_q = next(step for step in tfs_steps if scenario.q.uuid in step.required_uuids)
    assert hop_p.uuid != hop_q.uuid

    assert hop_p.uuid in scenario.step.tfs_ids
    assert hop_q.uuid in scenario.step.tfs_ids
