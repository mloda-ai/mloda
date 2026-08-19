"""Regression coverage for add_tfs: a deduped TransformFrameworkStep must still wire its uuid
into every consuming step, not just the one that first created it (Scenario A), and that two
same-shaped hops from genuinely different parents must not dedup into one (Scenario B).
"""

from typing import NamedTuple
from uuid import UUID, uuid4

import pytest

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
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


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
    producers: list[FeatureGroupStep]
    graph: Graph


def _root_node(graph: Graph, feature: Feature, fg: type[FeatureGroup]) -> None:
    # Graph.add_node is a plain dict assignment and this reset always assigns the same value, so
    # calling this twice for the same feature (the same_parent=True scenario below) is harmless.
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


def _producer_step(fg: type[FeatureGroup], feature: Feature, cfw: type[ComputeFramework]) -> FeatureGroupStep:
    feature_set = FeatureSet()
    feature_set.add(feature)
    return FeatureGroupStep(fg, feature_set, set(), cfw)


def _feature_group_step_dedup_scenario(same_parent: bool) -> FeatureGroupStepDedupScenario:
    """Two FeatureGroupSteps whose built ``TransformFrameworkStep``s share from/to framework and
    feature-group shape. ``same_parent=True`` pulls both from the same upstream feature (must
    dedup); ``same_parent=False`` pulls from different upstream features (must not dedup).

    Each parent is also produced by a ``FeatureGroupStep`` (``producers``), so the built hops key
    on ``owning_step_of`` for real instead of falling back to the raw parent uuid."""
    graph = Graph()

    parent_a = _feature("dedup_upstream_a", PyArrowTable)
    parent_b = parent_a if same_parent else _feature("dedup_upstream_b", PyArrowTable)
    _root_node(graph, parent_a, DedupUpstreamFG)
    _root_node(graph, parent_b, DedupUpstreamFG)

    producer_a = _producer_step(DedupUpstreamFG, parent_a, PyArrowTable)
    producer_b = producer_a if same_parent else _producer_step(DedupUpstreamFG, parent_b, PyArrowTable)
    producers = [producer_a] if same_parent else [producer_a, producer_b]

    dest_feature_a = _feature("dedup_dest_a", PandasDataFrame)
    dest_feature_b = _feature("dedup_dest_b", PandasDataFrame)
    step_a = _dest_step(DedupDestFG, dest_feature_a, PandasDataFrame, parent_a.uuid, graph)
    step_b = _dest_step(DedupDestFG, dest_feature_b, PandasDataFrame, parent_b.uuid, graph)

    return FeatureGroupStepDedupScenario(step_a, step_b, parent_a, parent_b, producers, graph)


def test_two_feature_group_steps_with_the_same_parent_and_shape_dedup_into_one_transform_step() -> None:
    """Guard against an overly-broad fix: genuinely shared parents must still dedup to one hop."""
    scenario = _feature_group_step_dedup_scenario(same_parent=True)

    new_plan = ExecutionPlan().add_tfs([*scenario.producers, scenario.step_a, scenario.step_b], scenario.graph)

    tfs_steps = [step for step in new_plan if isinstance(step, TransformFrameworkStep)]
    assert len(tfs_steps) == 1, f"expected the two same-parent hops to dedup into one, got: {tfs_steps}"
    tfs_uuid = tfs_steps[0].uuid

    assert scenario.step_a.tfs_ids == {tfs_uuid}
    assert scenario.step_b.tfs_ids == {tfs_uuid}

    assert tfs_uuid in scenario.step_a.required_uuids
    assert tfs_uuid in scenario.step_b.required_uuids


def test_two_feature_group_steps_with_different_parents_get_separate_transform_hops() -> None:
    """A hop only ever moves one physical source's data (execute() takes a single from_cfw), so
    collapsing two different-parent hops into one would starve whichever step's uuid lost the
    dedup of its own source data."""
    scenario = _feature_group_step_dedup_scenario(same_parent=False)
    producer_a, producer_b = scenario.producers

    new_plan = ExecutionPlan().add_tfs([*scenario.producers, scenario.step_a, scenario.step_b], scenario.graph)

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

    # Pins the owning-step keying itself, not just its fallback: each hop's source_step_uuid is
    # the producing FeatureGroupStep's uuid, not the raw parent uuid.
    assert hop_a.source_step_uuid == producer_a.uuid
    assert hop_b.source_step_uuid == producer_b.uuid

    assert scenario.step_a.tfs_ids == {hop_a.uuid}
    assert scenario.step_b.tfs_ids == {hop_b.uuid}
    assert scenario.step_a.required_uuids == {hop_a.uuid}
    assert scenario.step_b.required_uuids == {hop_b.uuid}


# ---------------------------------------------------------------------------
# Bug A / Bug B regression coverage
# ---------------------------------------------------------------------------


def test_two_hops_from_the_same_feature_group_class_do_not_raise() -> None:
    """One FeatureGroup class commonly splits into multiple steps by (framework, options,
    dependency level); two such steps feeding one consumer share one conceptual source and
    must not be mistaken for the missing-Link case."""
    graph = Graph()

    parent_a = _feature("dedup_shared_a", PyArrowTable)
    parent_b = _feature("dedup_shared_b", PyArrowTable)
    _root_node(graph, parent_a, DedupUpstreamFG)
    _root_node(graph, parent_b, DedupUpstreamFG)

    producer_a = _producer_step(DedupUpstreamFG, parent_a, PyArrowTable)
    producer_b = _producer_step(DedupUpstreamFG, parent_b, PyArrowTable)

    dest_feature = _feature("dedup_dest_multi", PandasDataFrame)
    feature_set = FeatureSet()
    feature_set.add(dest_feature)
    graph.parent_to_children_mapping[dest_feature.uuid] = {parent_a.uuid, parent_b.uuid}
    dest_step = FeatureGroupStep(DedupDestFG, feature_set, set(), PandasDataFrame)

    new_plan = ExecutionPlan().add_tfs([producer_a, producer_b, dest_step], graph)

    tfs_steps = [step for step in new_plan if isinstance(step, TransformFrameworkStep)]
    assert len(tfs_steps) == 2, f"expected two distinct hops (different producer instances), got: {tfs_steps}"
    assert dest_step.tfs_ids == {step.uuid for step in tfs_steps}


def test_parents_linked_by_join_requires_genuine_opposite_sides() -> None:
    """required_uuids on a JoinStep holds every parent of its consumers, not just its own two
    sides; two parents merely co-occurring there must not read as linked."""
    a, b, dest, src, unrelated = uuid4(), uuid4(), uuid4(), uuid4(), uuid4()
    link = Link.inner(JoinSpec(DedupLeftFG, "id"), JoinSpec(DedupRightFG, "id"))
    join_step = JoinStep(
        link=link,
        destination_framework=PandasDataFrame,
        source_framework=PyArrowTable,
        required_uuids={a, b, unrelated, dest, src},
        destination_framework_uuids={dest},
        source_framework_uuids={src},
    )

    assert ExecutionPlan._parents_linked_by_join(a, b, {join_step}) is False
    assert ExecutionPlan._parents_linked_by_join(dest, src, {join_step}) is True


# ---------------------------------------------------------------------------
# A parent matching more than one JoinStep must not depend on set iteration order
# ---------------------------------------------------------------------------


class MatchedJsHubFG(DedupBaseFG):
    """A hub genuinely joined to two different sides, so its uuid is a real side of two JoinSteps."""


class MatchedJsAFG(DedupBaseFG):
    pass


class MatchedJsYFG(DedupBaseFG):
    pass


class MatchedJsConsumerFG(DedupBaseFG):
    pass


def _hub_matches_two_joins_scenario(
    hub_link_token: UUID, other_link_token: UUID
) -> tuple[list[JoinStep | FeatureGroupStep], Graph]:
    """Hub parent P genuinely matches both JoinSteps; only the token order differs between calls. Q is the
    genuine source-side member of ``hub_join_step`` (what ``run_link`` actually produces: source uuids
    narrowed to genuine declared-side members), so P and Q are linked via real uuid adjacency, not merely
    by sharing a declared-side class."""
    graph = Graph()

    p = _feature("matched_js_hub_p", PandasDataFrame)
    q = _feature("matched_js_a_q", PyArrowTable)
    graph.add_node(p.uuid, NodeProperties(p, MatchedJsHubFG))
    graph.add_node(q.uuid, NodeProperties(q, MatchedJsAFG))
    graph.parent_to_children_mapping[p.uuid] = set()
    graph.parent_to_children_mapping[q.uuid] = set()

    hub_link = Link.inner(JoinSpec(MatchedJsAFG, "id"), JoinSpec(MatchedJsHubFG, "id"))
    other_link = Link.inner(JoinSpec(MatchedJsHubFG, "id2"), JoinSpec(MatchedJsYFG, "id2"))

    hub_join_step = JoinStep(
        link=hub_link,
        destination_framework=PandasDataFrame,
        source_framework=PyArrowTable,
        required_uuids=set(),
        destination_framework_uuids={p.uuid},
        source_framework_uuids={q.uuid},
        token=hub_link_token,
    )
    other_join_step = JoinStep(
        link=other_link,
        destination_framework=PandasDataFrame,
        source_framework=PythonDictFramework,
        required_uuids=set(),
        destination_framework_uuids={p.uuid},
        source_framework_uuids={uuid4()},
        token=other_link_token,
    )

    consumer = _feature("matched_js_consumer", PandasDataFrame)
    feature_set = FeatureSet()
    feature_set.add(consumer)
    graph.parent_to_children_mapping[consumer.uuid] = {p.uuid, q.uuid}
    consumer_step = FeatureGroupStep(MatchedJsConsumerFG, feature_set, set(), PandasDataFrame)

    return [hub_join_step, other_join_step, consumer_step], graph


@pytest.mark.parametrize(
    "hub_link_token, other_link_token",
    [(UUID(int=1), UUID(int=2)), (UUID(int=2), UUID(int=1))],
    ids=["hub_link_lower_token", "other_link_lower_token"],
)
def test_parent_matching_two_join_steps_does_not_raise_regardless_of_token_order(
    hub_link_token: UUID, other_link_token: UUID
) -> None:
    execution_plan, graph = _hub_matches_two_joins_scenario(hub_link_token, other_link_token)

    ExecutionPlan().add_tfs(execution_plan, graph)


# ---------------------------------------------------------------------------
# A join-served parent bridging two otherwise-unlinked explicit hops
# ---------------------------------------------------------------------------


class BridgeXFG(DedupBaseFG):
    pass


class BridgeYFG(DedupBaseFG):
    pass


class BridgeUnrelatedFG(DedupBaseFG):
    pass


class BridgeConsumerFG(DedupBaseFG):
    pass


class BridgeScenario(NamedTuple):
    producer_x: FeatureGroupStep
    producer_y: FeatureGroupStep
    join_step: JoinStep
    dest_step: FeatureGroupStep
    graph: Graph


def _bridge_scenario(bridged: bool) -> BridgeScenario:
    """Two genuinely unlinked explicit hops (producer_x, class BridgeXFG on PandasDataFrame;
    producer_y, class BridgeYFG on PyArrowTable) feed one consumer on PythonDictFramework, plus
    join-served parents whose class is an exact match for a hop's own class, on PythonDictFramework
    (what ``run_link`` actually produces: destination/source uuids are narrowed to genuine
    declared-side members). ``bridged=True`` gives BOTH hops a join-served counterpart, and the two
    counterparts sit on opposite sides of the same JoinStep, so they are themselves genuinely
    linked; this ties both explicit hops into one group. ``bridged=False`` gives only the left hop
    a counterpart, and the link's right side names an unrelated class, so the right hop stays
    genuinely unlinked."""
    graph = Graph()

    parent_x = _feature("bridge_parent_x", PandasDataFrame)
    parent_y = _feature("bridge_parent_y", PyArrowTable)
    bridge_left = _feature("bridge_left_served", PythonDictFramework)
    _root_node(graph, parent_x, BridgeXFG)
    _root_node(graph, parent_y, BridgeYFG)
    _root_node(graph, bridge_left, BridgeXFG)

    producer_x = _producer_step(BridgeXFG, parent_x, PandasDataFrame)
    producer_y = _producer_step(BridgeYFG, parent_y, PyArrowTable)

    right_side = BridgeYFG if bridged else BridgeUnrelatedFG
    link = Link.inner(JoinSpec(BridgeXFG, "id"), JoinSpec(right_side, "id"))

    destination_framework_uuids = {bridge_left.uuid}
    source_framework_uuids: set[UUID] = set()
    dest_parents = {parent_x.uuid, parent_y.uuid, bridge_left.uuid}

    if bridged:
        bridge_right = _feature("bridge_right_served", PythonDictFramework)
        _root_node(graph, bridge_right, BridgeYFG)
        source_framework_uuids = {bridge_right.uuid}
        dest_parents.add(bridge_right.uuid)

    join_step = JoinStep(
        link=link,
        destination_framework=PythonDictFramework,
        source_framework=PythonDictFramework,
        required_uuids=set(),
        destination_framework_uuids=destination_framework_uuids,
        source_framework_uuids=source_framework_uuids,
    )

    dest_feature = _feature("bridge_dest", PythonDictFramework)
    feature_set = FeatureSet()
    feature_set.add(dest_feature)
    graph.parent_to_children_mapping[dest_feature.uuid] = dest_parents
    dest_step = FeatureGroupStep(BridgeConsumerFG, feature_set, set(), PythonDictFramework)

    return BridgeScenario(producer_x, producer_y, join_step, dest_step, graph)


def test_join_served_parent_genuinely_bridging_two_hops_does_not_raise() -> None:
    """Two join-served parents, each an exact-class match for one explicit hop and genuinely
    adjacent to each other on the same JoinStep, must merge both hops into one group instead of
    tripping the missing-Links check on the bound-entries-only grouping pass."""
    scenario = _bridge_scenario(bridged=True)

    ExecutionPlan().add_tfs(
        [scenario.producer_x, scenario.producer_y, scenario.join_step, scenario.dest_step], scenario.graph
    )


def test_join_served_parent_linked_to_only_one_hop_still_raises() -> None:
    """Guard against an overly-broad fix: a join-served parent must genuinely bridge BOTH hops,
    not merely be present, for the missing-Links check to stand down."""
    scenario = _bridge_scenario(bridged=False)

    with pytest.raises(ValueError, match="two different, unlinked source feature"):
        ExecutionPlan().add_tfs(
            [scenario.producer_x, scenario.producer_y, scenario.join_step, scenario.dest_step], scenario.graph
        )


# ---------------------------------------------------------------------------
# A join-served parent must not bridge two hops that are merely siblings under a
# shared declared-side ancestor, with no genuine join between them
# ---------------------------------------------------------------------------


class SiblingBroadFG(DedupBaseFG):
    """The class a link declares on one side; two unrelated sibling subclasses share it."""


class SiblingS1FG(SiblingBroadFG):
    pass


class SiblingS2FG(SiblingBroadFG):
    pass


class SiblingRightFG(DedupBaseFG):
    pass


class SiblingConsumerFG(DedupBaseFG):
    pass


class SiblingBridgeScenario(NamedTuple):
    producer_s1: FeatureGroupStep
    producer_s2: FeatureGroupStep
    join_step: JoinStep
    consumer_step: FeatureGroupStep
    graph: Graph


def _sibling_bridge_scenario() -> SiblingBridgeScenario:
    """Consumer (PythonDict) pulls from S1 (Pandas) and S2 (PyArrow), two sibling subclasses of a
    link's declared left side with no Link between them, plus a genuinely join-served third parent
    whose class equals S1's exactly (a real declared-side member, as ``run_link`` narrows
    destination/source uuids to declared-side members only). The join-served parent legitimately
    binds to the S1 hop; it must not also bridge the unrelated S2 hop just because S2 happens to
    descend from the same declared-side base class."""
    graph = Graph()

    p_s1 = _feature("sibling_s1_pandas", PandasDataFrame)
    p_s2 = _feature("sibling_s2_pyarrow", PyArrowTable)
    p_served = _feature("sibling_s1_dict_served", PythonDictFramework)
    _root_node(graph, p_s1, SiblingS1FG)
    _root_node(graph, p_s2, SiblingS2FG)
    _root_node(graph, p_served, SiblingS1FG)

    producer_s1 = _producer_step(SiblingS1FG, p_s1, PandasDataFrame)
    producer_s2 = _producer_step(SiblingS2FG, p_s2, PyArrowTable)

    link = Link.inner(JoinSpec(SiblingBroadFG, "id"), JoinSpec(SiblingRightFG, "id"))
    right_parent = _feature("sibling_right_dict", PythonDictFramework)
    _root_node(graph, right_parent, SiblingRightFG)
    join_step = JoinStep(
        link=link,
        destination_framework=PythonDictFramework,
        source_framework=PythonDictFramework,
        required_uuids=set(),
        destination_framework_uuids={p_served.uuid, right_parent.uuid},
        source_framework_uuids={p_served.uuid, right_parent.uuid},
    )

    consumer = _feature("sibling_consumer", PythonDictFramework)
    feature_set = FeatureSet()
    feature_set.add(consumer)
    graph.parent_to_children_mapping[consumer.uuid] = {p_s1.uuid, p_s2.uuid, p_served.uuid}
    consumer_step = FeatureGroupStep(SiblingConsumerFG, feature_set, set(), PythonDictFramework)

    return SiblingBridgeScenario(producer_s1, producer_s2, join_step, consumer_step, graph)


def test_join_served_sibling_parent_does_not_bridge_two_unrelated_declared_side_subclasses() -> None:
    """A join-served parent classed as one declared-side subclass must not silently merge an
    unrelated sibling subclass's hop into its group; the two hops share no genuine Link."""
    scenario = _sibling_bridge_scenario()

    with pytest.raises(ValueError, match="two different, unlinked source feature"):
        ExecutionPlan().add_tfs(
            [scenario.producer_s1, scenario.producer_s2, scenario.join_step, scenario.consumer_step],
            scenario.graph,
        )
