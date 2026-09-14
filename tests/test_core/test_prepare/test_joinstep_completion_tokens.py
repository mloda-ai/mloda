"""One completion token per JoinStep, expanded into every step that waits on the link.

The cycle guard runs over the finished plan, so it sees join steps and feature group steps alike.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple
from uuid import UUID, uuid4

import pytest

from mloda.core.core.step.abstract_step import Step
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import NodeProperties
from mloda.core.prepare.resolve_links import LinkTrekker
from mloda.provider import BaseInputData
from mloda.provider import ComputeFramework
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import JoinSpec, Link
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from tests.test_core.test_prepare.join_plan_helpers import feature, trek


TOKEN_LEFT_INDEX = Index(("token_left_key",))
TOKEN_RIGHT_INDEX = Index(("token_right_key",))
OTHER_LEFT_INDEX = Index(("token_other_left_key",))
OTHER_RIGHT_INDEX = Index(("token_other_right_key",))

APPEND_HEAD_INDEX = Index(("token_append_head_key",))
APPEND_MIDDLE_INDEX = Index(("token_append_middle_key",))
APPEND_TAIL_INDEX = Index(("token_append_tail_key",))

SELF_SIDE = "token_self_side"
SELF_LEFT_KEY = "token_self_left_key"
SELF_LEFT_PAYLOAD = "token_self_left_payload"
SELF_RIGHT_KEY = "token_self_right_key"
SELF_RIGHT_PAYLOAD = "token_self_right_payload"


class TokenLeft(FeatureGroup):
    pass


class TokenRight(FeatureGroup):
    pass


class TokenChild(FeatureGroup):
    pass


class TokenOtherLeft(FeatureGroup):
    pass


class TokenOtherRight(FeatureGroup):
    pass


class TokenAppendSource(FeatureGroup):
    pass


class TokenSelfSource(FeatureGroup):
    """Serves both sides of the self join; the requested feature name picks the side."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={SELF_LEFT_PAYLOAD, SELF_RIGHT_PAYLOAD})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if SELF_LEFT_PAYLOAD in {str(feature.name) for feature in features.features}:
            return {SELF_LEFT_KEY: [1, 2], SELF_LEFT_PAYLOAD: ["l1", "l2"]}
        return {SELF_RIGHT_KEY: [1, 2], SELF_RIGHT_PAYLOAD: ["r1", "r2"]}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class TokenSelfConsumer(FeatureGroup):
    """Consumes both sides of the self join; the options are what the discriminators match on."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature(name=SELF_LEFT_PAYLOAD, options={SELF_SIDE: "left"}),
            Feature(name=SELF_RIGHT_PAYLOAD, options={SELF_SIDE: "right"}),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}


class Planned(NamedTuple):
    plan: ExecutionPlan
    graph: Graph
    link_trekker: LinkTrekker
    pre_execution_plan: list[Any]
    queue: list[Any]


class Branch(NamedTuple):
    consumer: FeatureGroupStep
    left_uuid: UUID
    right_uuid: UUID


def _step(fg: type[FeatureGroup], feature: Feature, required_uuids: set[UUID]) -> FeatureGroupStep:
    feature_set = FeatureSet()
    feature_set.add(feature)
    return FeatureGroupStep(fg, feature_set, required_uuids, feature.get_compute_framework())


def _planned() -> Planned:
    return Planned(ExecutionPlan(), Graph(), LinkTrekker(), [], [])


def _token_link() -> Link:
    return Link.inner(JoinSpec(TokenLeft, TOKEN_LEFT_INDEX), JoinSpec(TokenRight, TOKEN_RIGHT_INDEX))


def _other_link() -> Link:
    return Link.inner(JoinSpec(TokenOtherLeft, OTHER_LEFT_INDEX), JoinSpec(TokenOtherRight, OTHER_RIGHT_INDEX))


def _self_link(left_side: str, right_side: str) -> Link:
    return Link.left(
        JoinSpec(TokenSelfSource, Index((SELF_LEFT_KEY,))),
        JoinSpec(TokenSelfSource, Index((SELF_RIGHT_KEY,))),
        left_discriminator={SELF_SIDE: left_side},
        right_discriminator={SELF_SIDE: right_side},
    )


def _add_branch(
    planned: Planned,
    link: Link,
    name: str,
    left_cfw: type[ComputeFramework] = PyArrowTable,
    right_cfw: type[ComputeFramework] = PandasDataFrame,
    plan_the_link: bool = True,
) -> Branch:
    """Two parents joined by ``link`` plus the consumer, the smallest shape run_link accepts."""
    left = feature(f"{name}_left", left_cfw, link.left_index)
    right = feature(f"{name}_right", right_cfw, link.right_index)
    child = feature(f"{name}_child", left_cfw)

    graph = planned.graph
    graph.add_node(left.uuid, NodeProperties(left, link.left_feature_group))
    graph.add_node(right.uuid, NodeProperties(right, link.right_feature_group))
    graph.add_node(child.uuid, NodeProperties(child, TokenChild))
    graph.adjacency_list[left.uuid] = [child.uuid]
    graph.adjacency_list[right.uuid] = [child.uuid]
    graph.adjacency_list[child.uuid] = []
    graph.parent_to_children_mapping[child.uuid] = {left.uuid, right.uuid}

    consumer = _step(TokenChild, child, {left.uuid, right.uuid, link.uuid})
    planned.pre_execution_plan.append(_step(link.left_feature_group, left, set()))
    planned.pre_execution_plan.append(_step(link.right_feature_group, right, set()))
    planned.queue.append((link.left_feature_group, {left}))
    planned.queue.append((link.right_feature_group, {right}))
    if plan_the_link:
        planned.pre_execution_plan.append((link, left_cfw, right_cfw))
        planned.queue.append((link, left_cfw, right_cfw))
    planned.pre_execution_plan.append(consumer)
    planned.queue.append((TokenChild, {child}))

    trek(planned.link_trekker, link, (left_cfw, right_cfw), child.uuid)
    return Branch(consumer, left.uuid, right.uuid)


def _add_joinstep(planned: Planned) -> list[JoinStep | FeatureGroupStep]:
    """add_feature_group_step fills the collections on the queue path; this path has to do it itself."""
    planned.plan.feature_set_collections = [
        step.get_uuids() for step in planned.pre_execution_plan if isinstance(step, FeatureGroupStep)
    ]
    return planned.plan.add_joinstep(planned.pre_execution_plan, planned.link_trekker, planned.graph)


def _create_execution_plan(planned: Planned) -> list[Step]:
    planned.plan.create_execution_plan(planned.queue, planned.graph, planned.link_trekker)
    return list(planned.plan)


def _join_steps(steps: Sequence[Step], link: Link) -> list[JoinStep]:
    return [step for step in steps if isinstance(step, JoinStep) and step.link is link]


def _run_self_join(link: Link) -> list[Any]:
    return list(
        mloda.run_all(
            [Feature(name=TokenSelfConsumer.get_class_name())],
            links={link},
            compute_frameworks=["PyArrowTable"],
            plugin_collector=PluginCollector.enabled_feature_groups({TokenSelfSource, TokenSelfConsumer}),
        )
    )


def test_a_joinstep_stamps_only_its_own_completion_token() -> None:
    link = _token_link()
    step = JoinStep(link, PyArrowTable, PandasDataFrame, set(), set(), set())

    assert step.get_uuids() == {step.uuid}
    assert link.uuid not in step.get_uuids(), "the link uuid is an identity, not a completion token"


def test_two_joinsteps_of_one_link_carry_disjoint_completion_tokens() -> None:
    link = _token_link()
    declared = JoinStep(link, PyArrowTable, PandasDataFrame, set(), set(), set())
    inverted = JoinStep(link, PandasDataFrame, PyArrowTable, set(), set(), set())

    assert declared.get_uuids().isdisjoint(inverted.get_uuids()), (
        "a shared token lets a consumer unblock after whichever orientation finishes first"
    )


def test_add_joinstep_replaces_the_link_token_of_a_consumer_with_the_planned_joinstep() -> None:
    planned = _planned()
    link = _token_link()
    branch = _add_branch(planned, link, "single")

    fw_execution_plan = _add_joinstep(planned)

    join_steps = _join_steps(fw_execution_plan, link)
    assert len(join_steps) == 1, f"the declared orientation must plan one JoinStep; got: {join_steps}"
    assert link.uuid not in branch.consumer.required_uuids, "no step may keep waiting on the link uuid"
    assert join_steps[0].uuid in branch.consumer.required_uuids


def test_add_joinstep_leaves_the_parent_uuids_a_consumer_waits_for_untouched() -> None:
    """Only link uuids are expanded; the feature uuids of both parents stay required."""
    planned = _planned()
    branch = _add_branch(planned, _token_link(), "parents")

    _add_joinstep(planned)

    assert {branch.left_uuid, branch.right_uuid} <= branch.consumer.required_uuids, (
        f"the expansion swallowed a parent uuid; got: {branch.consumer.required_uuids}"
    )


def test_add_joinstep_makes_a_consumer_wait_for_both_orientations_of_its_link() -> None:
    planned = _planned()
    link = _token_link()
    declared = _add_branch(planned, link, "declared", PyArrowTable, PandasDataFrame)
    inverted = _add_branch(planned, link, "inverted", PandasDataFrame, PyArrowTable)

    fw_execution_plan = _add_joinstep(planned)

    join_uuids = {step.uuid for step in _join_steps(fw_execution_plan, link)}
    assert len(join_uuids) == 2, f"both orientations must plan a JoinStep; got: {join_uuids}"
    for branch in (declared, inverted):
        assert link.uuid not in branch.consumer.required_uuids, "no step may keep waiting on the link uuid"
        assert join_uuids.issubset(branch.consumer.required_uuids), (
            f"a consumer of the link must wait for every JoinStep of it; got: {branch.consumer.required_uuids}"
        )


def test_add_joinstep_rejects_a_link_token_that_no_joinstep_produces() -> None:
    """The link reaches the trekker but never the plan, so nothing would ever stamp its token."""
    planned = _planned()
    link = _token_link()
    _add_branch(planned, link, "unplanned", plan_the_link=False)

    with pytest.raises(ValueError) as excinfo:
        _add_joinstep(planned)

    assert str(link.uuid) in str(excinfo.value), f"the unproduced link must be named; got: {excinfo.value}"


def test_a_self_join_whose_discriminators_match_nothing_is_rejected_naming_the_link() -> None:
    """No orientation pairs the two nodes up, so no JoinStep stamps the token the consumer waits for."""
    link = _self_link("no_such_left", "no_such_right")

    with pytest.raises(ValueError) as excinfo:
        _run_self_join(link)

    assert str(link.uuid) in str(excinfo.value), f"the link that planned nothing must be named; got: {excinfo.value}"


def test_a_discriminator_that_matches_nothing_is_reported_as_configuration_not_as_an_mloda_bug() -> None:
    with pytest.raises(ValueError) as excinfo:
        _run_self_join(_self_link("no_such_left", "no_such_right"))

    message = str(excinfo.value)
    assert "discriminator" in message, f"the cause the user can fix must be named; got: {message}"
    assert "Internal error" not in message, f"this is user configuration, not an invariant breach; got: {message}"


def test_add_joinstep_never_makes_a_joinstep_wait_for_itself() -> None:
    """Hand built: the ordering entry names the link itself, which the expansion must not honour."""
    planned = _planned()
    link = _token_link()
    _add_branch(planned, link, "selfwait")
    planned.link_trekker.order[link.uuid] = {link.uuid}

    fw_execution_plan = _add_joinstep(planned)

    join_step = _join_steps(fw_execution_plan, link)[0]
    assert link.uuid not in join_step.required_uuids
    assert join_step.required_uuids.isdisjoint(join_step.get_uuids()), (
        f"a JoinStep may not wait for a token it produces itself; got: {join_step.required_uuids}"
    )


def test_a_chained_append_joinstep_waits_for_the_joinstep_of_the_link_it_was_handed() -> None:
    """Hand built JoinSteps stand in for run_link's output: the head appends what the tail first completed."""
    planned = _planned()
    head_link = Link.append(
        JoinSpec(TokenAppendSource, APPEND_HEAD_INDEX), JoinSpec(TokenAppendSource, APPEND_MIDDLE_INDEX)
    )
    tail_link = Link.append(
        JoinSpec(TokenAppendSource, APPEND_MIDDLE_INDEX), JoinSpec(TokenAppendSource, APPEND_TAIL_INDEX)
    )
    head_uuid, middle_uuid, tail_uuid = uuid4(), uuid4(), uuid4()

    head = JoinStep(head_link, PyArrowTable, PyArrowTable, {head_uuid, middle_uuid}, {head_uuid}, {middle_uuid})
    tail = JoinStep(tail_link, PyArrowTable, PyArrowTable, {middle_uuid, tail_uuid}, {middle_uuid}, {tail_uuid})
    planned.pre_execution_plan.extend([head, tail])
    trek(planned.link_trekker, head_link, (PyArrowTable, PyArrowTable), head_uuid)
    trek(planned.link_trekker, tail_link, (PyArrowTable, PyArrowTable), middle_uuid)

    _add_joinstep(planned)

    assert tail.uuid in head.required_uuids, f"the chained edge must name the tail JoinStep; got: {head.required_uuids}"
    assert tail_link.uuid not in head.required_uuids, "no step may keep waiting on the link uuid"
    assert {head_uuid, middle_uuid} <= head.required_uuids, "the expansion swallowed a parent uuid"


def test_create_execution_plan_rejects_a_cycle_the_expansion_leaves_between_two_joinsteps() -> None:
    """Hand built: each link is ordered after the other, so expanding both tokens deadlocks the run."""
    planned = _planned()
    link = _token_link()
    other = _other_link()
    _add_branch(planned, link, "cycle_first")
    _add_branch(planned, other, "cycle_second")
    planned.link_trekker.order[link.uuid] = {other.uuid}
    planned.link_trekker.order[other.uuid] = {link.uuid}

    with pytest.raises(ValueError, match="(?i)cycl"):
        _create_execution_plan(planned)


def test_create_execution_plan_plans_a_chain_of_two_ordered_links() -> None:
    """One ordering edge is a chain, not a cycle, and the finished plan keeps it as a JoinStep token."""
    planned = _planned()
    link = _token_link()
    other = _other_link()
    _add_branch(planned, link, "chain_first")
    _add_branch(planned, other, "chain_second")
    planned.link_trekker.order[link.uuid] = {other.uuid}

    execution_plan = _create_execution_plan(planned)

    first = _join_steps(execution_plan, link)[0]
    second = _join_steps(execution_plan, other)[0]
    assert first.uuid in second.required_uuids, (
        f"the chained JoinStep must wait for the first; got: {second.required_uuids}"
    )
    assert link.uuid not in second.required_uuids, "no step may keep waiting on the link uuid"


def test_raise_on_step_cycle_rejects_three_joinsteps_waiting_on_each_other() -> None:
    """The pair shape is broken upstream already, so a three step ring is what this guard is for."""
    steps: list[Step] = [JoinStep(_token_link(), PyArrowTable, PandasDataFrame, set(), set(), set()) for _ in range(3)]
    for step, waits_for in zip(steps, steps[1:] + steps[:1]):
        step.required_uuids.add(waits_for.uuid)

    with pytest.raises(ValueError, match="(?i)cycl"):
        ExecutionPlan().raise_on_step_cycle(steps)


def test_raise_on_step_cycle_rejects_a_cycle_running_through_a_feature_group_step() -> None:
    """A JoinStep waiting on a feature its consumer produces is as unrunnable as a JoinStep pair."""
    cycle_feature = feature("token_cycle_feature", PyArrowTable)
    join_step = JoinStep(_token_link(), PyArrowTable, PandasDataFrame, set(), set(), set())
    feature_group_step = _step(TokenChild, cycle_feature, {join_step.uuid})
    join_step.required_uuids.add(cycle_feature.uuid)
    steps: list[Step] = [join_step, feature_group_step]

    with pytest.raises(ValueError, match="(?i)cycl"):
        ExecutionPlan().raise_on_step_cycle(steps)


def test_raise_on_step_cycle_accepts_a_token_no_step_of_the_plan_produces() -> None:
    """The runtime already raises for an unproduced token; a missing producer is not a cycle."""
    steps: list[Step] = [JoinStep(_token_link(), PyArrowTable, PandasDataFrame, {uuid4()}, set(), set())]

    ExecutionPlan().raise_on_step_cycle(steps)


# A join hop's owed_tokens mirrors a plain hop's: the destination-side credit stamped on its SOURCE-side cfw.

CHAIN_INDEX = Index(("token_chain_key",))
GUARD_INDEX = Index(("token_guard_key",))


class TokenChainLeft(FeatureGroup):
    pass


class TokenChainMiddle(FeatureGroup):
    pass


class TokenChainRight(FeatureGroup):
    pass


class TokenChainConsumer(FeatureGroup):
    pass


class TokenGuardSource(FeatureGroup):
    pass


class TokenGuardSourceSibling(FeatureGroup):
    pass


class TokenGuardDest(FeatureGroup):
    pass


class TokenGuardDestConsumer(FeatureGroup):
    pass


class TokenGuardSiblingConsumer(FeatureGroup):
    pass


class ChainPlanned(NamedTuple):
    consumer: FeatureGroupStep
    plan: list[Step]


class GuardPlanned(NamedTuple):
    dest_consumer: FeatureGroupStep
    plan: list[Step]


def _plan_chained_three_framework_join() -> ChainPlanned:
    """Pandas<-PyArrow<-PythonDict chain: one consumer reads all three parents through two joins."""
    planned = _planned()
    graph = planned.graph

    left = feature("token_chain_left", PandasDataFrame, CHAIN_INDEX)
    middle = feature("token_chain_middle", PyArrowTable, CHAIN_INDEX)
    right = feature("token_chain_right", PythonDictFramework, CHAIN_INDEX)
    consumer_feature = feature("token_chain_consumer", PandasDataFrame)

    for member, fg in ((left, TokenChainLeft), (middle, TokenChainMiddle), (right, TokenChainRight)):
        graph.add_node(member.uuid, NodeProperties(member, fg))
        graph.adjacency_list[member.uuid] = [consumer_feature.uuid]
    graph.add_node(consumer_feature.uuid, NodeProperties(consumer_feature, TokenChainConsumer))
    graph.adjacency_list[consumer_feature.uuid] = []
    graph.parent_to_children_mapping[consumer_feature.uuid] = {left.uuid, middle.uuid, right.uuid}

    link_left_middle = Link.inner(JoinSpec(TokenChainLeft, CHAIN_INDEX), JoinSpec(TokenChainMiddle, CHAIN_INDEX))
    link_middle_right = Link.inner(JoinSpec(TokenChainMiddle, CHAIN_INDEX), JoinSpec(TokenChainRight, CHAIN_INDEX))

    planned.queue.append((TokenChainLeft, {left}))
    planned.queue.append((TokenChainMiddle, {middle}))
    planned.queue.append((TokenChainRight, {right}))
    planned.queue.append((link_left_middle, PandasDataFrame, PyArrowTable))
    planned.queue.append((link_middle_right, PyArrowTable, PythonDictFramework))
    planned.queue.append((TokenChainConsumer, {consumer_feature}))

    trek(planned.link_trekker, link_left_middle, (PandasDataFrame, PyArrowTable), consumer_feature.uuid)
    trek(planned.link_trekker, link_middle_right, (PyArrowTable, PythonDictFramework), consumer_feature.uuid)

    plan = _create_execution_plan(planned)
    consumer = next(s for s in plan if isinstance(s, FeatureGroupStep) and s.feature_group is TokenChainConsumer)
    return ChainPlanned(consumer, plan)


def _plan_join_hop_whose_source_framework_is_also_another_joins_framework() -> GuardPlanned:
    """A cross-framework join hop whose PythonDict source is ALSO the same-framework partner of an
    unrelated JoinStep elsewhere in the plan (mirrors a same-framework join re-pointing a shared cfw)."""
    planned = _planned()
    graph = planned.graph

    source = feature("token_guard_source", PythonDictFramework, GUARD_INDEX)
    sibling = feature("token_guard_source_sibling", PythonDictFramework, GUARD_INDEX)
    dest = feature("token_guard_dest", PyArrowTable, GUARD_INDEX)
    dest_consumer = feature("token_guard_dest_consumer", PyArrowTable)
    sibling_consumer = feature("token_guard_sibling_consumer", PythonDictFramework)

    graph.add_node(source.uuid, NodeProperties(source, TokenGuardSource))
    graph.adjacency_list[source.uuid] = [dest_consumer.uuid, sibling_consumer.uuid]
    graph.add_node(sibling.uuid, NodeProperties(sibling, TokenGuardSourceSibling))
    graph.adjacency_list[sibling.uuid] = [sibling_consumer.uuid]
    graph.add_node(dest.uuid, NodeProperties(dest, TokenGuardDest))
    graph.adjacency_list[dest.uuid] = [dest_consumer.uuid]
    graph.add_node(dest_consumer.uuid, NodeProperties(dest_consumer, TokenGuardDestConsumer))
    graph.adjacency_list[dest_consumer.uuid] = []
    graph.add_node(sibling_consumer.uuid, NodeProperties(sibling_consumer, TokenGuardSiblingConsumer))
    graph.adjacency_list[sibling_consumer.uuid] = []
    graph.parent_to_children_mapping[dest_consumer.uuid] = {source.uuid, dest.uuid}
    graph.parent_to_children_mapping[sibling_consumer.uuid] = {source.uuid, sibling.uuid}

    link_dest_source = Link.inner(JoinSpec(TokenGuardDest, GUARD_INDEX), JoinSpec(TokenGuardSource, GUARD_INDEX))
    link_source_sibling = Link.inner(
        JoinSpec(TokenGuardSource, GUARD_INDEX), JoinSpec(TokenGuardSourceSibling, GUARD_INDEX)
    )

    planned.queue.append((TokenGuardSource, {source}))
    planned.queue.append((TokenGuardSourceSibling, {sibling}))
    planned.queue.append((TokenGuardDest, {dest}))
    planned.queue.append((link_source_sibling, PythonDictFramework, PythonDictFramework))
    planned.queue.append((link_dest_source, PyArrowTable, PythonDictFramework))
    planned.queue.append((TokenGuardSiblingConsumer, {sibling_consumer}))
    planned.queue.append((TokenGuardDestConsumer, {dest_consumer}))

    trek(planned.link_trekker, link_source_sibling, (PythonDictFramework, PythonDictFramework), sibling_consumer.uuid)
    trek(planned.link_trekker, link_dest_source, (PyArrowTable, PythonDictFramework), dest_consumer.uuid)

    plan = _create_execution_plan(planned)
    dest_consumer_step = next(
        s for s in plan if isinstance(s, FeatureGroupStep) and s.feature_group is TokenGuardDestConsumer
    )
    return GuardPlanned(dest_consumer_step, plan)


def test_chained_cross_framework_join_hop_owed_tokens_exclude_a_third_framework_consumer() -> None:
    """Guards that the chain's Pandas consumer, which also reads the PythonDict source through a plain
    hop, is not credited by the PythonDict join hop."""
    chained = _plan_chained_three_framework_join()

    python_dict_hop = next(
        step
        for step in chained.plan
        if isinstance(step, TransformFrameworkStep)
        and step.link_id is not None
        and step.from_framework is PythonDictFramework
    )

    assert not (python_dict_hop.owed_tokens & chained.consumer.get_uuids()), (
        f"the PythonDict join hop must not credit the chain's own consumer directly; got "
        f"owed_tokens={python_dict_hop.owed_tokens}, consumer uuids={chained.consumer.get_uuids()}"
    )


def test_cross_framework_join_hop_owed_tokens_equal_its_destination_consumers_own_uuids() -> None:
    """A join hop's SOURCE-side cfw is only otherwise marked consumed by the join it serves; the hop
    itself must credit the destination-framework consumer's own uuids, mirroring a plain hop's owed_tokens."""
    planned = _planned()
    link = _token_link()
    branch = _add_branch(planned, link, "owed", PyArrowTable, PandasDataFrame)

    plan = _create_execution_plan(planned)

    hop = next(step for step in plan if isinstance(step, TransformFrameworkStep) and step.link_id == link.uuid)

    assert hop.owed_tokens == branch.consumer.get_uuids()


def test_cross_framework_append_join_hop_owed_tokens_stay_empty() -> None:
    """APPEND/UNION join hops stay uncredited (finalize-only timing)."""
    planned = _planned()
    link = Link.append(JoinSpec(TokenLeft, TOKEN_LEFT_INDEX), JoinSpec(TokenRight, TOKEN_RIGHT_INDEX))
    _add_branch(planned, link, "append_owed", PyArrowTable, PandasDataFrame)

    plan = _create_execution_plan(planned)

    hop = next(step for step in plan if isinstance(step, TransformFrameworkStep) and step.link_id == link.uuid)

    assert hop.owed_tokens == frozenset()


def test_join_hop_whose_source_framework_is_also_another_joins_framework_gets_empty_owed_tokens() -> None:
    """A join hop is ineligible for owed-token crediting when its source framework is also the source
    or destination of another JoinStep; here PythonDict is also an unrelated join's same-framework
    partner, so the hop stays uncredited even though a genuine destination-framework consumer exists."""
    guarded = _plan_join_hop_whose_source_framework_is_also_another_joins_framework()

    hop = next(
        step
        for step in guarded.plan
        if isinstance(step, TransformFrameworkStep)
        and step.link_id is not None
        and step.from_framework is PythonDictFramework
    )

    assert hop.owed_tokens == frozenset()


# One PythonDict source reaches one PyArrow consumer through both a join hop and a plain hop.

SHARED_SOURCE_INDEX = Index(("token_shared_source_key",))


class TokenSharedDerived(FeatureGroup):
    pass


class TokenSharedSource(TokenSharedDerived):
    pass


class TokenSharedDest(FeatureGroup):
    pass


class TokenSharedConsumer(FeatureGroup):
    pass


class SharedSourcePlanned(NamedTuple):
    consumer: FeatureGroupStep
    plan: list[Step]
    link: Link


def _plan_join_hop_and_plain_hop_share_one_source_framework() -> SharedSourcePlanned:
    """The source is split into a join-index feature and a payload feature so the consumer's index parent
    survives grandparent pruning; the join hop's source class subclasses the plain hop's, linking the two hops."""
    planned = _planned()
    graph = planned.graph

    source_key = feature("token_shared_source_key", PythonDictFramework, SHARED_SOURCE_INDEX)
    source_payload = feature("token_shared_source_payload", PythonDictFramework)
    derived = feature("token_shared_derived", PythonDictFramework)
    dest = feature("token_shared_dest", PyArrowTable, SHARED_SOURCE_INDEX)
    consumer_feature = feature("token_shared_consumer", PyArrowTable)

    graph.add_node(source_key.uuid, NodeProperties(source_key, TokenSharedSource))
    graph.adjacency_list[source_key.uuid] = [consumer_feature.uuid]
    graph.add_node(source_payload.uuid, NodeProperties(source_payload, TokenSharedSource))
    graph.adjacency_list[source_payload.uuid] = [derived.uuid]
    graph.add_node(derived.uuid, NodeProperties(derived, TokenSharedDerived))
    graph.adjacency_list[derived.uuid] = [consumer_feature.uuid]
    graph.parent_to_children_mapping[derived.uuid] = {source_payload.uuid}
    graph.add_node(dest.uuid, NodeProperties(dest, TokenSharedDest))
    graph.adjacency_list[dest.uuid] = [consumer_feature.uuid]
    graph.add_node(consumer_feature.uuid, NodeProperties(consumer_feature, TokenSharedConsumer))
    graph.adjacency_list[consumer_feature.uuid] = []
    graph.parent_to_children_mapping[consumer_feature.uuid] = {source_key.uuid, derived.uuid, dest.uuid}

    link = Link.inner(JoinSpec(TokenSharedDest, SHARED_SOURCE_INDEX), JoinSpec(TokenSharedSource, SHARED_SOURCE_INDEX))

    planned.queue.append((TokenSharedSource, {source_key}))
    planned.queue.append((TokenSharedSource, {source_payload}))
    planned.queue.append((TokenSharedDerived, {derived}))
    planned.queue.append((TokenSharedDest, {dest}))
    planned.queue.append((link, PyArrowTable, PythonDictFramework))
    planned.queue.append((TokenSharedConsumer, {consumer_feature}))

    trek(planned.link_trekker, link, (PyArrowTable, PythonDictFramework), consumer_feature.uuid)

    plan = _create_execution_plan(planned)
    consumer = next(s for s in plan if isinstance(s, FeatureGroupStep) and s.feature_group is TokenSharedConsumer)
    return SharedSourcePlanned(consumer, plan, link)


def test_a_consumer_reaching_two_hops_of_one_source_framework_is_credited_by_neither() -> None:
    """A consumer that reaches the same PythonDict source through a join hop and a plain hop must get
    no credit from either: crediting one lets the source cfw drop while the other hop is still pending."""
    shared = _plan_join_hop_and_plain_hop_share_one_source_framework()

    join_step = next(s for s in shared.plan if isinstance(s, JoinStep) and s.link is shared.link)
    join_hop = next(
        s
        for s in shared.plan
        if isinstance(s, TransformFrameworkStep) and s.link_id is not None and s.from_framework is PythonDictFramework
    )
    plain_hop = next(
        s
        for s in shared.plan
        if isinstance(s, TransformFrameworkStep) and s.link_id is None and s.from_framework is PythonDictFramework
    )

    # Guard: the fixture must actually reach both hops, or the assertions below would pass vacuously.
    assert plain_hop.uuid in shared.consumer.required_uuids, (
        f"the consumer must wait directly on the plain hop; got {shared.consumer.required_uuids}"
    )
    assert join_step.uuid in shared.consumer.required_uuids, (
        f"the consumer must wait on the JoinStep the join hop serves; got {shared.consumer.required_uuids}"
    )
    assert join_hop.uuid in join_step.required_uuids, (
        f"the JoinStep must wait on the join hop that serves it; got {join_step.required_uuids}"
    )

    consumer_uuids = shared.consumer.get_uuids()
    assert not (join_hop.owed_tokens & consumer_uuids), (
        f"the join hop must not credit a consumer that still needs a second, plain hop of the same "
        f"source framework; got owed_tokens={join_hop.owed_tokens}, consumer uuids={consumer_uuids}"
    )
    assert not (plain_hop.owed_tokens & consumer_uuids), (
        f"got owed_tokens={plain_hop.owed_tokens}, consumer uuids={consumer_uuids}"
    )
