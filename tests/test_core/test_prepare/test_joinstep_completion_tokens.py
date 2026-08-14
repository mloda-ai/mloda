"""One completion token per JoinStep, expanded into every step that waits on the link.

The cycle guard runs over the finished plan, so it sees join steps and feature group steps alike.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple, Optional
from uuid import UUID, uuid4

import pytest

from mloda.core.core.step.abstract_step import Step
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
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
    def input_data(cls) -> Optional[BaseInputData]:
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

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
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
