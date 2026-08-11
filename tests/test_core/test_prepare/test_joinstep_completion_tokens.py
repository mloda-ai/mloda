"""One completion token per JoinStep, expanded into every step that waits on the link."""

from __future__ import annotations

from typing import Any, NamedTuple
from uuid import UUID

import pytest

from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.graph.properties import NodeProperties
from mloda.core.prepare.resolve_links import LinkTrekker
from mloda.provider import ComputeFramework
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import Index
from mloda.user import JoinSpec, Link
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


TOKEN_LEFT_INDEX = Index(("token_left_key",))
TOKEN_RIGHT_INDEX = Index(("token_right_key",))
OTHER_LEFT_INDEX = Index(("token_other_left_key",))
OTHER_RIGHT_INDEX = Index(("token_other_right_key",))


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


class Planned(NamedTuple):
    plan: ExecutionPlan
    graph: Graph
    link_trekker: LinkTrekker
    pre_execution_plan: list[Any]


def _feature(name: str, cfw: type[ComputeFramework], index: Index | None = None) -> Feature:
    feature = Feature(name, index=index)
    feature.compute_frameworks = {cfw}
    return feature


def _step(fg: type[FeatureGroup], feature: Feature, required_uuids: set[UUID]) -> FeatureGroupStep:
    feature_set = FeatureSet()
    feature_set.add(feature)
    return FeatureGroupStep(fg, feature_set, required_uuids, feature.get_compute_framework())


def _planned() -> Planned:
    return Planned(ExecutionPlan(), Graph(), LinkTrekker(), [])


def _token_link() -> Link:
    return Link.inner(JoinSpec(TokenLeft, TOKEN_LEFT_INDEX), JoinSpec(TokenRight, TOKEN_RIGHT_INDEX))


def _other_link() -> Link:
    return Link.inner(JoinSpec(TokenOtherLeft, OTHER_LEFT_INDEX), JoinSpec(TokenOtherRight, OTHER_RIGHT_INDEX))


def _add_branch(
    planned: Planned,
    link: Link,
    name: str,
    left_cfw: type[ComputeFramework] = PyArrowTable,
    right_cfw: type[ComputeFramework] = PandasDataFrame,
    plan_the_link: bool = True,
) -> FeatureGroupStep:
    """Two parents joined by ``link`` plus the consumer, the smallest shape run_link accepts."""
    left = _feature(f"{name}_left", left_cfw, link.left_index)
    right = _feature(f"{name}_right", right_cfw, link.right_index)
    child = _feature(f"{name}_child", left_cfw)

    graph = planned.graph
    graph.add_node(left.uuid, NodeProperties(left, link.left_feature_group))
    graph.add_node(right.uuid, NodeProperties(right, link.right_feature_group))
    graph.add_node(child.uuid, NodeProperties(child, TokenChild))
    graph.adjacency_list[left.uuid] = [child.uuid]
    graph.adjacency_list[right.uuid] = [child.uuid]
    graph.adjacency_list[child.uuid] = []
    graph.parent_to_children_mapping[child.uuid] = {left.uuid, right.uuid}

    planned.plan.feature_set_collections.extend([{left.uuid}, {right.uuid}, {child.uuid}])

    consumer = _step(TokenChild, child, {left.uuid, right.uuid, link.uuid})
    planned.pre_execution_plan.append(_step(link.left_feature_group, left, set()))
    planned.pre_execution_plan.append(_step(link.right_feature_group, right, set()))
    if plan_the_link:
        planned.pre_execution_plan.append((link, left_cfw, right_cfw))
    planned.pre_execution_plan.append(consumer)

    # Production shares one set object between data and data_ordered, and invert_link relies on that.
    trekked = {child.uuid}
    planned.link_trekker.data[(link, left_cfw, right_cfw)] = trekked
    planned.link_trekker.data_ordered[(link, left_cfw, right_cfw)] = trekked
    return consumer


def _add_joinstep(planned: Planned) -> list[JoinStep | FeatureGroupStep]:
    return planned.plan.add_joinstep(planned.pre_execution_plan, planned.link_trekker, planned.graph)


def _join_steps(fw_execution_plan: list[JoinStep | FeatureGroupStep], link: Link) -> list[JoinStep]:
    return [step for step in fw_execution_plan if isinstance(step, JoinStep) and step.link is link]


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
    consumer = _add_branch(planned, link, "single")

    fw_execution_plan = _add_joinstep(planned)

    join_steps = _join_steps(fw_execution_plan, link)
    assert len(join_steps) == 1, f"the declared orientation must plan one JoinStep; got: {join_steps}"
    assert link.uuid not in consumer.required_uuids, "no step may keep waiting on the link uuid"
    assert join_steps[0].uuid in consumer.required_uuids


def test_add_joinstep_makes_a_consumer_wait_for_both_orientations_of_its_link() -> None:
    planned = _planned()
    link = _token_link()
    declared_consumer = _add_branch(planned, link, "declared", PyArrowTable, PandasDataFrame)
    inverted_consumer = _add_branch(planned, link, "inverted", PandasDataFrame, PyArrowTable)

    fw_execution_plan = _add_joinstep(planned)

    join_uuids = {step.uuid for step in _join_steps(fw_execution_plan, link)}
    assert len(join_uuids) == 2, f"both orientations must plan a JoinStep; got: {join_uuids}"
    for consumer in (declared_consumer, inverted_consumer):
        assert link.uuid not in consumer.required_uuids, "no step may keep waiting on the link uuid"
        assert join_uuids.issubset(consumer.required_uuids), (
            f"a consumer of the link must wait for every JoinStep of it; got: {consumer.required_uuids}"
        )


def test_add_joinstep_rejects_a_link_token_that_no_joinstep_produces() -> None:
    """The link reaches the trekker but never the plan, so nothing would ever stamp its token."""
    planned = _planned()
    link = _token_link()
    _add_branch(planned, link, "unplanned", plan_the_link=False)

    with pytest.raises(ValueError) as excinfo:
        _add_joinstep(planned)

    assert str(link.uuid) in str(excinfo.value), f"the unproduced link must be named; got: {excinfo.value}"


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


def test_add_joinstep_rejects_a_cycle_the_expansion_leaves_between_two_joinsteps() -> None:
    """Hand built: each link is ordered after the other, so expanding both tokens deadlocks the run."""
    planned = _planned()
    link = _token_link()
    other = _other_link()
    _add_branch(planned, link, "cycle_first")
    _add_branch(planned, other, "cycle_second")
    planned.link_trekker.order[link.uuid] = {other.uuid}
    planned.link_trekker.order[other.uuid] = {link.uuid}

    with pytest.raises(ValueError, match="(?i)cycl"):
        _add_joinstep(planned)
