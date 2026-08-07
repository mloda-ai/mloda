# mypy: disable-error-code="arg-type, method-assign"
"""The run loop must abort when a required completion token is never produced.

A dropped JoinStep leaves child steps waiting on a link uuid nothing produces. A pass that
schedules nothing, completes nothing and has nothing in flight must raise MlodaRunError.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest

from mloda.core.abstract_plugins.components.error_utils import MlodaRunError
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.core.cfw_manager import CfwManager
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.core.runtime.run import ExecutionOrchestrator, _describe_step
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.helpers.plugin_stubs import make_fg

# Below the suite-wide timeout on purpose: a stall regression must fail fast, not hang.
pytestmark = pytest.mark.timeout(5)


StallFeatureGroup = make_fg("StallFeatureGroup")


class ReiterablePlan:
    """A planner stub that yields the same steps on every pass, like a real ExecutionPlan."""

    def __init__(self, *steps: Any) -> None:
        self._steps = steps

    def __iter__(self) -> Iterator[Any]:
        return iter(self._steps)


class InFlightStep:
    """A step dispatched to a worker: it reports done only on the second poll."""

    def __init__(self) -> None:
        self.uuid = uuid4()
        self.required_uuids: set[UUID] = set()
        self.step_is_done = False
        self.polls = 0

    def get_uuids(self) -> set[UUID]:
        return {self.uuid}

    def poll(self) -> None:
        """Stands in for WorkerManager.poll_result_queues draining the worker's result queue."""
        self.polls += 1
        if self.polls >= 2:
            self.step_is_done = True


class ProducerStep:
    """A parent step that runs, finishes, and hands its uuid to a child as a satisfied token."""

    def __init__(self) -> None:
        self.uuid = uuid4()
        self.required_uuids: set[UUID] = set()
        self.step_is_done = False

    def get_uuids(self) -> set[UUID]:
        return {self.uuid}


def _finish_on_execute(step: Any) -> None:
    """Stands in for sync_execute_step, which sets step_is_done before returning."""
    step.step_is_done = True


def _orchestrator(*steps: Any) -> ExecutionOrchestrator:
    """Wire a SYNC-mode orchestrator over the given steps; no compute framework is ever created."""
    orchestrator = ExecutionOrchestrator(ReiterablePlan(*steps))
    orchestrator.cfw_register = CfwManager({ParallelizationMode.SYNC})
    return orchestrator


def _feature_group_step(feature_name: str, required_uuids: set[UUID]) -> FeatureGroupStep:
    """A real FeatureGroupStep; the plan stalls before anything ever executes it."""
    return FeatureGroupStep(StallFeatureGroup, FeatureSet([Feature(feature_name)]), required_uuids, PythonDictFramework)


def test_compute_raises_when_a_required_uuid_is_never_produced() -> None:
    dangling = uuid4()
    step = _feature_group_step("stall_orphan_feature", {dangling})
    orchestrator = _orchestrator(step)

    with pytest.raises(MlodaRunError) as exc_info:
        orchestrator.compute()

    message = str(exc_info.value)
    assert str(dangling) in message, f"the unsatisfied completion token must be named; got: {message}"
    assert "stall_orphan_feature" in message, f"the waiting step must be named by its feature; got: {message}"


def test_compute_stream_raises_when_a_required_uuid_is_never_produced() -> None:
    dangling = uuid4()
    step = _feature_group_step("stall_orphan_stream_feature", {dangling})
    orchestrator = _orchestrator(step)

    with pytest.raises(MlodaRunError) as exc_info:
        list(orchestrator.compute_stream())

    assert str(dangling) in str(exc_info.value), (
        f"the unsatisfied completion token must be named; got: {exc_info.value}"
    )


def test_stall_message_describes_every_step_waiting_on_the_dropped_link_token() -> None:
    """Both children of a dropped JoinStep wait on the same link uuid, so both must be described."""
    link_token = uuid4()
    left = _feature_group_step("stall_left_feature", {link_token})
    right = _feature_group_step("stall_right_feature", {link_token})
    orchestrator = _orchestrator(left, right)

    with pytest.raises(MlodaRunError) as exc_info:
        orchestrator.compute()

    message = str(exc_info.value)
    assert str(link_token) in message, f"the dropped link token must be named; got: {message}"
    for feature_name in ("stall_left_feature", "stall_right_feature"):
        assert feature_name in message, f"every root-cause step must be described; {feature_name} missing: {message}"


def test_stall_after_a_finished_step_names_only_the_token_still_missing() -> None:
    """The real bug shape: a parent runs and finishes, then its child stalls on the dropped token."""
    producer = ProducerStep()
    dangling = uuid4()
    child = _feature_group_step("stall_child_feature", {producer.uuid, dangling})
    orchestrator = _orchestrator(producer, child)
    orchestrator._execute_step = Mock(side_effect=_finish_on_execute)

    with pytest.raises(MlodaRunError) as exc_info:
        orchestrator.compute()

    message = str(exc_info.value)
    orchestrator._execute_step.assert_called_once_with(producer)
    assert "stall_child_feature" in message, f"the stalled child must be described; got: {message}"
    assert str(dangling) in message, f"the unsatisfied completion token must be named; got: {message}"
    assert str(producer.uuid) not in message, (
        f"a token the finished parent already produced is not a cause; got: {message}"
    )


def test_stall_message_names_the_root_cause_not_the_transitively_blocked_step() -> None:
    dangling = uuid4()
    root_cause = _feature_group_step("stall_root_cause_feature", {dangling})
    transitive = _feature_group_step("stall_transitive_feature", set())
    transitive.required_uuids = set(root_cause.get_uuids())
    orchestrator = _orchestrator(root_cause, transitive)

    with pytest.raises(MlodaRunError) as exc_info:
        orchestrator.compute()

    message = str(exc_info.value)
    assert "stall_root_cause_feature" in message, f"the root-cause step must be described; got: {message}"
    assert "stall_transitive_feature" not in message, (
        f"a step blocked only by a producible token is not a cause; got: {message}"
    )


def test_stall_message_is_bounded_when_many_steps_share_one_dangling_token() -> None:
    dangling = uuid4()
    steps = [_feature_group_step(f"stall_capped_feature_{index:02d}", {dangling}) for index in range(40)]
    orchestrator = _orchestrator(*steps)

    with pytest.raises(MlodaRunError) as exc_info:
        orchestrator.compute()

    message = str(exc_info.value)
    assert str(dangling) in message, f"the unsatisfied completion token must be named; got: {message}"
    assert "more" in message, f"a truncated listing must say how many steps it left out; got: {message}"
    assert len(message) < 4000, f"the stall message must stay bounded; got {len(message)} chars"


def test_stall_message_falls_back_to_unfinished_steps_when_every_token_is_producible() -> None:
    """A cycle has no never-produced token, so the root-cause filter must not render an empty list."""
    first = _feature_group_step("stall_cycle_first_feature", set())
    second = _feature_group_step("stall_cycle_second_feature", set())
    first.required_uuids = set(second.get_uuids())
    second.required_uuids = set(first.get_uuids())
    orchestrator = _orchestrator(first, second)

    with pytest.raises(MlodaRunError) as exc_info:
        orchestrator.compute()

    message = str(exc_info.value)
    for feature_name in ("stall_cycle_first_feature", "stall_cycle_second_feature"):
        assert feature_name in message, f"a deadlocked cycle must name both steps; {feature_name} missing: {message}"


def test_describe_step_renders_a_join_step() -> None:
    link = Link.inner(JoinSpec(StallFeatureGroup, "stall_idx"), JoinSpec(StallFeatureGroup, "stall_idx"))
    step = JoinStep(link, PythonDictFramework, PyArrowTable, set(), set(), set())

    rendered = _describe_step(step)

    assert str(link.uuid) in rendered, f"the link must be identifiable; got: {rendered}"
    assert PythonDictFramework.get_class_name() in rendered, f"the destination framework is missing: {rendered}"
    assert PyArrowTable.get_class_name() in rendered, f"the source framework is missing: {rendered}"


def test_describe_step_renders_a_transform_framework_step() -> None:
    step = TransformFrameworkStep(
        PythonDictFramework, PyArrowTable, set(), StallFeatureGroup, StallFeatureGroup, None, set()
    )

    rendered = _describe_step(step)

    assert PythonDictFramework.get_class_name() in rendered, f"the source framework is missing: {rendered}"
    assert PyArrowTable.get_class_name() in rendered, f"the target framework is missing: {rendered}"
    assert rendered.index(PythonDictFramework.get_class_name()) < rendered.index(PyArrowTable.get_class_name()), (
        f"the from framework must be rendered before the to framework; got: {rendered}"
    )


def test_describe_step_survives_a_step_without_the_attributes_it_reads() -> None:
    """An error path may not raise: a stall message must survive a step that answers nothing."""
    rendered = _describe_step(Mock(spec=FeatureGroupStep))

    assert isinstance(rendered, str)
    assert rendered, "a step must always render to something non-empty"


def test_compute_does_not_raise_while_a_step_is_in_flight() -> None:
    """A pass with no scheduling and no completion is progress while a step sits in flight."""
    step = InFlightStep()
    orchestrator = _orchestrator(step)
    orchestrator.worker_manager.poll_result_queues = step.poll
    orchestrator._execute_step = Mock()

    orchestrator.compute()

    assert step.polls >= 2, "the step must stay in flight for at least one full pass with no progress"
    assert step.step_is_done is True
    orchestrator._execute_step.assert_called_once_with(step)


def test_compute_stream_does_not_raise_while_a_step_is_in_flight() -> None:
    step = InFlightStep()
    orchestrator = _orchestrator(step)
    orchestrator.worker_manager.poll_result_queues = step.poll
    orchestrator._execute_step = Mock()

    list(orchestrator.compute_stream())

    assert step.polls >= 2, "the step must stay in flight for at least one full pass with no progress"
    assert step.step_is_done is True


def test_compute_returns_on_empty_plan() -> None:
    """Pins a second, independent hang: an empty plan spun forever in compute()."""
    orchestrator = _orchestrator()

    orchestrator.compute()


def test_compute_stream_yields_nothing_on_empty_plan() -> None:
    """Pins the ordering: the empty-plan break runs before the stall check, which would raise here."""
    orchestrator = _orchestrator()

    assert list(orchestrator.compute_stream()) == []
