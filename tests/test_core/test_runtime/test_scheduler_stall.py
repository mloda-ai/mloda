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
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.core.cfw_manager import CfwManager
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.runtime.run import ExecutionOrchestrator
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


def _orchestrator(*steps: Any) -> ExecutionOrchestrator:
    """Wire a SYNC-mode orchestrator over the given steps; no compute framework is ever created."""
    orchestrator = ExecutionOrchestrator(ReiterablePlan(*steps))
    orchestrator.cfw_register = CfwManager({ParallelizationMode.SYNC})
    return orchestrator


def _feature_group_step(feature_name: str, required_uuids: set[UUID]) -> FeatureGroupStep:
    """A real FeatureGroupStep; the plan stalls before anything ever executes it."""
    return FeatureGroupStep(StallFeatureGroup, FeatureSet([Feature(feature_name)]), required_uuids, PythonDictFramework)


def _unique_identifiers(step: FeatureGroupStep) -> tuple[str, ...]:
    """Tokens that point at this step and no other: its feature names and its uuids."""
    return (*step.features.get_all_names(), str(step.uuid), *(str(uuid) for uuid in step.get_uuids()))


def test_compute_raises_when_a_required_uuid_is_never_produced() -> None:
    dangling = uuid4()
    step = _feature_group_step("stall_orphan_feature", {dangling})
    orchestrator = _orchestrator(step)

    with pytest.raises(MlodaRunError) as exc_info:
        orchestrator.compute()

    message = str(exc_info.value)
    assert str(dangling) in message, f"the unsatisfied completion token must be named; got: {message}"
    identifiers = ("FeatureGroupStep", StallFeatureGroup.get_class_name(), *_unique_identifiers(step))
    assert any(token in message for token in identifiers), (
        f"the waiting step must be identifiable via one of {identifiers}; got: {message}"
    )


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
    for step in (left, right):
        identifiers = _unique_identifiers(step)
        assert any(token in message for token in identifiers), (
            f"every waiting step must be described via one of {identifiers}; got: {message}"
        )


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
    orchestrator = _orchestrator()

    orchestrator.compute()


def test_compute_stream_yields_nothing_on_empty_plan() -> None:
    orchestrator = _orchestrator()

    assert list(orchestrator.compute_stream()) == []
