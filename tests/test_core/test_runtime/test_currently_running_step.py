"""The running check must answer for the whole step, not for one arbitrary uuid."""

from __future__ import annotations

from uuid import UUID, uuid4

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.core.cfw_manager import CfwManager
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.runtime.run import ExecutionOrchestrator


def _orchestrator() -> ExecutionOrchestrator:
    """A SYNC orchestrator over an empty plan; the running check never touches a plan step."""
    orchestrator = ExecutionOrchestrator(ExecutionPlan())
    orchestrator.cfw_register = CfwManager({ParallelizationMode.SYNC})
    return orchestrator


def _step_uuids_and_a_later_one() -> tuple[set[UUID], UUID]:
    """A two-uuid step plus the uuid its iterator does not reach first."""
    step_uuids = {uuid4(), uuid4()}
    first = next(iter(step_uuids))
    later = next(uuid for uuid in step_uuids if uuid != first)
    return step_uuids, later


def test_currently_running_step_reports_a_uuid_the_iterator_reaches_second() -> None:
    step_uuids, later = _step_uuids_and_a_later_one()

    assert _orchestrator().currently_running_step(step_uuids, {later}) is True


def test_currently_running_step_reports_every_running_uuid_of_the_step() -> None:
    orchestrator = _orchestrator()
    step_uuids = {uuid4(), uuid4()}

    running = [uuid for uuid in step_uuids if orchestrator.currently_running_step(step_uuids, {uuid})]

    assert sorted(running) == sorted(step_uuids), f"every uuid of the step must count as running; got: {running}"


def test_currently_running_step_agrees_with_the_refusal_of_can_run_step() -> None:
    orchestrator = _orchestrator()
    step_uuids, later = _step_uuids_and_a_later_one()

    refused = not orchestrator._can_run_step(set(), step_uuids, set(), {later}, None)

    assert orchestrator.currently_running_step(step_uuids, {later}) is refused


def test_currently_running_step_is_false_when_no_uuid_of_the_step_is_running() -> None:
    step_uuids = {uuid4(), uuid4()}

    assert _orchestrator().currently_running_step(step_uuids, {uuid4()}) is False
