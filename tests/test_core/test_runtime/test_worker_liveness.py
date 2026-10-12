"""Failing tests: the orchestrator must abort when a worker dies without reporting an error.

``ExecutionOrchestrator.compute`` loops until every step UUID is finished. A
worker SIGKILL'd by the OOM killer never emits its step UUID (so it is never
marked finished) AND never runs its except block (so ``cfw_register.set_error``
is never called). ``_check_for_error`` therefore returns ``False`` and the loop
spins forever.

The fix adds ``WorkerManager.find_dead_workers`` (see test_worker_manager.py)
and has ``_check_for_error`` raise ``MlodaRunError`` when it returns a non-empty
list. This test pins that liveness check.

It FAILS today because ``_check_for_error`` does not inspect worker liveness: it
returns ``False`` and never raises, so ``pytest.raises(MlodaRunError)`` reports
DID NOT RAISE.
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.components.error_utils import MlodaRunError
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.runtime.run import ExecutionOrchestrator


def test_check_for_error_raises_when_worker_dead_without_reported_error() -> None:
    """_check_for_error must raise MlodaRunError for a dead worker even with no reported error.

    This is the SIGKILL/OOM case: ``get_error`` returns ``None`` (the worker
    never reached its except block), yet a registered process has a non-zero
    exitcode. Without a liveness check the run loop would never terminate.
    """
    orchestrator = ExecutionOrchestrator(Mock(spec=ExecutionPlan))

    cfw_register = MagicMock()
    cfw_register.get_error.return_value = None
    cfw_register.get_parallelization_modes.return_value = set()
    orchestrator.cfw_register = cfw_register

    dead_process = MagicMock()
    dead_process.exitcode = -9
    dead_process.is_alive.return_value = False
    orchestrator.worker_manager.process_register[uuid4()] = (dead_process, MagicMock(), MagicMock())

    with pytest.raises(MlodaRunError):
        orchestrator._check_for_error()


@pytest.mark.parametrize(
    ("exitcode", "assign_step"),
    [(-9, False), (0, True)],
    ids=["dead worker", "orphaned steps"],
)
def test_check_for_error_surfaces_error_reported_just_before_worker_exit(exitcode: int, assign_step: bool) -> None:
    """An error reported after the first read but before the liveness raise wins over MlodaRunError."""
    orchestrator = ExecutionOrchestrator(Mock(spec=ExecutionPlan))
    reads: list[int] = []

    def get_error() -> bool:
        reads.append(1)
        return len(reads) > 1

    original = SystemExit(3)
    cfw_register = MagicMock()
    cfw_register.get_error.side_effect = get_error
    cfw_register.take_error_exception.return_value = original
    cfw_register.get_parallelization_modes.return_value = set()
    orchestrator.cfw_register = cfw_register

    cfw_uuid = uuid4()
    process = MagicMock()
    process.exitcode = exitcode
    process.is_alive.return_value = False
    orchestrator.worker_manager.process_register[cfw_uuid] = (process, MagicMock(), MagicMock())
    if assign_step:
        orchestrator.worker_manager.assigned_steps[cfw_uuid] = {uuid4()}

    with pytest.raises(SystemExit) as exc_info:
        orchestrator._check_for_error()

    assert exc_info.value is original
