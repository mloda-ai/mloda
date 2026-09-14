"""ExecutionOrchestrator.__enter__ must reject a MULTIPROCESSING-capable connection before spawning a Manager."""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.runtime.run import ExecutionOrchestrator


class _ConnectionConflictOrchestratorCFW(ComputeFramework):
    """Supports MULTIPROCESSING by default; reports unavailable to avoid leaking into other tests."""

    @staticmethod
    def is_available() -> bool:
        return False


def _empty_plan() -> ExecutionPlan:
    """ExecutionPlan with execution_plan set, so an unrelated validator does not crash first."""
    plan = ExecutionPlan()
    plan.execution_plan = []
    return plan


def test_enter_with_multiprocessing_rejects_a_resolved_connection_before_spawning_a_manager() -> None:
    orchestrator = ExecutionOrchestrator(
        _empty_plan(), tfs_connection_map={_ConnectionConflictOrchestratorCFW: object()}
    )

    with pytest.raises(ValueError, match=_ConnectionConflictOrchestratorCFW.__name__):
        orchestrator.__enter__({ParallelizationMode.MULTIPROCESSING})

    assert orchestrator.manager is None, "no MyManager/worker process may be created on the rejection path"


def test_enter_with_sync_mode_does_not_reject_the_same_connection_map() -> None:
    orchestrator = ExecutionOrchestrator(
        _empty_plan(), tfs_connection_map={_ConnectionConflictOrchestratorCFW: object()}
    )

    orchestrator.__enter__({ParallelizationMode.SYNC})

    orchestrator.__exit__(None, None, None)
