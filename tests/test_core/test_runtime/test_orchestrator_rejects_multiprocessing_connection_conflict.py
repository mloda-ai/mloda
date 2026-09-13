"""ExecutionOrchestrator.__enter__ must reject a tfs_connection_map entry whose compute framework
class supports ParallelizationMode.MULTIPROCESSING before spawning any multiprocessing Manager,
mirroring test_orchestrator_rejects_unpicklable_step_feature_group.py's wiring-test shape. It must
not touch SYNC mode.
"""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.runtime.run import ExecutionOrchestrator


class _ConnectionConflictOrchestratorCFW(ComputeFramework):
    """A third-party framework that keeps the base default: supports all parallelization modes,
    including MULTIPROCESSING. Reports unavailable so it never leaks into the accessible-plugin
    pool for unrelated tests sharing this pytest-xdist worker."""

    @staticmethod
    def is_available() -> bool:
        return False


def _empty_plan() -> ExecutionPlan:
    """An ExecutionPlan with execution_plan explicitly set, mirroring
    _plan_with_unpicklable_feature_group_step()'s pattern so this test does not rely on which
    validator __enter__ happens to run first. A bare ExecutionPlan() has no execution_plan
    attribute until create_execution_plan() runs, and raise_on_unpicklable_join_link (which reads
    it) would crash with an unrelated AttributeError instead of the ValueError this test expects,
    were it ever reordered ahead of the connection-conflict check.
    """
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
