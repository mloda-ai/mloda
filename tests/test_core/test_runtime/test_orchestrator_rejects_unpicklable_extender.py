"""ExecutionOrchestrator.__enter__ must reject a function_extender it cannot pickle for
multiprocessing before spawning any Manager, and must not touch SYNC mode at all.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.runtime.run import ExecutionOrchestrator


class OrchestratorUnpicklableInstanceExtender(Extender):
    """Module-level Extender class whose instance holds a threading.Lock: never picklable."""

    def __init__(self) -> None:
        self.lock = threading.Lock()

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


def _empty_plan() -> ExecutionPlan:
    plan = ExecutionPlan()
    plan.execution_plan = []
    return plan


def test_enter_with_multiprocessing_rejects_an_unpicklable_extender_before_spawning_a_manager() -> None:
    orchestrator = ExecutionOrchestrator(_empty_plan())

    with pytest.raises(ValueError):
        orchestrator.__enter__({ParallelizationMode.MULTIPROCESSING}, {OrchestratorUnpicklableInstanceExtender()})

    assert orchestrator.manager is None, "no MyManager/worker process may be created on the rejection path"


def test_enter_with_sync_mode_does_not_reject_the_same_unpicklable_extender() -> None:
    orchestrator = ExecutionOrchestrator(_empty_plan())

    orchestrator.__enter__({ParallelizationMode.SYNC}, {OrchestratorUnpicklableInstanceExtender()})

    orchestrator.__exit__(None, None, None)
