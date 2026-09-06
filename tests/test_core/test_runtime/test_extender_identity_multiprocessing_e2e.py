"""Proves ExecutionOrchestrator keeps the caller's function_extender objects, not independent
unpickled copies, when constructing multiple ComputeFrameworks in the parent process.

Empirically checked first (see PR discussion): every ExtenderHook tied to computing a feature
(FEATURE_GROUP_CALCULATE_FEATURE, JOIN) runs its __call__ inside a spawned MULTIPROCESSING worker,
never in the parent, because a FeatureGroupStep/JoinStep is always dispatched via
multi_execute_step once MULTIPROCESSING is a register mode; there is no hook whose __call__
reliably fires in-parent for both THREADING and MULTIPROCESSING. So this test takes the
integration-style fallback: it drives ExecutionOrchestrator.__enter__ and
ComputeFrameworkExecutor.init_compute_framework directly (real machinery, no mocks) and inspects
the constructed ComputeFramework.function_extender in the parent process right after construction,
before anything would be shipped to a worker.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.user import ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class _CountingExtender(Extender):
    """Plain instance state (a counter), incremented directly in the test, not via __call__:
    proves shared identity through mutation-visibility, not merely two objects that started equal."""

    def __init__(self) -> None:
        self.calls = 0

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


def _empty_plan() -> ExecutionPlan:
    plan = ExecutionPlan()
    plan.execution_plan = []
    return plan


@pytest.mark.timeout(30)
@pytest.mark.parametrize("mode", [ParallelizationMode.THREADING, ParallelizationMode.MULTIPROCESSING])
class TestExtenderIdentityAcrossFrameworksBuiltInTheParent:
    def test_same_extender_object_is_used_to_construct_every_framework(self, mode: ParallelizationMode) -> None:
        extender = _CountingExtender()
        caller_function_extender: set[Extender] = {extender}

        orchestrator = ExecutionOrchestrator(_empty_plan())
        orchestrator.__enter__({mode}, caller_function_extender)
        try:
            orchestrator._init_run()

            uuid_one = orchestrator.executor.init_compute_framework(PythonDictFramework, mode, set(), uuid4())
            uuid_two = orchestrator.executor.init_compute_framework(PythonDictFramework, mode, set(), uuid4())

            cfw_one = orchestrator.executor.cfw_collection[uuid_one]
            cfw_two = orchestrator.executor.cfw_collection[uuid_two]

            extender_on_one = next(iter(cfw_one.function_extender))
            extender_on_two = next(iter(cfw_two.function_extender))

            # Identity, not equality: two independently-unpickled copies are never `is` each
            # other nor `is` the object the caller passed in, even though both start empty.
            assert extender_on_one is extender, "framework one must be built with the caller's own extender object"
            assert extender_on_two is extender, "framework two must be built with the caller's own extender object"
            assert extender_on_one is extender_on_two, "both frameworks must share the identical extender object"

            # Mutation-visibility: only possible if it is the same object, not copies that
            # merely started from the same initial state.
            extender_on_one.calls += 1
            assert extender_on_two.calls == 1, "a mutation via one framework's extender must be visible via the other's"
        finally:
            orchestrator.__exit__(None, None, None)
