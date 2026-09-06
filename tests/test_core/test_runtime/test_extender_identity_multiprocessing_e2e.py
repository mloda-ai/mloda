"""Proves ExecutionOrchestrator keeps the caller's function_extender objects, not independent
unpickled copies, across ComputeFrameworks built in the parent process, but only for frameworks
that themselves resolve to a non-MULTIPROCESSING mode. A framework resolved to MULTIPROCESSING is
dispatched to a spawned worker via Process(args=(cfw_register, cfw, from_cfw)), which pickles it;
it must keep getting an isolated, independently-fetched extender copy so an in-parent mutation of
the shared object (e.g. an extender lazily building an unpicklable handle) can never poison it.

Drives ExecutionOrchestrator.__enter__ and ComputeFrameworkExecutor.init_compute_framework directly
(real machinery, no mocks), since no ExtenderHook's __call__ reliably fires in-parent under
MULTIPROCESSING (a FeatureGroupStep/JoinStep is always dispatched to a worker once
MULTIPROCESSING is a register mode)."""

from __future__ import annotations

import pickle  # nosec B403
import threading
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


class _HandleHoldingExtender(Extender):
    """Simulates the documented pattern of lazily building a runtime handle inside __call__:
    `handle` starts None (picklable) and is later set to something pickle cannot handle."""

    def __init__(self) -> None:
        self.handle: Any = None

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


def _empty_plan() -> ExecutionPlan:
    plan = ExecutionPlan()
    plan.execution_plan = []
    return plan


@pytest.mark.timeout(30)
@pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
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


@pytest.mark.timeout(30)
class TestExtenderIdentityNotSharedWithMultiprocessingResolvedFramework:
    """MAJ-3 regression guard: a mixed run (overall MULTIPROCESSING-enabled, real proxy register)
    must still give in-parent-resident frameworks the caller's shared extender, while a framework
    resolved to MULTIPROCESSING (worker-bound) gets its own isolated copy through the real proxy,
    so it cannot be poisoned by another framework's later in-parent mutation of the shared object."""

    def test_parent_resident_shares_identity_worker_bound_gets_isolated_copy(self) -> None:
        extender = _HandleHoldingExtender()
        caller_function_extender: set[Extender] = {extender}

        orchestrator = ExecutionOrchestrator(_empty_plan())
        # A real manager/proxy is constructed here because MULTIPROCESSING is in the register modes.
        orchestrator.__enter__({ParallelizationMode.MULTIPROCESSING}, caller_function_extender)
        try:
            orchestrator._init_run()

            parent_resident_uuid = orchestrator.executor.init_compute_framework(
                PythonDictFramework, ParallelizationMode.SYNC, set(), uuid4()
            )
            worker_bound_uuid = orchestrator.executor.init_compute_framework(
                PythonDictFramework, ParallelizationMode.MULTIPROCESSING, set(), uuid4()
            )

            parent_resident_cfw = orchestrator.executor.cfw_collection[parent_resident_uuid]
            worker_bound_cfw = orchestrator.executor.cfw_collection[worker_bound_uuid]

            parent_resident_extender = next(iter(parent_resident_cfw.function_extender))
            worker_bound_extender = next(iter(worker_bound_cfw.function_extender))

            assert parent_resident_extender is extender, (
                "a framework resolved to a non-MULTIPROCESSING mode must still share the caller's "
                "real extender object, even inside a MULTIPROCESSING-enabled run"
            )
            assert worker_bound_extender is not extender, (
                "a framework resolved to MULTIPROCESSING must get its own independent copy from "
                "the register's proxy, not the caller's shared object"
            )

            # Simulate a hook that lazily built an unpicklable handle on the shared object, mutating
            # it only after both frameworks were already constructed.
            extender.handle = threading.Lock()

            # This is what Process(args=...) would pickle to dispatch the worker-bound framework.
            # It must succeed: the worker-bound framework's own extender copy is untouched by the
            # later mutation of the caller's shared object.
            pickle.dumps(worker_bound_cfw)  # nosec B301
        finally:
            orchestrator.__exit__(None, None, None)
