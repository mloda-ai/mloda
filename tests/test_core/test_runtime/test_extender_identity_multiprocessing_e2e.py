"""Proves ExecutionOrchestrator keeps the caller's function_extender objects, not independent
unpickled copies, across ComputeFrameworks built in the parent process, but only for frameworks
that themselves resolve to a non-MULTIPROCESSING mode. A framework resolved to MULTIPROCESSING is
dispatched to a spawned worker via Process(args=(cfw_register, cfw, from_cfw)), which pickles it;
it must keep a pending, still-pickled snapshot taken once at run entry, materialized only once the
worker actually unpickles it (ComputeFramework.__setstate__), so an in-parent mutation of the
shared object (e.g. an extender lazily building an unpicklable handle) can never poison it.

Drives ExecutionOrchestrator.__enter__ and ComputeFrameworkExecutor.init_compute_framework directly
(real machinery, no mocks), for precise control over which mode each individual framework resolves
to. test_extender_handle_survives_multiprocessing_run_e2e.py covers the same guarantee through a
real mlodaAPI run, where an ExtenderHook's __call__ does fire in-parent for a step whose compute
framework resolves to a non-MULTIPROCESSING mode."""

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


class _HandleStrippingExtender(Extender):
    """A live, unpicklable handle held from construction, stripped in __getstate__, so pickling
    this instance always succeeds regardless of when it runs."""

    def __init__(self) -> None:
        self.handle: Any = threading.Lock()

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["handle"] = None
        return state


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
    """A mixed run (overall MULTIPROCESSING-enabled, real manager/register) must still give
    in-parent-resident frameworks the caller's shared extender, while a framework resolved to
    MULTIPROCESSING (worker-bound) only carries a still-pickled snapshot (_pending_extender_payload),
    never a copy already unpickled in the parent, so it cannot be poisoned by another framework's
    later in-parent mutation of the shared object."""

    def test_parent_resident_shares_identity_worker_bound_defers_the_pending_payload(self) -> None:
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

            assert parent_resident_extender is extender, (
                "a framework resolved to a non-MULTIPROCESSING mode must still share the caller's "
                "real extender object, even inside a MULTIPROCESSING-enabled run"
            )
            assert worker_bound_cfw.function_extender == set(), (
                "a framework resolved to MULTIPROCESSING must not have its extender materialized "
                "in the parent; it must stay an empty set until the worker unpickles it"
            )
            assert worker_bound_cfw._pending_extender_payload is not None, (
                "the worker-bound framework must carry the still-pickled snapshot, materialized "
                "only by ComputeFramework.__setstate__ once the worker actually unpickles it"
            )

            # Simulate a hook that lazily built an unpicklable handle on the shared object, mutating
            # it only after both frameworks were already constructed.
            extender.handle = threading.Lock()

            # This is what Process(args=...) would pickle to dispatch the worker-bound framework.
            # It must succeed: the pending payload was snapshotted before the later mutation of the
            # caller's shared object, and is never touched here in the parent.
            pickle.dumps(worker_bound_cfw)
        finally:
            orchestrator.__exit__(None, None, None)


@pytest.mark.timeout(30)
class TestHandleStrippingExtenderNeverLosesStateToTheRegisterProxy:
    """Parent-resident construction keeps the caller's own extender and its live handle;
    MULTIPROCESSING-resolved construction only attaches the still-pickled snapshot as
    _pending_extender_payload, never a fetch through the register/proxy."""

    def test_parent_resident_keeps_live_handle_and_multiprocessing_never_asks_the_register_proxy(self) -> None:
        extender = _HandleStrippingExtender()
        live_handle = extender.handle
        caller_function_extender: set[Extender] = {extender}

        orchestrator = ExecutionOrchestrator(_empty_plan())
        orchestrator.__enter__({ParallelizationMode.MULTIPROCESSING}, caller_function_extender)
        try:
            orchestrator._init_run()

            parent_resident_uuid = orchestrator.executor.init_compute_framework(
                PythonDictFramework, ParallelizationMode.SYNC, set(), uuid4()
            )
            parent_resident_cfw = orchestrator.executor.cfw_collection[parent_resident_uuid]
            parent_resident_extender = next(iter(parent_resident_cfw.function_extender))

            assert parent_resident_extender is extender, "must be the caller's own extender object"
            assert parent_resident_extender.handle is live_handle, (
                "a parent-resident framework must never receive a proxy-fetched copy of the "
                "caller's own extender: that round trip would silently strip the live handle"
            )

            # Builds successfully without reaching for cfw_register.get_function_extender(), a
            # method the register no longer has: attaching worker_extender_payload as the pending
            # payload is the only path, materialized later in the worker, never here.
            orchestrator.executor.init_compute_framework(
                PythonDictFramework, ParallelizationMode.MULTIPROCESSING, set(), uuid4()
            )
        finally:
            orchestrator.__exit__(None, None, None)
