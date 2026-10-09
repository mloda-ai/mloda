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
framework resolves to a non-MULTIPROCESSING mode. The run-complete cases at the end check that the
caller's own object, not a worker's copy, is notified in the parent."""

from __future__ import annotations

import gc
import json
import os
import pickle  # nosec B403
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.close_context import CloseContext
from mloda.core.abstract_plugins.function_extender import (
    CompositeExtender,
    Extender,
    ExtenderHook,
    build_hook_extenders,
)
from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.steward import FeatureResolutionError
from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.helpers.uuid7_assertions import assert_valid_uuid7


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


class _IdentExtender(Extender):
    """Picklable same-class, equal-priority extender distinguished only by ident."""

    def __init__(self, ident: str) -> None:
        self.ident = ident

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


def _ident_of(extender: Extender) -> str:
    assert isinstance(extender, _IdentExtender)
    return extender.ident


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


@pytest.mark.timeout(30)
class TestWorkerPayloadShipsTheParentsSelectionTable:
    def test_payload_table_keeps_parent_order_and_the_shipped_extender_objects(self) -> None:
        hook = ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE
        caller_function_extender: set[Extender] = {_IdentExtender(name) for name in ("a", "b", "c", "d")}

        orchestrator = ExecutionOrchestrator(_empty_plan())
        orchestrator.__enter__({ParallelizationMode.MULTIPROCESSING}, caller_function_extender)
        try:
            assert orchestrator.worker_extender_payload is not None
            fe, table = pickle.loads(orchestrator.worker_extender_payload)  # nosec B301

            expected_composite = build_hook_extenders(caller_function_extender)[hook]
            assert isinstance(expected_composite, CompositeExtender)
            shipped_composite = table[hook]
            assert isinstance(shipped_composite, CompositeExtender)

            shipped = [_ident_of(e) for e in shipped_composite.extenders]
            expected = [_ident_of(e) for e in expected_composite.extenders]
            assert shipped == expected
            assert all(any(m is x for x in fe) for m in shipped_composite.extenders)
        finally:
            orchestrator.__exit__(None, None, None)


_RUN_COMPLETE_COLUMN = "run_complete_notification_e2e_col"


class _RunCompleteFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({_RUN_COMPLETE_COLUMN})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {_RUN_COMPLETE_COLUMN: [1, 2, 3]}


_RUN_COMPLETE_ENABLED = PluginCollector.enabled_feature_groups({_RunCompleteFeatureGroup})


class _RunCompleteProbeExtender(Extender):
    """Records (run_id, pid, sentinel_exists) in the caller's own object; close() runs only in a worker's copy."""

    def __init__(self, sentinel_path: Path, close_delay: float = 0.0, close_state_path: Path | None = None) -> None:
        self._sentinel_path = sentinel_path
        self._close_delay = close_delay
        self._close_state_path = close_state_path
        self.completions: list[tuple[str | None, int, bool]] = []

    def wraps(self) -> set[ExtenderHook]:
        return set()

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    def close(self) -> None:
        time.sleep(self._close_delay)
        self._sentinel_path.write_text("closed")
        if self._close_state_path is not None:
            ctx = CloseContext.current()
            assert ctx is not None
            self._close_state_path.write_text(json.dumps([len(self.completions), ctx.remaining(), ctx.reason]))

    def on_run_complete(self, run: Any, outcome: Any) -> None:
        self.completions.append((run.run_id, os.getpid(), self._sentinel_path.exists()))


def _prepare_run_complete_session(mode: ParallelizationMode, extenders: set[Extender] | None = None) -> mloda:
    return mloda.prepare(
        [Feature(name=_RUN_COMPLETE_COLUMN)],
        compute_frameworks=["PythonDictFramework"],
        plugin_collector=_RUN_COMPLETE_ENABLED,
        parallelization_modes={mode},
        function_extender=extenders,
    )


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING, ParallelizationMode.MULTIPROCESSING]
)
class TestRunCompleteNotifiesTheCallersOwnExtenderInTheParent:
    def test_notified_once_in_the_parent_with_a_fresh_run_id(
        self, mode: ParallelizationMode, tmp_path: Path, flight_server: Any
    ) -> None:
        probe = _RunCompleteProbeExtender(tmp_path / "closed.txt")
        session = _prepare_run_complete_session(mode, {probe})

        session.run(parallelization_modes={mode}, flight_server=flight_server)

        assert [pid for _, pid, _ in probe.completions] == [os.getpid()]
        run_id = probe.completions[0][0]
        assert run_id is not None
        assert_valid_uuid7(run_id)
        assert run_id != session.plan_id

    def test_running_the_session_twice_notifies_twice_with_a_different_run_id_each_time(
        self, mode: ParallelizationMode, tmp_path: Path, flight_server: Any
    ) -> None:
        probe = _RunCompleteProbeExtender(tmp_path / "closed.txt")
        session = _prepare_run_complete_session(mode, {probe})

        session.run(parallelization_modes={mode}, flight_server=flight_server)
        session.run(parallelization_modes={mode}, flight_server=flight_server)

        first, second = (run_id for run_id, _, _ in probe.completions)
        assert first is not None and second is not None
        assert first != second


@pytest.mark.timeout(30)
class TestRunCompleteFiresAfterAMultiprocessingWorkerClosedItsExtender:
    def test_worker_close_sentinel_already_exists_when_the_parent_is_notified(
        self, tmp_path: Path, flight_server: Any
    ) -> None:
        probe = _RunCompleteProbeExtender(tmp_path / "closed.txt", close_delay=0.5)
        session = _prepare_run_complete_session(ParallelizationMode.MULTIPROCESSING, {probe})

        session.run(
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
            graceful_shutdown_timeout=5.0,
        )

        assert [sentinel_seen for _, _, sentinel_seen in probe.completions] == [True]


class _UnpicklableExtender(Extender):
    def __init__(self) -> None:
        self.lock = threading.Lock()

    def wraps(self) -> set[ExtenderHook]:
        return set()

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


@pytest.mark.timeout(30)
class TestRunCompleteFiresWhenSetupFails:
    @pytest.mark.parametrize("call_site", ["run", "stream_run"])
    def test_multiprocessing_preflight_rejection_notifies_the_probe_once_in_the_parent(
        self, call_site: str, tmp_path: Path
    ) -> None:
        probe = _RunCompleteProbeExtender(tmp_path / "closed.txt")
        session = _prepare_run_complete_session(ParallelizationMode.MULTIPROCESSING, {probe, _UnpicklableExtender()})

        with pytest.raises(ValueError, match="cannot be pickled"):
            list(
                getattr(session, call_site)(
                    parallelization_modes={ParallelizationMode.MULTIPROCESSING},
                )
            )

        assert [pid for _, pid, _ in probe.completions] == [os.getpid()]
        assert probe.completions[0][0] is not None
        assert probe.completions[0][0] != session.plan_id


@pytest.mark.timeout(30)
class TestRunAllTwiceWithTheSameProbeCarriesPriorRunStateIntoTheSecondWorker:
    """The second worker sees the first run's on_run_complete state and its own graceful_shutdown_timeout."""

    def test_second_run_worker_sees_first_runs_completions_and_the_custom_budget(
        self, tmp_path: Path, flight_server: Any
    ) -> None:
        close_state_path = tmp_path / "close_state.json"
        probe = _RunCompleteProbeExtender(tmp_path / "closed.txt", close_state_path=close_state_path)

        mloda.run_all(
            [Feature(name=_RUN_COMPLETE_COLUMN)],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_RUN_COMPLETE_ENABLED,
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            function_extender={probe},
            flight_server=flight_server,
        )
        completions_after_first_run = len(probe.completions)
        assert completions_after_first_run == 1

        mloda.run_all(
            [Feature(name=_RUN_COMPLETE_COLUMN)],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_RUN_COMPLETE_ENABLED,
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            function_extender={probe},
            flight_server=flight_server,
            graceful_shutdown_timeout=7.5,
        )

        seen_completions, seen_remaining, seen_reason = json.loads(close_state_path.read_text())

        assert seen_completions == completions_after_first_run
        assert seen_remaining > 2.0
        assert seen_reason == "stop"


class _WorkerIdentityRecorder(Extender):
    """Appends (run_id, plan_id) per calculate to a file; the spawned worker's own copy writes it."""

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        context = HookContext.current()
        assert context is not None
        with self._output_path.open("a") as handle:
            handle.write(json.dumps([context.run_id, context.plan_id]) + "\n")
        return result


@pytest.mark.timeout(30)
class TestPrepareOnceRunTwiceMultiprocessingWorkerIdentity:
    def test_each_run_reaches_its_worker_with_a_fresh_run_id_and_the_same_plan_id(
        self, tmp_path: Path, flight_server: Any
    ) -> None:
        output = tmp_path / "worker_ids.jsonl"
        mode = ParallelizationMode.MULTIPROCESSING
        session = _prepare_run_complete_session(mode, {_WorkerIdentityRecorder(output)})

        session.run(parallelization_modes={mode}, flight_server=flight_server)
        session.run(parallelization_modes={mode}, flight_server=flight_server)

        (first_run, first_plan), (second_run, second_plan) = (
            json.loads(line) for line in output.read_text().splitlines()
        )
        assert first_run is not None and second_run is not None
        assert first_run != second_run
        assert first_plan == second_plan == session.plan_id


_FAIL_COLUMN = "lifecycle_matrix_fail_col"
_MISSING_COLUMN = "lifecycle_matrix_missing_col"


class _FailingLifecycleFeatureGroup(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({_FAIL_COLUMN})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise RuntimeError("calculate failed")


class _LifecycleMatrixExtender(Extender):
    """Logs every lifecycle hook and each calculate, with the ids it saw, on the caller's own object."""

    def __init__(self, refuse_run_start: bool = False) -> None:
        self.refuse_run_start = refuse_run_start
        self.events: list[tuple[Any, ...]] = []
        self.ids: list[tuple[str, str | None, str | None]] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        self.events.append(("calculate",))
        return func(*args, **kwargs)

    def on_plan_start(self, plan: Any) -> None:
        self.events.append(("plan_start",))
        self.ids.append(("plan_start", plan.plan_id, None))

    def on_plan_complete(self, plan: Any, outcome: Any) -> None:
        self.events.append(("plan_complete", outcome.status, outcome.error_type))
        self.ids.append(("plan_complete", plan.plan_id, None))

    def on_run_start(self, run: Any, plan: Any, steps: Any) -> None:
        self.events.append(("run_start",))
        self.ids.append(("run_start", plan.plan_id, run.run_id))
        assert run.plan_id == plan.plan_id
        assert isinstance(steps, tuple) and steps
        if self.refuse_run_start:
            raise RuntimeError("run start refused")

    def on_run_complete(self, run: Any, outcome: Any) -> None:
        self.events.append(("run_complete", outcome.status, outcome.error_type))
        self.ids.append(("run_complete", run.plan_id, run.run_id))


def _matrix_kwargs(extenders: set[Extender], column: str = _RUN_COMPLETE_COLUMN) -> dict[str, Any]:
    group = _FailingLifecycleFeatureGroup if column == _FAIL_COLUMN else _RunCompleteFeatureGroup
    return {
        "compute_frameworks": ["PythonDictFramework"],
        "plugin_collector": PluginCollector.enabled_feature_groups({group}),
        "function_extender": extenders,
    }


def _matrix_session(
    extender: Extender, column: str = _RUN_COMPLETE_COLUMN, mode: ParallelizationMode | None = None
) -> mloda:
    modes = {mode or ParallelizationMode.SYNC}
    return mloda.prepare([Feature(name=column)], parallelization_modes=modes, **_matrix_kwargs({extender}, column))


_SYNC = {ParallelizationMode.SYNC}
_OK_PLAN = [("plan_start",), ("plan_complete", "succeeded", None)]


def _ok_run(status: str = "succeeded", calculated: bool = True, error: str | None = None) -> list[tuple[Any, ...]]:
    return [("run_start",), *([("calculate",)] if calculated else []), ("run_complete", status, error)]


def _planning_failure_prepare(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    with pytest.raises(FeatureResolutionError):
        mloda.prepare([Feature(name=_MISSING_COLUMN)], parallelization_modes=_SYNC, **_matrix_kwargs({ext}))
    return [
        ("plan_start",),
        ("plan_complete", "failed", "mloda.core.prepare.identify_feature_group.FeatureResolutionError"),
    ]


def _planning_failure_diagnose(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    diagnosis = mloda.diagnose([Feature(name=_MISSING_COLUMN)], parallelization_modes=_SYNC, **_matrix_kwargs({ext}))
    assert not diagnosis.complete
    return [
        ("plan_start",),
        ("plan_complete", "failed", "mloda.core.prepare.identify_feature_group.FeatureResolutionError"),
    ]


def _planning_failure_explain(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    with pytest.raises(FeatureResolutionError):
        mloda.explain([Feature(name=_MISSING_COLUMN)], parallelization_modes=_SYNC, **_matrix_kwargs({ext}))
    return [
        ("plan_start",),
        ("plan_complete", "failed", "mloda.core.prepare.identify_feature_group.FeatureResolutionError"),
    ]


def _explain_success(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    mloda.explain([Feature(name=_RUN_COMPLETE_COLUMN)], parallelization_modes=_SYNC, **_matrix_kwargs({ext}))
    return list(_OK_PLAN)


def _diagnose_success(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    assert mloda.diagnose(
        [Feature(name=_RUN_COMPLETE_COLUMN)], parallelization_modes=_SYNC, **_matrix_kwargs({ext})
    ).complete
    return list(_OK_PLAN)


def _prepare_only(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    _matrix_session(ext)
    return list(_OK_PLAN)


def _batch_success(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    _matrix_session(ext).run(parallelization_modes=_SYNC)
    return [*_OK_PLAN, *_ok_run()]


def _run_all_success(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    mloda.run_all([Feature(name=_RUN_COMPLETE_COLUMN)], parallelization_modes=_SYNC, **_matrix_kwargs({ext}))
    return [*_OK_PLAN, *_ok_run()]


def _batch_twice(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    session = _matrix_session(ext)
    session.run(parallelization_modes=_SYNC)
    session.run(parallelization_modes=_SYNC)
    return [*_OK_PLAN, *_ok_run(), *_ok_run()]


def _setup_failure(call_site: str) -> Any:
    def scenario(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
        mode = ParallelizationMode.MULTIPROCESSING
        session = mloda.prepare(
            [Feature(name=_RUN_COMPLETE_COLUMN)],
            parallelization_modes={mode},
            **_matrix_kwargs({ext, _UnpicklableExtender()}),
        )
        with pytest.raises(ValueError, match="cannot be pickled"):
            list(getattr(session, call_site)(parallelization_modes={mode}))
        return [*_OK_PLAN, *_ok_run("failed", calculated=False, error="builtins.ValueError")]

    return scenario


def _execution_failure(call_site: str, error: type[BaseException] = RuntimeError) -> Any:
    def scenario(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
        if error is not RuntimeError:

            def interrupted(cls: Any, data: Any, features: FeatureSet) -> Any:
                raise error("calculate failed")

            mp.setattr(_FailingLifecycleFeatureGroup, "calculate_feature", classmethod(interrupted))
        session = _matrix_session(ext, _FAIL_COLUMN)
        with pytest.raises(error, match="calculate failed") as raised:
            list(getattr(session, call_site)(parallelization_modes=_SYNC))
        status = "failed" if issubclass(error, Exception) else "cancelled"
        return [
            *_OK_PLAN,
            *_ok_run(status, error=f"{type(raised.value).__module__}.{type(raised.value).__qualname__}"),
        ]

    return scenario


def _planning_interrupted_prepare(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    def interrupted(self: Any, *args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt()

    mp.setattr(mloda, "_plan", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _matrix_session(ext)
    return [("plan_start",), ("plan_complete", "cancelled", "builtins.KeyboardInterrupt")]


def _break_join(mp: pytest.MonkeyPatch) -> None:
    def broken(self: ExecutionOrchestrator) -> None:
        raise RuntimeError("join failed")

    mp.setattr(ExecutionOrchestrator, "join", broken)


def _finalizing_failure_batch(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    session = _matrix_session(ext)
    _break_join(mp)
    with pytest.raises(RuntimeError, match="join failed"):
        session.run(parallelization_modes=_SYNC)
    return [*_OK_PLAN, *_ok_run("failed", error="builtins.RuntimeError")]


def _finalizing_failure_stream_consumed(
    ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch
) -> list[Any]:
    session = _matrix_session(ext)
    _break_join(mp)
    with pytest.raises(RuntimeError, match="join failed"):
        list(session.stream_run(parallelization_modes=_SYNC))
    return [*_OK_PLAN, *_ok_run("failed", error="builtins.RuntimeError")]


def _finalizing_failure_stream_closed_early(
    ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch
) -> list[Any]:
    session = _matrix_session(ext)
    _break_join(mp)
    stream = session.stream_run(parallelization_modes=_SYNC)
    next(stream)
    with pytest.raises(RuntimeError, match="join failed"):
        stream.close()
    return [*_OK_PLAN, *_ok_run("failed", error="builtins.RuntimeError")]


def _refused_batch(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    ext.refuse_run_start = True
    session = _matrix_session(ext)
    with pytest.raises(RuntimeError, match="run start refused"):
        session.run(parallelization_modes=_SYNC)
    return [*_OK_PLAN, *_ok_run("failed", calculated=False, error="builtins.RuntimeError")]


def _refused_at_stream_run_call(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    ext.refuse_run_start = True
    session = _matrix_session(ext)
    with pytest.raises(RuntimeError, match="run start refused"):
        session.stream_run(parallelization_modes=_SYNC)
    return [*_OK_PLAN, *_ok_run("failed", calculated=False, error="builtins.RuntimeError")]


def _break_engine_setup(mp: pytest.MonkeyPatch) -> None:
    def broken(self: Any, parallelization_modes: Any, flight_server: Any) -> None:
        raise RuntimeError("engine setup failed")

    mp.setattr(mloda, "_setup_engine_runner", broken)


def _engine_setup_failure_batch(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    session = _matrix_session(ext)
    _break_engine_setup(mp)
    with pytest.raises(RuntimeError, match="engine setup failed"):
        session.run(parallelization_modes=_SYNC)
    return [*_OK_PLAN, *_ok_run("failed", calculated=False, error="builtins.RuntimeError")]


def _engine_setup_failure_at_stream_run_call(
    ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch
) -> list[Any]:
    session = _matrix_session(ext)
    _break_engine_setup(mp)
    with pytest.raises(RuntimeError, match="engine setup failed"):
        session.stream_run(parallelization_modes=_SYNC)
    return [*_OK_PLAN, *_ok_run("failed", calculated=False, error="builtins.RuntimeError")]


def _refused_at_stream_all_call(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    ext.refuse_run_start = True
    with pytest.raises(RuntimeError, match="run start refused"):
        mloda.stream_all([Feature(name=_RUN_COMPLETE_COLUMN)], parallelization_modes=_SYNC, **_matrix_kwargs({ext}))
    return [*_OK_PLAN, *_ok_run("failed", calculated=False, error="builtins.RuntimeError")]


def _stream_consumed(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    assert list(_matrix_session(ext).stream_run(parallelization_modes=_SYNC))
    return [*_OK_PLAN, *_ok_run()]


def _stream_all_consumed(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    stream = mloda.stream_all(
        [Feature(name=_RUN_COMPLETE_COLUMN)], parallelization_modes=_SYNC, **_matrix_kwargs({ext})
    )
    assert list(stream)
    return [*_OK_PLAN, *_ok_run()]


def _stream_closed_early(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    stream = _matrix_session(ext).stream_run(parallelization_modes=_SYNC)
    next(stream)
    stream.close()
    return [*_OK_PLAN, *_ok_run("cancelled")]


def _stream_never_iterated_closed(ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch) -> list[Any]:
    stream = _matrix_session(ext).stream_run(parallelization_modes=_SYNC)
    stream.close()
    return [*_OK_PLAN, *_ok_run("cancelled", calculated=False)]


def _stream_never_iterated_collected(
    ext: _LifecycleMatrixExtender, tmp_path: Path, mp: pytest.MonkeyPatch
) -> list[Any]:
    stream = _matrix_session(ext).stream_run(parallelization_modes=_SYNC)
    del stream
    gc.collect()
    return [*_OK_PLAN, *_ok_run("cancelled", calculated=False)]


_MATRIX: dict[str, Any] = {
    "prepare_only": _prepare_only,
    "explain_success": _explain_success,
    "diagnose_success": _diagnose_success,
    "planning_failure_prepare": _planning_failure_prepare,
    "planning_failure_diagnose": _planning_failure_diagnose,
    "planning_failure_explain": _planning_failure_explain,
    "batch_success": _batch_success,
    "run_all_success": _run_all_success,
    "batch_twice": _batch_twice,
    "setup_failure_run": _setup_failure("run"),
    "setup_failure_stream_run": _setup_failure("stream_run"),
    "execution_failure_run": _execution_failure("run"),
    "execution_failure_stream_run": _execution_failure("stream_run"),
    "interrupted_run_keyboard_interrupt": _execution_failure("run", KeyboardInterrupt),
    "interrupted_stream_run_keyboard_interrupt": _execution_failure("stream_run", KeyboardInterrupt),
    "interrupted_run_system_exit": _execution_failure("run", SystemExit),
    "interrupted_stream_run_system_exit": _execution_failure("stream_run", SystemExit),
    "planning_interrupted_prepare": _planning_interrupted_prepare,
    "finalizing_failure_batch": _finalizing_failure_batch,
    "finalizing_failure_stream_consumed": _finalizing_failure_stream_consumed,
    "finalizing_failure_stream_closed_early": _finalizing_failure_stream_closed_early,
    "run_start_refusal_batch": _refused_batch,
    "run_start_refusal_at_stream_run_call": _refused_at_stream_run_call,
    "run_start_refusal_at_stream_all_call": _refused_at_stream_all_call,
    "engine_setup_failure_batch": _engine_setup_failure_batch,
    "engine_setup_failure_at_stream_run_call": _engine_setup_failure_at_stream_run_call,
    "stream_consumed": _stream_consumed,
    "stream_all_consumed": _stream_all_consumed,
    "stream_closed_early": _stream_closed_early,
    "stream_never_iterated_closed": _stream_never_iterated_closed,
    "stream_never_iterated_collected": _stream_never_iterated_collected,
}


@pytest.mark.timeout(30)
class TestLifecycleHooksFireExactlyOnceInOrder:
    @pytest.mark.parametrize("scenario", list(_MATRIX))
    def test_hooks_fire_exactly_once_and_in_order(
        self, scenario: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extender = _LifecycleMatrixExtender()

        expected = _MATRIX[scenario](extender, tmp_path, monkeypatch)

        assert extender.events == expected

    @pytest.mark.parametrize("scenario", ["batch_twice", "stream_closed_early", "run_start_refusal_batch"])
    def test_ids_are_consistent_across_the_hooks(
        self, scenario: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extender = _LifecycleMatrixExtender()

        _MATRIX[scenario](extender, tmp_path, monkeypatch)

        plan_ids = {plan_id for _, plan_id, _ in extender.ids}
        assert len(plan_ids) == 1
        runs = [(hook, run_id) for hook, _, run_id in extender.ids if hook.startswith("run_")]
        assert all(run_id is not None and run_id not in plan_ids for _, run_id in runs)
        starts = [run_id for hook, run_id in runs if hook == "run_start"]
        completes = [run_id for hook, run_id in runs if hook == "run_complete"]
        assert starts == completes
        assert len(set(starts)) == len(starts)
