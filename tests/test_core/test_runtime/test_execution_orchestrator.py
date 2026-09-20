"""
Tests for ExecutionOrchestrator class (renamed from Runner).

This test file defines the requirements for the ExecutionOrchestrator class.
"""

from __future__ import annotations

import inspect
import logging
import threading
import uuid as uuid_mod
from collections.abc import Callable, Mapping
from typing import Any
from unittest.mock import Mock, patch, MagicMock
from uuid import UUID

import pytest

from mloda.provider import ComputeFramework, FeatureGroup  # noqa: F401
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.prepare.execution_plan import ExecutionPlan

from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.core.core.cfw_manager import CfwManager
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.run_context import RunContext
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class TestExecutionOrchestratorImport:
    """Tests for importing ExecutionOrchestrator."""

    def test_execution_orchestrator_can_be_imported(self) -> None:
        """ExecutionOrchestrator should be importable from mloda.core.runtime.run."""
        assert ExecutionOrchestrator is not None


class TestExecutionOrchestratorConstruction:
    """Tests for ExecutionOrchestrator initialization."""

    def test_constructor_accepts_execution_planner(self) -> None:
        """Constructor should accept an execution planner."""
        mock_planner = Mock(spec=ExecutionPlan)

        orchestrator = ExecutionOrchestrator(mock_planner)

        assert orchestrator.execution_planner is mock_planner

    def test_constructor_accepts_optional_flight_server(self) -> None:
        """Constructor should accept an optional flight server."""
        mock_planner = Mock(spec=ExecutionPlan)
        mock_flight_server = Mock()

        orchestrator = ExecutionOrchestrator(mock_planner, flight_server=mock_flight_server)

        assert orchestrator.flight_server is mock_flight_server

    def test_constructor_initializes_worker_manager(self) -> None:
        """Constructor should initialize a worker_manager instance."""
        mock_planner = Mock(spec=ExecutionPlan)

        orchestrator = ExecutionOrchestrator(mock_planner)

        assert hasattr(orchestrator, "worker_manager")
        assert orchestrator.worker_manager is not None

    def test_constructor_initializes_data_lifecycle_manager(self) -> None:
        """Constructor should initialize a data_lifecycle_manager instance."""
        mock_planner = Mock(spec=ExecutionPlan)

        orchestrator = ExecutionOrchestrator(mock_planner)

        assert hasattr(orchestrator, "data_lifecycle_manager")
        assert orchestrator.data_lifecycle_manager is not None

    def test_constructor_sets_location_to_none_by_default(self) -> None:
        """Constructor should set location to None by default."""
        mock_planner = Mock(spec=ExecutionPlan)

        orchestrator = ExecutionOrchestrator(mock_planner)

        assert orchestrator.location is None


class TestExecutionOrchestratorContextManager:
    """Tests for ExecutionOrchestrator context management."""

    def test_implements_context_manager_protocol(self) -> None:
        """ExecutionOrchestrator should implement __enter__ and __exit__ methods."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        assert hasattr(orchestrator, "__enter__")
        assert callable(orchestrator.__enter__)
        assert hasattr(orchestrator, "__exit__")
        assert callable(orchestrator.__exit__)

    def test_enter_accepts_parallelization_modes(self) -> None:
        """__enter__ should accept parallelization_modes parameter."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        # Check signature includes parallelization_modes parameter
        sig = inspect.signature(orchestrator.__enter__)
        assert "parallelization_modes" in sig.parameters

    def test_enter_accepts_function_extender(self) -> None:
        """__enter__ should accept function_extender parameter."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        sig = inspect.signature(orchestrator.__enter__)
        assert "function_extender" in sig.parameters

    def test_enter_accepts_api_data(self) -> None:
        """__enter__ should accept api_data parameter."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        sig = inspect.signature(orchestrator.__enter__)
        assert "api_data" in sig.parameters

    def test_exit_accepts_exception_info(self) -> None:
        """__exit__ should accept exc_type, exc_val, exc_tb parameters."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        sig = inspect.signature(orchestrator.__exit__)
        params = list(sig.parameters.keys())
        assert "exc_type" in params
        assert "exc_val" in params
        assert "exc_tb" in params


class TestExecutionOrchestratorOrchestrationMethods:
    """Tests for ExecutionOrchestrator orchestration methods."""

    def test_has_compute_method(self) -> None:
        """ExecutionOrchestrator should have a compute() method."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        assert hasattr(orchestrator, "compute")
        assert callable(orchestrator.compute)

    def test_has_is_step_done_method(self) -> None:
        """ExecutionOrchestrator should have a _is_step_done() method."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        assert hasattr(orchestrator, "_is_step_done")
        assert callable(orchestrator._is_step_done)

    def test_has_can_run_step_method(self) -> None:
        """ExecutionOrchestrator should have a _can_run_step() method."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        assert hasattr(orchestrator, "_can_run_step")
        assert callable(orchestrator._can_run_step)

    def test_has_mark_step_as_finished_method(self) -> None:
        """ExecutionOrchestrator should have a _mark_step_as_finished() method."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        assert hasattr(orchestrator, "_mark_step_as_finished")
        assert callable(orchestrator._mark_step_as_finished)

    def test_has_currently_running_step_method(self) -> None:
        """ExecutionOrchestrator should have a currently_running_step() method."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        assert hasattr(orchestrator, "currently_running_step")
        assert callable(orchestrator.currently_running_step)

    def test_has_execute_step_method(self) -> None:
        """ExecutionOrchestrator should have a _execute_step() method."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        assert hasattr(orchestrator, "_execute_step")
        assert callable(orchestrator._execute_step)

    def test_has_process_step_result_method(self) -> None:
        """ExecutionOrchestrator should have a _process_step_result() method."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        assert hasattr(orchestrator, "_process_step_result")
        assert callable(orchestrator._process_step_result)

    def test_has_join_method(self) -> None:
        """ExecutionOrchestrator should have a join() method."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        assert hasattr(orchestrator, "join")
        assert callable(orchestrator.join)

    def test_has_get_result_method(self) -> None:
        """ExecutionOrchestrator should have a get_result() method."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        assert hasattr(orchestrator, "get_result")
        assert callable(orchestrator.get_result)

    def test_has_get_artifacts_method(self) -> None:
        """ExecutionOrchestrator should have a get_artifacts() method."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        assert hasattr(orchestrator, "get_artifacts")
        assert callable(orchestrator.get_artifacts)


class TestExecutionOrchestratorMethodSignatures:
    """Tests for ExecutionOrchestrator method signatures to ensure correct interface."""

    def test_is_step_done_accepts_step_uuids_and_finished_ids(self) -> None:
        """_is_step_done should accept step_uuids and finished_ids parameters."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        sig = inspect.signature(orchestrator._is_step_done)
        params = list(sig.parameters.keys())
        assert "step_uuids" in params
        assert "finished_ids" in params

    def test_can_run_step_signature(self) -> None:
        """_can_run_step should have correct parameters."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        sig = inspect.signature(orchestrator._can_run_step)
        params = list(sig.parameters.keys())
        assert "required_uuids" in params
        assert "step_uuid" in params
        assert "finished_steps" in params
        assert "currently_running_steps" in params

    def test_mark_step_as_finished_signature(self) -> None:
        """_mark_step_as_finished should have correct parameters."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        sig = inspect.signature(orchestrator._mark_step_as_finished)
        params = list(sig.parameters.keys())
        assert "step_uuid" in params
        assert "finished_steps" in params
        assert "currently_running_steps" in params

    def test_currently_running_step_signature(self) -> None:
        """currently_running_step should have correct parameters."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        sig = inspect.signature(orchestrator.currently_running_step)
        params = list(sig.parameters.keys())
        assert "step_uuids" in params
        assert "currently_running_steps" in params

    def test_execute_step_accepts_step_parameter(self) -> None:
        """_execute_step should accept a step parameter."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        sig = inspect.signature(orchestrator._execute_step)
        params = list(sig.parameters.keys())
        assert "step" in params

    def test_process_step_result_accepts_step_parameter(self) -> None:
        """_process_step_result should accept a step parameter."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        sig = inspect.signature(orchestrator._process_step_result)
        params = list(sig.parameters.keys())
        assert "step" in params


class TestSyncModeSkipsMyManager:
    """Tests that SYNC mode does not spawn a BaseManager server process.

    In SYNC mode, multiprocessing is not used, so spawning a MyManager
    (which starts a separate server process) is unnecessary overhead.
    The ExecutionOrchestrator should create a direct CfwManager instance
    instead and set self.manager to None.
    """

    def test_sync_mode_does_not_create_manager(self) -> None:
        """In SYNC-only mode, __enter__ should not spawn a MyManager server process.

        When parallelization_modes contains only SYNC, the orchestrator should:
        - Set self.manager to None (no BaseManager server process)
        - Still create a usable cfw_register (direct CfwManager instance)
        - Allow __exit__ to complete without raising
        """
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        orchestrator.__enter__({ParallelizationMode.SYNC})

        assert orchestrator.manager is None
        assert orchestrator.cfw_register is not None
        assert isinstance(orchestrator.cfw_register, CfwManager)

        # __exit__ should not raise when manager is None
        orchestrator.__exit__(None, None, None)

    def test_sync_mode_cfw_register_is_direct_instance(self) -> None:
        """In SYNC mode, cfw_register should be a direct CfwManager instance.

        The cfw_register should be a real CfwManager object (not a proxy),
        and it should have the correct parallelization_modes set.
        """
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        orchestrator.__enter__({ParallelizationMode.SYNC})

        assert isinstance(orchestrator.cfw_register, CfwManager)
        assert orchestrator.cfw_register.parallelization_modes == {ParallelizationMode.SYNC}

        orchestrator.__exit__(None, None, None)

    def test_sync_mode_exit_with_none_manager(self) -> None:
        """__exit__ should handle manager being None without raising."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        orchestrator.manager = None

        orchestrator.__exit__(None, None, None)


class TestEnterAcceptsRunContext:
    def test_enter_accepts_run_context_parameter(self) -> None:
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        sig = inspect.signature(orchestrator.__enter__)
        assert "run_context" in sig.parameters

    def test_run_context_parameter_defaults_to_none(self) -> None:
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        sig = inspect.signature(orchestrator.__enter__)
        assert sig.parameters["run_context"].default is None

    def test_enter_signature_has_exactly_five_parameters_in_order(self) -> None:
        """No run_id/carrier/child_bootstrap parameters remain."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        sig = inspect.signature(orchestrator.__enter__)
        param_names = list(sig.parameters.keys())
        assert param_names == ["parallelization_modes", "function_extender", "api_data", "artifacts", "run_context"]


class TestEnterSetsRunContextOnCfwRegister:
    def test_sync_mode_cfw_register_round_trips_run_context(self) -> None:
        def _bootstrap() -> None:
            pass

        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        orchestrator.__enter__({ParallelizationMode.SYNC}, None, None, None, RunContext(child_bootstrap=_bootstrap))

        assert orchestrator.cfw_register.get_run_context().child_bootstrap is _bootstrap

        orchestrator.__exit__(None, None, None)

    def test_sync_mode_cfw_register_run_context_defaults_to_empty_run_context(self) -> None:
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        orchestrator.__enter__({ParallelizationMode.SYNC})

        assert orchestrator.cfw_register.get_run_context() == RunContext()

        orchestrator.__exit__(None, None, None)


class TestEnterRejectsUnpicklableChildBootstrapBeforeSpawningAManager:
    def test_multiprocessing_mode_with_unpicklable_bootstrap_raises_before_any_manager_starts(self) -> None:
        class _Unpicklable:
            """threading.Lock is never picklable, so a closure over this is unpicklable too."""

            def __init__(self) -> None:
                self.lock = threading.Lock()

        def _make_closure_over_unpicklable() -> Callable[[], None]:
            unpicklable = _Unpicklable()

            def _bootstrap() -> None:
                unpicklable.lock.acquire()

            return _bootstrap

        empty_planner = ExecutionPlan()
        empty_planner.execution_plan = []
        orchestrator = ExecutionOrchestrator(empty_planner)

        with pytest.raises(ValueError, match="child_bootstrap"):
            orchestrator.__enter__(
                {ParallelizationMode.MULTIPROCESSING},
                None,
                None,
                None,
                RunContext(child_bootstrap=_make_closure_over_unpicklable()),
            )

        assert orchestrator.manager is None, "no MyManager/worker process may be created on the rejection path"


class TestExecutionOrchestratorStepLock:
    def test_has_step_lock_attribute_after_construction(self) -> None:
        """ExecutionOrchestrator should have a _step_lock attribute after __init__."""
        mock_planner = Mock(spec=ExecutionPlan)

        orchestrator = ExecutionOrchestrator(mock_planner)

        assert hasattr(orchestrator, "_step_lock"), (
            "ExecutionOrchestrator.__init__ must create a self._step_lock attribute"
        )

    def test_step_lock_is_threading_lock_instance(self) -> None:
        """_step_lock should be an instance of threading.Lock."""
        mock_planner = Mock(spec=ExecutionPlan)

        orchestrator = ExecutionOrchestrator(mock_planner)

        assert isinstance(orchestrator._step_lock, type(threading.Lock())), (
            "_step_lock must be a threading.Lock instance, not a new lock per call"
        )


class TestSyncModeSkipsSleep:
    """Tests that SYNC mode does not call time.sleep in the compute loop.

    In SYNC mode, all steps complete inline within sync_execute_step (which
    sets step.step_is_done = True before returning). There is no need to poll
    for results from worker threads or processes, so the time.sleep(0.01) call
    in the compute() while-loop is pure waste. The compute loop should skip
    the sleep when running in SYNC mode.
    """

    def test_sync_mode_does_not_call_time_sleep(self) -> None:
        """In SYNC mode, compute() should not call time.sleep."""
        step_uuid = uuid_mod.uuid4()

        mock_step = MagicMock()
        mock_step.get_uuids.return_value = {step_uuid}
        mock_step.required_uuids = set()
        mock_step.step_is_done = False
        mock_step.uuid = step_uuid

        class ReiterablePlan:
            """A planner mock that yields the same step on each iteration."""

            def __init__(self, step: object) -> None:
                self._step = step

            def __iter__(self):  # type: ignore[no-untyped-def]
                yield self._step

        planner = ReiterablePlan(mock_step)

        orchestrator = ExecutionOrchestrator(planner)  # type: ignore[arg-type]
        orchestrator.cfw_register = CfwManager({ParallelizationMode.SYNC})

        def fake_execute_step(step: object) -> None:
            step.step_is_done = True  # type: ignore[attr-defined]

        orchestrator._execute_step = Mock(side_effect=fake_execute_step)  # type: ignore[method-assign]
        orchestrator._drop_data_for_finished_cfws = Mock()  # type: ignore[method-assign]
        orchestrator.data_lifecycle_manager.set_artifacts = Mock()  # type: ignore[method-assign]
        orchestrator.join = Mock()  # type: ignore[method-assign]

        with patch("mloda.core.runtime.run.time.sleep") as mock_sleep:
            orchestrator.compute()
            mock_sleep.assert_not_called()


class TestGetResultItemsPlanOrder:
    """get_result_items()/get_result() must report plan order, not result_data_collection insertion order."""

    def _make_step(self, step_uuid: UUID) -> Mock:
        step = Mock(spec=FeatureGroupStep)
        step.uuid = step_uuid
        return step

    def _orchestrator_with_plan(self, uuid_a: UUID, uuid_b: UUID, uuid_c: UUID) -> ExecutionOrchestrator:
        planner = ExecutionPlan()
        planner.execution_plan = [self._make_step(uuid_a), self._make_step(uuid_b), self._make_step(uuid_c)]
        orchestrator = ExecutionOrchestrator(planner)

        # Insertion order deliberately differs from plan order (c, a, b vs a, b, c).
        collection = orchestrator.data_lifecycle_manager.result_data_collection
        collection[uuid_c] = "result_c"
        collection[uuid_a] = "result_a"
        collection[uuid_b] = "result_b"
        return orchestrator

    def test_get_result_items_returns_plan_order_not_insertion_order(self) -> None:
        uuid_a, uuid_b, uuid_c = uuid_mod.uuid4(), uuid_mod.uuid4(), uuid_mod.uuid4()
        orchestrator = self._orchestrator_with_plan(uuid_a, uuid_b, uuid_c)

        assert orchestrator.get_result_items() == [
            (uuid_a, "result_a"),
            (uuid_b, "result_b"),
            (uuid_c, "result_c"),
        ]

    def test_get_result_returns_values_in_plan_order(self) -> None:
        uuid_a, uuid_b, uuid_c = uuid_mod.uuid4(), uuid_mod.uuid4(), uuid_mod.uuid4()
        orchestrator = self._orchestrator_with_plan(uuid_a, uuid_b, uuid_c)

        assert orchestrator.get_result() == ["result_a", "result_b", "result_c"]

    def test_uuid_missing_from_plan_sorts_after_known_order_and_does_not_raise(self) -> None:
        uuid_a, uuid_b, uuid_c, uuid_unplanned = (
            uuid_mod.uuid4(),
            uuid_mod.uuid4(),
            uuid_mod.uuid4(),
            uuid_mod.uuid4(),
        )
        orchestrator = self._orchestrator_with_plan(uuid_a, uuid_b, uuid_c)
        orchestrator.data_lifecycle_manager.result_data_collection[uuid_unplanned] = "result_unplanned"

        items = orchestrator.get_result_items()

        assert items[:3] == [(uuid_a, "result_a"), (uuid_b, "result_b"), (uuid_c, "result_c")]
        assert items[3] == (uuid_unplanned, "result_unplanned")


class _DropTfsSourceFromFG(FeatureGroup):
    """Marker feature group, only used as a type reference on the transform hop."""


class _DropTfsSourceToFG(FeatureGroup):
    """Marker feature group, only used as a type reference on the transform hop."""


class TestDropTfsSourceIfPossibleResolvesViaSourceFrameworkUuid:
    """`next(iter(required_uuids))` can resolve to a uuid with no cfw at all, or to a different cfw
    instance of the same framework; only source_framework_uuid reliably names the hop's own source cfw."""

    def test_resolves_source_cfw_via_source_framework_uuid_not_required_uuids(self) -> None:
        destination_uuid = uuid_mod.uuid4()
        source_uuid = uuid_mod.uuid4()
        consumer_uuid = uuid_mod.uuid4()
        cfw_uuid = uuid_mod.uuid4()
        source_cfw = object()

        step = TransformFrameworkStep(
            from_framework=PythonDictFramework,
            to_framework=PyArrowTable,
            required_uuids={destination_uuid},
            from_feature_group=_DropTfsSourceFromFG,
            to_feature_group=_DropTfsSourceToFG,
            link_id=uuid_mod.uuid4(),
            source_framework_uuids={source_uuid},
        )
        step.owed_tokens = frozenset({consumer_uuid})

        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)
        orchestrator.cfw_register = Mock()
        orchestrator.cfw_register.get_cfw_uuid.side_effect = lambda _class_name, uuid: (
            cfw_uuid if uuid == source_uuid else None
        )
        orchestrator.executor = Mock()
        orchestrator.executor.cfw_collection = {cfw_uuid: source_cfw}
        orchestrator._mark_children_and_track = Mock()  # type: ignore[method-assign]

        orchestrator._drop_tfs_source_if_possible(step)

        orchestrator._mark_children_and_track.assert_called_once_with(source_cfw, {consumer_uuid})


class TestMarkChildrenAndTrackNeverBlocksOnWorkerOwnedCfw:
    """The planner thread must not stall on a worker's drop ack; the flyway fallback still tracks."""

    def test_worker_owned_branch_never_calls_wait_for_drop_completion(self) -> None:
        cfw_uuid = uuid_mod.uuid4()
        child_uuid = uuid_mod.uuid4()
        tracked_uuid = uuid_mod.uuid4()

        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        process, command_queue, result_queue = Mock(), Mock(), Mock()
        orchestrator.worker_manager.process_register[cfw_uuid] = (process, command_queue, result_queue)
        orchestrator.worker_manager.wait_for_drop_completion = Mock(return_value=None)  # type: ignore[method-assign]

        orchestrator.cfw_register = Mock()
        orchestrator.cfw_register.get_uuid_flyway_datasets.return_value = None

        cfw = Mock()
        cfw.uuid = cfw_uuid
        cfw.children_if_root = frozenset({tracked_uuid})

        children = {child_uuid}

        orchestrator._mark_children_and_track(cfw, children)

        command_queue.put.assert_called_once_with(children)
        orchestrator.worker_manager.wait_for_drop_completion.assert_not_called()
        assert orchestrator.data_lifecycle_manager.track_data_to_drop[cfw.uuid] == set(cfw.children_if_root)


class TestDropCfwDataRoutedNeverBlocksWhenWorkerAlive:
    """_drop_cfw_data_routed's return value is never read; the wait only ever blocked the caller."""

    def test_never_calls_wait_for_drop_completion_when_worker_alive(self) -> None:
        cfw_uuid = uuid_mod.uuid4()
        child_uuid = uuid_mod.uuid4()

        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        process = Mock()
        process.is_alive.return_value = True
        command_queue, result_queue = Mock(), Mock()
        orchestrator.worker_manager.process_register[cfw_uuid] = (process, command_queue, result_queue)
        orchestrator.worker_manager.wait_for_drop_completion = Mock(return_value=None)  # type: ignore[method-assign]

        cfw = Mock()
        cfw.children_if_root = frozenset({child_uuid})

        orchestrator._drop_cfw_data_routed(cfw_uuid, cfw)

        command_queue.put.assert_called_once_with(set(cfw.children_if_root))
        orchestrator.worker_manager.wait_for_drop_completion.assert_not_called()


class TestDropCfwDataRoutedFallsBackToDirectDropWhenWorkerDead:
    """A dead worker's queue must not be touched; the fallback drops directly instead."""

    def test_drops_directly_without_touching_worker_queue_when_worker_not_alive(self) -> None:
        cfw_uuid = uuid_mod.uuid4()
        child_uuid = uuid_mod.uuid4()

        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        process = Mock()
        process.is_alive.return_value = False
        command_queue, result_queue = Mock(), Mock()
        orchestrator.worker_manager.process_register[cfw_uuid] = (process, command_queue, result_queue)

        cfw = Mock()
        cfw.children_if_root = frozenset({child_uuid})

        orchestrator._drop_cfw_data_routed(cfw_uuid, cfw)

        cfw.drop_last_data.assert_called_once_with(orchestrator.location)
        command_queue.put.assert_not_called()


_RUN_COMPLETE_BOOM = "run-complete-boom"

_RunLog = list[tuple[str, str | None]]


class _UnprintableError(Exception):
    def __str__(self) -> str:
        raise ValueError("str() of this exception is broken")


class _RunCompleteRecorder(Extender):
    def __init__(
        self,
        label: str,
        log: _RunLog,
        priority: int = 100,
        hooks: set[ExtenderHook] | None = None,
        raises: bool = False,
        error: type[BaseException] = RuntimeError,
    ) -> None:
        self.label = label
        self.log = log
        self.priority = priority
        self.hooks = {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE} if hooks is None else hooks
        self.raises = raises
        self.error = error

    def wraps(self) -> set[ExtenderHook]:
        return self.hooks

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    def on_run_complete(self, run_id: str | None) -> None:
        self.log.append((self.label, run_id))
        if self.raises:
            raise self.error(_RUN_COMPLETE_BOOM)


def _entered_orchestrator(extenders: set[Extender]) -> ExecutionOrchestrator:
    orchestrator = ExecutionOrchestrator(Mock(spec=ExecutionPlan))
    orchestrator.__enter__({ParallelizationMode.SYNC}, extenders, None, None, RunContext(run_id="run-1"))
    return orchestrator


def _finalize_then_exit(orchestrator: ExecutionOrchestrator) -> None:
    orchestrator._finalize()
    orchestrator.__exit__(None, None, None)


class TestExitNotifiesExtendersOfRunCompletion:
    def test_notifies_with_the_run_id_after_join_and_the_flight_table_sweep(self) -> None:
        log: _RunLog = []
        orchestrator = _entered_orchestrator({_RunCompleteRecorder("extender", log)})
        orchestrator.join = Mock(side_effect=lambda: log.append(("join", None)))  # type: ignore[method-assign]
        orchestrator._drop_all_uploaded_flight_tables = Mock(  # type: ignore[method-assign]
            side_effect=lambda: log.append(("sweep", None))
        )

        orchestrator._finalize()

        assert log == [("join", None), ("sweep", None)]

        orchestrator.__exit__(None, None, None)

        assert log == [("join", None), ("sweep", None), ("extender", "run-1")]

    def test_exit_shuts_the_manager_down_when_an_extender_raises_a_base_exception(self) -> None:
        log: _RunLog = []
        extender = _RunCompleteRecorder("extender", log, raises=True, error=KeyboardInterrupt)
        orchestrator = _entered_orchestrator({extender})
        orchestrator.manager = Mock()
        orchestrator.manager.shutdown.side_effect = lambda: log.append(("shutdown", None))

        with pytest.raises(KeyboardInterrupt):
            orchestrator.__exit__(None, None, None)

        assert log == [("extender", "run-1"), ("shutdown", None)]

    def test_exit_on_a_never_entered_orchestrator_does_not_raise(self) -> None:
        ExecutionOrchestrator(Mock(spec=ExecutionPlan)).__exit__(None, None, None)

    def test_notifies_when_compute_raises_and_the_run_exception_propagates(self) -> None:
        log: _RunLog = []
        orchestrator = _entered_orchestrator({_RunCompleteRecorder("extender", log)})
        orchestrator.join = Mock(side_effect=lambda: log.append(("join", None)))  # type: ignore[method-assign]
        failure = RuntimeError("compute failed")
        orchestrator._check_for_error = Mock(side_effect=failure)  # type: ignore[method-assign]

        with pytest.raises(RuntimeError) as raised:
            try:
                orchestrator.compute()
            finally:
                orchestrator.__exit__(None, None, None)

        assert raised.value is failure
        assert log == [("join", None), ("extender", "run-1")]

    @pytest.mark.parametrize("failing_call", ["join", "set_artifacts"])
    def test_does_not_notify_when_the_worker_join_did_not_complete(
        self, failing_call: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log: _RunLog = []
        orchestrator = _entered_orchestrator({_RunCompleteRecorder("extender", log)})
        owner = orchestrator.data_lifecycle_manager if failing_call == "set_artifacts" else orchestrator
        monkeypatch.setattr(owner, failing_call, Mock(side_effect=Exception("teardown failed")))

        with pytest.raises(Exception, match="teardown failed"):
            orchestrator._finalize()
        orchestrator.__exit__(None, None, None)

        assert log == []

    def test_notifies_in_ascending_priority_order(self) -> None:
        log: _RunLog = []
        priorities = [50, 20, 60, 10, 40, 30]
        orchestrator = _entered_orchestrator({_RunCompleteRecorder(f"p{p}", log, priority=p) for p in priorities})

        _finalize_then_exit(orchestrator)

        assert [label for label, _ in log] == [f"p{p}" for p in sorted(priorities)]

    def test_notifies_an_extender_that_wraps_no_hook(self) -> None:
        log: _RunLog = []
        orchestrator = _entered_orchestrator({_RunCompleteRecorder("wraps_nothing", log, hooks=set())})

        _finalize_then_exit(orchestrator)

        assert log == [("wraps_nothing", "run-1")]

    def test_raising_extender_is_logged_and_later_extenders_are_still_notified(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        log: _RunLog = []
        raiser = _RunCompleteRecorder("raiser", log, priority=10, raises=True)
        survivor = _RunCompleteRecorder("survivor", log, priority=20)
        orchestrator = _entered_orchestrator({raiser, survivor})

        with caplog.at_level(logging.ERROR):
            _finalize_then_exit(orchestrator)

        assert log == [("raiser", "run-1"), ("survivor", "run-1")]
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) == 1
        record = error_records[0]
        assert _RUN_COMPLETE_BOOM in record.getMessage()
        assert "RuntimeError" in record.getMessage()
        assert record.exc_info is None
        args = record.args
        arg_values = args.values() if isinstance(args, Mapping) else (args or ())
        assert not [a for a in arg_values if isinstance(a, BaseException)]

    def test_extender_whose_exception_str_raises_does_not_escape_exit(self, caplog: pytest.LogCaptureFixture) -> None:
        log: _RunLog = []
        raiser = _RunCompleteRecorder("raiser", log, priority=10, raises=True, error=_UnprintableError)
        survivor = _RunCompleteRecorder("survivor", log, priority=20)
        orchestrator = _entered_orchestrator({raiser, survivor})

        with caplog.at_level(logging.ERROR):
            _finalize_then_exit(orchestrator)

        assert log == [("raiser", "run-1"), ("survivor", "run-1")]
        assert [r for r in caplog.records if r.levelno == logging.ERROR]

    def test_raising_extender_does_not_replace_the_run_exception(self) -> None:
        log: _RunLog = []
        orchestrator = _entered_orchestrator({_RunCompleteRecorder("raiser", log, raises=True)})
        failure = RuntimeError("compute failed")
        orchestrator._check_for_error = Mock(side_effect=failure)  # type: ignore[method-assign]

        with pytest.raises(RuntimeError) as raised:
            try:
                orchestrator.compute()
            finally:
                orchestrator.__exit__(None, None, None)

        assert log == [("raiser", "run-1")]
        assert raised.value is failure
