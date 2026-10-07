"""
Tests for ExecutionOrchestrator class (renamed from Runner).

This test file defines the requirements for the ExecutionOrchestrator class.
"""

from __future__ import annotations

import inspect
import logging
import threading
import uuid as uuid_mod
from collections.abc import Callable
from typing import Any
from unittest.mock import Mock, patch, MagicMock
from uuid import UUID

import pytest

from mloda.provider import (  # noqa: F401
    BaseInputData,
    ComputeFramework,
    DataCreator,
    FeatureGroup,
    FeatureSet,
)
from mloda.user import Feature, PluginCollector, mloda
from mloda.core.runtime.data_lifecycle_manager import DataLifecycleManager
from tests.helpers.uuid7_assertions import assert_valid_uuid7
from tests.helpers.plan_stubs import ReiterablePlan
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.prepare.execution_plan import ExecutionPlan

from mloda.core.runtime.run import ExecutionOrchestrator, planned_worker_count
from tests.test_core.test_prepare.test_multi_link_group_resolution import SidePathPandasP, _side_path_prepare
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


def _single_step_mock() -> MagicMock:
    step_uuid = uuid_mod.uuid4()
    step = MagicMock()
    step.get_uuids.return_value = {step_uuid}
    step.required_uuids = set()
    step.step_is_done = False
    step.uuid = step_uuid
    return step


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
        planner = ReiterablePlan(_single_step_mock())

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


class TestRegisterModesCache:
    """The compute loop and _execute_step read the parallelization modes cached on the orchestrator."""

    def test_multiprocessing_loop_reads_parallelization_modes_once(self) -> None:
        """Across several loop passes the Manager is asked for the modes only once."""
        orchestrator = ExecutionOrchestrator(ReiterablePlan(_single_step_mock()))  # type: ignore[arg-type]
        register = Mock(wraps=CfwManager({ParallelizationMode.MULTIPROCESSING}))
        orchestrator.cfw_register = register

        polls = 0

        def fake_process_step_result(step: object) -> bool:
            nonlocal polls
            polls += 1
            return polls >= 3

        orchestrator._execute_step = Mock()  # type: ignore[method-assign]
        orchestrator._process_step_result = Mock(side_effect=fake_process_step_result)  # type: ignore[method-assign]
        orchestrator._drop_data_for_finished_cfws = Mock()  # type: ignore[method-assign]
        orchestrator.data_lifecycle_manager.set_artifacts = Mock()  # type: ignore[method-assign]
        orchestrator.join = Mock()  # type: ignore[method-assign]

        with patch("mloda.core.runtime.run.time.sleep") as mock_sleep:
            orchestrator.compute()

        mock_sleep.assert_called()
        register.get_parallelization_modes.assert_called_once()

    def test_execute_step_uses_cached_register_modes(self) -> None:
        orchestrator = ExecutionOrchestrator(MagicMock())
        register = Mock(wraps=CfwManager({ParallelizationMode.SYNC}))
        orchestrator.cfw_register = register
        cached_modes = {ParallelizationMode.THREADING}
        orchestrator._register_modes = cached_modes
        orchestrator.executor = Mock()
        step = Mock()

        orchestrator._execute_step(step)

        register.get_parallelization_modes.assert_not_called()
        orchestrator.executor._get_execution_function.assert_called_once_with(
            cached_modes, step.get_parallelization_mode()
        )


_CHAIN_COLUMN = "planned_worker_chain_root_col"


class _ChainRootFG(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({_CHAIN_COLUMN})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {_CHAIN_COLUMN: [1, 2, 3]}


class _ChainChildFG(FeatureGroup):
    def input_features(self, options: Any, feature_name: Any) -> Any:
        return {Feature(_CHAIN_COLUMN)}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"planned_worker_chain_child_col": [1, 2, 3]}


def _chain_plan() -> Any:
    session = mloda.prepare(
        [Feature(name="_ChainChildFG")],
        compute_frameworks=["PythonDictFramework"],
        plugin_collector=PluginCollector.enabled_feature_groups({_ChainRootFG, _ChainChildFG}),
        parallelization_modes={ParallelizationMode.MULTIPROCESSING},
    )
    return session.engine.execution_planner


class TestPlannedWorkerCount:
    """planned_worker_count mirrors how many workers an MP run will create."""

    @pytest.mark.parametrize(
        ("register_modes", "expected"),
        [({ParallelizationMode.MULTIPROCESSING}, 4), ({ParallelizationMode.SYNC}, 0)],
        ids=["multiprocessing", "sync"],
    )
    def test_side_path_plan_count(self, register_modes: set[ParallelizationMode], expected: int) -> None:
        plan = _side_path_prepare(
            SidePathPandasP, False, False, ParallelizationMode.MULTIPROCESSING
        ).engine.execution_planner

        assert planned_worker_count(plan, register_modes) == expected

    def test_same_framework_chain_shares_one_worker(self) -> None:
        assert planned_worker_count(_chain_plan(), {ParallelizationMode.MULTIPROCESSING}) == 1


class TestEnterPrestartsStandbyWorkers:
    """__enter__ prestarts standbys (MULTIPROCESSING only, after preflight); compute never does."""

    @staticmethod
    def _orchestrator(mode: ParallelizationMode) -> ExecutionOrchestrator:
        orchestrator = ExecutionOrchestrator(ReiterablePlan(_single_step_mock()))  # type: ignore[arg-type]
        orchestrator.cfw_register = CfwManager({mode})
        orchestrator.worker_manager = Mock()
        orchestrator.worker_manager.find_dead_workers.return_value = []
        orchestrator.worker_manager.find_orphaned_steps.return_value = []
        orchestrator._execute_step = Mock(  # type: ignore[method-assign]
            side_effect=lambda step: setattr(step, "step_is_done", True)
        )
        orchestrator._process_step_result = Mock(return_value=True)  # type: ignore[method-assign]
        orchestrator._drop_data_for_finished_cfws = Mock()  # type: ignore[method-assign]
        orchestrator.data_lifecycle_manager.set_artifacts = Mock()  # type: ignore[method-assign]
        orchestrator.join = Mock()  # type: ignore[method-assign]
        return orchestrator

    @staticmethod
    def _entering() -> ExecutionOrchestrator:
        plan = ExecutionPlan()
        plan.execution_plan = []
        orchestrator = ExecutionOrchestrator(plan)
        orchestrator.worker_manager = Mock()
        orchestrator.flight_server = None
        return orchestrator

    def test_multiprocessing_enter_prestarts_the_planned_count(self) -> None:
        from mloda.core.runtime.worker.multiprocessing_worker import standby_worker

        orchestrator = self._entering()
        modes = {ParallelizationMode.MULTIPROCESSING}

        with (
            patch("mloda.core.runtime.run.planned_worker_count", return_value=3) as count,
            patch("mloda.core.runtime.run.MyManager"),
            patch("mloda.core.runtime.run.mp_start_context"),
        ):
            orchestrator.__enter__(modes)

        count.assert_called_once_with(orchestrator.execution_planner, modes)
        orchestrator.worker_manager.prestart_workers.assert_called_once_with(3, standby_worker)

    def test_prestart_happens_before_the_manager_starts(self) -> None:
        orchestrator = self._entering()
        calls: list[str] = []
        orchestrator.worker_manager.prestart_workers.side_effect = lambda *a: calls.append("prestart")

        with (
            patch("mloda.core.runtime.run.planned_worker_count", return_value=1),
            patch("mloda.core.runtime.run.MyManager") as manager_cls,
            patch("mloda.core.runtime.run.mp_start_context"),
        ):
            manager_instance = Mock()

            def _enter() -> Mock:
                calls.append("manager")
                return manager_instance

            manager_cls.return_value.__enter__.side_effect = _enter
            orchestrator.__enter__({ParallelizationMode.MULTIPROCESSING})

        assert calls == ["prestart", "manager"]

    def test_sync_enter_does_not_prestart(self) -> None:
        orchestrator = self._entering()

        orchestrator.__enter__({ParallelizationMode.SYNC})

        orchestrator.worker_manager.prestart_workers.assert_not_called()

    def test_failing_preflight_does_not_prestart(self) -> None:
        orchestrator = self._entering()

        with (
            patch("mloda.core.runtime.run.raise_on_unpicklable_join_link", side_effect=ValueError("bad link")),
            patch("mloda.core.runtime.run.planned_worker_count", return_value=1),
        ):
            with pytest.raises(ValueError, match="bad link"):
                orchestrator.__enter__({ParallelizationMode.MULTIPROCESSING})

        orchestrator.worker_manager.prestart_workers.assert_not_called()

    def test_exit_stops_unbound_standbys_before_shutting_the_manager_down(self) -> None:
        orchestrator = self._entering()
        calls: list[str] = []
        orchestrator.worker_manager.stop_standbys.side_effect = lambda: calls.append("standbys")
        orchestrator.manager = Mock()
        orchestrator.manager.shutdown.side_effect = lambda: calls.append("shutdown")

        orchestrator.__exit__(None, None, None)

        assert calls == ["standbys", "shutdown"]

    def test_exit_stops_standbys_without_a_manager(self) -> None:
        orchestrator = self._entering()

        orchestrator.__exit__(None, None, None)

        orchestrator.worker_manager.stop_standbys.assert_called_once_with()

    @pytest.mark.parametrize(
        "run",
        [lambda o: o.compute(), lambda o: list(o.compute_stream())],
        ids=["compute", "compute_stream"],
    )
    def test_multiprocessing_compute_does_not_prestart(self, run: Callable[[Any], Any]) -> None:
        orchestrator = self._orchestrator(ParallelizationMode.MULTIPROCESSING)

        with patch("mloda.core.runtime.run.time.sleep"):
            run(orchestrator)

        orchestrator.worker_manager.prestart_workers.assert_not_called()


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
    """The planner never reads a worker's result queue; the flyway fallback still tracks."""

    def test_worker_owned_branch_never_touches_result_queue(self) -> None:
        cfw_uuid = uuid_mod.uuid4()
        child_uuid = uuid_mod.uuid4()
        tracked_uuid = uuid_mod.uuid4()

        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        process, command_queue, result_queue = Mock(), Mock(), Mock()
        orchestrator.worker_manager.process_register[cfw_uuid] = (process, command_queue, result_queue)

        orchestrator.cfw_register = Mock()
        orchestrator.cfw_register.get_uuid_flyway_datasets.return_value = None

        cfw = Mock()
        cfw.uuid = cfw_uuid
        cfw.children_if_root = frozenset({tracked_uuid})

        children = {child_uuid}

        orchestrator._mark_children_and_track(cfw, children)

        command_queue.put.assert_called_once_with(children)
        assert result_queue.mock_calls == []
        assert orchestrator.data_lifecycle_manager.track_data_to_drop[cfw.uuid] == set(cfw.children_if_root)


class TestAddUuidFlywayDatasetsAccumulates:
    """Registering readers for one frame twice unions them; a narrower later set must not drop earlier readers."""

    def test_second_registration_unions_with_first(self) -> None:
        manager = CfwManager({ParallelizationMode.SYNC})
        cf_uuid = uuid_mod.uuid4()
        a, b, c, d = (uuid_mod.uuid4() for _ in range(4))

        manager.add_uuid_flyway_datasets(cf_uuid, {a, b, c})
        manager.add_uuid_flyway_datasets(cf_uuid, {b, d})

        assert manager.get_uuid_flyway_datasets(cf_uuid) == {a, b, c, d}


class TestDropCfwDataRoutedNeverBlocksWhenWorkerAlive:
    """_drop_cfw_data_routed must only queue the drop command; it never reads the worker's result queue."""

    def test_never_touches_result_queue_when_worker_alive(self) -> None:
        cfw_uuid = uuid_mod.uuid4()
        child_uuid = uuid_mod.uuid4()

        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        process = Mock()
        process.is_alive.return_value = True
        command_queue, result_queue = Mock(), Mock()
        orchestrator.worker_manager.process_register[cfw_uuid] = (process, command_queue, result_queue)

        cfw = Mock()
        cfw.children_if_root = frozenset({child_uuid})

        orchestrator._drop_cfw_data_routed(cfw_uuid, cfw)

        command_queue.put.assert_called_once_with(set(cfw.children_if_root))
        assert result_queue.mock_calls == []


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
        raises: bool = False,
        error: type[BaseException] | BaseException = RuntimeError,
        raise_on_run_complete: bool = False,
    ) -> None:
        self.raise_on_run_complete = raise_on_run_complete
        self.label = label
        self.log = log
        self.priority = priority
        self.raises = raises
        self.error = error

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    def on_run_complete(self, run: Any, outcome: Any) -> None:
        self.log.append((self.label, run.run_id))
        if self.raises:
            if isinstance(self.error, BaseException):
                raise self.error
            raise self.error(_RUN_COMPLETE_BOOM)


_RUN_COLUMN = "orchestrator_run_complete_request_col"


class _RunCompleteFG(FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({_RUN_COLUMN})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {_RUN_COLUMN: [1, 2, 3]}


class _FailingRunFG(_RunCompleteFG):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raise RuntimeError("compute failed")


def _session(extenders: set[Extender], failing: bool = False) -> Any:
    return mloda.prepare(
        [Feature(name=_RUN_COLUMN)],
        compute_frameworks=["PythonDictFramework"],
        plugin_collector=PluginCollector.enabled_feature_groups({_FailingRunFG if failing else _RunCompleteFG}),
        parallelization_modes={ParallelizationMode.SYNC},
        function_extender=extenders,
    )


def _run(session: Any) -> Any:
    return session.run(parallelization_modes={ParallelizationMode.SYNC})


class TestDropAllUploadedFlightTablesScrubsCredentials:
    """A FlightServer.drop_tables failure logged as a best-effort WARNING must not leak a credential."""

    def test_drop_tables_failure_warning_scrubs_credential(self, caplog: pytest.LogCaptureFixture) -> None:
        leak_marker = "hunter2z9"
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)
        orchestrator.location = "flight-location"

        cfw = Mock()
        cfw.get_object_ids.return_value = []
        orchestrator.executor = Mock()
        orchestrator.executor.cfw_collection = {uuid_mod.uuid4(): cfw}

        with patch(
            "mloda.core.runtime.run.FlightServer.drop_tables",
            side_effect=Exception(f"drop failed for postgres://u:{leak_marker}@h/db"),
        ):
            with caplog.at_level(logging.WARNING):
                orchestrator._drop_all_uploaded_flight_tables()

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_records, "expected a WARNING record for the failed flight table drop"
        assert not any(leak_marker in r.getMessage() for r in warning_records), (
            f"credential leaked into a WARNING record: {[r.getMessage() for r in warning_records]}"
        )


class TestExitOfANeverEnteredOrchestrator:
    def test_exit_on_a_never_entered_orchestrator_does_not_raise(self) -> None:
        ExecutionOrchestrator(Mock(spec=ExecutionPlan)).__exit__(None, None, None)


class TestRequestNotifiesExtendersOfRunCompletion:
    """Ported from the orchestrator-level exit notification: run completion now fires per request."""

    def test_notifies_with_the_run_id_after_join_and_the_flight_table_sweep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log: _RunLog = []
        monkeypatch.setattr(ExecutionOrchestrator, "join", lambda self: log.append(("join", None)))
        monkeypatch.setattr(
            ExecutionOrchestrator, "_drop_all_uploaded_flight_tables", lambda self: log.append(("sweep", None))
        )

        _run(_session({_RunCompleteRecorder("extender", log)}))

        assert [label for label, _ in log] == ["join", "sweep", "extender"]
        run_id = log[-1][1]
        assert run_id is not None
        assert_valid_uuid7(run_id)

    def test_notifies_after_the_runner_exit_and_a_base_exception_from_the_extender_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Replaces the orchestrator-level manager shutdown case: the runner exit must precede the notification."""
        log: _RunLog = []
        real_exit = ExecutionOrchestrator.__exit__
        manager_spy = Mock()

        def logged_exit(self: ExecutionOrchestrator, *args: Any) -> None:
            if self.manager is None:
                self.manager = manager_spy
            real_exit(self, *args)
            log.append(("exit", None))

        monkeypatch.setattr(ExecutionOrchestrator, "__exit__", logged_exit)
        extender = _RunCompleteRecorder("extender", log, raises=True, error=KeyboardInterrupt)

        with pytest.raises(KeyboardInterrupt):
            _run(_session({extender}))

        assert [label for label, _ in log] == ["exit", "extender"]
        manager_spy.shutdown.assert_called_once_with()

    def test_notifies_when_compute_raises_and_the_run_exception_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log: _RunLog = []
        monkeypatch.setattr(ExecutionOrchestrator, "join", lambda self: log.append(("join", None)))

        with pytest.raises(Exception, match="compute failed"):
            _run(_session({_RunCompleteRecorder("extender", log)}, failing=True))

        assert [label for label, _ in log] == ["join", "extender"]

    @pytest.mark.parametrize("failing_call", ["join", "set_artifacts"])
    def test_notifies_exactly_once_as_failed_when_finalizing_raised(
        self, failing_call: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flipped: a failed worker join used to suppress the notification, now it reports failed."""
        outcomes: list[Any] = []

        class _OutcomeRecorder(_RunCompleteRecorder):
            def on_run_complete(self, run: Any, outcome: Any) -> None:
                outcomes.append(outcome)

        owner = DataLifecycleManager if failing_call == "set_artifacts" else ExecutionOrchestrator
        monkeypatch.setattr(owner, failing_call, Mock(side_effect=Exception("teardown failed")))

        with pytest.raises(Exception, match="teardown failed"):
            _run(_session({_OutcomeRecorder("extender", [])}))

        assert [o.status for o in outcomes] == ["failed"]
        assert outcomes[0].error_type == "builtins.Exception"

    def test_extender_whose_exception_str_raises_does_not_escape_the_run(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        log: _RunLog = []
        raiser = _RunCompleteRecorder("raiser", log, priority=10, raises=True, error=_UnprintableError)
        survivor = _RunCompleteRecorder("survivor", log, priority=20)

        with caplog.at_level(logging.ERROR):
            _run(_session({raiser, survivor}))

        assert [label for label, _ in log] == ["raiser", "survivor"]
        assert [r for r in caplog.records if r.levelno == logging.ERROR]

    @pytest.mark.parametrize("opt_in", [False, True])
    def test_raising_extender_does_not_replace_the_run_exception(self, opt_in: bool) -> None:
        log: _RunLog = []
        failure = RuntimeError("compute failed")

        class _IdentityFG(_FailingRunFG):
            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                raise failure

        session = mloda.prepare(
            [Feature(name=_RUN_COLUMN)],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=PluginCollector.enabled_feature_groups({_IdentityFG}),
            parallelization_modes={ParallelizationMode.SYNC},
            function_extender={_RunCompleteRecorder("raiser", log, raises=True, raise_on_run_complete=opt_in)},
        )

        with pytest.raises(Exception) as raised:
            _run(session)

        assert raised.value is failure
        assert [label for label, _ in log] == ["raiser"]

    def test_opt_in_failure_on_a_successful_run_propagates_after_later_extenders_are_notified(self) -> None:
        log: _RunLog = []
        failure = RuntimeError("opt-in boom")
        raiser = _RunCompleteRecorder(
            "raiser", log, priority=10, raises=True, error=failure, raise_on_run_complete=True
        )
        survivor = _RunCompleteRecorder("survivor", log, priority=20)

        with pytest.raises(RuntimeError) as raised:
            _run(_session({raiser, survivor}))

        assert raised.value is failure
        assert [label for label, _ in log] == ["raiser", "survivor"]

    @pytest.mark.parametrize(
        "first_error, first_opt_in, second_error, second_opt_in",
        [
            (ValueError, False, RuntimeError, True),
            (RuntimeError, True, ValueError, True),
        ],
        ids=["default_then_opt_in", "opt_in_then_opt_in"],
    )
    def test_only_the_propagating_failure_escapes_and_the_other_is_logged(
        self,
        first_error: type[BaseException],
        first_opt_in: bool,
        second_error: type[BaseException],
        second_opt_in: bool,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        log: _RunLog = []
        first = _RunCompleteRecorder(
            "first", log, priority=10, raises=True, error=first_error, raise_on_run_complete=first_opt_in
        )
        second = _RunCompleteRecorder(
            "second", log, priority=20, raises=True, error=second_error, raise_on_run_complete=second_opt_in
        )

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match=_RUN_COMPLETE_BOOM):
                _run(_session({first, second}))

        assert [label for label, _ in log] == ["first", "second"]
        errors = [
            r.getMessage() for r in caplog.records if r.levelno == logging.ERROR and "on_run_complete" in r.getMessage()
        ]
        assert len(errors) == 1
        assert "ValueError" in errors[0]
