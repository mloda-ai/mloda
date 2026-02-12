"""
Tests for ExecutionOrchestrator class (renamed from Runner).

This test file defines the requirements for the ExecutionOrchestrator class.
"""

from __future__ import annotations

import uuid as uuid_mod
from unittest.mock import Mock, patch, MagicMock


from mloda.provider import ComputeFramework  # noqa: F401
from mloda.core.prepare.execution_plan import ExecutionPlan

from mloda.core.runtime.run import ExecutionOrchestrator
from mloda.core.core.cfw_manager import CfwManager, MyManager
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode


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
        import inspect

        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        # Check signature includes parallelization_modes parameter
        sig = inspect.signature(orchestrator.__enter__)
        assert "parallelization_modes" in sig.parameters

    def test_enter_accepts_function_extender(self) -> None:
        """__enter__ should accept function_extender parameter."""
        import inspect

        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        sig = inspect.signature(orchestrator.__enter__)
        assert "function_extender" in sig.parameters

    def test_enter_accepts_api_data(self) -> None:
        """__enter__ should accept api_data parameter."""
        import inspect

        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        sig = inspect.signature(orchestrator.__enter__)
        assert "api_data" in sig.parameters

    def test_exit_accepts_exception_info(self) -> None:
        """__exit__ should accept exc_type, exc_val, exc_tb parameters."""
        import inspect

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
        import inspect

        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        sig = inspect.signature(orchestrator._is_step_done)
        params = list(sig.parameters.keys())
        assert "step_uuids" in params
        assert "finished_ids" in params

    def test_can_run_step_signature(self) -> None:
        """_can_run_step should have correct parameters."""
        import inspect

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
        import inspect

        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        sig = inspect.signature(orchestrator._mark_step_as_finished)
        params = list(sig.parameters.keys())
        assert "step_uuid" in params
        assert "finished_steps" in params
        assert "currently_running_steps" in params

    def test_currently_running_step_signature(self) -> None:
        """currently_running_step should have correct parameters."""
        import inspect

        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        sig = inspect.signature(orchestrator.currently_running_step)
        params = list(sig.parameters.keys())
        assert "step_uuids" in params
        assert "currently_running_steps" in params

    def test_execute_step_accepts_step_parameter(self) -> None:
        """_execute_step should accept a step parameter."""
        import inspect

        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        sig = inspect.signature(orchestrator._execute_step)
        params = list(sig.parameters.keys())
        assert "step" in params

    def test_process_step_result_accepts_step_parameter(self) -> None:
        """_process_step_result should accept a step parameter."""
        import inspect

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
