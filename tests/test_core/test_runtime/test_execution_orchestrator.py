"""
Tests for ExecutionOrchestrator class (renamed from Runner).

This test file defines the requirements for the ExecutionOrchestrator class.
"""

from __future__ import annotations

from unittest.mock import Mock


from mloda.provider import ComputeFramework  # noqa: F401
from mloda.core.prepare.execution_plan import ExecutionPlan

from mloda.core.runtime.run import ExecutionOrchestrator


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
