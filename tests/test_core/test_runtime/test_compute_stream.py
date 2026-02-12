# mypy: disable-error-code="arg-type, unused-ignore, method-assign, assignment"
"""
Tests for ExecutionOrchestrator.compute_stream() method.

compute_stream() is a generator variant of compute() that yields (UUID, Any) tuples
as results become available, instead of accumulating them internally.
"""

from __future__ import annotations

import types
from typing import Any, Generator, Iterator, Tuple
from unittest.mock import Mock, MagicMock
from uuid import UUID, uuid4

from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.runtime.run import ExecutionOrchestrator


class TestComputeStreamExists:
    """Tests that compute_stream exists and has the correct interface."""

    def test_compute_stream_method_exists(self) -> None:
        """ExecutionOrchestrator should have a compute_stream method."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        assert hasattr(orchestrator, "compute_stream")

    def test_compute_stream_is_callable(self) -> None:
        """compute_stream should be callable."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        assert callable(orchestrator.compute_stream)


class TestComputeStreamReturnsGenerator:
    """Tests that compute_stream returns a generator."""

    def test_compute_stream_returns_generator(self) -> None:
        """compute_stream() should return a generator object."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        mock_cfw_register = MagicMock()
        mock_cfw_register.get_error.return_value = None
        mock_cfw_register.get_artifacts.return_value = {}
        orchestrator.cfw_register = mock_cfw_register

        # Empty execution planner so the loop finishes immediately
        mock_planner.__iter__ = Mock(return_value=iter([]))

        result = orchestrator.compute_stream()

        assert isinstance(result, types.GeneratorType)
        # Consume to trigger finally block
        list(result)


class TestComputeStreamYieldsResults:
    """Tests that compute_stream yields (UUID, Any) tuples from pop_result_data_collection."""

    def test_yields_uuid_and_data_tuples(self) -> None:
        """compute_stream should yield (UUID, Any) tuples from pop_result_data_collection."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        mock_cfw_register = MagicMock()
        mock_cfw_register.get_error.return_value = None
        mock_cfw_register.get_artifacts.return_value = {}
        orchestrator.cfw_register = mock_cfw_register

        expected_uuid = uuid4()
        expected_data = {"column": [1, 2, 3]}

        # Make the planner iterate once then stop (simulating all steps finishing)
        mock_step = MagicMock()
        step_uuid = uuid4()
        mock_step.get_uuids.return_value = {step_uuid}

        call_count = 0

        def planner_iter() -> Iterator[Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return iter([])
            return iter([])

        mock_planner.__iter__ = Mock(side_effect=planner_iter)

        # Mock pop_result_data_collection to yield one result, then empty on subsequent calls
        pop_call_count = 0

        def mock_pop() -> Generator[Tuple[UUID, Any], None, None]:
            nonlocal pop_call_count
            pop_call_count += 1
            if pop_call_count == 1:
                yield expected_uuid, expected_data

        orchestrator.data_lifecycle_manager = MagicMock()
        orchestrator.data_lifecycle_manager.pop_result_data_collection = mock_pop

        gen = orchestrator.compute_stream()
        results = list(gen)

        assert len(results) >= 1
        first_result = results[0]
        assert first_result[0] == expected_uuid
        assert first_result[1] == expected_data

    def test_yields_multiple_results_across_iterations(self) -> None:
        """compute_stream should yield multiple (UUID, Any) tuples as they become available."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        mock_cfw_register = MagicMock()
        mock_cfw_register.get_error.return_value = None
        mock_cfw_register.get_artifacts.return_value = {}
        orchestrator.cfw_register = mock_cfw_register

        uuid_1 = uuid4()
        uuid_2 = uuid4()
        data_1 = "result_data_1"
        data_2 = "result_data_2"

        mock_planner.__iter__ = Mock(return_value=iter([]))

        pop_call_count = 0

        def mock_pop() -> Generator[Tuple[UUID, Any], None, None]:
            nonlocal pop_call_count
            pop_call_count += 1
            if pop_call_count == 1:
                yield uuid_1, data_1
                yield uuid_2, data_2

        orchestrator.data_lifecycle_manager = MagicMock()
        orchestrator.data_lifecycle_manager.pop_result_data_collection = mock_pop

        gen = orchestrator.compute_stream()
        results = list(gen)

        yielded_uuids = {r[0] for r in results}
        assert uuid_1 in yielded_uuids
        assert uuid_2 in yielded_uuids

    def test_yielded_items_are_tuples_of_uuid_and_any(self) -> None:
        """Each yielded item should be a tuple where first element is a UUID."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        mock_cfw_register = MagicMock()
        mock_cfw_register.get_error.return_value = None
        mock_cfw_register.get_artifacts.return_value = {}
        orchestrator.cfw_register = mock_cfw_register

        test_uuid = uuid4()
        test_data = [1, 2, 3]

        mock_planner.__iter__ = Mock(return_value=iter([]))

        pop_call_count = 0

        def mock_pop() -> Generator[Tuple[UUID, Any], None, None]:
            nonlocal pop_call_count
            pop_call_count += 1
            if pop_call_count == 1:
                yield test_uuid, test_data

        orchestrator.data_lifecycle_manager = MagicMock()
        orchestrator.data_lifecycle_manager.pop_result_data_collection = mock_pop

        gen = orchestrator.compute_stream()
        results = list(gen)

        assert len(results) >= 1
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], UUID)


class TestComputeStreamCleanup:
    """Tests that compute_stream calls set_artifacts and join in a finally block."""

    def test_calls_set_artifacts_after_full_consumption(self) -> None:
        """compute_stream should call set_artifacts after generator is fully consumed."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        mock_cfw_register = MagicMock()
        mock_cfw_register.get_error.return_value = None
        mock_cfw_register.get_artifacts.return_value = {"key": "value"}
        orchestrator.cfw_register = mock_cfw_register

        mock_planner.__iter__ = Mock(return_value=iter([]))

        orchestrator.data_lifecycle_manager = MagicMock()
        orchestrator.data_lifecycle_manager.pop_result_data_collection = Mock(return_value=iter([]))
        orchestrator.join = Mock()

        gen = orchestrator.compute_stream()
        list(gen)  # fully consume

        orchestrator.data_lifecycle_manager.set_artifacts.assert_called_once_with(mock_cfw_register.get_artifacts())

    def test_calls_join_after_full_consumption(self) -> None:
        """compute_stream should call join after generator is fully consumed."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        mock_cfw_register = MagicMock()
        mock_cfw_register.get_error.return_value = None
        mock_cfw_register.get_artifacts.return_value = {}
        orchestrator.cfw_register = mock_cfw_register

        mock_planner.__iter__ = Mock(return_value=iter([]))

        orchestrator.data_lifecycle_manager = MagicMock()
        orchestrator.data_lifecycle_manager.pop_result_data_collection = Mock(return_value=iter([]))
        orchestrator.join = Mock()

        gen = orchestrator.compute_stream()
        list(gen)  # fully consume

        orchestrator.join.assert_called_once()

    def test_cleanup_happens_when_generator_not_fully_consumed(self) -> None:
        """set_artifacts and join should be called even if generator is closed early."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        mock_cfw_register = MagicMock()
        mock_cfw_register.get_error.return_value = None
        mock_cfw_register.get_artifacts.return_value = {}
        orchestrator.cfw_register = mock_cfw_register

        uuid_1 = uuid4()
        uuid_2 = uuid4()

        mock_planner.__iter__ = Mock(return_value=iter([]))

        pop_call_count = 0

        def mock_pop() -> Generator[Tuple[UUID, Any], None, None]:
            nonlocal pop_call_count
            pop_call_count += 1
            if pop_call_count == 1:
                yield uuid_1, "data_1"
                yield uuid_2, "data_2"

        orchestrator.data_lifecycle_manager = MagicMock()
        orchestrator.data_lifecycle_manager.pop_result_data_collection = mock_pop
        orchestrator.join = Mock()

        gen = orchestrator.compute_stream()
        # Only consume one item, then close the generator
        next(gen)
        gen.close()

        orchestrator.data_lifecycle_manager.set_artifacts.assert_called_once()
        orchestrator.join.assert_called_once()


class TestComputeStreamErrorHandling:
    """Tests for compute_stream error handling."""

    def test_raises_valueerror_when_cfw_register_is_none(self) -> None:
        """compute_stream should raise ValueError when cfw_register is None."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        orchestrator.cfw_register = None

        import pytest

        with pytest.raises(ValueError, match="CfwManager not initialized"):
            # For a generator, the ValueError may be raised on first next() call
            gen = orchestrator.compute_stream()
            next(gen)

    def test_raises_exception_on_cfw_error(self) -> None:
        """compute_stream should raise an exception when cfw_register reports an error."""
        mock_planner = Mock(spec=ExecutionPlan)
        orchestrator = ExecutionOrchestrator(mock_planner)

        mock_cfw_register = MagicMock()
        mock_cfw_register.get_error.return_value = True
        mock_cfw_register.get_error_exc_info.return_value = "error info"
        mock_cfw_register.get_error_msg.return_value = "error message"
        mock_cfw_register.get_artifacts.return_value = {}
        orchestrator.cfw_register = mock_cfw_register

        mock_planner.__iter__ = Mock(return_value=iter([]))

        orchestrator.data_lifecycle_manager = MagicMock()
        orchestrator.data_lifecycle_manager.pop_result_data_collection = Mock(return_value=iter([]))
        orchestrator.join = Mock()

        import pytest

        # First verify compute_stream exists (will fail until implemented)
        assert hasattr(orchestrator, "compute_stream"), "compute_stream method must exist"

        with pytest.raises(Exception):
            gen = orchestrator.compute_stream()
            list(gen)
