"""Tests for WorkerManager class that manages thread/process lifecycle for parallel execution."""

import multiprocessing
import queue
import threading
import time
from typing import Any, Callable, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch
from uuid import UUID, uuid4

import pytest

from mloda_core.runtime.worker_manager import WorkerManager


class TestWorkerManagerInit:
    """Test WorkerManager initialization."""

    def test_init_creates_empty_state(self) -> None:
        """WorkerManager should initialize with empty collections."""
        manager = WorkerManager()

        assert manager.tasks == []
        assert manager.process_register == {}
        assert manager.result_queues_collection == set()
        assert manager.result_uuids_collection == set()


class TestWorkerManagerThreadTasks:
    """Test thread task management."""

    def test_add_thread_task_appends_to_tasks(self) -> None:
        """add_thread_task should append thread to tasks list."""
        manager = WorkerManager()
        mock_thread = Mock(spec=threading.Thread)

        manager.add_thread_task(mock_thread)

        assert len(manager.tasks) == 1
        assert manager.tasks[0] == mock_thread

    def test_add_thread_task_starts_thread(self) -> None:
        """add_thread_task should call start() on the thread."""
        manager = WorkerManager()
        mock_thread = Mock(spec=threading.Thread)

        manager.add_thread_task(mock_thread)

        mock_thread.start.assert_called_once()

    def test_add_multiple_thread_tasks(self) -> None:
        """Should be able to add multiple thread tasks."""
        manager = WorkerManager()
        mock_thread1 = Mock(spec=threading.Thread)
        mock_thread2 = Mock(spec=threading.Thread)

        manager.add_thread_task(mock_thread1)
        manager.add_thread_task(mock_thread2)

        assert len(manager.tasks) == 2
        assert mock_thread1 in manager.tasks
        assert mock_thread2 in manager.tasks


class TestWorkerManagerProcessCreation:
    """Test worker process creation and registration."""

    def test_create_worker_process_creates_process(self) -> None:
        """create_worker_process should create a new Process."""
        manager = WorkerManager()
        cfw_uuid = uuid4()
        target_func = Mock()
        args = ("arg1", "arg2")

        process, cmd_queue, result_queue = manager.create_worker_process(cfw_uuid, target_func, args)

        assert isinstance(process, multiprocessing.Process)
        # Queues have put/get methods - verify interface
        assert hasattr(cmd_queue, "put") and hasattr(cmd_queue, "get")
        assert hasattr(result_queue, "put") and hasattr(result_queue, "get")

    def test_create_worker_process_registers_in_process_register(self) -> None:
        """create_worker_process should register process with CFW UUID."""
        manager = WorkerManager()
        cfw_uuid = uuid4()
        target_func = Mock()
        args = ("arg1", "arg2")

        process, cmd_queue, result_queue = manager.create_worker_process(cfw_uuid, target_func, args)

        assert cfw_uuid in manager.process_register
        registered_process, registered_cmd_queue, registered_result_queue = manager.process_register[cfw_uuid]
        assert registered_process == process
        assert registered_cmd_queue == cmd_queue
        assert registered_result_queue == result_queue

    def test_create_worker_process_adds_result_queue_to_collection(self) -> None:
        """create_worker_process should add result queue to collection."""
        manager = WorkerManager()
        cfw_uuid = uuid4()
        target_func = Mock()
        args = ("arg1", "arg2")

        process, cmd_queue, result_queue = manager.create_worker_process(cfw_uuid, target_func, args)

        assert result_queue in manager.result_queues_collection

    def test_create_worker_process_adds_process_to_tasks(self) -> None:
        """create_worker_process should add process to tasks list."""
        manager = WorkerManager()
        cfw_uuid = uuid4()
        target_func = Mock()
        args = ("arg1", "arg2")

        process, cmd_queue, result_queue = manager.create_worker_process(cfw_uuid, target_func, args)

        assert process in manager.tasks

    def test_create_worker_process_starts_process(self) -> None:
        """create_worker_process should start the process."""
        manager = WorkerManager()
        cfw_uuid = uuid4()
        target_func = Mock()
        args = ("arg1", "arg2")

        with patch("multiprocessing.Process") as mock_process_class:
            mock_process = Mock()
            mock_process_class.return_value = mock_process

            manager.create_worker_process(cfw_uuid, target_func, args)

            mock_process.start.assert_called_once()


class TestWorkerManagerProcessRetrieval:
    """Test retrieving existing process information."""

    def test_get_process_queues_returns_existing_process(self) -> None:
        """get_process_queues should return existing process/queues."""
        manager = WorkerManager()
        cfw_uuid = uuid4()
        target_func = Mock()
        args = ()

        original_process, original_cmd_queue, original_result_queue = manager.create_worker_process(
            cfw_uuid, target_func, args
        )

        retrieved = manager.get_process_queues(cfw_uuid)

        assert retrieved is not None
        process, cmd_queue, result_queue = retrieved
        assert process == original_process
        assert cmd_queue == original_cmd_queue
        assert result_queue == original_result_queue

    def test_get_process_queues_returns_none_for_unknown_uuid(self) -> None:
        """get_process_queues should return None for unknown CFW UUID."""
        manager = WorkerManager()
        unknown_uuid = uuid4()

        result = manager.get_process_queues(unknown_uuid)

        assert result is None


class TestWorkerManagerCommandSending:
    """Test sending commands to worker processes."""

    def test_send_command_puts_command_in_queue(self) -> None:
        """send_command should put command in process command queue."""
        manager = WorkerManager()
        cfw_uuid = uuid4()
        command = {"action": "execute", "data": "test"}

        mock_cmd_queue = MagicMock()
        mock_process = Mock(spec=multiprocessing.Process)
        manager.process_register[cfw_uuid] = (mock_process, mock_cmd_queue, MagicMock())

        manager.send_command(cfw_uuid, command)

        mock_cmd_queue.put.assert_called_once_with(command)

    def test_send_command_raises_for_unknown_uuid(self) -> None:
        """send_command should raise ValueError for unknown CFW UUID."""
        manager = WorkerManager()
        unknown_uuid = uuid4()
        command = {"action": "execute"}

        with pytest.raises(ValueError, match="No process found for CFW UUID"):
            manager.send_command(unknown_uuid, command)


class TestWorkerManagerResultPolling:
    """Test polling result queues for completed steps."""

    def test_poll_result_queues_collects_uuids_from_all_queues(self) -> None:
        """poll_result_queues should collect UUIDs from all result queues."""
        manager = WorkerManager()

        # Create mock queues with UUIDs
        uuid1 = str(uuid4())
        uuid2 = str(uuid4())

        mock_queue1 = MagicMock()
        mock_queue1.get.side_effect = [uuid1, queue.Empty()]

        mock_queue2 = MagicMock()
        mock_queue2.get.side_effect = [uuid2, queue.Empty()]

        manager.result_queues_collection.add(mock_queue1)
        manager.result_queues_collection.add(mock_queue2)

        manager.poll_result_queues()

        assert UUID(uuid1) in manager.result_uuids_collection
        assert UUID(uuid2) in manager.result_uuids_collection

    def test_poll_result_queues_handles_empty_queues(self) -> None:
        """poll_result_queues should handle empty queues gracefully."""
        manager = WorkerManager()

        mock_queue = MagicMock()
        mock_queue.get.side_effect = queue.Empty()

        manager.result_queues_collection.add(mock_queue)

        # Should not raise exception
        manager.poll_result_queues()

        assert len(manager.result_uuids_collection) == 0

    def test_poll_result_queues_is_non_blocking(self) -> None:
        """poll_result_queues should use non-blocking get."""
        manager = WorkerManager()

        mock_queue = MagicMock()
        mock_queue.get.side_effect = queue.Empty()
        manager.result_queues_collection.add(mock_queue)

        manager.poll_result_queues()

        mock_queue.get.assert_called_with(block=False)

    def test_poll_result_queues_accumulates_over_multiple_calls(self) -> None:
        """poll_result_queues should accumulate UUIDs across multiple calls."""
        manager = WorkerManager()

        uuid1 = str(uuid4())
        uuid2 = str(uuid4())

        mock_queue = MagicMock()
        manager.result_queues_collection.add(mock_queue)

        # First poll
        mock_queue.get.side_effect = [uuid1, queue.Empty()]
        manager.poll_result_queues()

        # Second poll
        mock_queue.get.side_effect = [uuid2, queue.Empty()]
        manager.poll_result_queues()

        assert UUID(uuid1) in manager.result_uuids_collection
        assert UUID(uuid2) in manager.result_uuids_collection
        assert len(manager.result_uuids_collection) == 2


class TestWorkerManagerStepCompletion:
    """Test checking if steps are completed."""

    def test_is_step_done_returns_true_for_completed_step(self) -> None:
        """is_step_done should return True if step UUID is in collection."""
        manager = WorkerManager()
        step_uuid = uuid4()
        manager.result_uuids_collection.add(step_uuid)

        assert manager.is_step_done(step_uuid) is True

    def test_is_step_done_returns_false_for_incomplete_step(self) -> None:
        """is_step_done should return False if step UUID not in collection."""
        manager = WorkerManager()
        step_uuid = uuid4()

        assert manager.is_step_done(step_uuid) is False


class TestWorkerManagerDropCompletion:
    """Test waiting for drop completion messages."""

    def test_wait_for_drop_completion_returns_on_drop_complete_message(self) -> None:
        """wait_for_drop_completion should return when DROP_COMPLETE message received."""
        manager = WorkerManager()
        cfw_uuid = uuid4()

        mock_queue = MagicMock()
        mock_queue.get.return_value = ("DROP_COMPLETE", cfw_uuid)

        manager.wait_for_drop_completion(mock_queue, cfw_uuid, timeout=1.0)

        # Should complete without timeout
        mock_queue.get.assert_called()

    def test_wait_for_drop_completion_puts_back_other_messages(self) -> None:
        """wait_for_drop_completion should put back non-drop messages."""
        manager = WorkerManager()
        cfw_uuid = uuid4()
        other_message = str(uuid4())

        mock_queue = MagicMock()
        mock_queue.get.side_effect = [other_message, ("DROP_COMPLETE", cfw_uuid)]

        manager.wait_for_drop_completion(mock_queue, cfw_uuid, timeout=1.0)

        # Should put back the other message
        mock_queue.put.assert_called_with(other_message, block=False)

    def test_wait_for_drop_completion_times_out(self) -> None:
        """wait_for_drop_completion should timeout if no message received."""
        manager = WorkerManager()
        cfw_uuid = uuid4()

        mock_queue = MagicMock()
        mock_queue.get.side_effect = queue.Empty()

        start_time = time.time()
        manager.wait_for_drop_completion(mock_queue, cfw_uuid, timeout=0.1)
        elapsed = time.time() - start_time

        # Should timeout after approximately 0.1 seconds
        assert 0.09 < elapsed < 0.2

    def test_wait_for_drop_completion_uses_non_blocking_get(self) -> None:
        """wait_for_drop_completion should use non-blocking queue get."""
        manager = WorkerManager()
        cfw_uuid = uuid4()

        mock_queue = MagicMock()
        mock_queue.get.return_value = ("DROP_COMPLETE", cfw_uuid)

        manager.wait_for_drop_completion(mock_queue, cfw_uuid, timeout=1.0)

        mock_queue.get.assert_called_with(block=False)


class TestWorkerManagerJoinAll:
    """Test joining and terminating all tasks."""

    def test_join_all_terminates_processes(self) -> None:
        """join_all should terminate all multiprocessing processes."""
        manager = WorkerManager()

        mock_process1 = Mock(spec=multiprocessing.Process)
        mock_process2 = Mock(spec=multiprocessing.Process)

        manager.tasks.append(mock_process1)
        manager.tasks.append(mock_process2)

        manager.join_all()

        mock_process1.terminate.assert_called_once()
        mock_process2.terminate.assert_called_once()

    def test_join_all_joins_all_tasks(self) -> None:
        """join_all should call join() on all tasks."""
        manager = WorkerManager()

        mock_thread = Mock(spec=threading.Thread)
        mock_process = Mock(spec=multiprocessing.Process)

        manager.tasks.append(mock_thread)
        manager.tasks.append(mock_process)

        manager.join_all()

        mock_thread.join.assert_called_once()
        mock_process.join.assert_called_once()

    def test_join_all_does_not_terminate_threads(self) -> None:
        """join_all should not call terminate on threads (only processes)."""
        manager = WorkerManager()

        mock_thread = Mock(spec=threading.Thread)
        manager.tasks.append(mock_thread)

        manager.join_all()

        # Threads don't have terminate method, should only join
        assert not hasattr(mock_thread, "terminate") or mock_thread.terminate.call_count == 0
        mock_thread.join.assert_called_once()

    def test_join_all_handles_join_errors(self) -> None:
        """join_all should handle errors during join and raise exception."""
        manager = WorkerManager()

        mock_process = Mock(spec=multiprocessing.Process)
        mock_process.join.side_effect = Exception("Join failed")

        manager.tasks.append(mock_process)

        with pytest.raises(Exception, match="Error while joining tasks"):
            manager.join_all()

    def test_join_all_continues_after_single_join_error(self) -> None:
        """join_all should continue joining other tasks even if one fails."""
        manager = WorkerManager()

        mock_process1 = Mock(spec=multiprocessing.Process)
        mock_process1.join.side_effect = Exception("Join failed")

        mock_process2 = Mock(spec=multiprocessing.Process)

        manager.tasks.append(mock_process1)
        manager.tasks.append(mock_process2)

        with pytest.raises(Exception, match="Error while joining tasks"):
            manager.join_all()

        # Second process should still be joined despite first failure
        mock_process2.terminate.assert_called_once()
        mock_process2.join.assert_called_once()


class TestWorkerManagerIntegration:
    """Integration tests for WorkerManager with multiple operations."""

    def test_complete_workflow_with_process(self) -> None:
        """Test complete workflow: create process, send command, poll results, join."""
        manager = WorkerManager()
        cfw_uuid = uuid4()

        # Create a mock worker process
        target_func = Mock()
        process, cmd_queue, result_queue = manager.create_worker_process(cfw_uuid, target_func, ())

        # Verify process is registered and started
        assert cfw_uuid in manager.process_register
        assert process in manager.tasks
        assert result_queue in manager.result_queues_collection

        # Send command
        command = {"action": "test"}
        manager.send_command(cfw_uuid, command)

        # Simulate result - use a mock queue for reliable testing
        step_uuid = uuid4()
        mock_result_queue = MagicMock()
        mock_result_queue.get.side_effect = [str(step_uuid), queue.Empty()]
        manager.result_queues_collection.clear()
        manager.result_queues_collection.add(mock_result_queue)

        # Poll results
        manager.poll_result_queues()
        assert manager.is_step_done(step_uuid)

        # Join all
        manager.join_all()

    def test_multiple_processes_for_different_cfws(self) -> None:
        """Test managing multiple processes for different CFWs."""
        manager = WorkerManager()

        cfw_uuid1 = uuid4()
        cfw_uuid2 = uuid4()

        target_func = Mock()

        # Create two processes
        process1, cmd_queue1, result_queue1 = manager.create_worker_process(cfw_uuid1, target_func, ())
        process2, cmd_queue2, result_queue2 = manager.create_worker_process(cfw_uuid2, target_func, ())

        # Verify both are registered separately
        assert len(manager.process_register) == 2
        assert len(manager.tasks) == 2
        assert len(manager.result_queues_collection) == 2

        # Verify we can retrieve each independently
        retrieved1 = manager.get_process_queues(cfw_uuid1)
        retrieved2 = manager.get_process_queues(cfw_uuid2)

        assert retrieved1 is not None
        assert retrieved2 is not None
        assert retrieved1[0] == process1
        assert retrieved2[0] == process2

    def test_mixed_threads_and_processes(self) -> None:
        """Test managing both threads and processes together."""
        manager = WorkerManager()

        # Add threads
        mock_thread1 = Mock(spec=threading.Thread)
        mock_thread2 = Mock(spec=threading.Thread)
        manager.add_thread_task(mock_thread1)
        manager.add_thread_task(mock_thread2)

        # Add process
        cfw_uuid = uuid4()
        target_func = Mock()
        process, cmd_queue, result_queue = manager.create_worker_process(cfw_uuid, target_func, ())

        # Verify all are tracked
        assert len(manager.tasks) == 3
        assert mock_thread1 in manager.tasks
        assert mock_thread2 in manager.tasks
        assert process in manager.tasks

        # Join should handle both types
        manager.join_all()
