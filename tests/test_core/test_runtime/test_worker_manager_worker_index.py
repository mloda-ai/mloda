"""Tests for WorkerManager.create_worker_process's zero-based worker_index, appended as an
extra trailing positional arg to the spawned process's target."""

from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

from mloda.core.runtime.worker_manager import WorkerManager


def _noop_target(*args: Any, **kwargs: Any) -> None:
    return None


def _mock_spawn_context() -> Any:
    mock_process = Mock()
    mock_ctx = Mock()
    mock_ctx.Process.return_value = mock_process
    mock_ctx.Queue.return_value = Mock()
    return mock_ctx


class TestCreateWorkerProcessWorkerIndex:
    def test_first_worker_created_in_a_run_gets_index_zero(self) -> None:
        manager = WorkerManager()
        mock_ctx = _mock_spawn_context()

        with patch("mloda.core.runtime.worker_manager.mp_spawn_context", return_value=mock_ctx):
            manager.create_worker_process(uuid4(), _noop_target, ())

        _, kwargs = mock_ctx.Process.call_args
        assert kwargs["args"][-1] == 0

    def test_second_worker_created_in_a_run_gets_index_one_regardless_of_cfw_uuid(self) -> None:
        manager = WorkerManager()
        mock_ctx = _mock_spawn_context()

        with patch("mloda.core.runtime.worker_manager.mp_spawn_context", return_value=mock_ctx):
            manager.create_worker_process(uuid4(), _noop_target, ())
            manager.create_worker_process(uuid4(), _noop_target, ())

        first_args = mock_ctx.Process.call_args_list[0].kwargs["args"]
        second_args = mock_ctx.Process.call_args_list[1].kwargs["args"]
        assert first_args[-1] == 0
        assert second_args[-1] == 1
