"""Tests for multiprocessing_worker.worker(): worker_index assignment, the child_bootstrap
seam, and the extender-close cleanup path.
"""

import inspect
import logging
import multiprocessing
import queue
from collections.abc import Mapping
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.core.cfw_manager import CfwManager
from mloda.core.runtime.mp_context import mp_spawn_context
from mloda.core.runtime.worker.multiprocessing_worker import _close_extenders, worker
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class _CloseRecordingExtender(Extender):
    def __init__(self) -> None:
        self.close_calls: list[bool] = []

    def wraps(self) -> set[ExtenderHook]:
        return set()

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    def close(self) -> None:
        self.close_calls.append(True)


class _RaisingCloseExtender(Extender):
    """close() always raises the configured error type."""

    def __init__(self, error: type[BaseException] = RuntimeError) -> None:
        self.error = error

    def wraps(self) -> set[ExtenderHook]:
        return set()

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    def close(self) -> None:
        raise self.error("close boom")


class _UnprintableError(Exception):
    def __str__(self) -> str:
        raise ValueError("str() of this exception is broken")


class _RaisingCommand:
    """Module-level so it pickles through the spawn-context queue."""

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("command boom")


class _UnprintableRaisingCommand:
    """Module-level so it pickles through the spawn-context queue."""

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise _UnprintableError()


_WORKER_LOGGER_NAME = "mloda.core.runtime.worker.multiprocessing_worker"


class TestWorkerSetsWorkerIndexBeforeTheCommandLoop:
    def test_worker_index_is_set_on_the_cfw_instance(self) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=5)

        assert cfw.worker_index == 5


class TestWorkerHasNoConnectionBindPrologue:
    """worker()'s signature carries no connection-binding parameter."""

    def test_worker_signature_has_no_needs_tfs_connection(self) -> None:
        signature = inspect.signature(worker)

        assert "needs_tfs_connection" not in signature.parameters


class TestWorkerRunsChildBootstrapBeforeTheCommandLoop:
    def test_bootstrap_callable_is_invoked_exactly_once(self) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        bootstrap = Mock()
        cfw_register.get_run_context.return_value = RunContext(child_bootstrap=bootstrap)
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        bootstrap.assert_called_once()

    def test_none_child_bootstrap_does_not_raise(self) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)


class TestWorkerReportsChildBootstrapExceptionThroughTheErrorChannel:
    def test_bootstrap_exception_is_reported_via_set_error_and_stop_without_propagating(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        boom = RuntimeError("boom")
        bootstrap = Mock(side_effect=boom)
        cfw_register.get_run_context.return_value = RunContext(child_bootstrap=bootstrap)
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())

        # The call itself must not raise, even though bootstrap() does.
        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        call_args = cfw_register.set_error.call_args
        error_message = call_args.args[0]
        assert "boom" in error_message
        assert call_args.kwargs.get("exception") is boom

        # worker() must have put "STOP" on command_queue itself (via _handle_stop_command),
        # exactly like the existing except block at the bottom of the while-True loop does.
        stopped_command = command_queue.get(timeout=2)
        assert stopped_command == "STOP"

        # The parent logs the traceback once; the child must not log it, least of all on the root logger.
        assert "root" not in {r.name for r in caplog.records}
        assert not [r for r in caplog.records if "Traceback" in r.getMessage()]

    def test_bootstrap_exception_whose_str_raises_is_still_reported_and_stops(self) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        boom = _UnprintableError()
        bootstrap = Mock(side_effect=boom)
        cfw_register.get_run_context.return_value = RunContext(child_bootstrap=bootstrap)
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        call_args = cfw_register.set_error.call_args
        assert call_args.args[0].startswith("An error occurred: _UnprintableError\n")
        assert call_args.kwargs.get("exception") is boom
        assert command_queue.get(timeout=2) == "STOP"


class TestWorkerReportsCommandExceptionThroughTheErrorChannel:
    def test_command_exception_is_reported_via_set_error_and_stop_without_logging_a_traceback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        command_queue.put(_RaisingCommand())

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        call_args = cfw_register.set_error.call_args
        assert "command boom" in call_args.args[0]
        assert isinstance(call_args.kwargs.get("exception"), RuntimeError)
        assert command_queue.get(timeout=2) == "STOP"

        assert "root" not in {r.name for r in caplog.records}
        assert not [r for r in caplog.records if "Traceback" in r.getMessage()]

    def test_command_exception_whose_str_raises_is_still_reported_and_stops(self) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        command_queue.put(_UnprintableRaisingCommand())

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        call_args = cfw_register.set_error.call_args
        assert call_args.args[0].startswith("An error occurred: _UnprintableError\n")
        assert isinstance(call_args.kwargs.get("exception"), _UnprintableError)
        assert command_queue.get(timeout=2) == "STOP"


class TestWorkerLogsCriticalLocationErrorOnItsOwnLogger:
    def test_missing_location_error_is_not_logged_on_the_root_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = None
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        assert [r.name for r in caplog.records if "critical error" in r.getMessage()] == [_WORKER_LOGGER_NAME]


class TestWorkerExitsWhenTheParentProcessIsNoLongerAlive:
    """worker() exits its command loop once parent_process() reports the parent as dead, even without a STOP."""

    @pytest.mark.timeout(5)
    def test_worker_returns_without_a_stop_command_when_parent_is_dead(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeDeadParent:
            def is_alive(self) -> bool:
                return False

        monkeypatch.setattr(multiprocessing, "parent_process", lambda: _FakeDeadParent())

        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())

        # No STOP is queued; the dead-parent check alone must end the loop.
        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)


class TestWorkerClosesExtendersOnStop:
    """worker() must call every function_extender's close() exactly once, via the STOP
    command loop-break path."""

    def test_close_called_exactly_once_on_stop(self) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        extender = _CloseRecordingExtender()
        cfw.function_extender = {extender}
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        assert extender.close_calls == [True]


class TestWorkerSwallowsExtenderCloseExceptions:
    """A raising close() must not propagate out of worker(), and must not prevent other
    extenders' close() from running."""

    def test_worker_does_not_raise_and_other_extender_still_closes(self, caplog: pytest.LogCaptureFixture) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        ok_extender = _CloseRecordingExtender()
        raising_extender = _RaisingCloseExtender()
        cfw.function_extender = {ok_extender, raising_extender}
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        assert ok_extender.close_calls == [True]
        assert "root" not in {r.name for r in caplog.records}
        assert [r.name for r in caplog.records if "close boom" in r.getMessage()] == [_WORKER_LOGGER_NAME]

    def test_close_exceptions_whose_str_raises_do_not_escape_and_other_extender_still_closes(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ok_extender = _CloseRecordingExtender()
        raising_extender = _RaisingCloseExtender(error=_UnprintableError)
        # A list pins the raising extender first; a real cfw's function_extender is an unordered set.
        cfw = Mock(spec=ComputeFramework)
        cfw.function_extender = [raising_extender, ok_extender]

        with caplog.at_level(logging.ERROR):
            _close_extenders(cfw)

        assert ok_extender.close_calls == [True]
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) == 1
        record = error_records[0]
        assert record.name == _WORKER_LOGGER_NAME
        assert "_RaisingCloseExtender" in record.getMessage()
        assert "_UnprintableError" in record.getMessage()
        assert record.exc_info is None
        args = record.args
        arg_values = args.values() if isinstance(args, Mapping) else (args or ())
        assert not [a for a in arg_values if isinstance(a, BaseException)]


class TestWorkerClosesExtendersEvenWhenChildBootstrapRaises:
    """close() must fire on the child_bootstrap-exception exit path too, even though the
    command loop is never entered."""

    def test_close_called_when_bootstrap_raises(self) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        bootstrap = Mock(side_effect=RuntimeError("boom"))
        cfw_register.get_run_context.return_value = RunContext(child_bootstrap=bootstrap)
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        extender = _CloseRecordingExtender()
        cfw.function_extender = {extender}

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        bootstrap.assert_called_once()
        assert extender.close_calls == [True]


class TestWorkerProcessesQueuedCommandsBeforeClosingExtendersOnStop:
    """A command already queued ahead of STOP must be fully processed before the worker exits;
    STOP must not jump ahead of already-queued work, and close() must only run once the loop
    has actually exited."""

    def test_queued_drop_command_effect_precedes_close(self) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue = Mock()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        # A non-empty children_if_root, unsatisfied by the queued drop command, so
        # _handle_data_dropping resolves False and the loop continues to the next queued
        # command (STOP) instead of breaking on the drop command itself.
        child = uuid4()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset({uuid4()}))
        # A truthy but unresolved add_already_calculated_children_and_drop_if_possible()
        # return, so a worker that mistakes any truthy result for "resolved" is caught too.
        cfw.object_ids = ["x"]
        extender = _CloseRecordingExtender()
        cfw.function_extender = {extender}
        command_queue.put({child})
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        # The queued drop command's effect ran (worker() runs in-process, so cfw is shared).
        assert child in cfw.already_calculated_children_tracker
        # A drop command posts nothing on the result queue.
        result_queue.put.assert_not_called()
        # No stray STOP left behind on the command queue.
        with pytest.raises(queue.Empty):
            command_queue.get(timeout=0.2)
        assert extender.close_calls == [True]

    def test_resolved_drop_command_stops_worker_without_posting_a_result(self) -> None:
        """The resolved-drop path (all children_if_root satisfied) queues its own STOP."""
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue = Mock()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        child = uuid4()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset({child}))
        extender = _CloseRecordingExtender()
        cfw.function_extender = {extender}
        command_queue.put({child})

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        assert child in cfw.already_calculated_children_tracker
        assert command_queue.get(timeout=2) == "STOP"
        result_queue.put.assert_not_called()
        assert extender.close_calls == [True]
