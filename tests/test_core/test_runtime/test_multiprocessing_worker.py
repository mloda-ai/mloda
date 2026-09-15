"""Tests for multiprocessing_worker.worker(): worker_index assignment and the child_bootstrap
seam (invoked once before the command loop, exceptions reported via the standard error channel).
"""

import inspect
import multiprocessing
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.core.cfw_manager import CfwManager
from mloda.core.runtime.mp_context import mp_spawn_context
from mloda.core.runtime.worker.multiprocessing_worker import worker
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
    """close() always raises."""

    def wraps(self) -> set[ExtenderHook]:
        return set()

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    def close(self) -> None:
        raise RuntimeError("close boom")


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
    def test_bootstrap_exception_is_reported_via_set_error_and_stop_without_propagating(self) -> None:
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

    def test_worker_does_not_raise_and_other_extender_still_closes(self) -> None:
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
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        # A non-empty children_if_root, unsatisfied by the queued (empty) drop command, so
        # _handle_data_dropping resolves False and the loop continues to the next queued
        # command (STOP) instead of breaking on the drop command itself.
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset({uuid4()}))
        extender = _CloseRecordingExtender()
        cfw.function_extender = {extender}
        command_queue.put(set())
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        # The queued drop command's effect (its DROP_COMPLETE ack) must have been produced.
        drop_ack = result_queue.get(timeout=2)
        assert drop_ack == ("DROP_COMPLETE", cfw.uuid, False)
        assert extender.close_calls == [True]
