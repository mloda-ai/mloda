"""Tests for multiprocessing_worker.worker(): worker_index assignment and the child_bootstrap
seam (invoked once before the command loop, exceptions reported via the standard error channel).
"""

import inspect
import logging
import multiprocessing
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.core.cfw_manager import CfwManager
from mloda.core.runtime.mp_context import mp_spawn_context
from mloda.core.runtime.worker.multiprocessing_worker import worker
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


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


class TestWorkerBindsTfsConnectionWhenNeededBeforeTheCommandLoop:
    """Binds the TFS connection worker-side so the unpicklable live connection object never
    crosses into the worker."""

    def test_needs_tfs_connection_true_and_unset_connection_calls_set_framework_connection_object_with_none(
        self,
    ) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        assert cfw.framework_connection_object is None
        cfw.set_framework_connection_object = Mock()  # type: ignore[method-assign]
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), True, worker_index=0)

        cfw.set_framework_connection_object.assert_called_once_with(None)

    def test_needs_tfs_connection_true_and_already_set_connection_is_not_clobbered(self) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        sentinel_connection = object()
        cfw.framework_connection_object = sentinel_connection
        cfw.set_framework_connection_object = Mock()  # type: ignore[method-assign]
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), True, worker_index=0)

        cfw.set_framework_connection_object.assert_not_called()
        assert cfw.framework_connection_object is sentinel_connection

    def test_needs_tfs_connection_false_does_not_call_set_framework_connection_object(self) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        assert cfw.framework_connection_object is None
        cfw.set_framework_connection_object = Mock()  # type: ignore[method-assign]
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), False, worker_index=0)

        cfw.set_framework_connection_object.assert_not_called()

    def test_needs_tfs_connection_defaults_to_false_when_omitted(self) -> None:
        """Default False keeps existing direct worker(...) calls elsewhere in this file working."""
        signature = inspect.signature(worker)

        assert "needs_tfs_connection" in signature.parameters
        assert signature.parameters["needs_tfs_connection"].default is False

        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        cfw.set_framework_connection_object = Mock()  # type: ignore[method-assign]
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw.set_framework_connection_object.assert_not_called()


class TestWorkerReportsTfsConnectionBindExceptionThroughTheErrorChannel:
    """set_framework_connection_object(None) must be guarded the same way the child_bootstrap
    call immediately above it is: a raising framework (e.g. DuckDBFramework's real implementation
    on a MULTIPROCESSING-capable subclass) must not crash the worker process."""

    def test_bind_exception_is_reported_via_set_error_and_stop_without_propagating(self) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        boom = ValueError("boom")
        cfw.set_framework_connection_object = Mock(side_effect=boom)  # type: ignore[method-assign]

        # The call itself must not raise, even though set_framework_connection_object() does.
        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), True, worker_index=0)

        cfw_register.set_error.assert_called_once()
        call_args = cfw_register.set_error.call_args
        error_message = call_args.args[0]
        assert "boom" in error_message
        assert call_args.kwargs.get("exception") is boom

        # worker() must have put "STOP" on command_queue itself (via _handle_stop_command),
        # exactly like the bootstrap exception path does.
        stopped_command = command_queue.get(timeout=2)
        assert stopped_command == "STOP"


class TestWorkerWarnsWhenTfsConnectionBindSilentlyNoOps:
    """A framework (e.g. IcebergFramework) whose set_framework_connection_object(None) is a silent
    no-op leaves framework_connection_object None with no exception raised. The worker must not
    crash or hang, but must log a warning so the eventual downstream failure is easy to trace back
    to a framework that does not yet support worker-side self-construction."""

    def test_silent_no_op_bind_logs_a_warning_and_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING)

        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        assert cfw.framework_connection_object is None
        # No-op: does not raise, and does not actually set framework_connection_object.
        cfw.set_framework_connection_object = Mock()  # type: ignore[method-assign]
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), True, worker_index=0)

        assert len(caplog.records) >= 1
        assert any(record.levelno == logging.WARNING for record in caplog.records)
