"""Tests for multiprocessing_worker.worker(): worker_index assignment, the child_bootstrap
seam, and the extender-close cleanup path.
"""

import inspect
import logging
import multiprocessing
import pickle  # nosec B403
import queue
import threading
import time
from collections.abc import Mapping
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.close_context import CloseContext
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.core.cfw_manager import CfwManager
from mloda.core.core.step.join_step import JoinStep
from mloda.core.runtime.mp_context import mp_start_context
from mloda.core.runtime.worker import multiprocessing_worker as worker_module
from mloda.core.runtime.worker.multiprocessing_worker import _close_extenders, _execute_command, worker
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class _CloseRecordingExtender(Extender):
    def __init__(
        self,
        sleep_after_record: float = 0.0,
        contexts: list[CloseContext | None] | None = None,
        remainings: list[float] | None = None,
    ) -> None:
        self.close_calls: list[bool] = []
        self._sleep_after_record = sleep_after_record
        self._contexts = contexts
        self._remainings = remainings

    def wraps(self) -> set[ExtenderHook]:
        return set()

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    def close(self) -> None:
        self.close_calls.append(True)
        ctx = CloseContext.current()
        if self._contexts is not None:
            self._contexts.append(ctx)
        if self._remainings is not None:
            assert ctx is not None
            self._remainings.append(ctx.remaining())
        time.sleep(self._sleep_after_record)


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


# Only this token, never an identifier or comment, must be searched for in scrub assertions below.
_LEAK_MARKER = "hunter2z9"
_LEAK_MESSAGE = f"failed for https://u:p@h/db?sig={_LEAK_MARKER}"


class _UnprintableError(Exception):
    def __str__(self) -> str:
        raise ValueError("str() of this exception is broken")


class _RaisingCommand:
    """Module-level so it pickles through the spawn-context queue."""

    def __init__(self, error: type[BaseException] = RuntimeError) -> None:
        self.error = error

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise self.error("command boom")


class _UnprintableRaisingCommand:
    """Module-level so it pickles through the spawn-context queue."""

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise _UnprintableError()


class _LeakRaisingCommand:
    """Module-level so it pickles through the spawn-context queue."""

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(_LEAK_MESSAGE)


_REGISTERED = "Zq81Lm5Vx0Rb7Ye3Hw29"  # nosec B105


class _ContainerLeakError(Exception):
    """Module-level so it pickles."""

    def __init__(self, message: str = f"failed {_REGISTERED}") -> None:
        super().__init__(message)
        self.as_dict = {"k": _REGISTERED}
        self.as_list = [_REGISTERED]


class _ContainerLeakRaisingCommand:
    """Module-level so it pickles through the spawn-context queue."""

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise _ContainerLeakError()


class _ChainedLeakRaisingCommand:
    """Module-level so it pickles through the spawn-context queue."""

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("load failed") from RuntimeError(_LEAK_MESSAGE)


_WORKER_LOGGER_NAME = "mloda.core.runtime.worker.multiprocessing_worker"

_BASE_ERRORS = pytest.mark.parametrize("error", [RuntimeError, SystemExit, KeyboardInterrupt])


class TestWorkerSetsWorkerIndexBeforeTheCommandLoop:
    def test_worker_index_is_set_on_the_cfw_instance(self) -> None:
        ctx = mp_start_context()
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


_STEP_A_SECRET = "step-a-secret-Qw3Er5Ty7"  # nosec B105
_STEP_A_IDENTITY = "step-a-identity-Ui9Op1As2"
_STEP_B_SECRET = "step-b-secret-Df4Gh6Jk8"  # nosec B105
_SEEN_BY_STEPS: list[tuple[frozenset[str], frozenset[str]]] = []


class _SecretsRecordingCommand:
    """Module-level so it pickles; records the secrets visible while the step runs."""

    def __init__(self) -> None:
        self.uuid = uuid4()

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        from mloda.core.abstract_plugins.components.credential_scrub import (
            current_identity_secrets,
            current_registered_secrets,
        )

        _SEEN_BY_STEPS.append((current_registered_secrets(), current_identity_secrets()))
        return "done"


class TestWorkerRegisteredSecrets:
    """The worker starts with no secrets; each step command carries only its own, scoped to that step."""

    def test_the_worker_bootstrap_sees_no_secrets(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import (
            current_identity_secrets,
            current_registered_secrets,
        )

        seen: list[tuple[frozenset[str], frozenset[str]]] = []
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext(
            child_bootstrap=lambda: seen.append((current_registered_secrets(), current_identity_secrets()))
        )
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        assert seen == [(frozenset(), frozenset())]
        assert current_registered_secrets() == frozenset()
        assert current_identity_secrets() == frozenset()

    def test_each_step_command_sees_only_its_own_secrets_and_they_reset_after(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import (
            current_identity_secrets,
            current_registered_secrets,
        )

        _SEEN_BY_STEPS.clear()
        command_queue, result_queue, cfw_register, cfw = _grant_worker_setup()
        first: tuple[frozenset[str], frozenset[str]] = (frozenset({_STEP_A_SECRET}), frozenset({_STEP_A_IDENTITY}))
        second: tuple[frozenset[str], frozenset[str]] = (frozenset({_STEP_B_SECRET}), frozenset())
        empty: tuple[frozenset[str], frozenset[str]] = (frozenset(), frozenset())
        for secrets, identity in (first, second, empty):
            command_queue.put((_SecretsRecordingCommand(), None, secrets, identity))
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        assert _SEEN_BY_STEPS == [first, second, empty]
        assert current_registered_secrets() == frozenset()
        assert current_identity_secrets() == frozenset()

    def test_a_failure_in_result_handling_is_masked_with_the_step_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            worker_module, "_handle_command_result", Mock(side_effect=RuntimeError(f"upload failed {_REGISTERED}"))
        )
        command_queue, result_queue, cfw_register, cfw = _grant_worker_setup()
        command_queue.put((_SecretsRecordingCommand(), None, frozenset({_REGISTERED}), frozenset()))

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        msg, exc_info = cfw_register.set_error.call_args.args
        exc = cfw_register.set_error.call_args.kwargs["exception"]
        assert "upload failed" in msg
        assert _REGISTERED not in msg
        assert _REGISTERED not in exc_info
        assert _REGISTERED not in str(exc)
        assert command_queue.get(timeout=2) == "STOP"


class TestWorkerRunsChildBootstrapBeforeTheCommandLoop:
    def test_bootstrap_callable_is_invoked_exactly_once(self) -> None:
        ctx = mp_start_context()
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
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)


class TestWorkerReportsChildBootstrapExceptionThroughTheErrorChannel:
    @_BASE_ERRORS
    def test_bootstrap_exception_is_reported_via_set_error_and_stop_without_propagating(
        self, error: type[BaseException], caplog: pytest.LogCaptureFixture
    ) -> None:
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        boom = error("boom")
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

    def test_bootstrap_report_falling_back_to_string_only_still_stops(self) -> None:
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.set_error.side_effect = [TypeError("cannot pickle"), None]
        boom = SystemExit(3)
        cfw_register.get_run_context.return_value = RunContext(child_bootstrap=Mock(side_effect=boom))
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        assert cfw_register.set_error.call_count == 2
        first, second = cfw_register.set_error.call_args_list
        assert first.kwargs.get("exception") is boom
        assert len(second.args) == 2 and not second.kwargs
        assert command_queue.get(timeout=2) == "STOP"

    def test_bootstrap_exception_whose_str_raises_is_still_reported_and_stops(self) -> None:
        ctx = mp_start_context()
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

    def test_bootstrap_exception_with_secret_is_scrubbed(self) -> None:
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        boom = RuntimeError(_LEAK_MESSAGE)
        bootstrap = Mock(side_effect=boom)
        cfw_register.get_run_context.return_value = RunContext(child_bootstrap=bootstrap)
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        call_args = cfw_register.set_error.call_args
        error_msg, exc_info = call_args.args
        assert _LEAK_MARKER not in error_msg
        assert _LEAK_MARKER not in exc_info
        assert call_args.kwargs.get("exception") is boom
        assert command_queue.get(timeout=2) == "STOP"


class TestWorkerReportsCommandExceptionThroughTheErrorChannel:
    @_BASE_ERRORS
    def test_command_exception_is_reported_via_set_error_and_stop_without_logging_a_traceback(
        self, error: type[BaseException], caplog: pytest.LogCaptureFixture
    ) -> None:
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        contexts: list[CloseContext | None] = []
        extender = _CloseRecordingExtender(contexts=contexts)
        cfw.function_extender = {extender}
        command_queue.put((_RaisingCommand(error), None, frozenset(), frozenset()))

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        call_args = cfw_register.set_error.call_args
        assert "command boom" in call_args.args[0]
        assert isinstance(call_args.kwargs.get("exception"), error)
        assert command_queue.get(timeout=2) == "STOP"

        assert "root" not in {r.name for r in caplog.records}
        assert not [r for r in caplog.records if "Traceback" in r.getMessage()]
        assert len(contexts) == 1
        recorded = contexts[0]
        assert recorded is not None and recorded.reason == "error"

    @_BASE_ERRORS
    def test_idle_poll_exception_is_reported_and_stops_without_propagating(
        self, error: type[BaseException], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        contexts: list[CloseContext | None] = []
        extender = _CloseRecordingExtender(contexts=contexts)
        cfw.function_extender = {extender}
        boom = error("poll boom")
        monkeypatch.setattr(worker_module, "_parent_gone", Mock(side_effect=boom))

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        assert cfw_register.set_error.call_args.kwargs.get("exception") is boom
        assert command_queue.get(timeout=2) == "STOP"
        assert extender.close_calls == [True]
        assert len(contexts) == 1
        recorded = contexts[0]
        assert recorded is not None and recorded.reason == "error"

    @_BASE_ERRORS
    def test_data_dropping_exception_is_reported_and_stops_without_propagating(
        self, error: type[BaseException], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        contexts: list[CloseContext | None] = []
        extender = _CloseRecordingExtender(contexts=contexts)
        cfw.function_extender = {extender}
        boom = error("drop boom")
        monkeypatch.setattr(worker_module, "_handle_data_dropping", Mock(side_effect=boom))
        command_queue.put({uuid4()})

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        assert cfw_register.set_error.call_args.kwargs.get("exception") is boom
        assert command_queue.get(timeout=2) == "STOP"
        assert extender.close_calls == [True]
        assert len(contexts) == 1
        recorded = contexts[0]
        assert recorded is not None and recorded.reason == "error"

    def test_set_error_raising_keyboard_interrupt_still_stops_closes_and_propagates(self) -> None:
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw_register.set_error.side_effect = KeyboardInterrupt()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        extender = _CloseRecordingExtender()
        cfw.function_extender = {extender}
        command_queue.put((_RaisingCommand(RuntimeError), None, frozenset(), frozenset()))

        with pytest.raises(KeyboardInterrupt):
            worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        assert command_queue.get(timeout=2) == "STOP"
        assert extender.close_calls == [True]

    def test_command_report_falling_back_to_string_only_still_stops(self) -> None:
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw_register.set_error.side_effect = [TypeError("cannot pickle"), None]
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        command_queue.put((_RaisingCommand(SystemExit), None, frozenset(), frozenset()))

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        assert cfw_register.set_error.call_count == 2
        first, second = cfw_register.set_error.call_args_list
        assert isinstance(first.kwargs.get("exception"), SystemExit)
        assert len(second.args) == 2 and not second.kwargs
        assert command_queue.get(timeout=2) == "STOP"

    def test_command_exception_whose_str_raises_is_still_reported_and_stops(self) -> None:
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        command_queue.put((_UnprintableRaisingCommand(), None, frozenset(), frozenset()))

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        call_args = cfw_register.set_error.call_args
        assert call_args.args[0].startswith("An error occurred: _UnprintableError\n")
        assert isinstance(call_args.kwargs.get("exception"), _UnprintableError)
        assert command_queue.get(timeout=2) == "STOP"

    def test_command_exception_with_secret_is_scrubbed(self) -> None:
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        command_queue.put((_LeakRaisingCommand(), None, frozenset(), frozenset()))

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        call_args = cfw_register.set_error.call_args
        error_msg, exc_info = call_args.args
        assert _LEAK_MARKER not in error_msg
        assert _LEAK_MARKER not in exc_info
        assert command_queue.get(timeout=2) == "STOP"

    def test_registered_secret_in_container_attributes_is_masked_before_set_error(self) -> None:
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        command_queue.put((_ContainerLeakRaisingCommand(), None, frozenset({_REGISTERED}), frozenset()))

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        exc = cfw_register.set_error.call_args.kwargs["exception"]
        assert _REGISTERED not in str(exc)
        assert _REGISTERED not in repr(exc.as_dict)
        assert _REGISTERED not in repr(exc.as_list)
        restored = pickle.loads(pickle.dumps(exc))  # nosec B301
        assert _REGISTERED not in repr(restored.as_dict)
        assert _REGISTERED not in repr(restored.as_list)

    def test_command_exception_with_secret_in_chained_cause_is_scrubbed(self) -> None:
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        command_queue.put((_ChainedLeakRaisingCommand(), None, frozenset(), frozenset()))

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        call_args = cfw_register.set_error.call_args
        error_msg, exc_info = call_args.args
        assert _LEAK_MARKER not in error_msg
        assert _LEAK_MARKER not in exc_info
        assert command_queue.get(timeout=2) == "STOP"


class TestWorkerLogsCriticalLocationErrorOnItsOwnLogger:
    def test_missing_location_error_is_not_logged_on_the_root_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        ctx = mp_start_context()
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

        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        contexts: list[CloseContext | None] = []
        extender = _CloseRecordingExtender(contexts=contexts)
        cfw.function_extender = {extender}

        # No STOP is queued; the dead-parent check alone must end the loop.
        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        assert len(contexts) == 1
        recorded = contexts[0]
        assert recorded is not None and recorded.reason == "parent_gone"


class TestWorkerClosesExtendersOnStop:
    """worker() must call every function_extender's close() exactly once, via the STOP
    command loop-break path."""

    def test_close_called_exactly_once_on_stop(self) -> None:
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        contexts: list[CloseContext | None] = []
        extender = _CloseRecordingExtender(contexts=contexts)
        cfw.function_extender = {extender}
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        assert extender.close_calls == [True]
        assert len(contexts) == 1
        recorded = contexts[0]
        assert recorded is not None and recorded.reason == "stop"


class TestWorkerSwallowsExtenderCloseExceptions:
    """A raising close() must not propagate out of worker(), and must not prevent other
    extenders' close() from running."""

    def test_worker_does_not_raise_and_other_extender_still_closes(self, caplog: pytest.LogCaptureFixture) -> None:
        ctx = mp_start_context()
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

        context = CloseContext(deadline=time.monotonic() + 5.0, reason="stop")
        with caplog.at_level(logging.ERROR):
            _close_extenders(cfw, context)

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

    @pytest.mark.parametrize("error", [SystemExit, KeyboardInterrupt])
    def test_close_base_exception_does_not_stop_other_extenders_and_is_reraised(
        self, error: type[BaseException], caplog: pytest.LogCaptureFixture
    ) -> None:
        ok_extender = _CloseRecordingExtender()
        cfw = Mock(spec=ComputeFramework)
        cfw.function_extender = [_RaisingCloseExtender(error=error), ok_extender]
        context = CloseContext(deadline=time.monotonic() + 5.0, reason="stop")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(error):
                _close_extenders(cfw, context)

        assert ok_extender.close_calls == [True]
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) == 1
        assert error_records[0].name == _WORKER_LOGGER_NAME
        assert "_RaisingCloseExtender" in error_records[0].getMessage()

    @pytest.mark.parametrize("error", [SystemExit, KeyboardInterrupt])
    def test_worker_closes_remaining_extenders_and_reraises_close_base_exception(
        self, error: type[BaseException]
    ) -> None:
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        ok_extender = _CloseRecordingExtender()
        cfw.function_extender = [_RaisingCloseExtender(error=error), ok_extender]  # type: ignore[assignment]
        command_queue.put("STOP")

        with pytest.raises(error):
            worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        assert ok_extender.close_calls == [True]


class TestWorkerClosesExtendersEvenWhenChildBootstrapRaises:
    """close() must fire on the child_bootstrap-exception exit path too, even though the
    command loop is never entered."""

    def test_close_called_when_bootstrap_raises(self) -> None:
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        bootstrap = Mock(side_effect=RuntimeError("boom"))
        cfw_register.get_run_context.return_value = RunContext(child_bootstrap=bootstrap)
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        contexts: list[CloseContext | None] = []
        extender = _CloseRecordingExtender(contexts=contexts)
        cfw.function_extender = {extender}

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        bootstrap.assert_called_once()
        assert extender.close_calls == [True]
        assert len(contexts) == 1
        recorded = contexts[0]
        assert recorded is not None and recorded.reason == "error"


class TestWorkerProcessesQueuedCommandsBeforeClosingExtendersOnStop:
    """A command already queued ahead of STOP must be fully processed before the worker exits;
    STOP must not jump ahead of already-queued work, and close() must only run once the loop
    has actually exited."""

    def test_queued_drop_command_effect_precedes_close(self) -> None:
        ctx = mp_start_context()
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
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue = Mock()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_run_context.return_value = RunContext()
        child = uuid4()
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset({child}))
        contexts: list[CloseContext | None] = []
        extender = _CloseRecordingExtender(contexts=contexts)
        cfw.function_extender = {extender}
        command_queue.put({child})

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        assert child in cfw.already_calculated_children_tracker
        assert command_queue.get(timeout=2) == "STOP"
        result_queue.put.assert_not_called()
        assert extender.close_calls == [True]
        assert len(contexts) == 1
        recorded = contexts[0]
        assert recorded is not None and recorded.reason == "stop"


class TestCloseContextRemainingTracksTheSharedDeadline:
    """CloseContext.remaining() reflects one deadline shared across every extender closing in the same worker."""

    def test_second_extender_sees_less_remaining_than_the_first(self) -> None:
        contexts: list[CloseContext | None] = []
        remainings: list[float] = []
        first = _CloseRecordingExtender(sleep_after_record=0.05, contexts=contexts, remainings=remainings)
        second = _CloseRecordingExtender(contexts=contexts, remainings=remainings)
        cfw = Mock(spec=ComputeFramework)
        cfw.function_extender = [first, second]
        context = CloseContext(deadline=time.monotonic() + 5.0, reason="stop")

        _close_extenders(cfw, context)

        assert len(contexts) == 2
        first_ctx, second_ctx = contexts
        assert first_ctx is not None and second_ctx is not None
        assert first_ctx is second_ctx

        assert len(remainings) == 2
        first_remaining, second_remaining = remainings
        assert first_remaining - second_remaining >= 0.04

    def test_current_is_none_outside_close(self) -> None:
        assert CloseContext.current() is None

        contexts: list[CloseContext | None] = []
        extender = _CloseRecordingExtender(contexts=contexts)
        cfw = Mock(spec=ComputeFramework)
        cfw.function_extender = [extender]
        context = CloseContext(deadline=time.monotonic() + 5.0, reason="stop")

        _close_extenders(cfw, context)

        assert len(contexts) == 1
        assert contexts[0] is context
        assert CloseContext.current() is None

    def test_remaining_is_clamped_to_zero_once_the_deadline_has_passed(self) -> None:
        remainings: list[float] = []
        slow = _CloseRecordingExtender(sleep_after_record=0.05, remainings=remainings)
        late = _CloseRecordingExtender(remainings=remainings)
        cfw = Mock(spec=ComputeFramework)
        cfw.function_extender = [slow, late]
        context = CloseContext(deadline=time.monotonic() + 0.01, reason="stop")

        _close_extenders(cfw, context)

        assert len(remainings) == 2
        assert remainings[1] == 0.0

    def test_worker_run_context_graceful_shutdown_timeout_reaches_close_extenders(self) -> None:
        ctx = mp_start_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        run_context = RunContext(
            run_id="run-close-ctx",
            plan_id="plan-close-ctx",
            carrier={"traceparent": "abc"},
            tenant_id="tenant-1",
            project_id="project-1",
            principal="principal-1",
            graceful_shutdown_timeout=7.5,
        )
        cfw_register.get_run_context.return_value = run_context
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        contexts: list[CloseContext | None] = []
        remainings: list[float] = []
        extender = _CloseRecordingExtender(contexts=contexts, remainings=remainings)
        cfw.function_extender = {extender}
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=3)

        assert len(contexts) == 1
        recorded = contexts[0]
        assert recorded is not None
        assert len(remainings) == 1
        assert remainings[0] > 2.0
        assert recorded.reason == "stop"
        assert recorded.run_id == "run-close-ctx"
        assert recorded.plan_id == "plan-close-ctx"
        assert recorded.worker_index == 3
        assert recorded.carrier == {"traceparent": "abc"}
        assert recorded.tenant_id == "tenant-1"
        assert recorded.project_id == "project-1"
        assert recorded.principal == "principal-1"

        assert recorded.carrier is not None
        recorded.carrier["mutated"] = "yes"
        assert "mutated" not in (run_context.carrier or {})


class TestCloseContextIsUsableFromAThreadStartedInClose:
    """A thread started inside close() reads a live, positive remaining() off the captured instance."""

    def test_thread_reads_a_positive_remaining_within_the_timeout(self) -> None:
        captured_remaining: list[float] = []

        class _ThreadCapturingExtender(Extender):
            def wraps(self) -> set[ExtenderHook]:
                return set()

            def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            def close(self) -> None:
                captured = CloseContext.current()
                assert captured is not None

                def _read() -> None:
                    captured_remaining.append(captured.remaining())

                thread = threading.Thread(target=_read)
                thread.start()
                thread.join(timeout=2)

        cfw = Mock(spec=ComputeFramework)
        cfw.function_extender = [_ThreadCapturingExtender()]
        timeout = 5.0
        context = CloseContext(deadline=time.monotonic() + timeout, reason="stop")

        _close_extenders(cfw, context)

        assert len(captured_remaining) == 1
        assert captured_remaining[0] > 0.0
        assert captured_remaining[0] <= timeout


class TestExecuteCommandJoinStepWithoutSource:
    def test_unregistered_join_token_raises(self) -> None:
        cfw_register = Mock(spec=CfwManager)
        step = Mock(spec=JoinStep)
        step.uuid = uuid4()
        step.link = Mock()
        step.destination_framework = Mock()
        step.destination_framework.get_class_name.return_value = "DestinationCFW"
        step.source_framework_uuids = {uuid4()}
        cfw_register.get_join_source_cfw_uuid.return_value = None

        with pytest.raises(ValueError, match="no single frame lists the join token"):
            _execute_command(step, cfw_register, Mock(spec=ComputeFramework), None, None)


_FIRST_SECRET = "first-grant-Xc4Vb7Nm2"  # nosec B105
_SECOND_SECRET = "second-grant-Jd9Lp3Qs5"  # nosec B105


class _GrantReadingCommand:
    """Module-level so it pickles through the spawn-context queue."""

    def __init__(self) -> None:
        self.uuid = uuid4()

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        from mloda.core.abstract_plugins.components.credential_grant import _read_granted

        return _read_granted("first")["token"] and "done"


class _GrantObservingCommand:
    """Reports what a reused worker still holds from the previous command, through its error text."""

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        from mloda.core.abstract_plugins.components.credential_grant import _read_granted
        from mloda.core.abstract_plugins.components.credential_scrub import current_registered_secrets

        try:
            _read_granted("first")
            first_refused = False
        except Exception:
            first_refused = True
        second_readable = _read_granted("second")["token"] == _SECOND_SECRET
        first_leaf_registered = _FIRST_SECRET in current_registered_secrets()
        second_leaf_registered = _SECOND_SECRET in current_registered_secrets()
        raise RuntimeError(
            f"first_refused={first_refused} second_readable={second_readable} "
            f"first_leaf={first_leaf_registered} second_leaf={second_leaf_registered}"
        )


class _GrantEchoingCommand:
    """Reads its granted value and fails with an error that echoes it."""

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        from mloda.core.abstract_plugins.components.credential_grant import _read_granted

        raise RuntimeError(f"failed with {_read_granted('first')['token']}")


def _grant_worker_setup() -> tuple[Any, Any, Mock, ComputeFramework]:
    ctx = mp_start_context()
    command_queue: multiprocessing.Queue[Any] = ctx.Queue()
    result_queue: multiprocessing.Queue[Any] = ctx.Queue()
    cfw_register = Mock(spec=CfwManager)
    cfw_register.get_location.return_value = "grpc://localhost:9999"
    cfw_register.get_run_context.return_value = RunContext()
    cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
    return command_queue, result_queue, cfw_register, cfw


class _Unwalked:
    """Plain object whose attributes redaction does not walk."""

    def __init__(self, value: str) -> None:
        self.value = value


class _SetStateError(Exception):
    """Module-level so it pickles; __str__ renders state held in a set and an object."""

    def __init__(self, secret: str) -> None:
        super().__init__("opaque failure")
        self.in_set = {secret}
        self.in_object = _Unwalked(secret)

    def __str__(self) -> str:
        return f"failed {sorted(self.in_set)} {self.in_object.value}"


class _SetStateRaisingCommand:
    """Raises an error whose __str__ renders the granted secret from unwalked state."""

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        from mloda.core.abstract_plugins.components.credential_grant import _read_granted

        raise _SetStateError(_read_granted("first")["token"])


_PUBLIC_KEY_VALUE = "worker-public-Tg6Hn1Wd8"  # nosec B105


class _GrantPolicyObservingCommand:
    """Reports through its error text whether a declared public key's value is registered as secret."""

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        from mloda.core.abstract_plugins.components.credential_scrub import current_registered_secrets

        raise RuntimeError(
            f"public_registered={_PUBLIC_KEY_VALUE in current_registered_secrets()} "
            f"token_registered={_FIRST_SECRET in current_registered_secrets()}"
        )


class TestWorkerCredentialGrant:
    """The worker opens each command's own grant and drops it when the command ends."""

    def test_a_reused_worker_sees_only_the_current_commands_grant(self) -> None:
        from mloda.core.abstract_plugins.components.credential_grant import StaticGrant

        command_queue, result_queue, cfw_register, cfw = _grant_worker_setup()
        command_queue.put(
            (_GrantReadingCommand(), StaticGrant({"first": {"token": _FIRST_SECRET}}), frozenset(), frozenset())
        )
        command_queue.put(
            (_GrantObservingCommand(), StaticGrant({"second": {"token": _SECOND_SECRET}}), frozenset(), frozenset())
        )

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        message = cfw_register.set_error.call_args.args[0]
        assert "first_refused=True" in message
        assert "second_readable=True" in message
        assert "first_leaf=False" in message
        assert "second_leaf=True" in message

    def test_the_holder_and_secrets_are_gone_after_the_worker_returns(self) -> None:
        from mloda.core.abstract_plugins.components.credential_grant import StaticGrant, _read_granted
        from mloda.core.abstract_plugins.components.credential_scrub import current_registered_secrets

        command_queue, result_queue, cfw_register, cfw = _grant_worker_setup()
        command_queue.put(
            (_GrantReadingCommand(), StaticGrant({"first": {"token": _FIRST_SECRET}}), frozenset(), frozenset())
        )
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        assert _FIRST_SECRET not in current_registered_secrets()
        with pytest.raises(RuntimeError, match="not available outside a step"):
            _read_granted("first")

    def test_a_grant_only_value_echoed_by_a_failing_command_is_masked(self) -> None:
        from mloda.core.abstract_plugins.components.credential_grant import StaticGrant

        command_queue, result_queue, cfw_register, cfw = _grant_worker_setup()
        command_queue.put(
            (_GrantEchoingCommand(), StaticGrant({"first": {"token": _FIRST_SECRET}}), frozenset(), frozenset())
        )

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        call_args = cfw_register.set_error.call_args
        msg, exc_info = call_args.args
        exc = call_args.kwargs["exception"]
        assert _FIRST_SECRET not in str(exc)
        assert _FIRST_SECRET not in repr(exc.args)
        assert _FIRST_SECRET not in msg
        assert _FIRST_SECRET not in exc_info

    def test_a_grant_only_value_in_unwalked_exception_state_is_masked_in_the_report(self) -> None:
        from mloda.core.abstract_plugins.components.credential_grant import StaticGrant

        command_queue, result_queue, cfw_register, cfw = _grant_worker_setup()
        command_queue.put(
            (_SetStateRaisingCommand(), StaticGrant({"first": {"token": _FIRST_SECRET}}), frozenset(), frozenset())
        )

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        msg, exc_info = cfw_register.set_error.call_args.args
        assert _FIRST_SECRET not in msg
        assert _FIRST_SECRET not in exc_info

    def test_a_declared_public_key_is_not_masked_in_the_worker_and_the_secret_is(self) -> None:
        from mloda.core.abstract_plugins.components.credential_grant import StaticGrant
        from mloda.core.abstract_plugins.components.credential_leaves import CredentialKeyPolicy

        policy = CredentialKeyPolicy(public=frozenset({"zzworkerpublickey"}), secret=frozenset())
        value = {"zz_worker_public_key": _PUBLIC_KEY_VALUE, "token": _FIRST_SECRET}
        command_queue, result_queue, cfw_register, cfw = _grant_worker_setup()
        command_queue.put(
            (_GrantPolicyObservingCommand(), StaticGrant({"first": value}, policy=policy), frozenset(), frozenset())
        )

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)

        cfw_register.set_error.assert_called_once()
        message = cfw_register.set_error.call_args.args[0]
        assert "public_registered=False" in message
        assert "token_registered=True" in message
