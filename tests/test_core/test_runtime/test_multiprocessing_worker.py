"""Tests for multiprocessing_worker.worker().

Covers three responsibilities of worker(), all handled before it enters its main while-True
command loop:

- Accepting a trailing worker_index parameter and setting cfw.worker_index = worker_index.
  This is the only place the spawned worker process learns which index it is among the
  workers spawned so far in this run: create_worker_process (see
  test_worker_manager_worker_index.py) computes the index and passes it as an extra trailing
  argument; worker() is the callable started as that process's target and is responsible for
  landing it on the ComputeFramework instance running inside it.
- Reading cfw_register.get_child_bootstrap() and, if not None, calling it exactly once before
  any command (feature group run) is processed. This is the child-process bootstrap seam: a
  caller may register a plain, picklable, no-argument callable that mloda invokes once inside
  a spawned worker process before that worker does anything else.
- If that bootstrap callable raises, worker() must report the exception through the SAME error
  channel every other worker failure uses (cfw_register.set_error(...) followed by
  _handle_stop_command(command_queue)), and must not let the exception propagate out of
  worker() itself. Every other failure path in this module (_execute_command's except block,
  error_out) already follows this pattern; the bootstrap call sits outside any try/except today.
"""

import multiprocessing
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.core.cfw_manager import CfwManager
from mloda.core.runtime.mp_context import mp_spawn_context
from mloda.core.runtime.worker.multiprocessing_worker import worker
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class TestWorkerSetsWorkerIndexBeforeTheCommandLoop:
    """worker() must set cfw.worker_index from its new worker_index parameter."""

    def test_worker_index_is_set_on_the_cfw_instance(self) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        cfw_register.get_child_bootstrap.return_value = None
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=5)

        assert cfw.worker_index == 5


class TestWorkerRunsChildBootstrapBeforeTheCommandLoop:
    """worker() must invoke cfw_register.get_child_bootstrap(), if not None, exactly once
    before entering its while-True command loop, i.e. before any command is processed."""

    def test_bootstrap_callable_is_invoked_exactly_once(self) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        bootstrap = Mock()
        cfw_register.get_child_bootstrap.return_value = bootstrap
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
        cfw_register.get_child_bootstrap.return_value = None
        cfw = PythonDictFramework(mode=ParallelizationMode.MULTIPROCESSING, children_if_root=frozenset())
        command_queue.put("STOP")

        worker(command_queue, result_queue, cfw_register, cfw, uuid4(), worker_index=0)


class TestWorkerReportsChildBootstrapExceptionThroughTheErrorChannel:
    """If the child_bootstrap callable raises, worker() must report it via
    cfw_register.set_error(...) + _handle_stop_command(command_queue), exactly like every other
    failure path in this module, instead of letting the exception propagate out of worker()."""

    def test_bootstrap_exception_is_reported_via_set_error_and_stop_without_propagating(self) -> None:
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_location.return_value = "grpc://localhost:9999"
        boom = RuntimeError("boom")
        bootstrap = Mock(side_effect=boom)
        cfw_register.get_child_bootstrap.return_value = bootstrap
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
