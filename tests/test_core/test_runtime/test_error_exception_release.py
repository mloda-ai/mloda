"""The error register hands the caught exception over instead of holding it."""

from __future__ import annotations

import gc
import weakref
from unittest.mock import Mock

import pytest

from mloda.core.abstract_plugins.components.error_utils import MlodaRunError
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.core.cfw_manager import CfwManager
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.runtime.run import ExecutionOrchestrator


class _WorkerFailure(RuntimeError):
    """Stand-in for an exception raised inside a worker step."""


def _raise_worker_failure() -> None:
    raise _WorkerFailure("worker failed")


def _register_with_error() -> tuple[CfwManager, _WorkerFailure]:
    register = CfwManager({ParallelizationMode.SYNC})
    error = _WorkerFailure("worker failed")
    register.set_error("worker failed", "formatted traceback", exception=error)
    return register, error


class TestTakeErrorException:
    def test_take_returns_the_stored_exception(self) -> None:
        register, error = _register_with_error()

        assert register.take_error_exception() is error

    def test_a_second_take_returns_none(self) -> None:
        register, _ = _register_with_error()

        register.take_error_exception()

        assert register.take_error_exception() is None

    def test_take_without_an_exception_returns_none(self) -> None:
        register = CfwManager({ParallelizationMode.SYNC})
        register.set_error("worker failed", "formatted traceback")

        assert register.take_error_exception() is None

    def test_take_leaves_flag_message_and_exc_info_untouched(self) -> None:
        register, _ = _register_with_error()

        register.take_error_exception()

        assert register.get_error() is True
        assert register.get_error_msg() == "worker failed"
        assert register.get_error_exc_info() == "formatted traceback"

    def test_take_releases_the_exception_for_collection(self) -> None:
        register = CfwManager({ParallelizationMode.SYNC})
        with pytest.raises(_WorkerFailure) as excinfo:
            _raise_worker_failure()
        register.set_error("worker failed", "formatted traceback", exception=excinfo.value)
        collected = weakref.ref(excinfo.value)

        taken = register.take_error_exception()
        assert taken is excinfo.value

        # The taken reference and pytest's ExceptionInfo both pin the traceback, whose frames form
        # a cycle that only a collect resolves.
        del taken
        del excinfo
        gc.collect()
        assert collected() is None


class TestCheckForErrorRaisePath:
    def test_the_original_exception_is_still_reraised(self) -> None:
        orchestrator = ExecutionOrchestrator(Mock(spec=ExecutionPlan))
        register, error = _register_with_error()
        orchestrator.cfw_register = register

        with pytest.raises(_WorkerFailure) as excinfo:
            orchestrator._check_for_error()

        assert excinfo.value is error

    def test_the_register_no_longer_holds_the_exception_after_the_raise(self) -> None:
        orchestrator = ExecutionOrchestrator(Mock(spec=ExecutionPlan))
        register, _ = _register_with_error()
        orchestrator.cfw_register = register

        with pytest.raises(_WorkerFailure):
            orchestrator._check_for_error()

        assert register.take_error_exception() is None

    def test_an_error_without_an_exception_still_raises_the_typed_fallback(self) -> None:
        orchestrator = ExecutionOrchestrator(Mock(spec=ExecutionPlan))
        register = CfwManager({ParallelizationMode.SYNC})
        register.set_error("critical error_out", "formatted traceback")
        orchestrator.cfw_register = register

        with pytest.raises(MlodaRunError, match="critical error_out"):
            orchestrator._check_for_error()
