"""
Tests for ComputeFrameworkExecutor class.

This test file defines the requirements for the ComputeFrameworkExecutor class
through failing tests (TDD Red Phase). The class should extract step execution
and CFW initialization logic from the Runner class.
"""

from __future__ import annotations

import pickle  # nosec B403
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock, patch
from uuid import UUID, uuid4

import pytest

from mloda.user import ParallelizationMode
from mloda.provider import ComputeFramework
from mloda.core.abstract_plugins.hook_context import input_data_load_gate_scope
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook, build_hook_extenders
from mloda.core.abstract_plugins.components.credential_scrub import registered_secrets_scope
from mloda.core.abstract_plugins.components.credential_leaves import (
    credential_identity_literals,
    credential_secret_literals,
)
from mloda.core.abstract_plugins.components.input_data.api_data_scope import ApiDataRef
from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.abstract_plugins.components.input_data.read_db_fg import DBQuery, DBTable, ReadDBFG
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.core.cfw_manager import CfwManager
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep, resolve_join_source_cfw
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.core.runtime.compute_framework_executor import ComputeFrameworkExecutor
from mloda.core.runtime.worker.thread_worker import thread_worker
from mloda.core.runtime.worker_manager import WorkerManager
from tests.helpers.credential_handle_stubs import ChsConsumerFG
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import ToySourceFG
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework

# Only this token, never an identifier or comment, must be searched for in scrub assertions below.
_LEAK_MARKER = "hunter2z9"
_LEAK_MESSAGE = f"failed for https://u:p@h/db?sig={_LEAK_MARKER}"
_REGISTERED = "Zq81Lm5Vx0Rb7Ye3Hw29"  # nosec B105


def _worker_manager_mock() -> Mock:
    """A WorkerManager double carrying the per-run done set the executor marks steps in."""
    manager = Mock(spec=WorkerManager)
    manager.result_uuids_collection = set()
    return manager


def _ignore_done(_step_uuid: UUID) -> None:
    """A mark_done callback for thread_worker tests that do not track completion."""


class _ContainerLeakError(Exception):
    def __init__(self, message: str = f"failed {_REGISTERED}") -> None:
        super().__init__(message)
        self.as_dict = {"k": _REGISTERED}
        self.as_list = [_REGISTERED]


class _StateHoldingExtender(Extender):
    """Minimal concrete Extender carrying inspectable state, usable as a real pickle payload."""

    def __init__(self, marker: str) -> None:
        self.marker = marker

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


class TestComputeFrameworkExecutorConstruction:
    """Tests for ComputeFrameworkExecutor constructor and initialization."""

    def test_constructor_initializes_empty_cfw_collection(self) -> None:
        """Constructor should initialize an empty cfw_collection dictionary."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()

        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        assert hasattr(executor, "cfw_collection")
        assert isinstance(executor.cfw_collection, dict)
        assert len(executor.cfw_collection) == 0

    def test_constructor_stores_cfw_register_dependency(self) -> None:
        """Constructor should store the cfw_register dependency."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()

        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        assert executor.cfw_register is cfw_register

    def test_constructor_stores_worker_manager_dependency(self) -> None:
        """Constructor should store the worker_manager dependency."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()

        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        assert executor.worker_manager is worker_manager


class TestInitComputeFramework:
    """Tests for init_compute_framework method."""

    def test_creates_new_compute_framework_instance(self) -> None:
        """run_context must be assigned post-construction, not passed as a constructor kwarg."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        function_extender: set[Extender] = {Mock(spec=Extender)}
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager, function_extender=function_extender)

        # Mock CFW class
        mock_cfw_class = Mock()
        mock_cfw_instance = Mock(spec=ComputeFramework)
        mock_cfw_class.return_value = mock_cfw_instance
        mock_cfw_class.get_class_name.return_value = "TestCFW"

        test_uuid = uuid4()
        mock_cfw_instance.get_uuid.return_value = test_uuid

        run_id = "01909a3b-1234-7abc-8def-0123456789ab"
        carrier = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
        cfw_register.get_run_context.return_value = RunContext(run_id=run_id, carrier=carrier)

        children = {uuid4(), uuid4()}
        mode = ParallelizationMode.SYNC

        cfw_uuid = executor.init_compute_framework(mock_cfw_class, mode, children, test_uuid)

        mock_cfw_class.assert_called_once_with(
            mode, frozenset(children), test_uuid, function_extender=function_extender
        )
        assert mock_cfw_instance.run_context == RunContext(run_id=run_id, carrier=carrier)
        assert cfw_uuid == test_uuid

    def test_generates_uuid_when_not_provided(self) -> None:
        """Should generate a UUID if none is provided."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        mock_cfw_class = Mock()
        mock_cfw_instance = Mock(spec=ComputeFramework)
        mock_cfw_class.return_value = mock_cfw_instance

        generated_uuid = uuid4()
        mock_cfw_instance.get_uuid.return_value = generated_uuid

        cfw_register.get_run_context.return_value = RunContext()

        cfw_uuid = executor.init_compute_framework(mock_cfw_class, ParallelizationMode.SYNC, set())

        # Should receive some UUID (generated internally)
        assert isinstance(cfw_uuid, UUID)

    def test_adds_cfw_to_register(self) -> None:
        """Should add the CFW to the cfw_register."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        mock_cfw_class = Mock()
        mock_cfw_instance = Mock(spec=ComputeFramework)
        mock_cfw_class.return_value = mock_cfw_instance
        mock_cfw_class.get_class_name.return_value = "TestCFW"

        test_uuid = uuid4()
        mock_cfw_instance.get_uuid.return_value = test_uuid

        cfw_register.get_run_context.return_value = RunContext()

        children = {uuid4()}
        executor.init_compute_framework(mock_cfw_class, ParallelizationMode.SYNC, children, test_uuid)

        cfw_register.add_cfw_to_compute_frameworks.assert_called_once_with(test_uuid, "TestCFW", children)

    def test_adds_cfw_to_collection(self) -> None:
        """Should add the CFW instance to the cfw_collection."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        mock_cfw_class = Mock()
        mock_cfw_instance = Mock(spec=ComputeFramework)
        mock_cfw_class.return_value = mock_cfw_instance

        test_uuid = uuid4()
        mock_cfw_instance.get_uuid.return_value = test_uuid

        cfw_register.get_run_context.return_value = RunContext()

        executor.init_compute_framework(mock_cfw_class, ParallelizationMode.SYNC, set(), test_uuid)

        assert test_uuid in executor.cfw_collection
        assert executor.cfw_collection[test_uuid] is mock_cfw_instance

    def test_each_initialized_framework_gets_its_own_carrier_copy(self) -> None:
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        mock_cfw_class_a = Mock()
        mock_cfw_instance_a = Mock(spec=ComputeFramework)
        mock_cfw_class_a.return_value = mock_cfw_instance_a
        mock_cfw_instance_a.get_uuid.return_value = uuid4()

        mock_cfw_class_b = Mock()
        mock_cfw_instance_b = Mock(spec=ComputeFramework)
        mock_cfw_class_b.return_value = mock_cfw_instance_b
        mock_cfw_instance_b.get_uuid.return_value = uuid4()

        register_run_context = RunContext(
            carrier={"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
        )
        cfw_register.get_run_context.return_value = register_run_context

        executor.init_compute_framework(mock_cfw_class_a, ParallelizationMode.SYNC, set())
        executor.init_compute_framework(mock_cfw_class_b, ParallelizationMode.SYNC, set())

        assert mock_cfw_instance_a.run_context == mock_cfw_instance_b.run_context
        assert mock_cfw_instance_a.run_context.carrier is not mock_cfw_instance_b.run_context.carrier
        assert mock_cfw_instance_a.run_context.carrier is not register_run_context.carrier
        assert mock_cfw_instance_b.run_context.carrier is not register_run_context.carrier


class TestInitComputeFrameworkWithDirectFunctionExtender:
    """A constructor-supplied function_extender must be used directly, skipping the cfw_register
    round trip that, under MULTIPROCESSING, would return a freshly-unpickled copy on every call."""

    def test_constructor_accepts_function_extender_keyword_argument(self) -> None:
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        extender = Mock(spec=Extender)

        executor = ComputeFrameworkExecutor(cfw_register, worker_manager, function_extender={extender})

        assert executor.function_extender == {extender}

    def test_init_compute_framework_uses_the_provided_function_extender_directly(self) -> None:
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        extender = Mock(spec=Extender)
        provided_function_extender: set[Extender] = {extender}
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager, function_extender=provided_function_extender)

        mock_cfw_class = Mock()
        mock_cfw_instance = Mock(spec=ComputeFramework)
        mock_cfw_class.return_value = mock_cfw_instance
        mock_cfw_class.get_class_name.return_value = "TestCFW"

        test_uuid = uuid4()
        mock_cfw_instance.get_uuid.return_value = test_uuid
        cfw_register.get_run_context.return_value = RunContext()

        cfw_uuid = executor.init_compute_framework(mock_cfw_class, ParallelizationMode.SYNC, set(), test_uuid)

        mock_cfw_class.assert_called_once_with(
            ParallelizationMode.SYNC, frozenset(), test_uuid, function_extender=provided_function_extender
        )
        assert cfw_uuid == test_uuid

    def test_init_compute_framework_uses_none_when_function_extender_omitted_for_non_multiprocessing_mode(
        self,
    ) -> None:
        """No constructor-supplied extender and a non-MULTIPROCESSING mode: there is no register fallback,
        the register never carries extenders."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        mock_cfw_class = Mock()
        mock_cfw_instance = Mock(spec=ComputeFramework)
        mock_cfw_class.return_value = mock_cfw_instance
        mock_cfw_class.get_class_name.return_value = "TestCFW"

        test_uuid = uuid4()
        mock_cfw_instance.get_uuid.return_value = test_uuid
        cfw_register.get_run_context.return_value = RunContext()

        executor.init_compute_framework(mock_cfw_class, ParallelizationMode.SYNC, set(), test_uuid)

        mock_cfw_class.assert_called_once_with(ParallelizationMode.SYNC, frozenset(), test_uuid, function_extender=None)

    def test_multiprocessing_mode_defers_worker_extender_payload_materialization_to_the_worker(self) -> None:
        """Builds with function_extender=None and the raw payload attached as _pending_extender_payload;
        unpickling is deferred to ComputeFramework.__setstate__ in the worker, never done here."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        caller_extender = _StateHoldingExtender("caller")
        provided_function_extender: set[Extender] = {caller_extender}
        payload_extender = _StateHoldingExtender("payload")
        payload = pickle.dumps(({payload_extender}, build_hook_extenders({payload_extender})))
        executor = ComputeFrameworkExecutor(
            cfw_register,
            worker_manager,
            function_extender=provided_function_extender,
            worker_extender_payload=payload,
        )

        mock_cfw_class = Mock()
        mock_cfw_instance = Mock(spec=ComputeFramework)
        mock_cfw_class.return_value = mock_cfw_instance
        mock_cfw_class.get_class_name.return_value = "TestCFW"
        mock_cfw_class.supported_parallelization_modes.return_value = {
            ParallelizationMode.SYNC,
            ParallelizationMode.THREADING,
            ParallelizationMode.MULTIPROCESSING,
        }

        test_uuid = uuid4()
        mock_cfw_instance.get_uuid.return_value = test_uuid
        cfw_register.get_run_context.return_value = RunContext()

        executor.init_compute_framework(mock_cfw_class, ParallelizationMode.MULTIPROCESSING, set(), test_uuid)

        assert mock_cfw_class.call_args.kwargs["function_extender"] is None
        assert mock_cfw_instance._pending_extender_payload == payload

    def test_multiprocessing_call_on_a_sync_only_framework_keeps_the_callers_extender_identity(self) -> None:
        """A framework whose supported_parallelization_modes() excludes MULTIPROCESSING can never
        actually run in a worker, so it must be built with the caller's own extender objects even
        when init_compute_framework is called with parallelization_mode=MULTIPROCESSING, never a
        stripped copy from worker_extender_payload."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        caller_extender = _StateHoldingExtender("caller")
        provided_function_extender: set[Extender] = {caller_extender}
        payload_extender = _StateHoldingExtender("payload")
        payload = pickle.dumps(({payload_extender}, build_hook_extenders({payload_extender})))
        executor = ComputeFrameworkExecutor(
            cfw_register,
            worker_manager,
            function_extender=provided_function_extender,
            worker_extender_payload=payload,
        )

        mock_cfw_class = Mock()
        mock_cfw_instance = Mock(spec=ComputeFramework)
        mock_cfw_class.return_value = mock_cfw_instance
        mock_cfw_class.get_class_name.return_value = "TestSyncOnlyCFW"
        mock_cfw_class.supported_parallelization_modes.return_value = {ParallelizationMode.SYNC}

        test_uuid = uuid4()
        mock_cfw_instance.get_uuid.return_value = test_uuid
        cfw_register.get_run_context.return_value = RunContext()

        executor.init_compute_framework(mock_cfw_class, ParallelizationMode.MULTIPROCESSING, set(), test_uuid)

        built_function_extender = mock_cfw_class.call_args.kwargs["function_extender"]
        assert built_function_extender is provided_function_extender


class TestComputeFrameworkExecutorCfwLock:
    def test_has_cfw_lock_attribute_after_construction(self) -> None:
        """ComputeFrameworkExecutor should have a _cfw_lock attribute after __init__."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()

        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        assert hasattr(executor, "_cfw_lock"), (
            "ComputeFrameworkExecutor.__init__ must create a self._cfw_lock attribute"
        )

    def test_cfw_lock_is_threading_lock_instance(self) -> None:
        """_cfw_lock should be an instance of threading.Lock (not multiprocessing.Lock)."""
        import threading

        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()

        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        assert isinstance(executor._cfw_lock, type(threading.Lock())), "_cfw_lock must be a threading.Lock instance"


class TestAddComputeFramework:
    """Tests for add_compute_framework method."""

    def test_returns_existing_cfw_uuid_if_found(self) -> None:
        """Should return existing CFW UUID without creating new one."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        existing_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = existing_uuid

        step = Mock()
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        result = executor.add_compute_framework(step, ParallelizationMode.SYNC, uuid4(), set())

        assert result == existing_uuid
        # Should not call init_compute_framework
        cfw_register.add_cfw_to_compute_frameworks.assert_not_called()

    def test_creates_new_cfw_if_not_found(self) -> None:
        """Should create new CFW if none exists for the feature UUID."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        # First call returns None (not found), second call would return the new UUID
        cfw_register.get_cfw_uuid.return_value = None

        step = Mock()
        mock_cfw_class = Mock()
        mock_cfw_instance = Mock(spec=ComputeFramework)
        mock_cfw_class.return_value = mock_cfw_instance
        mock_cfw_class.get_class_name.return_value = "TestCFW"

        step.compute_framework = mock_cfw_class

        new_uuid = uuid4()
        mock_cfw_instance.get_uuid.return_value = new_uuid
        cfw_register.get_run_context.return_value = RunContext()

        feature_uuid = uuid4()
        children = {uuid4()}

        result = executor.add_compute_framework(step, ParallelizationMode.THREADING, feature_uuid, children)

        assert result == new_uuid
        assert new_uuid in executor.cfw_collection

    def test_uses_instance_cfw_lock_not_per_call_lock(self) -> None:
        """Should use self._cfw_lock for thread safety."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        mock_lock = MagicMock()
        executor._cfw_lock = mock_lock

        cfw_register.get_cfw_uuid.return_value = uuid4()

        step = Mock()
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        executor.add_compute_framework(step, ParallelizationMode.SYNC, uuid4(), set())

        # Verify the instance lock was used as context manager
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()


class TestGetCfw:
    """Tests for get_cfw method."""

    def test_retrieves_cfw_by_type_and_feature_uuid(self) -> None:
        """Should retrieve CFW from collection using register lookup."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        # Set up CFW in collection
        cfw_uuid = uuid4()
        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        # Mock register lookup
        mock_cfw_class = Mock()
        mock_cfw_class.get_class_name.return_value = "PandasDataFrame"
        feature_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = cfw_uuid

        result = executor.get_cfw(mock_cfw_class, feature_uuid)

        assert result is mock_cfw
        cfw_register.get_cfw_uuid.assert_called_once_with("PandasDataFrame", feature_uuid)

    def test_raises_value_error_if_cfw_uuid_is_none(self) -> None:
        """Should raise ValueError if CFW UUID is not found in register."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        cfw_register.get_cfw_uuid.return_value = None

        mock_cfw_class = Mock()
        mock_cfw_class.get_class_name.return_value = "PandasDataFrame"

        with pytest.raises(ValueError, match="cfw_uuid should not be none"):
            executor.get_cfw(mock_cfw_class, uuid4())


class TestGetExecutionFunction:
    """Tests for _get_execution_function method."""

    def test_returns_multi_execute_step_for_multiprocessing(self) -> None:
        """Should return multi_execute_step when multiprocessing mode is in intersection."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        mode_by_cfw = {ParallelizationMode.MULTIPROCESSING, ParallelizationMode.SYNC}
        mode_by_step = {ParallelizationMode.MULTIPROCESSING}

        result = executor._get_execution_function(mode_by_cfw, mode_by_step)

        assert result == executor.multi_execute_step

    def test_returns_thread_execute_step_for_threading(self) -> None:
        """Should return thread_execute_step when threading mode is in intersection."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        mode_by_cfw = {ParallelizationMode.THREADING, ParallelizationMode.SYNC}
        mode_by_step = {ParallelizationMode.THREADING}

        result = executor._get_execution_function(mode_by_cfw, mode_by_step)

        assert result == executor.thread_execute_step

    def test_returns_sync_execute_step_as_default(self) -> None:
        """Should return sync_execute_step when no other mode matches."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        mode_by_cfw = {ParallelizationMode.SYNC}
        mode_by_step = {ParallelizationMode.SYNC}

        result = executor._get_execution_function(mode_by_cfw, mode_by_step)

        assert result == executor.sync_execute_step

    def test_prioritizes_multiprocessing_over_threading(self) -> None:
        """Should prioritize multiprocessing when both are available."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        mode_by_cfw = {
            ParallelizationMode.MULTIPROCESSING,
            ParallelizationMode.THREADING,
            ParallelizationMode.SYNC,
        }
        mode_by_step = {ParallelizationMode.MULTIPROCESSING, ParallelizationMode.THREADING}

        result = executor._get_execution_function(mode_by_cfw, mode_by_step)

        assert result == executor.multi_execute_step

    def test_returns_sync_execute_step_for_sync_only_tfs_step_even_when_register_supports_multiprocessing(
        self,
    ) -> None:
        """Should return sync_execute_step for a SYNC-only TFS step even when multiprocessing is registered."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = TransformFrameworkStep(
            from_framework=DuckDBFramework,
            to_framework=DuckDBFramework,
            required_uuids=set(),
            from_feature_group=MagicMock(),
            to_feature_group=MagicMock(),
        )

        mode_by_cfw = {ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING}
        mode_by_step = step.get_parallelization_mode()

        result = executor._get_execution_function(mode_by_cfw, mode_by_step)

        assert result == executor.sync_execute_step

    def test_returns_sync_execute_step_for_sync_only_join_step_even_when_register_supports_multiprocessing(
        self,
    ) -> None:
        """Should return sync_execute_step for a SYNC-only JoinStep even when multiprocessing is registered."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = JoinStep(
            link=MagicMock(),
            destination_framework=DuckDBFramework,
            source_framework=DuckDBFramework,
            required_uuids=set(),
            destination_framework_uuids=set(),
            source_framework_uuids=set(),
        )

        mode_by_cfw = {ParallelizationMode.SYNC, ParallelizationMode.MULTIPROCESSING}
        mode_by_step = step.get_parallelization_mode()

        result = executor._get_execution_function(mode_by_cfw, mode_by_step)

        assert result == executor.sync_execute_step


def _own_frame_step(
    feature_uuid: UUID, children: frozenset[UUID], step_uuid: UUID, compute_framework: Any = PythonDictFramework
) -> Mock:
    step = Mock(spec=FeatureGroupStep)
    step.own_frame = True
    step.uuid = step_uuid
    step.tfs_ids = set()
    step.join_tokens = frozenset()
    step.reads_join_result = False
    step.compute_framework = compute_framework
    step.features = Mock()
    step.features.any_uuid = feature_uuid
    step.children_if_root = children
    return step


class TestPrepareExecuteStep:
    """Tests for prepare_execute_step method."""

    @pytest.mark.parametrize("join_tokens", [frozenset(), frozenset({uuid4()})])
    def test_handles_feature_group_step_with_existing_cfw(self, join_tokens: frozenset[UUID]) -> None:
        cfw_register = Mock(spec=CfwManager)
        executor = ComputeFrameworkExecutor(cfw_register, _worker_manager_mock())
        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.tfs_ids = [uuid4()]
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.join_tokens = join_tokens
        existing_uuid = uuid4()
        cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = existing_uuid

        result = executor.prepare_execute_step(step, ParallelizationMode.SYNC)

        assert result == existing_uuid
        if join_tokens:
            cfw_register.add_join_tokens.assert_called_once_with(existing_uuid, step.join_tokens)
        else:
            cfw_register.add_join_tokens.assert_not_called()

    def test_join_resolves_to_a_frame_reused_by_the_source_step_with_a_real_register(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        executor = ComputeFrameworkExecutor(cfw_register, _worker_manager_mock())
        cls_name = "TestCFW"
        frame_uuid, tfs_id, token = uuid4(), uuid4(), uuid4()
        cfw_register.add_cfw_to_compute_frameworks(frame_uuid, cls_name, {tfs_id})
        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.tfs_ids = {tfs_id}
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = cls_name
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.reads_join_result = False
        step.join_tokens = frozenset({token})
        join_step = Mock(spec=JoinStep)
        join_step.uuid = token
        join_step.destination_framework = Mock()
        join_step.destination_framework.get_class_name.return_value = cls_name

        assert executor.prepare_execute_step(step, ParallelizationMode.SYNC) == frame_uuid
        assert resolve_join_source_cfw(join_step, cfw_register) == frame_uuid

    def test_creates_new_cfw_for_feature_group_step_without_existing(self) -> None:
        """Should create new CFW for FeatureGroupStep when none exists."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.tfs_ids = [uuid4()]
        cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = None
        # add_compute_framework's own dedup check, separate from resolve_cfw_uuid_by_tfs_ids above.
        cfw_register.get_cfw_uuid.return_value = None

        feature_uuid = uuid4()
        step.features = Mock()
        step.features.any_uuid = feature_uuid
        step.children_if_root = [uuid4()]

        mock_cfw_class = Mock()
        mock_cfw_instance = Mock(spec=ComputeFramework)
        mock_cfw_class.return_value = mock_cfw_instance
        mock_cfw_class.get_class_name.return_value = "TestCFW"
        step.compute_framework = mock_cfw_class

        new_uuid = uuid4()
        mock_cfw_instance.get_uuid.return_value = new_uuid
        cfw_register.get_run_context.return_value = RunContext()

        result = executor.prepare_execute_step(step, ParallelizationMode.SYNC)

        assert result == new_uuid

    def test_raises_error_for_feature_group_step_without_feature_uuid(self) -> None:
        """Should raise ValueError if FeatureGroupStep has no feature UUID."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = None
        step.compute_framework = Mock()
        cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = None

        with pytest.raises(ValueError, match="from_feature_uuid should not be none"):
            executor.prepare_execute_step(step, ParallelizationMode.SYNC)

    def test_own_frame_step_gets_a_new_cfw_keyed_by_its_step_uuid(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        executor = ComputeFrameworkExecutor(cfw_register, _worker_manager_mock())
        feature_uuid, child_uuid, other_uuid, step_uuid = uuid4(), uuid4(), uuid4(), uuid4()
        source_uuid = uuid4()
        executor.init_compute_framework(
            PythonDictFramework, ParallelizationMode.SYNC, {feature_uuid, child_uuid, other_uuid}, source_uuid
        )
        step = _own_frame_step(feature_uuid, frozenset({feature_uuid, child_uuid}), step_uuid)

        assert executor.prepare_execute_step(step, ParallelizationMode.SYNC) == step_uuid

        assert cfw_register.compute_frameworks[step_uuid] == ("PythonDictFramework", {feature_uuid, child_uuid})
        assert executor.own_frame_sources[step_uuid] == source_uuid
        assert executor.prepare_tfs_and_joinstep(step) is executor.cfw_collection[source_uuid]
        assert cfw_register.compute_frameworks[source_uuid] == ("PythonDictFramework", {other_uuid})
        assert cfw_register.get_cfw_uuid("PythonDictFramework", child_uuid) == step_uuid

    def test_own_frame_step_claims_all_of_a_fully_covered_source(self) -> None:
        cfw_register = CfwManager({ParallelizationMode.SYNC})
        executor = ComputeFrameworkExecutor(cfw_register, _worker_manager_mock())
        feature_uuid, child_uuid, step_uuid, source_uuid = uuid4(), uuid4(), uuid4(), uuid4()
        executor.init_compute_framework(
            PythonDictFramework, ParallelizationMode.SYNC, {feature_uuid, child_uuid}, source_uuid
        )
        step = _own_frame_step(feature_uuid, frozenset({feature_uuid, child_uuid}), step_uuid)

        executor.prepare_execute_step(step, ParallelizationMode.SYNC)

        assert cfw_register.compute_frameworks[source_uuid] == ("PythonDictFramework", set())

    def test_own_frame_step_without_a_resolvable_source_raises(self) -> None:
        cfw_register = Mock(spec=CfwManager)
        cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = None
        cfw_register.get_cfw_uuid.return_value = None
        cfw_register.get_run_context.return_value = RunContext()
        executor = ComputeFrameworkExecutor(cfw_register, _worker_manager_mock())
        compute_framework = Mock()
        compute_framework.get_class_name.return_value = "TestCFW"
        step = _own_frame_step(uuid4(), frozenset({uuid4()}), uuid4(), compute_framework)

        with pytest.raises(ValueError, match="own frame"):
            executor.prepare_execute_step(step, ParallelizationMode.SYNC)

    def test_handles_transform_framework_step(self) -> None:
        """Should handle TransformFrameworkStep by creating new CFW with children."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=TransformFrameworkStep)
        from_feature_uuid = uuid4()
        step.required_uuids = [from_feature_uuid]
        step.from_framework = Mock()
        step.from_framework.get_class_name.return_value = "FromCFW"

        from_cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = from_cfw_uuid

        # Mock the from_cfw in collection
        from_cfw = Mock(spec=ComputeFramework)
        child_uuid = uuid4()
        from_cfw.children_if_root = {child_uuid}
        executor.cfw_collection[from_cfw_uuid] = from_cfw

        step.link_id = None
        step.private_copy = False
        step.uuid = uuid4()

        # Mock to_framework
        mock_to_cfw_class = Mock()
        mock_to_cfw_instance = Mock(spec=ComputeFramework)
        mock_to_cfw_class.return_value = mock_to_cfw_instance
        step.to_framework = mock_to_cfw_class

        new_uuid = uuid4()
        mock_to_cfw_instance.get_uuid.return_value = new_uuid
        cfw_register.get_run_context.return_value = RunContext()

        with patch("mloda.core.runtime.compute_framework_executor.multiprocessing.Lock"):
            result = executor.prepare_execute_step(step, ParallelizationMode.THREADING)

        assert result == new_uuid

    def test_handles_join_step(self) -> None:
        """Should handle JoinStep by retrieving destination framework CFW."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=JoinStep)
        destination_uuid = uuid4()
        step.destination_framework_uuids = {destination_uuid}
        step.destination_framework = Mock()
        step.destination_framework.get_class_name.return_value = "DestinationCFW"

        cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = cfw_uuid

        result = executor.prepare_execute_step(step, ParallelizationMode.SYNC)

        assert result == cfw_uuid

    def test_raises_error_when_cfw_uuid_is_none(self) -> None:
        """Should raise ValueError if CFW UUID is None after processing."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        # Create a step that doesn't match any type (edge case)
        step = Mock()
        step.__class__.__name__ = "UnknownStep"

        with pytest.raises(ValueError, match="Internal error: cfw_uuid is None"):
            executor.prepare_execute_step(step, ParallelizationMode.SYNC)


class TestPrepareTfsRightCfw:
    """Tests for prepare_tfs_right_cfw method."""

    def test_uses_source_framework_uuid_when_available(self) -> None:
        """Should use source_framework_uuid when set."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=TransformFrameworkStep)
        source_uuid = uuid4()
        step.source_framework_uuid = source_uuid
        step.from_framework = Mock()
        step.from_framework.get_class_name.return_value = "FromCFW"

        cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = cfw_uuid

        result = executor.prepare_tfs_right_cfw(step)

        assert result == cfw_uuid
        cfw_register.get_cfw_uuid.assert_called_once_with("FromCFW", source_uuid)

    def test_uses_first_required_uuid_when_source_framework_uuid_is_none(self) -> None:
        """Should use first required UUID when source_framework_uuid is None."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=TransformFrameworkStep)
        step.source_framework_uuid = None
        required_uuid = uuid4()
        step.required_uuids = [required_uuid, uuid4()]
        step.from_framework = Mock()
        step.from_framework.get_class_name.return_value = "FromCFW"

        cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = cfw_uuid

        result = executor.prepare_tfs_right_cfw(step)

        assert result == cfw_uuid
        cfw_register.get_cfw_uuid.assert_called_once_with("FromCFW", required_uuid)

    def test_raises_error_if_cfw_uuid_is_none(self) -> None:
        """Should raise ValueError if CFW UUID is None."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=TransformFrameworkStep)
        step.source_framework_uuid = None
        step.required_uuids = [uuid4()]
        step.from_framework = Mock()
        step.from_framework.get_class_name.return_value = "FromCFW"

        cfw_register.get_cfw_uuid.return_value = None

        with pytest.raises(ValueError, match="cfw_uuid should not be none in prepare_tfs"):
            executor.prepare_tfs_right_cfw(step)


class TestPrepareTfsAndJoinStep:
    """Tests for prepare_tfs_and_joinstep method."""

    def test_returns_from_cfw_for_transform_framework_step(self) -> None:
        """Should return from_cfw instance for TransformFrameworkStep."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        worker_manager.get_process_queues.return_value = None
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=TransformFrameworkStep)
        step.source_framework_uuid = uuid4()
        step.from_framework = Mock()
        step.from_framework.get_class_name.return_value = "FromCFW"

        from_cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = from_cfw_uuid

        from_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[from_cfw_uuid] = from_cfw

        result = executor.prepare_tfs_and_joinstep(step)

        assert result is from_cfw

    def test_returns_source_uuid_for_transform_framework_step_when_a_worker_owns_the_source(self) -> None:
        """Characterization pin: a worker-owned transform source is returned by uuid, not resolved to a cfw instance."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        worker_manager.get_process_queues.return_value = (Mock(), Mock(), Mock())
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=TransformFrameworkStep)
        step.source_framework_uuid = uuid4()
        step.from_framework = Mock()
        step.from_framework.get_class_name.return_value = "FromCFW"

        from_cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = from_cfw_uuid

        result = executor.prepare_tfs_and_joinstep(step)

        assert result is from_cfw_uuid
        worker_manager.get_process_queues.assert_called_once_with(from_cfw_uuid)

    def test_returns_from_cfw_for_join_step_keyed_by_its_own_uuid(self) -> None:
        """Should return from_cfw for a JoinStep, looked up by the step uuid."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=JoinStep)
        step.uuid = uuid4()
        step.link = Mock()
        step.link.uuid = uuid4()
        step.destination_framework = Mock()
        step.destination_framework.get_class_name.return_value = "DestinationCFW"
        step.source_framework_uuids = {uuid4()}

        from_cfw_uuid = uuid4()
        cfw_register.get_join_source_cfw_uuid.return_value = from_cfw_uuid

        from_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[from_cfw_uuid] = from_cfw

        result = executor.prepare_tfs_and_joinstep(step)

        assert result is from_cfw
        cfw_register.get_join_source_cfw_uuid.assert_called_once_with("DestinationCFW", step.uuid)

    def test_raises_error_for_join_step_if_no_frame_lists_the_join_token(self) -> None:
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=JoinStep)
        step.uuid = uuid4()
        step.link = Mock()
        step.link.uuid = uuid4()
        step.destination_framework = Mock()
        step.destination_framework.get_class_name.return_value = "DestinationCFW"
        step.source_framework_uuids = {uuid4()}
        cfw_register.get_join_source_cfw_uuid.return_value = None

        with pytest.raises(ValueError, match="no single frame lists the join token"):
            executor.prepare_tfs_and_joinstep(step)

    def test_returns_none_for_other_step_types(self) -> None:
        """Should return None for step types that are neither TFS nor JoinStep."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False

        result = executor.prepare_tfs_and_joinstep(step)

        assert result is None

    def test_returns_from_cfw_instance_for_join_step_even_when_uuid_owned_by_a_worker(self) -> None:
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        worker_manager.get_process_queues.return_value = (Mock(), Mock(), Mock())
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=JoinStep)
        step.uuid = uuid4()
        step.link = Mock()
        step.link.uuid = uuid4()
        step.destination_framework = Mock()
        step.destination_framework.get_class_name.return_value = "DestinationCFW"
        step.source_framework_uuids = {uuid4()}

        from_cfw_uuid = uuid4()
        cfw_register.get_join_source_cfw_uuid.return_value = from_cfw_uuid

        from_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[from_cfw_uuid] = from_cfw

        result = executor.prepare_tfs_and_joinstep(step)

        assert result is from_cfw
        worker_manager.get_process_queues.assert_not_called()


class TestSyncExecuteStep:
    """Tests for sync_execute_step method."""

    def test_prepares_step_with_sync_mode(self) -> None:
        """Should call prepare_execute_step with SYNC mode."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        cfw_uuid = uuid4()
        cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        executor.sync_execute_step(step)

        # Verify step execution was called
        step.execute.assert_called_once()

    def test_marks_step_as_done_on_success(self) -> None:
        """Should record the step uuid in the run state on successful execution."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"
        step.uuid = uuid4()

        cfw_uuid = uuid4()
        cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        executor.sync_execute_step(step)

        assert step.uuid in worker_manager.result_uuids_collection

    def test_handles_exception_and_sets_error(self) -> None:
        """Should catch exceptions and set error in cfw_register."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        cfw_uuid = uuid4()
        cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        # Make step.execute raise an exception
        step.execute.side_effect = RuntimeError("Test error")

        executor.sync_execute_step(step)

        # Verify error was set
        cfw_register.set_error.assert_called_once()
        error_msg, exc_info = cfw_register.set_error.call_args[0]
        assert "Test error" in error_msg

    def test_handles_exception_scrubs_secret_and_preserves_exception_object(self) -> None:
        """A secret in the raised exception's text must not reach set_error's msg/exc_info."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        cfw_uuid = uuid4()
        cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        boom = RuntimeError(_LEAK_MESSAGE)
        step.execute.side_effect = boom

        executor.sync_execute_step(step)

        cfw_register.set_error.assert_called_once()
        call_args = cfw_register.set_error.call_args
        error_msg, exc_info = call_args.args
        assert _LEAK_MARKER not in error_msg
        assert _LEAK_MARKER not in exc_info
        assert call_args.kwargs["exception"] is boom
        assert _LEAK_MARKER in str(boom)

    def test_registered_secret_in_container_attributes_is_masked_in_place(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import registered_secrets_scope

        cfw_register = Mock(spec=CfwManager)
        executor = ComputeFrameworkExecutor(cfw_register, _worker_manager_mock())
        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"
        cfw_uuid = uuid4()
        cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = cfw_uuid
        executor.cfw_collection[cfw_uuid] = Mock(spec=ComputeFramework)
        boom = _ContainerLeakError()
        step.execute.side_effect = boom

        with registered_secrets_scope(frozenset({_REGISTERED})):
            executor.sync_execute_step(step)

        cfw_register.set_error.assert_called_once()
        assert cfw_register.set_error.call_args.kwargs["exception"] is boom
        assert _REGISTERED not in str(boom)
        assert _REGISTERED not in repr(boom.as_dict)
        assert _REGISTERED not in repr(boom.as_list)


class TestThreadExecuteStep:
    """Tests for thread_execute_step method."""

    def test_prepares_step_with_threading_mode(self) -> None:
        """Should call prepare_execute_step with THREADING mode."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        cfw_uuid = uuid4()
        cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        with patch("mloda.core.runtime.compute_framework_executor.threading.Thread"):
            executor.thread_execute_step(step)

        # Verify CFW was retrieved
        assert cfw_uuid in executor.cfw_collection

    @patch("mloda.core.runtime.compute_framework_executor.thread_worker")
    @patch("mloda.core.runtime.compute_framework_executor.threading.Thread")
    def test_creates_thread_with_correct_target_and_args(self, mock_thread_class: Any, mock_worker: Any) -> None:
        """Should create thread with thread_worker target and correct arguments."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        cfw_uuid = uuid4()
        cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        executor.thread_execute_step(step)

        # Verify thread was created with correct parameters
        mock_thread_class.assert_called_once_with(
            target=mock_worker,
            args=(
                step,
                cfw_register,
                mock_cfw,
                None,
                frozenset(),
                frozenset(),
                None,
                worker_manager.result_uuids_collection.add,
                False,
            ),
            daemon=True,
        )

    @patch("mloda.core.runtime.compute_framework_executor.threading.Thread")
    def test_adds_thread_to_worker_manager(self, mock_thread_class: Any) -> None:
        """Should add created thread to worker_manager."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        cfw_uuid = uuid4()
        cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        executor.thread_execute_step(step)

        worker_manager.add_thread_task.assert_called_once_with(mock_thread)

    @pytest.mark.parametrize("error_type", [RuntimeError, SystemExit])
    def test_thread_worker_reports_error_without_reraising(self, error_type: type[BaseException]) -> None:
        cfw_register = Mock(spec=CfwManager)
        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        boom = error_type("Test error")
        step.execute.side_effect = boom

        thread_worker(step, cfw_register, Mock(spec=ComputeFramework), None, mark_done=_ignore_done)

        cfw_register.set_error.assert_called_once()
        assert cfw_register.set_error.call_args.kwargs["exception"] is boom

    def test_thread_worker_scrubs_secret_and_preserves_exception_object(self) -> None:
        """A secret in the raised exception's text must not reach set_error's msg/exc_info."""
        cfw_register = Mock(spec=CfwManager)
        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        boom = RuntimeError(_LEAK_MESSAGE)
        step.execute.side_effect = boom

        thread_worker(step, cfw_register, Mock(spec=ComputeFramework), None, mark_done=_ignore_done)

        cfw_register.set_error.assert_called_once()
        call_args = cfw_register.set_error.call_args
        error_msg, exc_info = call_args.args
        assert _LEAK_MARKER not in error_msg
        assert _LEAK_MARKER not in exc_info
        assert call_args.kwargs["exception"] is boom

    def test_thread_worker_masks_registered_secret_in_container_attributes(self) -> None:
        cfw_register = Mock(spec=CfwManager)
        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        boom = _ContainerLeakError()
        step.execute.side_effect = boom

        thread_worker(
            step,
            cfw_register,
            Mock(spec=ComputeFramework),
            None,
            registered_secrets=frozenset({_REGISTERED}),
            mark_done=_ignore_done,
        )

        cfw_register.set_error.assert_called_once()
        assert cfw_register.set_error.call_args.kwargs["exception"] is boom
        assert _REGISTERED not in str(boom)
        assert _REGISTERED not in repr(boom.as_dict)
        assert _REGISTERED not in repr(boom.as_list)

    def test_thread_worker_scopes_the_identity_secrets(self) -> None:
        from mloda.core.abstract_plugins.components.credential_scrub import current_identity_secrets

        seen: list[frozenset[str]] = []
        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.uuid = uuid4()
        step.execute.side_effect = lambda *a, **k: seen.append(current_identity_secrets())

        thread_worker(
            step,
            Mock(spec=CfwManager),
            Mock(spec=ComputeFramework),
            None,
            registered_secrets=frozenset({"a-registered-secret-value"}),
            identity_secrets=frozenset({"an-identity-value"}),
            mark_done=_ignore_done,
        )

        assert seen == [frozenset({"an-identity-value"})]
        assert current_identity_secrets() == frozenset()

    def test_thread_worker_scrubs_secret_in_chained_cause(self) -> None:
        """A secret in the exception's __cause__ must not reach set_error's msg/exc_info either."""
        cfw_register = Mock(spec=CfwManager)
        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        boom = RuntimeError("load failed")
        boom.__cause__ = RuntimeError(_LEAK_MESSAGE)
        step.execute.side_effect = boom

        thread_worker(step, cfw_register, Mock(spec=ComputeFramework), None, mark_done=_ignore_done)

        cfw_register.set_error.assert_called_once()
        call_args = cfw_register.set_error.call_args
        error_msg, exc_info = call_args.args
        assert _LEAK_MARKER not in error_msg
        assert _LEAK_MARKER not in exc_info
        assert call_args.kwargs["exception"] is boom

    def test_thread_worker_marks_the_step_done_on_success(self) -> None:
        step = Mock(spec=FeatureGroupStep)
        step.uuid = uuid4()
        done: set[UUID] = set()

        thread_worker(step, Mock(spec=CfwManager), Mock(spec=ComputeFramework), None, mark_done=done.add)

        assert done == {step.uuid}

    def test_thread_worker_does_not_mark_a_failed_step_done(self) -> None:
        step = Mock(spec=FeatureGroupStep)
        step.uuid = uuid4()
        step.execute.side_effect = RuntimeError("boom")
        done: set[UUID] = set()

        thread_worker(step, Mock(spec=CfwManager), Mock(spec=ComputeFramework), None, mark_done=done.add)

        assert done == set()


class TestMultiExecuteStep:
    """Tests for multi_execute_step method."""

    def test_prepares_step_with_multiprocessing_mode(self) -> None:
        """Should call prepare_execute_step with MULTIPROCESSING mode."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"
        # multi_execute_step records the dispatch, so the step must answer get_uuids()
        # with real uuids like a FeatureGroupStep does.
        step.get_uuids.return_value = {uuid4()}

        cfw_uuid = uuid4()
        cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        worker_manager.get_process_queues.return_value = (Mock(), Mock(), Mock())

        executor.multi_execute_step(step)

        # Verify process queues were checked
        worker_manager.get_process_queues.assert_called_once_with(cfw_uuid)

    @patch("mloda.core.runtime.compute_framework_executor.worker")
    def test_creates_worker_process_if_not_exists(self, mock_worker: Any) -> None:
        """Should create new worker process if none exists for CFW UUID."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"
        # multi_execute_step records the dispatch, so the step must answer get_uuids()
        # with real uuids like a FeatureGroupStep does.
        step.get_uuids.return_value = {uuid4()}

        cfw_uuid = uuid4()
        cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        worker_manager.get_process_queues.return_value = None

        mock_process = Mock()
        mock_cmd_queue = Mock()
        mock_result_queue = Mock()
        worker_manager.create_worker_process.return_value = (mock_process, mock_cmd_queue, mock_result_queue)

        executor.multi_execute_step(step)

        worker_manager.create_worker_process.assert_called_once_with(
            cfw_uuid, mock_worker, (cfw_register, mock_cfw, None)
        )

    def test_uses_existing_worker_process_if_exists(self) -> None:
        """Should use existing worker process if one exists for CFW UUID."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"
        # multi_execute_step records the dispatch, so the step must answer get_uuids()
        # with real uuids like a FeatureGroupStep does.
        step.get_uuids.return_value = {uuid4()}

        cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        existing_process = (Mock(), Mock(), Mock())
        worker_manager.get_process_queues.return_value = existing_process

        executor.multi_execute_step(step)

        # Should not create new process
        worker_manager.create_worker_process.assert_not_called()

    def test_sends_command_to_worker(self) -> None:
        """Should send step command to worker process."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.tfs_ids = []
        step.features = Mock()
        step.features.input_data_match = None
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"
        # multi_execute_step records the dispatch, so the step must answer get_uuids()
        # with real uuids like a FeatureGroupStep does.
        step.get_uuids.return_value = {uuid4()}

        cfw_uuid = uuid4()
        cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        worker_manager.get_process_queues.return_value = (Mock(), Mock(), Mock())

        executor.multi_execute_step(step)

        worker_manager.send_command.assert_called_once_with(cfw_uuid, (step, None, frozenset(), frozenset(), False))

    def test_records_assignment_with_the_steps_own_result_uuid_not_get_uuids(self) -> None:
        """record_assignment must use step.get_result_uuid(), not the feature uuids from get_uuids()."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"
        step.uuid = uuid4()
        step.get_result_uuid.return_value = step.uuid
        # Deliberately disjoint from step.uuid, like the real feature uuids on a FeatureGroupStep.
        step.get_uuids.return_value = {uuid4(), uuid4()}

        cfw_uuid = uuid4()
        cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        worker_manager.get_process_queues.return_value = (Mock(), Mock(), Mock())

        executor.multi_execute_step(step)

        worker_manager.record_assignment.assert_called_once_with(cfw_uuid, {step.uuid})

    def test_prepares_from_cfw_for_transform_framework_step(self) -> None:
        """Should prepare from_cfw for TransformFrameworkStep."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=TransformFrameworkStep)
        step.source_framework_uuid = uuid4()
        step.from_framework = Mock()
        step.from_framework.get_class_name.return_value = "FromCFW"
        step.required_uuids = [uuid4()]
        # multi_execute_step records the dispatch, so the step must answer get_uuids()
        # with real uuids like a TransformFrameworkStep does.
        step.get_uuids.return_value = {uuid4()}

        from_cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.side_effect = [from_cfw_uuid, from_cfw_uuid]

        from_cfw = Mock(spec=ComputeFramework)
        from_cfw.children_if_root = set()
        executor.cfw_collection[from_cfw_uuid] = from_cfw

        step.link_id = None
        step.private_copy = False
        step.uuid = uuid4()

        mock_to_cfw_class = Mock()
        mock_to_cfw_instance = Mock(spec=ComputeFramework)
        mock_to_cfw_class.return_value = mock_to_cfw_instance
        mock_to_cfw_class.supported_parallelization_modes.return_value = {
            ParallelizationMode.SYNC,
            ParallelizationMode.THREADING,
            ParallelizationMode.MULTIPROCESSING,
        }
        step.to_framework = mock_to_cfw_class

        new_uuid = uuid4()
        mock_to_cfw_instance.get_uuid.return_value = new_uuid
        cfw_register.get_run_context.return_value = RunContext()

        worker_manager.get_process_queues.return_value = (Mock(), Mock(), Mock())

        with patch("mloda.core.runtime.compute_framework_executor.multiprocessing.Lock"):
            executor.multi_execute_step(step)

        # Verify from_cfw_uuid was prepared
        assert cfw_register.get_cfw_uuid.call_count >= 1

    def test_does_not_bind_connection(self) -> None:
        """multi_execute_step must not call _bind_connection: the cfw is pickled to a spawned worker process."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.own_frame = False
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"
        step.get_uuids.return_value = {uuid4()}

        cfw_uuid = uuid4()
        cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        worker_manager.get_process_queues.return_value = (Mock(), Mock(), Mock())

        executor._bind_connection = Mock()  # type: ignore[method-assign]

        executor.multi_execute_step(step)

        executor._bind_connection.assert_not_called()

    @patch("mloda.core.runtime.compute_framework_executor.worker")
    def test_passes_cfw_register_cfw_and_from_cfw_to_the_worker(self, mock_worker: Any) -> None:
        """create_worker_process receives (cfw_register, cfw, from_cfw), no secrets and no needs_tfs_connection."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = _worker_manager_mock()
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=TransformFrameworkStep)
        step.source_framework_uuid = uuid4()
        step.from_framework = Mock()
        step.from_framework.get_class_name.return_value = "FromCFW"
        step.required_uuids = [uuid4()]
        # multi_execute_step records the dispatch, so the step must answer get_uuids()
        # with real uuids like a TransformFrameworkStep does.
        step.get_uuids.return_value = {uuid4()}

        from_cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.side_effect = [from_cfw_uuid, from_cfw_uuid]

        from_cfw = Mock(spec=ComputeFramework)
        from_cfw.children_if_root = set()
        executor.cfw_collection[from_cfw_uuid] = from_cfw

        step.link_id = None
        step.private_copy = False
        step.uuid = uuid4()

        mock_to_cfw_class = Mock()
        mock_to_cfw_instance = Mock(spec=ComputeFramework)
        mock_to_cfw_class.return_value = mock_to_cfw_instance
        mock_to_cfw_class.supported_parallelization_modes.return_value = {
            ParallelizationMode.SYNC,
            ParallelizationMode.THREADING,
            ParallelizationMode.MULTIPROCESSING,
        }
        step.to_framework = mock_to_cfw_class

        new_uuid = uuid4()
        mock_to_cfw_instance.get_uuid.return_value = new_uuid
        cfw_register.get_run_context.return_value = RunContext()

        executor.connections = {"h": Mock()}

        worker_manager.get_process_queues.return_value = None

        mock_process = Mock()
        mock_cmd_queue = Mock()
        mock_result_queue = Mock()
        worker_manager.create_worker_process.return_value = (mock_process, mock_cmd_queue, mock_result_queue)

        with patch("mloda.core.runtime.compute_framework_executor.multiprocessing.Lock"):
            executor.multi_execute_step(step)

        worker_manager.create_worker_process.assert_called_once()
        call_args = worker_manager.create_worker_process.call_args
        args_tuple = call_args.args[2]
        assert len(args_tuple) == 3
        assert args_tuple == (cfw_register, mock_to_cfw_instance, from_cfw_uuid)


class _DropsFunctionExtenderFramework(PythonDictFramework):
    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("function_extender", None)
        return state


class TestExecutorSealsTheAttachedFramework:
    def _attached(
        self,
        function_extender: set[Extender] | None = None,
        hook_extenders: dict[ExtenderHook, Extender] | None = None,
        framework_class: type[ComputeFramework] = PythonDictFramework,
    ) -> ComputeFramework:
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_run_context.return_value = RunContext(run_id="run-1", carrier={"k": "v"})
        executor = ComputeFrameworkExecutor(
            cfw_register,
            _worker_manager_mock(),
            function_extender=function_extender,
            hook_extenders=hook_extenders,
        )
        cfw_uuid = executor.init_compute_framework(framework_class, ParallelizationMode.SYNC, set())
        return executor.cfw_collection[cfw_uuid]

    def test_framework_carries_the_executors_transformer_registry(self) -> None:
        from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import (
            ComputeFrameworkTransformer,
        )

        registry = ComputeFrameworkTransformer()
        cfw_register = Mock(spec=CfwManager)
        cfw_register.get_run_context.return_value = RunContext(run_id="run-1")
        executor = ComputeFrameworkExecutor(cfw_register, _worker_manager_mock(), transformer=registry)
        cfw_uuid = executor.init_compute_framework(PythonDictFramework, ParallelizationMode.SYNC, set())

        assert executor.cfw_collection[cfw_uuid].transformer is registry

    def test_unpickled_framework_without_function_extender_state_does_not_invent_one(self) -> None:
        restored = pickle.loads(pickle.dumps(self._attached(framework_class=_DropsFunctionExtenderFramework)))  # nosec B301

        with pytest.raises(AttributeError):
            restored.function_extender

    def test_attached_framework_has_the_run_context(self) -> None:
        cfw = self._attached()

        assert cfw.run_context.run_id == "run-1"
        assert cfw.run_context.carrier == {"k": "v"}

    @pytest.mark.parametrize(
        "name,value",
        [
            ("run_context", RunContext(run_id="forged")),
            ("worker_index", 5),
            ("function_extender", set()),
            ("_hook_extenders", {}),
        ],
    )
    def test_attached_framework_rejects_assignment(self, name: str, value: Any) -> None:
        cfw = self._attached()
        cfw.get_function_extender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        function_extender = cfw.function_extender
        hook_extenders = cfw._hook_extenders

        with pytest.raises(AttributeError):
            setattr(cfw, name, value)

        assert cfw.run_context.run_id == "run-1"
        assert cfw.worker_index is None
        assert cfw.function_extender is function_extender
        assert cfw._hook_extenders is hook_extenders

    @pytest.mark.parametrize(
        "name", ["run_context", "worker_index", "_run_context_sealed", "function_extender", "_hook_extenders"]
    )
    def test_attached_framework_rejects_deletion(self, name: str) -> None:
        cfw = self._attached()
        cfw.get_function_extender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)

        with pytest.raises(AttributeError):
            delattr(cfw, name)

        assert "function_extender" in cfw.__dict__
        assert "_hook_extenders" in cfw.__dict__

        assert cfw.run_context.run_id == "run-1"
        assert cfw.worker_index is None
        assert cfw.__dict__["_run_context_sealed"] is True
        with pytest.raises(AttributeError):
            cfw.run_context = RunContext(run_id="forged")

    @pytest.mark.parametrize(
        "name,value",
        [
            ("run_context", RunContext(run_id="forged")),
            ("worker_index", 5),
            ("function_extender", set()),
            ("_hook_extenders", {}),
        ],
    )
    def test_sealed_framework_stays_sealed_after_pickle_round_trip(self, name: str, value: Any) -> None:
        restored = pickle.loads(pickle.dumps(self._attached()))  # nosec B301

        assert restored.run_context.run_id == "run-1"
        with pytest.raises(AttributeError):
            setattr(restored, name, value)

    @pytest.mark.parametrize("name,error", [("function_extender", AttributeError), ("_hook_extenders", TypeError)])
    def test_sealed_framework_rejects_in_place_clear_after_pickle_round_trip(
        self, name: str, error: type[Exception]
    ) -> None:
        restored = pickle.loads(pickle.dumps(self._attached({_StateHoldingExtender("a")})))  # nosec B301
        restored.get_function_extender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        target: Any = getattr(restored, name)

        with pytest.raises(error):
            target.clear()

        assert len(restored.function_extender) == 1
        assert ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE in restored._hook_extenders

    @pytest.mark.parametrize(
        "name,method,error",
        [
            ("function_extender", "clear", AttributeError),
            ("function_extender", "add", AttributeError),
            ("_hook_extenders", "clear", TypeError),
            ("_hook_extenders", "pop", TypeError),
            ("_hook_extenders", "__setitem__", TypeError),
        ],
    )
    def test_attached_framework_rejects_in_place_mutation(self, name: str, method: str, error: type[Exception]) -> None:
        hook = ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE
        caller_set: set[Extender] = {_StateHoldingExtender("a")}
        caller_table = build_hook_extenders(caller_set)
        table_before = dict(caller_table)
        cfw = self._attached(caller_set, caller_table)
        target: Any = getattr(cfw, name)
        args: tuple[Any, ...] = {
            "clear": (),
            "add": (_StateHoldingExtender("b"),),
            "pop": (hook,),
            "__setitem__": (hook, _StateHoldingExtender("b")),
        }[method]

        with pytest.raises(error):
            getattr(target, method)(*args)

        assert len(caller_set) == 1
        assert caller_table == table_before
        assert cfw.function_extender is not caller_set
        assert cfw._hook_extenders is not caller_table

    def test_direct_setstate_on_attached_framework_is_rejected(self) -> None:
        cfw = self._attached({_StateHoldingExtender("a")})
        cfw.get_function_extender(ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE)
        run_context = cfw.run_context
        function_extender = cfw.function_extender
        hook_extenders = cfw._hook_extenders
        state = dict(cfw.__dict__)
        state.update(run_context=RunContext(run_id="forged"), function_extender=set(), _hook_extenders={})

        with pytest.raises(AttributeError):
            cfw.__setstate__(state)

        assert cfw.run_context is run_context
        assert cfw.function_extender is function_extender
        assert cfw._hook_extenders is hook_extenders

    def test_unattached_framework_pickle_round_trip_stays_assignable(self) -> None:
        cfw = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        restored = pickle.loads(pickle.dumps(cfw))  # nosec B301
        restored.run_context = RunContext(run_id="free")
        restored.worker_index = 1
        restored.function_extender = set()

        assert restored.function_extender == set()
        assert restored.run_context.run_id == "free"


_GRANT_HMAC = "exec-grant-hmac-Rt6Yp3Nk"  # nosec B105
_GRANT_HOST = "exec-grant.internal.example"


def _grant_collection() -> Any:
    from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection

    return DataAccessCollection(
        credentials={
            "declared": {"host": _GRANT_HOST, "hmac": _GRANT_HMAC},
            "undeclared": {"token": "undeclared-token-Bv2Qs9Wm"},  # nosec B105
        }
    )


def _grant_step(handles: tuple[str, ...]) -> Mock:
    step = Mock(spec=FeatureGroupStep)
    step.own_frame = False
    step.uuid = uuid4()
    step.tfs_ids = []
    step.features = Mock()
    step.features.any_uuid = uuid4()
    step.children_if_root = []
    step.compute_framework = Mock()
    step.compute_framework.get_class_name.return_value = "TestCFW"
    step.get_uuids.return_value = {uuid4()}
    step.credential_handles = handles
    return step


def _grant_executor(step: Mock, collection: Any = None) -> tuple[ComputeFrameworkExecutor, Mock, Mock, UUID]:
    cfw_register = Mock(spec=CfwManager)
    worker_manager = _worker_manager_mock()
    executor = ComputeFrameworkExecutor(cfw_register, worker_manager, data_access_collection=collection)
    cfw_uuid = uuid4()
    cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = cfw_uuid
    executor.cfw_collection[cfw_uuid] = Mock(spec=ComputeFramework)
    worker_manager.get_process_queues.return_value = (Mock(), Mock(), Mock())
    return executor, cfw_register, worker_manager, cfw_uuid


class _RecordingGrantKind:
    instances: list[_RecordingGrantKind] = []

    def __init__(self, values: Any, policy: Any = None) -> None:
        self.values = dict(values)
        _RecordingGrantKind.instances.append(self)


_STEP_PASSWORD = "step-match-password-Hj2Kl4Zx"  # nosec B105
_RUN_SECRET = "run-wide-secret-Zx9Cv8Bn"  # nosec B105
_RUN_IDENTITY = "run-wide-identity-Lk7Jh6Gf"


def _match_step(access: Any, group: Any = ReadDBFG) -> Mock:
    step = _grant_step(())
    step.feature_group = group
    step.features.input_data_match = (Mock(), SourceMatch("src", access))
    return step


def _no_match_step() -> Mock:
    step = _grant_step(())
    step.features.input_data_match = None
    return step


class TestCredentialGrantHelper:
    """_credential_grant(step) builds a fresh grant of only the step's declared handles."""

    def test_the_default_grant_kind_is_static_grant(self) -> None:
        from mloda.core.abstract_plugins.components.credential_grant import StaticGrant

        assert ComputeFrameworkExecutor._grant_kind is StaticGrant

    def test_copies_only_the_declared_handles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _RecordingGrantKind.instances = []
        monkeypatch.setattr(ComputeFrameworkExecutor, "_grant_kind", _RecordingGrantKind)
        collection = _grant_collection()
        executor, *_ = _grant_executor(_grant_step(("declared",)), collection)

        grant = executor._credential_grant(_grant_step(("declared",)))

        assert isinstance(grant, _RecordingGrantKind)
        assert list(grant.values) == ["declared"]
        assert grant.values["declared"] is collection.credentials["declared"]

    def test_the_default_kind_opens_to_the_declared_value_only(self) -> None:
        executor, *_ = _grant_executor(_grant_step(("declared",)), _grant_collection())

        grant = executor._credential_grant(_grant_step(("declared",)))

        assert grant is not None
        lookup = grant.open()
        assert lookup.credential("declared")["host"] == _GRANT_HOST
        with pytest.raises(KeyError, match="undeclared"):
            lookup.credential("undeclared")

    def test_the_default_kind_carries_the_collection_key_policy(self) -> None:
        from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection

        collection = DataAccessCollection(
            credentials={"declared": {"zz_exec_policy_key": "exec-policy-value-1"}},
            public_credential_keys={"zz_exec_policy_key"},
            secret_credential_keys={"host"},
        )
        executor, *_ = _grant_executor(_grant_step(("declared",)), collection)

        grant = executor._credential_grant(_grant_step(("declared",)))

        assert grant is not None
        assert grant.open().policy == collection.credential_key_policy()

    def test_a_custom_grant_kind_receives_the_collection_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from mloda.core.abstract_plugins.components.credential_grant import StaticGrant
        from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection

        seen: list[Any] = []

        class _PolicyRecordingGrant(StaticGrant):
            def __init__(self, values: Any, policy: Any = None) -> None:
                seen.append(policy)
                super().__init__(values, policy)

        monkeypatch.setattr(ComputeFrameworkExecutor, "_grant_kind", _PolicyRecordingGrant)
        collection = DataAccessCollection(
            credentials={"declared": {"zz_custom_kind_key": "custom-kind-value-1"}},
            public_credential_keys={"zz_custom_kind_key"},
        )
        executor, *_ = _grant_executor(_grant_step(("declared",)), collection)

        executor._credential_grant(_grant_step(("declared",)))

        assert seen == [collection.credential_key_policy()]

    def test_each_dispatch_gets_a_fresh_grant(self) -> None:
        executor, *_ = _grant_executor(_grant_step(("declared",)), _grant_collection())
        step = _grant_step(("declared",))

        assert executor._credential_grant(step) is not executor._credential_grant(step)

    @pytest.mark.parametrize(
        "with_collection, make_step",
        [
            pytest.param(False, lambda: _grant_step(("declared",)), id="no_collection"),
            pytest.param(True, lambda: _grant_step(()), id="no_declared_handles"),
            pytest.param(
                True, lambda: SimpleNamespace(credential_handles=("declared",)), id="namespace_not_feature_group_step"
            ),
            pytest.param(True, lambda: Mock(spec=JoinStep), id="join_step"),
            pytest.param(False, lambda: Mock(spec=FeatureGroupStep), id="mock_step_without_collection"),
        ],
    )
    def test_none_grant(self, with_collection: bool, make_step: Callable[[], Any]) -> None:
        executor, *_ = _grant_executor(_grant_step(()), _grant_collection() if with_collection else None)

        assert executor._credential_grant(make_step()) is None

    @pytest.mark.parametrize("access_kind", ["table", "query"])
    def test_step_secrets_are_the_match_credentials_literals(self, access_kind: str) -> None:
        creds = {"host": "db.step.internal", "password": _STEP_PASSWORD}
        collection = _grant_collection()
        access = DBTable(creds, "t") if access_kind == "table" else DBQuery(creds, "select 1")
        step = _match_step(access)
        executor, *_ = _grant_executor(step, collection)

        secrets, identity = executor._step_secrets(step)

        assert secrets == credential_secret_literals(creds, collection.credential_key_policy())
        assert identity == credential_identity_literals(creds)
        assert _STEP_PASSWORD in secrets
        assert _STEP_PASSWORD in identity

    def test_step_secrets_use_no_policy_without_a_collection(self) -> None:
        creds = {"password": _STEP_PASSWORD}
        step = _match_step(DBTable(creds, "t"))
        executor, *_ = _grant_executor(step, None)

        secrets, _ = executor._step_secrets(step)

        assert secrets == credential_secret_literals(creds, None)
        assert _STEP_PASSWORD in secrets

    @pytest.mark.parametrize(
        "make_step",
        [
            pytest.param(lambda: _no_match_step(), id="no_match_on_a_mock_step"),
            pytest.param(lambda: SimpleNamespace(features=SimpleNamespace(input_data_match=None)), id="not_a_fg_step"),
            pytest.param(lambda: Mock(spec=JoinStep), id="join_step"),
            pytest.param(lambda: _match_step(ApiDataRef("key", ("col",))), id="api_data_ref"),
            pytest.param(lambda: _match_step("s3://bucket/other.csv", ChsConsumerFG), id="str_on_a_non_source_group"),
            pytest.param(lambda: _match_step(object(), ChsConsumerFG), id="unknown_on_a_non_source_group"),
        ],
    )
    def test_step_secrets_are_empty_without_a_credential(self, make_step: Callable[[], Any]) -> None:
        executor, *_ = _grant_executor(_grant_step(()), _grant_collection())

        with registered_secrets_scope(frozenset({_RUN_SECRET}), identity=frozenset({_RUN_IDENTITY})):
            assert executor._step_secrets(make_step()) == (frozenset(), frozenset())

    @pytest.mark.parametrize(
        "access, expected_secrets, expected_identity",
        [
            pytest.param(
                f"s3://bucket/{_RUN_SECRET}/data.csv?id={_RUN_IDENTITY}",
                frozenset({_RUN_SECRET}),
                frozenset({_RUN_IDENTITY}),
                id="both_contained",
            ),
            pytest.param(f"/data/{_RUN_SECRET}.csv", frozenset({_RUN_SECRET}), frozenset(), id="secret_only"),
            pytest.param("/data/plain.csv", frozenset(), frozenset(), id="none_contained"),
        ],
    )
    def test_a_str_access_gets_only_the_run_literals_it_contains(
        self, access: str, expected_secrets: frozenset[str], expected_identity: frozenset[str]
    ) -> None:
        step = _match_step(access, ToySourceFG)
        executor, *_ = _grant_executor(step, _grant_collection())

        with registered_secrets_scope(
            frozenset({_RUN_SECRET, "other-run-secret-Qa1Ws2Ed"}),
            identity=frozenset({_RUN_IDENTITY, "other-run-identity-Rf3Tg4Yh"}),
        ):
            assert executor._step_secrets(step) == (expected_secrets, expected_identity)

    @pytest.mark.parametrize("group", [ReadDBFG, ToySourceFG])
    def test_a_source_group_with_unknown_access_gets_the_full_run_set(self, group: Any) -> None:
        step = _match_step(object(), group)
        executor, *_ = _grant_executor(step, _grant_collection())

        with registered_secrets_scope(frozenset({_RUN_SECRET}), identity=frozenset({_RUN_IDENTITY})):
            secrets, identity = executor._step_secrets(step)

        assert secrets == frozenset({_RUN_SECRET})
        assert identity == frozenset({_RUN_IDENTITY})


class TestGrantDispatch:
    """Each mode hands the step's grant to its worker."""

    def test_multi_execute_step_sends_the_step_and_its_grant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _RecordingGrantKind.instances = []
        monkeypatch.setattr(ComputeFrameworkExecutor, "_grant_kind", _RecordingGrantKind)
        step = _grant_step(("declared",))
        executor, _, worker_manager, cfw_uuid = _grant_executor(step, _grant_collection())

        executor.multi_execute_step(step)

        assert len(_RecordingGrantKind.instances) == 1
        worker_manager.send_command.assert_called_once_with(
            cfw_uuid, (step, _RecordingGrantKind.instances[0], frozenset(), frozenset(), False)
        )

        worker_manager.send_command.reset_mock()
        with input_data_load_gate_scope():
            executor.multi_execute_step(step)
        assert worker_manager.send_command.call_args.args[1][4] is True

    @patch("mloda.core.runtime.compute_framework_executor.worker")
    def test_multi_execute_step_sends_only_the_steps_secrets_and_none_in_the_worker_args(
        self, mock_worker: Any
    ) -> None:
        creds = {"password": _STEP_PASSWORD}
        step = _match_step(DBTable(creds, "t"))
        executor, cfw_register, worker_manager, cfw_uuid = _grant_executor(step, _grant_collection())
        worker_manager.get_process_queues.return_value = None
        worker_manager.create_worker_process.return_value = (Mock(), Mock(), Mock())

        with registered_secrets_scope(frozenset({"run-wide-secret-value"}), identity=frozenset({"run-wide-id"})):
            executor.multi_execute_step(step)

        args = worker_manager.create_worker_process.call_args.args[2]
        assert args == (cfw_register, executor.cfw_collection[cfw_uuid], None)
        _, grant, secrets, identity, _in_gate_scope = worker_manager.send_command.call_args.args[1]
        assert grant is None
        assert _STEP_PASSWORD in secrets and _STEP_PASSWORD in identity
        assert "run-wide-secret-value" not in secrets
        assert "run-wide-id" not in identity

    @patch("mloda.core.runtime.compute_framework_executor.threading.Thread")
    def test_thread_execute_step_passes_the_grant_to_the_worker(
        self, mock_thread_class: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _RecordingGrantKind.instances = []
        monkeypatch.setattr(ComputeFrameworkExecutor, "_grant_kind", _RecordingGrantKind)
        step = _grant_step(("declared",))
        executor, cfw_register, worker_manager, cfw_uuid = _grant_executor(step, _grant_collection())

        executor.thread_execute_step(step)

        args = mock_thread_class.call_args.kwargs["args"]
        assert len(_RecordingGrantKind.instances) == 1
        assert args == (
            step,
            cfw_register,
            executor.cfw_collection[cfw_uuid],
            None,
            frozenset(),
            frozenset(),
            _RecordingGrantKind.instances[0],
            worker_manager.result_uuids_collection.add,
            False,
        )

        mock_thread_class.reset_mock()
        with input_data_load_gate_scope():
            executor.thread_execute_step(step)
        assert mock_thread_class.call_args.kwargs["args"][8] is True


def _reading_execute(handle: str) -> Any:
    from mloda.core.abstract_plugins.components.credential_grant import _read_granted

    def execute(*args: Any, **kwargs: Any) -> None:
        value = _read_granted(handle)
        raise RuntimeError(f"failed host={value['host']} hmac={value['hmac']}")

    return execute


def _capturing_execute(handle: str) -> tuple[Callable[..., None], list[BaseException]]:
    """Return an execute callable and the list it appends the raised exception to."""
    inner = _reading_execute(handle)
    raised: list[BaseException] = []

    def execute(*args: Any, **kwargs: Any) -> None:
        try:
            inner()
        except RuntimeError as e:
            raised.append(e)
            raise

    return execute, raised


def _run_sync(step: Mock, cfw_register: Mock) -> None:
    executor, original_register, _, _ = _grant_executor(step, _grant_collection())
    cfw_register.resolve_cfw_uuid_by_tfs_ids.return_value = original_register.resolve_cfw_uuid_by_tfs_ids.return_value
    executor.cfw_register = cfw_register
    executor.sync_execute_step(step)


def _run_thread(step: Mock, cfw_register: Mock) -> None:
    from mloda.core.abstract_plugins.components.credential_grant import StaticGrant

    grant = StaticGrant({"declared": {"host": _GRANT_HOST, "hmac": _GRANT_HMAC}})
    thread_worker(step, cfw_register, Mock(spec=ComputeFramework), None, grant=grant, mark_done=_ignore_done)


class _UnwalkedHolder:
    """Plain object whose attributes redaction does not walk."""

    def __init__(self, value: str) -> None:
        self.value = value


class _UnwalkedStateError(Exception):
    """__str__ renders a value kept in a set and an object, not in args."""

    def __init__(self, secret: str) -> None:
        super().__init__("opaque failure")
        self.in_set = {secret}
        self.in_object = _UnwalkedHolder(secret)

    def __str__(self) -> str:
        return f"failed {sorted(self.in_set)} {self.in_object.value}"


def _unwalked_state_execute(handle: str) -> Callable[..., None]:
    from mloda.core.abstract_plugins.components.credential_grant import _read_granted

    def execute(*args: Any, **kwargs: Any) -> None:
        raise _UnwalkedStateError(_read_granted(handle)["hmac"])

    return execute


class TestGrantOnlyValueIsMaskedInEveryFailureReport:
    """A grant-only value echoed by a failing step is absent from the exception, msg and exc_info."""

    @pytest.mark.parametrize("run", [_run_sync, _run_thread], ids=["sync", "thread"])
    def test_masked(self, run: Callable[[Mock, Mock], None]) -> None:
        cfw_register = Mock(spec=CfwManager)
        step = _grant_step(("declared",))
        step.execute.side_effect, raised = _capturing_execute("declared")

        run(step, cfw_register)

        cfw_register.set_error.assert_called_once()
        call_args = cfw_register.set_error.call_args
        msg, exc_info = call_args.args
        boom = raised[0]
        assert call_args.kwargs["exception"] is boom
        assert _GRANT_HMAC not in str(boom)
        assert _GRANT_HMAC not in repr(boom.args)
        assert _GRANT_HMAC not in msg
        assert _GRANT_HMAC not in exc_info
        assert _GRANT_HOST in str(boom)

    @pytest.mark.parametrize("run", [_run_sync, _run_thread], ids=["sync", "thread"])
    def test_masked_when_a_custom_str_renders_unwalked_state(self, run: Callable[[Mock, Mock], None]) -> None:
        cfw_register = Mock(spec=CfwManager)
        step = _grant_step(("declared",))
        step.execute.side_effect = _unwalked_state_execute("declared")

        run(step, cfw_register)

        cfw_register.set_error.assert_called_once()
        msg, exc_info = cfw_register.set_error.call_args.args
        assert _GRANT_HMAC not in msg
        assert _GRANT_HMAC not in exc_info

    def test_thread_worker_without_a_grant_still_works(self) -> None:
        cfw_register = Mock(spec=CfwManager)
        step = Mock(spec=FeatureGroupStep)
        step.uuid = uuid4()

        thread_worker(step, cfw_register, Mock(spec=ComputeFramework), None, mark_done=_ignore_done)

        step.execute.assert_called_once()
        cfw_register.set_error.assert_not_called()


class TestLogMaskingCommandDispatch:
    """The worker gets the disable or enable-process-wide command for the parent state right before the step."""

    @pytest.mark.parametrize("enabled", [False, True], ids=["not_enabled", "enabled"])
    @pytest.mark.parametrize("disabled", [True, False])
    def test_the_worker_gets_the_log_masking_command_for_the_parent_state(
        self, monkeypatch: pytest.MonkeyPatch, disabled: bool, enabled: bool
    ) -> None:
        from mloda.core.abstract_plugins.components import log_masking

        monkeypatch.setattr(log_masking, "_disabled", disabled)
        monkeypatch.setattr(log_masking, "_process_wide", enabled)
        step = _grant_step(())
        executor, _, worker_manager, cfw_uuid = _grant_executor(step, _grant_collection())

        executor.multi_execute_step(step)

        step_command: tuple[Any, Any] = (cfw_uuid, (step, None, frozenset(), frozenset(), False))
        disable_command: tuple[Any, Any] = (cfw_uuid, log_masking.DISABLE_LOG_MASKING_COMMAND)
        enable_command: tuple[Any, Any] = (
            cfw_uuid,
            log_masking.ENABLE_PROCESS_WIDE_LOG_MASKING_COMMAND,
        )
        expected: list[tuple[Any, Any]] = []
        if disabled:
            expected.append(disable_command)
        elif enabled:
            expected.append(enable_command)
        expected.append(step_command)
        assert [call.args for call in worker_manager.send_command.call_args_list] == expected
