"""
Tests for ComputeFrameworkExecutor class.

This test file defines the requirements for the ComputeFrameworkExecutor class
through failing tests (TDD Red Phase). The class should extract step execution
and CFW initialization logic from the Runner class.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, Mock, call, patch
from uuid import UUID, uuid4

import pytest

from mloda.user import ParallelizationMode
from mloda import ComputeFramework
from mloda.core.core.cfw_manager import CfwManager
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.core.runtime.compute_framework_executor import ComputeFrameworkExecutor
from mloda.core.runtime.worker_manager import WorkerManager


class TestComputeFrameworkExecutorConstruction:
    """Tests for ComputeFrameworkExecutor constructor and initialization."""

    def test_constructor_initializes_empty_cfw_collection(self) -> None:
        """Constructor should initialize an empty cfw_collection dictionary."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)

        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        assert hasattr(executor, "cfw_collection")
        assert isinstance(executor.cfw_collection, dict)
        assert len(executor.cfw_collection) == 0

    def test_constructor_stores_cfw_register_dependency(self) -> None:
        """Constructor should store the cfw_register dependency."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)

        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        assert executor.cfw_register is cfw_register

    def test_constructor_stores_worker_manager_dependency(self) -> None:
        """Constructor should store the worker_manager dependency."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)

        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        assert executor.worker_manager is worker_manager


class TestInitComputeFramework:
    """Tests for init_compute_framework method."""

    def test_creates_new_compute_framework_instance(self) -> None:
        """Should create a new CFW instance with provided parameters."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        # Mock CFW class and function extender
        mock_cfw_class = Mock()
        mock_cfw_instance = Mock(spec=ComputeFramework)
        mock_cfw_class.return_value = mock_cfw_instance
        mock_cfw_class.get_class_name.return_value = "TestCFW"

        test_uuid = uuid4()
        mock_cfw_instance.get_uuid.return_value = test_uuid

        function_extender = Mock()
        cfw_register.get_function_extender.return_value = function_extender

        children = {uuid4(), uuid4()}
        mode = ParallelizationMode.SYNC

        cfw_uuid = executor.init_compute_framework(mock_cfw_class, mode, children, test_uuid)

        # Verify CFW was created with correct parameters
        mock_cfw_class.assert_called_once_with(
            mode, frozenset(children), test_uuid, function_extender=function_extender
        )
        assert cfw_uuid == test_uuid

    def test_generates_uuid_when_not_provided(self) -> None:
        """Should generate a UUID if none is provided."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        mock_cfw_class = Mock()
        mock_cfw_instance = Mock(spec=ComputeFramework)
        mock_cfw_class.return_value = mock_cfw_instance

        generated_uuid = uuid4()
        mock_cfw_instance.get_uuid.return_value = generated_uuid

        cfw_register.get_function_extender.return_value = None

        cfw_uuid = executor.init_compute_framework(mock_cfw_class, ParallelizationMode.SYNC, set())

        # Should receive some UUID (generated internally)
        assert isinstance(cfw_uuid, UUID)

    def test_adds_cfw_to_register(self) -> None:
        """Should add the CFW to the cfw_register."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        mock_cfw_class = Mock()
        mock_cfw_instance = Mock(spec=ComputeFramework)
        mock_cfw_class.return_value = mock_cfw_instance
        mock_cfw_class.get_class_name.return_value = "TestCFW"

        test_uuid = uuid4()
        mock_cfw_instance.get_uuid.return_value = test_uuid

        cfw_register.get_function_extender.return_value = None

        children = {uuid4()}
        executor.init_compute_framework(mock_cfw_class, ParallelizationMode.SYNC, children, test_uuid)

        cfw_register.add_cfw_to_compute_frameworks.assert_called_once_with(test_uuid, "TestCFW", children)

    def test_adds_cfw_to_collection(self) -> None:
        """Should add the CFW instance to the cfw_collection."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        mock_cfw_class = Mock()
        mock_cfw_instance = Mock(spec=ComputeFramework)
        mock_cfw_class.return_value = mock_cfw_instance

        test_uuid = uuid4()
        mock_cfw_instance.get_uuid.return_value = test_uuid

        cfw_register.get_function_extender.return_value = None

        executor.init_compute_framework(mock_cfw_class, ParallelizationMode.SYNC, set(), test_uuid)

        assert test_uuid in executor.cfw_collection
        assert executor.cfw_collection[test_uuid] is mock_cfw_instance


class TestAddComputeFramework:
    """Tests for add_compute_framework method."""

    def test_returns_existing_cfw_uuid_if_found(self) -> None:
        """Should return existing CFW UUID without creating new one."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        existing_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = existing_uuid

        step = Mock()
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        result = executor.add_compute_framework(step, ParallelizationMode.SYNC, uuid4(), set())

        assert result == existing_uuid
        # Should not call init_compute_framework
        cfw_register.get_function_extender.assert_not_called()

    def test_creates_new_cfw_if_not_found(self) -> None:
        """Should create new CFW if none exists for the feature UUID."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
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
        cfw_register.get_function_extender.return_value = None

        feature_uuid = uuid4()
        children = {uuid4()}

        result = executor.add_compute_framework(step, ParallelizationMode.THREADING, feature_uuid, children)

        assert result == new_uuid
        assert new_uuid in executor.cfw_collection

    @patch("mloda.core.runtime.compute_framework_executor.multiprocessing.Lock")
    def test_uses_multiprocessing_lock(self, mock_lock_class: Any) -> None:
        """Should use multiprocessing.Lock for thread safety."""
        mock_lock = MagicMock()
        mock_lock_class.return_value = mock_lock

        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        cfw_register.get_cfw_uuid.return_value = uuid4()

        step = Mock()
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        executor.add_compute_framework(step, ParallelizationMode.SYNC, uuid4(), set())

        # Verify lock was used as context manager
        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()


class TestGetCfw:
    """Tests for get_cfw method."""

    def test_retrieves_cfw_by_type_and_feature_uuid(self) -> None:
        """Should retrieve CFW from collection using register lookup."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        # Set up CFW in collection
        cfw_uuid = uuid4()
        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        # Mock register lookup
        mock_cfw_class = Mock()
        feature_uuid = uuid4()
        cfw_register.get_initialized_compute_framework_uuid.return_value = cfw_uuid

        result = executor.get_cfw(mock_cfw_class, feature_uuid)

        assert result is mock_cfw
        cfw_register.get_initialized_compute_framework_uuid.assert_called_once_with(
            mock_cfw_class, feature_uuid=feature_uuid
        )

    def test_raises_value_error_if_cfw_uuid_is_none(self) -> None:
        """Should raise ValueError if CFW UUID is not found in register."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        cfw_register.get_initialized_compute_framework_uuid.return_value = None

        mock_cfw_class = Mock()

        with pytest.raises(ValueError, match="cfw_uuid should not be none"):
            executor.get_cfw(mock_cfw_class, uuid4())


class TestGetExecutionFunction:
    """Tests for _get_execution_function method."""

    def test_returns_multi_execute_step_for_multiprocessing(self) -> None:
        """Should return multi_execute_step when multiprocessing mode is in intersection."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        mode_by_cfw = {ParallelizationMode.MULTIPROCESSING, ParallelizationMode.SYNC}
        mode_by_step = {ParallelizationMode.MULTIPROCESSING}

        result = executor._get_execution_function(mode_by_cfw, mode_by_step)

        assert result == executor.multi_execute_step

    def test_returns_thread_execute_step_for_threading(self) -> None:
        """Should return thread_execute_step when threading mode is in intersection."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        mode_by_cfw = {ParallelizationMode.THREADING, ParallelizationMode.SYNC}
        mode_by_step = {ParallelizationMode.THREADING}

        result = executor._get_execution_function(mode_by_cfw, mode_by_step)

        assert result == executor.thread_execute_step

    def test_returns_sync_execute_step_as_default(self) -> None:
        """Should return sync_execute_step when no other mode matches."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        mode_by_cfw = {ParallelizationMode.SYNC}
        mode_by_step = {ParallelizationMode.SYNC}

        result = executor._get_execution_function(mode_by_cfw, mode_by_step)

        assert result == executor.sync_execute_step

    def test_prioritizes_multiprocessing_over_threading(self) -> None:
        """Should prioritize multiprocessing when both are available."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        mode_by_cfw = {
            ParallelizationMode.MULTIPROCESSING,
            ParallelizationMode.THREADING,
            ParallelizationMode.SYNC,
        }
        mode_by_step = {ParallelizationMode.MULTIPROCESSING, ParallelizationMode.THREADING}

        result = executor._get_execution_function(mode_by_cfw, mode_by_step)

        assert result == executor.multi_execute_step


class TestPrepareExecuteStep:
    """Tests for prepare_execute_step method."""

    def test_handles_feature_group_step_with_existing_cfw(self) -> None:
        """Should return existing CFW UUID for FeatureGroupStep."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        tfs_id = uuid4()
        step.tfs_ids = [tfs_id]
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        existing_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = existing_uuid

        result = executor.prepare_execute_step(step, ParallelizationMode.SYNC)

        assert result == existing_uuid

    def test_creates_new_cfw_for_feature_group_step_without_existing(self) -> None:
        """Should create new CFW for FeatureGroupStep when none exists."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.tfs_ids = [uuid4()]
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
        cfw_register.get_function_extender.return_value = None

        result = executor.prepare_execute_step(step, ParallelizationMode.SYNC)

        assert result == new_uuid

    def test_raises_error_for_feature_group_step_without_feature_uuid(self) -> None:
        """Should raise ValueError if FeatureGroupStep has no feature UUID."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = None

        with pytest.raises(ValueError, match="from_feature_uuid should not be none"):
            executor.prepare_execute_step(step, ParallelizationMode.SYNC)

    def test_handles_transform_framework_step(self) -> None:
        """Should handle TransformFrameworkStep by creating new CFW with children."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
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
        step.uuid = uuid4()

        # Mock to_framework
        mock_to_cfw_class = Mock()
        mock_to_cfw_instance = Mock(spec=ComputeFramework)
        mock_to_cfw_class.return_value = mock_to_cfw_instance
        step.to_framework = mock_to_cfw_class

        new_uuid = uuid4()
        mock_to_cfw_instance.get_uuid.return_value = new_uuid
        cfw_register.get_function_extender.return_value = None

        with patch("mloda.core.runtime.compute_framework_executor.multiprocessing.Lock"):
            result = executor.prepare_execute_step(step, ParallelizationMode.THREADING)

        assert result == new_uuid

    def test_handles_join_step(self) -> None:
        """Should handle JoinStep by retrieving left framework CFW."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=JoinStep)
        left_uuid = uuid4()
        step.left_framework_uuids = {left_uuid}
        step.left_framework = Mock()
        step.left_framework.get_class_name.return_value = "LeftCFW"

        cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = cfw_uuid

        result = executor.prepare_execute_step(step, ParallelizationMode.SYNC)

        assert result == cfw_uuid

    def test_raises_error_when_cfw_uuid_is_none(self) -> None:
        """Should raise ValueError if CFW UUID is None after processing."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        # Create a step that doesn't match any type (edge case)
        step = Mock()
        step.__class__.__name__ = "UnknownStep"

        with pytest.raises(ValueError, match="This should not occur"):
            executor.prepare_execute_step(step, ParallelizationMode.SYNC)


class TestPrepareTfsRightCfw:
    """Tests for prepare_tfs_right_cfw method."""

    def test_uses_right_framework_uuid_when_available(self) -> None:
        """Should use right_framework_uuid when set."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=TransformFrameworkStep)
        right_uuid = uuid4()
        step.right_framework_uuid = right_uuid
        step.from_framework = Mock()
        step.from_framework.get_class_name.return_value = "FromCFW"

        cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = cfw_uuid

        result = executor.prepare_tfs_right_cfw(step)

        assert result == cfw_uuid
        cfw_register.get_cfw_uuid.assert_called_once_with("FromCFW", right_uuid)

    def test_uses_first_required_uuid_when_right_framework_uuid_is_none(self) -> None:
        """Should use first required UUID when right_framework_uuid is None."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=TransformFrameworkStep)
        step.right_framework_uuid = None
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
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=TransformFrameworkStep)
        step.right_framework_uuid = None
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
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=TransformFrameworkStep)
        step.right_framework_uuid = uuid4()
        step.from_framework = Mock()
        step.from_framework.get_class_name.return_value = "FromCFW"

        from_cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = from_cfw_uuid

        from_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[from_cfw_uuid] = from_cfw

        result = executor.prepare_tfs_and_joinstep(step)

        assert result is from_cfw

    def test_returns_from_cfw_for_join_step_with_link_uuid(self) -> None:
        """Should return from_cfw for JoinStep using link UUID."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=JoinStep)
        link_uuid = uuid4()
        step.link = Mock()
        step.link.uuid = link_uuid
        step.left_framework = Mock()
        step.left_framework.get_class_name.return_value = "LeftCFW"
        step.right_framework_uuids = {uuid4()}

        from_cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = from_cfw_uuid

        from_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[from_cfw_uuid] = from_cfw

        result = executor.prepare_tfs_and_joinstep(step)

        assert result is from_cfw
        # Should first try link.uuid
        assert cfw_register.get_cfw_uuid.call_args_list[0] == call("LeftCFW", link_uuid)

    def test_falls_back_to_right_framework_uuids_for_join_step(self) -> None:
        """Should fallback to right_framework_uuids if link UUID not found."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=JoinStep)
        link_uuid = uuid4()
        right_uuid = uuid4()
        step.link = Mock()
        step.link.uuid = link_uuid
        step.left_framework = Mock()
        step.left_framework.get_class_name.return_value = "LeftCFW"
        step.right_framework_uuids = {right_uuid}

        # First call returns None, second returns valid UUID
        from_cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.side_effect = [None, from_cfw_uuid]

        from_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[from_cfw_uuid] = from_cfw

        result = executor.prepare_tfs_and_joinstep(step)

        assert result is from_cfw
        assert cfw_register.get_cfw_uuid.call_count == 2

    def test_raises_error_for_join_step_if_from_cfw_uuid_is_none(self) -> None:
        """Should raise ValueError if from_cfw_uuid is None for JoinStep."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=JoinStep)
        step.link = Mock()
        step.link.uuid = uuid4()
        step.left_framework = Mock()
        step.left_framework.get_class_name.return_value = "LeftCFW"
        step.right_framework_uuids = {uuid4()}

        cfw_register.get_cfw_uuid.return_value = None

        with pytest.raises(ValueError, match="from_cfw_uuid should not be none"):
            executor.prepare_tfs_and_joinstep(step)

    def test_returns_none_for_other_step_types(self) -> None:
        """Should return None for step types that are neither TFS nor JoinStep."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)

        result = executor.prepare_tfs_and_joinstep(step)

        assert result is None


class TestSyncExecuteStep:
    """Tests for sync_execute_step method."""

    def test_prepares_step_with_sync_mode(self) -> None:
        """Should call prepare_execute_step with SYNC mode."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        executor.sync_execute_step(step)

        # Verify step execution was called
        step.execute.assert_called_once()

    def test_marks_step_as_done_on_success(self) -> None:
        """Should set step_is_done to True on successful execution."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"
        step.step_is_done = False

        cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        executor.sync_execute_step(step)

        assert step.step_is_done is True

    def test_handles_exception_and_sets_error(self) -> None:
        """Should catch exceptions and set error in cfw_register."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        # Make step.execute raise an exception
        step.execute.side_effect = RuntimeError("Test error")

        executor.sync_execute_step(step)

        # Verify error was set
        cfw_register.set_error.assert_called_once()
        error_msg, exc_info = cfw_register.set_error.call_args[0]
        assert "Test error" in error_msg


class TestThreadExecuteStep:
    """Tests for thread_execute_step method."""

    def test_prepares_step_with_threading_mode(self) -> None:
        """Should call prepare_execute_step with THREADING mode."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = cfw_uuid

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
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        executor.thread_execute_step(step)

        # Verify thread was created with correct parameters
        mock_thread_class.assert_called_once_with(target=mock_worker, args=(step, cfw_register, mock_cfw, None))

    @patch("mloda.core.runtime.compute_framework_executor.threading.Thread")
    def test_adds_thread_to_worker_manager(self, mock_thread_class: Any) -> None:
        """Should add created thread to worker_manager."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        executor.thread_execute_step(step)

        worker_manager.add_thread_task.assert_called_once_with(mock_thread)


class TestMultiExecuteStep:
    """Tests for multi_execute_step method."""

    def test_prepares_step_with_multiprocessing_mode(self) -> None:
        """Should call prepare_execute_step with MULTIPROCESSING mode."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = cfw_uuid

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
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = cfw_uuid

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
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

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
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=FeatureGroupStep)
        step.tfs_ids = []
        step.features = Mock()
        step.features.any_uuid = uuid4()
        step.children_if_root = []
        step.compute_framework = Mock()
        step.compute_framework.get_class_name.return_value = "TestCFW"

        cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.return_value = cfw_uuid

        mock_cfw = Mock(spec=ComputeFramework)
        executor.cfw_collection[cfw_uuid] = mock_cfw

        worker_manager.get_process_queues.return_value = (Mock(), Mock(), Mock())

        executor.multi_execute_step(step)

        worker_manager.send_command.assert_called_once_with(cfw_uuid, step)

    def test_prepares_from_cfw_for_transform_framework_step(self) -> None:
        """Should prepare from_cfw for TransformFrameworkStep."""
        cfw_register = Mock(spec=CfwManager)
        worker_manager = Mock(spec=WorkerManager)
        executor = ComputeFrameworkExecutor(cfw_register, worker_manager)

        step = Mock(spec=TransformFrameworkStep)
        step.right_framework_uuid = uuid4()
        step.from_framework = Mock()
        step.from_framework.get_class_name.return_value = "FromCFW"
        step.required_uuids = [uuid4()]

        from_cfw_uuid = uuid4()
        cfw_register.get_cfw_uuid.side_effect = [from_cfw_uuid, from_cfw_uuid]

        from_cfw = Mock(spec=ComputeFramework)
        from_cfw.children_if_root = set()
        executor.cfw_collection[from_cfw_uuid] = from_cfw

        step.link_id = None
        step.uuid = uuid4()

        mock_to_cfw_class = Mock()
        mock_to_cfw_instance = Mock(spec=ComputeFramework)
        mock_to_cfw_class.return_value = mock_to_cfw_instance
        step.to_framework = mock_to_cfw_class

        new_uuid = uuid4()
        mock_to_cfw_instance.get_uuid.return_value = new_uuid
        cfw_register.get_function_extender.return_value = None

        worker_manager.get_process_queues.return_value = (Mock(), Mock(), Mock())

        with patch("mloda.core.runtime.compute_framework_executor.multiprocessing.Lock"):
            executor.multi_execute_step(step)

        # Verify from_cfw_uuid was prepared
        assert cfw_register.get_cfw_uuid.call_count >= 1
