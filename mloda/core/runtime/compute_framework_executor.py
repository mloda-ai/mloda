from __future__ import annotations

import multiprocessing
import threading
import traceback
import logging
from typing import Any, Callable, Dict, Optional, Set, Type
from uuid import UUID, uuid4

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.core.cfw_manager import CfwManager
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.core.runtime.worker_manager import WorkerManager
from mloda.core.runtime.worker.thread_worker import thread_worker
from mloda.core.runtime.worker.multiprocessing_worker import worker

logger = logging.getLogger(__name__)


class ComputeFrameworkExecutor:
    """
    Manages compute framework initialization and step execution.

    Extracted from Runner class to handle CFW lifecycle and step execution logic.
    """

    def __init__(self, cfw_register: CfwManager, worker_manager: WorkerManager) -> None:
        """
        Initialize the executor with dependencies.

        Args:
            cfw_register: The CFW manager for registering compute frameworks.
            worker_manager: The worker manager for handling parallel execution.
        """
        self.cfw_collection: Dict[UUID, ComputeFramework] = {}
        self.cfw_register = cfw_register
        self.worker_manager = worker_manager

    def init_compute_framework(
        self,
        cf_class: Type[ComputeFramework],
        parallelization_mode: ParallelizationMode,
        children_if_root: Set[UUID],
        uuid: Optional[UUID] = None,
    ) -> UUID:
        """
        Initializes a compute framework.

        Returns:
            The UUID of the compute framework.
        """
        # get function_extender
        function_extender = self.cfw_register.get_function_extender()

        # init framework
        new_cfw = cf_class(
            parallelization_mode,
            frozenset(children_if_root),
            uuid or uuid4(),
            function_extender=function_extender,
        )

        # add to register
        self.cfw_register.add_cfw_to_compute_frameworks(new_cfw.get_uuid(), cf_class.get_class_name(), children_if_root)

        # add to collection
        self.cfw_collection[new_cfw.get_uuid()] = new_cfw

        return new_cfw.get_uuid()

    def add_compute_framework(
        self,
        step: Any,
        parallelization_mode: ParallelizationMode,
        feature_uuid: UUID,
        children_if_root: Set[UUID],
    ) -> UUID:
        """
        Adds a compute framework to the CFW register and CFW collection.

        Returns:
            The UUID of the compute framework.
        """
        with multiprocessing.Lock():
            cfw_uuid = self.cfw_register.get_cfw_uuid(step.compute_framework.get_class_name(), feature_uuid)
            # if cfw does not exist, create a new one
            if cfw_uuid is None:
                cfw_uuid = self.init_compute_framework(step.compute_framework, parallelization_mode, children_if_root)

            return cfw_uuid

    def get_cfw(self, compute_framework: Type[ComputeFramework], feature_uuid: UUID) -> ComputeFramework:
        """
        Retrieves a compute framework based on its type and a feature UUID.

        Args:
            compute_framework: The type of compute framework to retrieve.
            feature_uuid: The UUID of the feature associated with the compute framework.
        """
        cfw_uuid = self.cfw_register.get_initialized_compute_framework_uuid(
            compute_framework, feature_uuid=feature_uuid
        )
        if cfw_uuid is None:
            raise ValueError(f"cfw_uuid should not be none: {compute_framework}.")
        return self.cfw_collection[cfw_uuid]

    def _get_execution_function(
        self, mode_by_cfw_register: Set[ParallelizationMode], mode_by_step: Set[ParallelizationMode]
    ) -> Callable[[Any], None]:
        """
        Identifies the execution mode and returns the corresponding execute step function.

        Returns:
            The execute step function corresponding to the identified mode.
        """
        modes = mode_by_cfw_register.intersection(mode_by_step)

        if ParallelizationMode.MULTIPROCESSING in modes:
            return self.multi_execute_step
        elif ParallelizationMode.THREADING in modes:
            return self.thread_execute_step
        return self.sync_execute_step

    def prepare_execute_step(self, step: Any, parallelization_mode: ParallelizationMode) -> UUID:
        """
        Prepares a step for execution by initializing or retrieving the associated CFW.
        """
        cfw_uuid: Optional[UUID] = None

        if isinstance(step, FeatureGroupStep):
            for tfs_id in step.tfs_ids:
                cfw_uuid = self.cfw_register.get_cfw_uuid(step.compute_framework.get_class_name(), tfs_id)
                if cfw_uuid:
                    return cfw_uuid

            feature_uuid = step.features.any_uuid

            if feature_uuid is None:
                raise ValueError(f"from_feature_uuid should not be none. {step, feature_uuid}")

            cfw_uuid = self.add_compute_framework(step, parallelization_mode, feature_uuid, set(step.children_if_root))
        elif isinstance(step, TransformFrameworkStep):
            from_feature_uuid, from_cfw_uuid = None, None
            for r_f in step.required_uuids:
                from_cfw_uuid = self.cfw_register.get_cfw_uuid(step.from_framework.get_class_name(), r_f)
                if from_cfw_uuid:
                    from_feature_uuid = r_f
                    break

            if from_feature_uuid is None or from_cfw_uuid is None:
                raise ValueError(
                    f"from_feature_uuid or from_cfw_uuid should not be none. {step, from_feature_uuid, from_cfw_uuid}"
                )

            from_cfw = self.cfw_collection[from_cfw_uuid]
            childrens = set(from_cfw.children_if_root)

            if step.link_id:
                from_feature_uuid = step.link_id
                childrens.add(from_feature_uuid)

            with multiprocessing.Lock():
                cfw_uuid = self.init_compute_framework(step.to_framework, parallelization_mode, childrens, step.uuid)

        elif isinstance(step, JoinStep):
            cfw_uuid = self.cfw_register.get_cfw_uuid(
                step.left_framework.get_class_name(), next(iter(step.left_framework_uuids))
            )

        if cfw_uuid is None:
            raise ValueError(f"This should not occur. {step}")

        return cfw_uuid

    def prepare_tfs_right_cfw(self, step: TransformFrameworkStep) -> UUID:
        """
        Prepares the right CFW for a TransformFrameworkStep.
        """
        uuid = step.right_framework_uuid if step.right_framework_uuid else next(iter(step.required_uuids))

        cfw_uuid = self.cfw_register.get_cfw_uuid(step.from_framework.get_class_name(), uuid)

        if cfw_uuid is None or isinstance(cfw_uuid, UUID) is False:
            raise ValueError(
                f"cfw_uuid should not be none in prepare_tfs: {step.from_framework.get_class_name()}, {uuid}"
            )

        return cfw_uuid

    def prepare_tfs_and_joinstep(self, step: Any) -> Any:
        """
        Prepares CFWs required for TransformFrameworkStep or JoinStep.
        """
        from_cfw: Optional[Any] = None
        if isinstance(step, TransformFrameworkStep):
            from_cfw = self.prepare_tfs_right_cfw(step)
            from_cfw = self.cfw_collection[from_cfw]
        elif isinstance(step, JoinStep):
            # Left framework here, because it is already transformed beforehand
            from_cfw_uuid = self.cfw_register.get_cfw_uuid(step.left_framework.get_class_name(), step.link.uuid)

            if from_cfw_uuid is None:
                from_cfw_uuid = self.cfw_register.get_cfw_uuid(
                    step.left_framework.get_class_name(), next(iter(step.right_framework_uuids))
                )

            if from_cfw_uuid is None:
                raise ValueError(
                    f"from_cfw_uuid should not be none: {step.left_framework.get_class_name()}, {step.link.uuid}"
                )

            from_cfw = self.cfw_collection[from_cfw_uuid]
        return from_cfw

    def sync_execute_step(self, step: Any) -> None:
        """
        Executes a step synchronously.
        """
        cfw_uuid = self.prepare_execute_step(step, ParallelizationMode.SYNC)

        try:
            from_cfw = self.prepare_tfs_and_joinstep(step) or None
            step.execute(self.cfw_register, self.cfw_collection[cfw_uuid], from_cfw=from_cfw)
            step.step_is_done = True

        except Exception as e:
            error_message = f"An error occurred: {e}"
            msg = f"{error_message}\nFull traceback:\n{traceback.format_exc()}"
            logging.error(msg)
            exc_info = traceback.format_exc()
            self.cfw_register.set_error(msg, exc_info)

    def thread_execute_step(self, step: Any) -> None:
        """
        Executes a step in a separate thread.
        """
        cfw_uuid = self.prepare_execute_step(step, ParallelizationMode.THREADING)
        from_cfw = self.prepare_tfs_and_joinstep(step) or None

        task = threading.Thread(
            target=thread_worker,
            args=(step, self.cfw_register, self.cfw_collection[cfw_uuid], from_cfw),
        )

        self.worker_manager.add_thread_task(task)

    def multi_execute_step(self, step: Any) -> None:
        """
        Executes a step in a separate process.
        """
        cfw_uuid = self.prepare_execute_step(step, ParallelizationMode.MULTIPROCESSING)

        from_cfw = None
        if isinstance(step, TransformFrameworkStep):
            from_cfw = self.prepare_tfs_right_cfw(step)

        existing = self.worker_manager.get_process_queues(cfw_uuid)

        if existing is None:
            process, command_queue, result_queue = self.worker_manager.create_worker_process(
                cfw_uuid,
                worker,
                (self.cfw_register, self.cfw_collection[cfw_uuid], from_cfw),
            )
        else:
            process, command_queue, result_queue = existing

        self.worker_manager.send_command(cfw_uuid, step)
