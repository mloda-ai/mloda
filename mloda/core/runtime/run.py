from __future__ import annotations

import multiprocessing
import threading
import time
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID
import logging

from mloda.core.abstract_plugins.function_extender import Extender
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.core.runtime.worker_manager import WorkerManager
from mloda.core.runtime.data_lifecycle_manager import DataLifecycleManager
from mloda.core.runtime.compute_framework_executor import ComputeFrameworkExecutor
from mloda.core.core.cfw_manager import CfwManager, MyManager
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda.core.abstract_plugins.components.feature_set import FeatureSet


logger = logging.getLogger(__name__)


class ExecutionOrchestrator:
    """
    Orchestrates the execution of an mloda based on a given execution plan.

    This class manages compute frameworks (CFWs), data dependencies, and parallel execution
    using threads or multiprocessing. It handles the execution of feature group steps,
    transform framework steps, and join steps, while also managing data dropping and result collection.
    """

    def __init__(
        self,
        execution_planner: ExecutionPlan,
        flight_server: Optional[ParallelRunnerFlightServer] = None,
    ) -> None:
        """
        Initializes the ExecutionOrchestrator with an execution plan and optional flight server.

        Args:
            execution_planner: The execution plan that defines the steps to be executed.
            flight_server: An optional flight server for data transfer.
        """
        self.execution_planner = execution_planner

        self.cfw_register: CfwManager

        # multiprocessing - delegate to WorkerManager
        self.location: Optional[str] = None
        self.worker_manager = WorkerManager()

        # Data lifecycle - delegate to DataLifecycleManager
        self.data_lifecycle_manager = DataLifecycleManager()

        self.flight_server = None
        if flight_server:
            self.flight_server = flight_server

    def _is_step_done(self, step_uuids: Set[UUID], finished_ids: Set[UUID]) -> bool:
        """
        Checks if all steps identified by the given UUIDs have already been finished.
        """
        return all(uuid in finished_ids for uuid in step_uuids)

    def _drop_data_for_finished_cfws(self, finished_ids: Set[UUID]) -> None:
        """
        Handles the dropping of intermediate data based on finished steps.
        """
        self.data_lifecycle_manager.drop_data_for_finished_cfws(
            finished_ids, self.executor.cfw_collection, self.location
        )

    def compute(self) -> None:
        """
        Executes the mloda pipeline based on the execution plan.

        This method iterates through the execution plan, checks dependencies,
        and executes steps using the appropriate parallelization mode.
        It also handles errors, result collection, and data dropping.
        """
        if self.cfw_register is None:
            raise ValueError("CfwManager not initialized")

        self.executor = ComputeFrameworkExecutor(self.cfw_register, self.worker_manager)

        finished_ids: Set[UUID] = set()
        to_finish_ids: Set[UUID] = set()
        currently_running_steps: Set[UUID] = set()

        try:
            while to_finish_ids != finished_ids or len(finished_ids) == 0:
                if self.cfw_register:
                    error = self.cfw_register.get_error()
                    if error:
                        logger.error(self.cfw_register.get_error_exc_info())
                        raise Exception(self.cfw_register.get_error_exc_info(), self.cfw_register.get_error_msg())
                else:
                    break

                for step in self.execution_planner:
                    to_finish_ids.update(step.get_uuids())

                    if isinstance(step, FeatureGroupStep):
                        self._drop_data_for_finished_cfws(finished_ids)

                    if self._is_step_done(step.get_uuids(), finished_ids):
                        continue

                    # check if step is currently running
                    if self.currently_running_step(step.get_uuids(), currently_running_steps):
                        if self._process_step_result(step):
                            self._mark_step_as_finished(step.get_uuids(), finished_ids, currently_running_steps)
                        continue

                    if not self._can_run_step(
                        step.required_uuids, step.get_uuids(), finished_ids, currently_running_steps
                    ):
                        continue
                    self._execute_step(step)

                time.sleep(0.01)

        finally:
            self.data_lifecycle_manager.set_artifacts(self.cfw_register.get_artifacts())
            self.join()

    def _process_step_result(self, step: Any) -> Union[Any, bool]:
        """
        Handles the result of a step based on its type.

        This method checks if a step is done, then performs specific actions based
        on the step's type, such as adding results to the data collection or dropping data.
        """
        # set step.is_done from other processes via result queue
        self.worker_manager.poll_result_queues()
        if step.uuid in self.worker_manager.result_uuids_collection:
            step.step_is_done = True

        if not step.step_is_done:
            return False

        if isinstance(step, (TransformFrameworkStep, JoinStep)):
            return True

        if isinstance(step, FeatureGroupStep):
            if step.features.any_uuid is None:
                raise ValueError(f"from_feature_uuid should not be none. {step}")

            cfw = self.executor.get_cfw(step.compute_framework, step.features.any_uuid)
            self.add_to_result_data_collection(cfw, step.features, step.uuid)
            self._drop_data_if_possible(cfw, step)

        return True

    def _drop_data_if_possible(self, cfw: ComputeFramework, step: Any) -> None:
        """
        Drops data associated with a compute framework if possible.

        This method checks if data can be dropped based on the CFW's dependencies
        and either drops the data directly or sends a command to a worker process to do so.
        """
        process, command_queue, result_queue = self.worker_manager.process_register.get(cfw.uuid, (None, None, None))

        feature_uuids_to_possible_drop = {f.uuid for f in step.features.features}

        if command_queue is None:
            data_to_drop = cfw.add_already_calculated_children_and_drop_if_possible(
                feature_uuids_to_possible_drop, self.location
            )
            if isinstance(data_to_drop, frozenset):
                self.data_lifecycle_manager.track_data_to_drop[cfw.uuid] = set(data_to_drop)
        else:
            command_queue.put(feature_uuids_to_possible_drop)

            flyway_datasets = self.cfw_register.get_uuid_flyway_datasets(cfw.uuid)
            if flyway_datasets:
                self.data_lifecycle_manager.track_data_to_drop[cfw.uuid] = flyway_datasets

            if result_queue is not None:
                self._wait_for_drop_completion(result_queue, cfw.uuid)

    def _wait_for_drop_completion(
        self, result_queue: multiprocessing.Queue[Any], cfw_uuid: UUID, timeout: float = 5.0
    ) -> None:
        """
        Wait for drop operation to complete from worker process.

        Args:
            result_queue: The queue to receive completion signals from the worker.
            cfw_uuid: The UUID of the compute framework being dropped.
            timeout: Maximum time to wait for completion in seconds.
        """
        self.worker_manager.wait_for_drop_completion(result_queue, cfw_uuid, timeout)

    def _execute_step(self, step: Any) -> None:
        """
        Executes a step based on its parallelization mode.
        """
        execution_function = self.executor._get_execution_function(
            self.cfw_register.get_parallelization_modes(), step.get_parallelization_mode()
        )
        execution_function(step)

    def join(self) -> None:
        """
        Joins all tasks (threads or processes) and terminates multiprocessing processes.
        """
        self.worker_manager.join_all()

    def add_to_result_data_collection(self, cfw: ComputeFramework, features: FeatureSet, step_uuid: UUID) -> None:
        """
        Adds the result data to the result data collection.
        """
        self.data_lifecycle_manager.add_to_result_data_collection(cfw, features, step_uuid, self.location)

    def get_result_data(
        self, cfw: ComputeFramework, selected_feature_names: Set[FeatureName], location: Optional[str] = None
    ) -> Any:
        """
        Gets result data from the compute framework.
        """
        return self.data_lifecycle_manager.get_result_data(cfw, selected_feature_names, location)

    def currently_running_step(self, step_uuids: Set[UUID], currently_running_steps: Set[UUID]) -> bool:
        """
        Checks if a step is currently running.

        Returns:
            True if the step is currently running, False otherwise.
        """
        if next(iter(step_uuids)) not in currently_running_steps:
            return False
        return True

    def __enter__(
        self,
        parallelization_modes: Set[ParallelizationMode] = {ParallelizationMode.SYNC},
        function_extender: Optional[Set[Extender]] = None,
        api_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Enters the context of the ExecutionOrchestrator.
        """
        MyManager.register("CfwManager", CfwManager)
        self.manager = MyManager().__enter__()
        self.cfw_register = self.manager.CfwManager(parallelization_modes, function_extender)  # type: ignore[attr-defined]

        if self.flight_server:
            if self.flight_server.flight_server_process is None:
                self.flight_server.start_flight_server_process()

        if self.flight_server:
            self.location = self.flight_server.get_location()

            if self.location is None:
                raise ValueError("Location should not be None.")

            self.cfw_register.set_location(self.location)

        if api_data:
            self.cfw_register.set_api_data(api_data)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Exits the context of the ExecutionOrchestrator.

        Args:
            exc_type: The exception type.
            exc_val: The exception value.
            exc_tb: The exception traceback.
        """
        self.manager.shutdown()

    def get_artifacts(self) -> Dict[str, Any]:
        """
        Gets the artifacts.
        """
        return self.data_lifecycle_manager.get_artifacts()

    def _can_run_step(
        self,
        required_uuids: Set[UUID],
        step_uuid: Set[UUID],
        finished_steps: Set[UUID],
        currently_running_steps: Set[UUID],
    ) -> bool:
        """
        Checks if a step can be run. If it can, add it to the currently_running_steps set.
        """

        with threading.Lock():
            if required_uuids.issubset(finished_steps) and not step_uuid.intersection(currently_running_steps):
                currently_running_steps.update(step_uuid)
                return True
            return False

    def _mark_step_as_finished(
        self, step_uuid: Set[UUID], finished_steps: Set[UUID], currently_running_steps: Set[UUID]
    ) -> None:
        """
        Marks a step as finished.
        """
        with threading.Lock():
            currently_running_steps.difference_update(step_uuid)
            finished_steps.update(step_uuid)

    def get_result(self) -> List[Any]:
        """
        Gets the results.
        """
        return self.data_lifecycle_manager.get_results()
