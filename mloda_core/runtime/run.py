from collections import defaultdict
import multiprocessing
import queue
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from uuid import UUID, uuid4
import logging

from mloda_core.abstract_plugins.components.framework_transformer.cfw_transformer import ComputeFrameworkTransformer
from mloda_core.abstract_plugins.function_extender import WrapperFunctionExtender
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.prepare.execution_plan import ExecutionPlan
from mloda_core.runtime.worker.multiprocessing_worker import worker
from mloda_core.runtime.worker.thread_worker import thread_worker
from mloda_core.core.cfw_manager import CfwManager, MyManager
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer
from mloda_core.core.step.feature_group_step import FeatureGroupStep
from mloda_core.core.step.join_step import JoinStep
from mloda_core.core.step.transform_frame_work_step import TransformFrameworkStep
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.runtime.flight.flight_server import FlightServer


logger = logging.getLogger(__name__)


class Runner:
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
        Initializes the Runner with an execution plan and optional flight server.

        Args:
            execution_planner: The execution plan that defines the steps to be executed.
            flight_server: An optional flight server for data transfer.
        """
        self.execution_planner = execution_planner

        self.cfw_register: CfwManager
        self.result_data_collection: Dict[UUID, Any] = {}
        self.track_data_to_drop: Dict[UUID, Set[UUID]] = {}
        self.artifacts: Dict[str, Any] = {}

        # multiprocessing
        self.location: Optional[str] = None
        self.tasks: List[Union[threading.Thread, multiprocessing.Process]] = []
        self.process_register: Dict[  # type: ignore
            UUID, Tuple[multiprocessing.Process, multiprocessing.Queue, multiprocessing.Queue]
        ] = defaultdict()
        self.result_queues_collection: Set[multiprocessing.Queue] = set()  # type: ignore
        self.result_uuids_collection: Set[UUID] = set()

        # Initialize framework transformer
        self.transformer = ComputeFrameworkTransformer()

        self.flight_server = None
        if flight_server:
            self.flight_server = flight_server

        # This can be reduced in realtime.
        # It is set currently for convenience on this high level
        self.wait_for_drop_data = 0.01

    def _is_step_done(self, step_uuids: Set[UUID], finished_ids: Set[UUID]) -> bool:
        """
        Checks if all steps identified by the given UUIDs have already been finished.
        """
        return all(uuid in finished_ids for uuid in step_uuids)

    def _drop_data_for_finished_cfws(self, finished_ids: Set[UUID]) -> None:
        """
        Handles the dropping of intermediate data based on finished steps.
        """
        if not finished_ids:
            return

        cfw_to_delete = set()
        for cfw_uuid, step_uuids in self.track_data_to_drop.items():
            if all(step_id in finished_ids for step_id in step_uuids):
                self._drop_cfw_data(cfw_uuid)
                cfw_to_delete.add(cfw_uuid)

        for cfw_uuid in cfw_to_delete:
            del self.track_data_to_drop[cfw_uuid]

    def _drop_cfw_data(self, cfw_uuid: UUID) -> None:
        """Drops data associated with a CFW."""
        if self.location:
            # FlightServer.drop_tables(self.location, {str(self.cfw_collection[cfw_uuid].uuid)})
            pass
        else:
            self.cfw_collection[cfw_uuid].drop_last_data()

    def compute(self) -> None:
        """
        Executes the mloda pipeline based on the execution plan.

        This method iterates through the execution plan, checks dependencies,
        and executes steps using the appropriate parallelization mode.
        It also handles errors, result collection, and data dropping.
        """
        if self.cfw_register is None:
            raise ValueError("CfwManager not initialized")

        finished_ids: Set[UUID] = set()
        to_finish_ids: Set[UUID] = set()
        currently_running_steps: Set[UUID] = set()

        self.cfw_collection: Dict[UUID, ComputeFrameWork] = {}

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
            self.artifacts = self.cfw_register.get_artifacts()
            self.join()

    def get_done_steps_of_multiprocessing_result_queue(self) -> None:
        """
        Retrieves UUIDs of finished steps from multiprocessing result queues.

        This method iterates through the result queues and adds any available UUIDs
        to the collection of finished UUIDs.
        """
        for r_queue in self.result_queues_collection:
            try:
                result_uuid = r_queue.get(block=False)
                self.result_uuids_collection.add(UUID(result_uuid))
            except queue.Empty:
                continue

    def _process_step_result(self, step: Any) -> Union[Any, bool]:
        """
        Handles the result of a step based on its type.

        This method checks if a step is done, then performs specific actions based
        on the step's type, such as adding results to the data collection or dropping data.
        """
        # set step.is_done from other processes via result queue
        self.get_done_steps_of_multiprocessing_result_queue()
        if step.uuid in self.result_uuids_collection:
            step.step_is_done = True

        if not step.step_is_done:
            return False

        if isinstance(step, (TransformFrameworkStep, JoinStep)):
            return True

        if isinstance(step, FeatureGroupStep):
            if step.features.any_uuid is None:
                raise ValueError(f"from_feature_uuid should not be none. {step}")

            cfw = self.get_cfw(step.compute_framework, step.features.any_uuid)
            self.add_to_result_data_collection(cfw, step.features, step.uuid)
            self._drop_data_if_possible(cfw, step)

        return True

    def _drop_data_if_possible(self, cfw: ComputeFrameWork, step: Any) -> None:
        """
        Drops data associated with a compute framework if possible.

        This method checks if data can be dropped based on the CFW's dependencies
        and either drops the data directly or sends a command to a worker process to do so.
        """
        process, command_queue, result_queue = self.process_register.get(cfw.uuid, (None, None, None))

        feature_uuids_to_possible_drop = {f.uuid for f in step.features.features}

        if command_queue is None:
            data_to_drop = cfw.add_already_calculated_children_and_drop_if_possible(
                feature_uuids_to_possible_drop, self.location
            )
            if isinstance(data_to_drop, frozenset):
                self.track_data_to_drop[cfw.uuid] = set(data_to_drop)
        else:
            command_queue.put(feature_uuids_to_possible_drop)

            flyway_datasets = self.cfw_register.get_uuid_flyway_datasets(cfw.uuid)
            if flyway_datasets:
                self.track_data_to_drop[cfw.uuid] = flyway_datasets

            time.sleep(self.wait_for_drop_data)

    def get_cfw(self, compute_framework: Type[ComputeFrameWork], feature_uuid: UUID) -> ComputeFrameWork:
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

    def prepare_execute_step(self, step: Any, parallelization_mode: ParallelizationModes) -> UUID:
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
        from_cfw: Optional[Union[ComputeFrameWork, UUID]] = None
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

    def _execute_step(self, step: Any) -> None:
        """
        Executes a step based on its parallelization mode.
        """
        execution_function = self._get_execution_function(
            self.cfw_register.get_parallelization_modes(), step.get_parallelization_mode()
        )
        execution_function(step)

    def sync_execute_step(self, step: Any) -> None:
        """
        Executes a step synchronously.
        """
        cfw_uuid = self.prepare_execute_step(step, ParallelizationModes.SYNC)

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
        cfw_uuid = self.prepare_execute_step(step, ParallelizationModes.THREADING)
        from_cfw = self.prepare_tfs_and_joinstep(step) or None

        task = threading.Thread(
            target=thread_worker,
            args=(step, self.cfw_register, self.cfw_collection[cfw_uuid], from_cfw),
        )

        self.tasks.append(task)
        task.start()

    def multi_execute_step(self, step: Any) -> None:
        """
        Executes a step in a separate process.
        """
        cfw_uuid = self.prepare_execute_step(step, ParallelizationModes.MULTIPROCESSING)

        from_cfw = None
        if isinstance(step, TransformFrameworkStep):
            from_cfw = self.prepare_tfs_right_cfw(step)

        process, command_queue, result_queue = self.process_register.get(
            cfw_uuid, (None, multiprocessing.Queue(), multiprocessing.Queue())
        )

        if process is None:
            process = multiprocessing.Process(
                target=worker,
                args=(command_queue, result_queue, self.cfw_register, self.cfw_collection[cfw_uuid], from_cfw),
            )
            self.process_register[cfw_uuid] = (process, command_queue, result_queue)
            process.start()
            self.tasks.append(process)
            self.result_queues_collection.add(result_queue)

        if command_queue:
            command_queue.put(step)
        else:
            raise ValueError("Command queue should not be None.")

    def join(self) -> None:
        """
        Joins all tasks (threads or processes) and terminates multiprocessing processes.
        """
        failed = False
        for task in self.tasks:
            try:
                if isinstance(task, multiprocessing.Process):
                    task.terminate()

                task.join()
            except Exception as e:
                logger.error(f"Error joining task: {e}")
                failed = True

        if failed:
            raise Exception("Error while joining tasks")

    def add_to_result_data_collection(self, cfw: ComputeFrameWork, features: FeatureSet, step_uuid: UUID) -> None:
        """
        Adds the result data to the result data collection.
        """
        if initial_requested_features := features.get_initial_requested_features():
            result = None
            result = self.get_result_data(cfw, initial_requested_features, self.location)
            if result is not None:
                self.result_data_collection[step_uuid] = result

    def get_result_data(
        self, cfw: ComputeFrameWork, selected_feature_names: Set[FeatureName], location: Optional[str] = None
    ) -> Any:
        """
        Gets result data from the compute framework.
        """
        if cfw.data is not None:
            data = cfw.data
        elif location:
            data = FlightServer.download_table(location, str(cfw.uuid))
            data = cfw.convert_flyserver_data_back(data, self.transformer)
        else:
            raise ValueError("Not implemented.")

        return cfw.select_data_by_column_names(data, selected_feature_names)

    def add_compute_framework(
        self,
        step: Any,
        parallelization_mode: ParallelizationModes,
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

    def init_compute_framework(
        self,
        cf_class: Type[ComputeFrameWork],
        parallelization_mode: ParallelizationModes,
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
        parallelization_modes: Set[ParallelizationModes] = {ParallelizationModes.SYNC},
        function_extender: Optional[Set[WrapperFunctionExtender]] = None,
        api_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Enters the context of the Runner.
        """
        MyManager.register("CfwManager", CfwManager)
        self.manager = MyManager().__enter__()
        self.cfw_register = self.manager.CfwManager(parallelization_modes, function_extender)  # type: ignore

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
        Exits the context of the Runner.

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
        return self.artifacts

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

    def _get_execution_function(
        self, mode_by_cfw_register: Set[ParallelizationModes], mode_by_step: Set[ParallelizationModes]
    ) -> Callable:  # type: ignore
        """
        Identifies the execution mode and returns the corresponding execute step function.

        Returns:
            The execute step function corresponding to the identified mode.
        """
        modes = mode_by_cfw_register.intersection(mode_by_step)

        if ParallelizationModes.MULTIPROCESSING in modes:
            return self.multi_execute_step
        elif ParallelizationModes.THREADING in modes:
            return self.thread_execute_step
        return self.sync_execute_step

    def get_result(self) -> List[Any]:
        """
        Gets the results.
        """
        # TODO: This is a temporary solution. We need to return the data in a more structured way.
        # Idea: return a dictionary with the feature name as key and the data as value.
        # Idea: list can keep history for debug more
        results = [v for k, v in self.result_data_collection.items()]
        if len(results) > 0:
            return results
        raise ValueError("No results found.")
