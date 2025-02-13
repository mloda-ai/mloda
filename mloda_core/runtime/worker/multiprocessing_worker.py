import logging
import multiprocessing
import time
import traceback
from typing import Any, Union
from uuid import UUID
from queue import Empty

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.core.cfw_manager import CfwManager
from mloda_core.core.step.feature_group_step import FeatureGroupStep
from mloda_core.core.step.join_step import JoinStep
from mloda_core.core.step.transform_frame_work_step import TransformFrameworkStep


logger = logging.getLogger(__name__)


def _handle_stop_command(command_queue: multiprocessing.Queue) -> None:  # type: ignore
    """Puts a 'STOP' command in the command queue."""
    if command_queue:
        command_queue.put("STOP", block=False)


def _handle_data_dropping(
    command_queue: multiprocessing.Queue,  # type: ignore
    cfw: ComputeFrameWork,
    command: set,  # type: ignore
    location: str,
) -> bool:
    """Handles dropping already calculated data based on the provided command."""
    data_to_drop = cfw.add_already_calculated_children_and_drop_if_possible(command, location)
    if data_to_drop is True:
        _handle_stop_command(command_queue)
        return True
    return False


def _execute_command(
    command: Union[JoinStep, TransformFrameworkStep, FeatureGroupStep],
    cfw_register: CfwManager,
    cfw: ComputeFrameWork,
    data: Any,
    from_cfw: UUID,
) -> Any:
    """Executes a given command based on its type."""
    if isinstance(command, JoinStep):
        # Left framework here, because it is already transformed beforehand
        from_cfw = cfw_register.get_cfw_uuid(command.left_framework.get_class_name(), command.link.uuid)  # type: ignore

        if from_cfw is None:
            from_cfw = cfw_register.get_cfw_uuid(
                command.left_framework.get_class_name(), next(iter(command.right_framework_uuids))
            )

        if from_cfw is None:
            raise ValueError(f"from_cfw should not be none: {command}")

    if isinstance(command, TransformFrameworkStep):
        # from cfw is not None, if the TFS is done due to a join
        if from_cfw is None:
            from_cfw = cfw_register.get_cfw_uuid(
                command.from_framework.get_class_name(),
                command.right_framework_uuid,
            )

    data = command.execute(cfw_register, cfw, data=data, from_cfw=from_cfw)
    cfw_register.add_column_names_to_cf_uuid(cfw.uuid, cfw.get_column_names())
    return data


def _handle_command_result(
    command: FeatureGroupStep,
    cfw: ComputeFrameWork,
    location: str,
    data: Any,
    result_queue: multiprocessing.Queue,  # type: ignore
) -> None:
    """Handles the result of a command execution, including uploading data if necessary."""
    if not isinstance(data, str) and isinstance(command, FeatureGroupStep):
        # uploaded if requested
        if command.features.get_initial_requested_features():
            if location is None:
                raise ValueError("Location is not set. This should not happen.")
            cfw.upload_finished_data(location)

    if result_queue:
        result_queue.put(str(command.uuid), block=False)


def worker(
    command_queue: multiprocessing.Queue,  # type: ignore
    result_queue: multiprocessing.Queue,  # type: ignore
    cfw_register: CfwManager,
    cfw: ComputeFrameWork,
    from_cfw: UUID,
) -> None:
    data = None
    location = cfw_register.get_location()

    if location is None:
        error_out(cfw_register, command_queue)
        return

    while True:
        try:
            command = command_queue.get(block=False)  # Waits up to 10 seconds
        except Empty:
            time.sleep(0.01)
            continue

        if command == "STOP":
            break

        if isinstance(command, set):
            if _handle_data_dropping(command_queue, cfw, command, location):
                break
            continue

        try:
            data = _execute_command(command, cfw_register, cfw, data, from_cfw)
            _handle_command_result(command, cfw, location, data, result_queue)

        except Exception as e:
            error_message = f"An error occurred: {e}"
            msg = f"{error_message}\nFull traceback:\n{traceback.format_exc()}"
            logging.error(msg)
            exc_info = traceback.format_exc()
            if cfw_register:
                cfw_register.set_error(msg, exc_info)

            _handle_stop_command(command_queue)
            break

        time.sleep(0.0001)


def error_out(cfw_register: CfwManager, command_queue: multiprocessing.Queue) -> None:  # type: ignore
    msg = """This is a critical error, the location should not be None."""
    logging.error(msg)
    exc_info = traceback.format_exc()
    if cfw_register:
        cfw_register.set_error(msg, exc_info)
    _handle_stop_command(command_queue)
