import logging
import traceback
from typing import Any

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork


logger = logging.getLogger(__name__)


def thread_worker(command: Any, cfw_register: Any, cfw: ComputeFrameWork, from_cfw: ComputeFrameWork) -> None:
    try:
        command.execute(cfw_register, cfw, from_cfw=from_cfw)
        command.step_is_done = True
    except Exception as e:
        error_message = f"An error occurred: {e}"
        msg = f"{error_message}\nFull traceback:\n{traceback.format_exc()}"
        exc_info = traceback.format_exc()
        cfw_register.set_error(msg, exc_info)
        raise Exception(msg, exc_info)
