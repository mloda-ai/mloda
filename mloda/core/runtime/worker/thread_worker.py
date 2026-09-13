import logging
import traceback
from typing import Any
from uuid import UUID

from mloda.core.abstract_plugins.compute_framework import ComputeFramework


logger = logging.getLogger(__name__)


def thread_worker(
    command: Any, cfw_register: Any, cfw: ComputeFramework, from_cfw: ComputeFramework | UUID | None
) -> None:
    try:
        command.execute(cfw_register, cfw, from_cfw=from_cfw)
        cfw_register.add_column_names_to_cf_uuid(cfw.uuid, cfw.get_column_names())
        command.step_is_done = True
    except Exception as e:
        error_message = f"An error occurred: {e}"
        msg = f"{error_message}\nFull traceback:\n{traceback.format_exc()}"
        exc_info = traceback.format_exc()
        cfw_register.set_error(msg, exc_info, exception=e)
        raise
