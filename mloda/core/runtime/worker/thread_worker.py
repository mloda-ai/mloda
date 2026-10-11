import logging
from typing import Any
from uuid import UUID

from mloda.core.abstract_plugins.components.utils import failure_report
from mloda.core.abstract_plugins.compute_framework import ComputeFramework


logger = logging.getLogger(__name__)


def thread_worker(
    command: Any, cfw_register: Any, cfw: ComputeFramework, from_cfw: ComputeFramework | UUID | None
) -> None:
    try:
        command.execute(cfw_register, cfw, from_cfw=from_cfw)
        command.step_is_done = True
    except BaseException as e:  # not only Exception: anything escaping a thread is lost and the run hangs
        msg, exc_info = failure_report(e)
        cfw_register.set_error(msg, exc_info, exception=e)
