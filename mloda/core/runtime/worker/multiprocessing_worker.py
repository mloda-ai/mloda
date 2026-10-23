from __future__ import annotations

import logging
import multiprocessing
import pickle  # nosec B403
import time
import traceback
from typing import Any
from uuid import UUID
from queue import Empty

from mloda.core.abstract_plugins.close_context import CloseContext, CloseReason
from mloda.core.abstract_plugins.components.credential_scrub import registered_secrets_scope
from mloda.core.abstract_plugins.components.error_utils import internal_invariant_error
from mloda.core.abstract_plugins.components.utils import contained_raise_reason, failure_report
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import ComputeFrameworkTransformer
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.core.cfw_manager import CfwManager
from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep, resolve_join_source_cfw
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep


logger = logging.getLogger(__name__)


def _handle_stop_command(command_queue: multiprocessing.Queue[Any]) -> None:
    """Puts a 'STOP' command in the command queue."""
    if command_queue:
        command_queue.put("STOP", block=False)


def _parent_gone() -> bool:
    parent = multiprocessing.parent_process()
    return parent is not None and not parent.is_alive()


def standby_worker(command_queue: multiprocessing.Queue[Any], result_queue: multiprocessing.Queue[Any]) -> None:
    """Pre-warm backend imports, then wait to be bound to a cfw and run `worker`."""
    ComputeFrameworkTransformer()  # called for its import side effect: loads every backend a bound cfw needs
    while True:
        try:
            command = command_queue.get(block=False)
        except Empty:
            if _parent_gone():
                return
            time.sleep(0.01)
            continue
        if command == "STOP":
            return
        process: Any = multiprocessing.current_process()
        # Like spawn unpickling Process args: the register proxy then takes no server-side reference.
        process._inheriting = True
        try:
            target, args, worker_index = pickle.loads(command)  # nosec B301  # bytes from this run's parent queue
        finally:
            del process._inheriting
        target(command_queue, result_queue, *args, worker_index)
        return


def _close_extenders(cfw: ComputeFramework, context: CloseContext) -> None:
    """A raising extender's close() must not stop the others from running.

    Activates one shared CloseContext, exposed via ``CloseContext.current()``, for every close() call.
    """
    pending: BaseException | None = None
    with context.activate():
        for extender in getattr(cfw, "function_extender", None) or ():
            try:
                extender.close()
            except BaseException as e:
                logger.error("Extender %s.close() %s", extender.__class__.__name__, contained_raise_reason(e))
                if not isinstance(e, Exception) and pending is None:
                    pending = e
    if pending is not None:
        raise pending


def _handle_data_dropping(
    command_queue: multiprocessing.Queue[Any],
    cfw: ComputeFramework,
    command: set[Any],
    location: str,
) -> bool:
    """Handles dropping already calculated data based on the provided command."""
    if cfw.add_already_calculated_children_and_drop_if_possible(command, location) is True:
        _handle_stop_command(command_queue)
        return True
    return False


def _execute_command(
    command: JoinStep | TransformFrameworkStep | FeatureGroupStep,
    cfw_register: CfwManager,
    cfw: ComputeFramework,
    data: Any,
    from_cfw: UUID | None,
) -> Any:
    """Executes a given command based on its type."""
    if isinstance(command, JoinStep):
        # Destination framework here, because it is already transformed beforehand
        from_cfw = resolve_join_source_cfw(command, cfw_register)

    if isinstance(command, TransformFrameworkStep):
        # from cfw is not None, if the TFS is done due to a join
        if from_cfw is None:
            if command.source_framework_uuid is None:
                raise ValueError(f"source_framework_uuid should not be none: {command}")
            from_cfw = cfw_register.get_cfw_uuid(
                command.from_framework.get_class_name(),
                command.source_framework_uuid,
            )

    data = command.execute(cfw_register, cfw, data=data, from_cfw=from_cfw)
    return data


def _handle_command_result(
    command: FeatureGroupStep,
    cfw: ComputeFramework,
    location: str,
    data: Any,
    result_queue: multiprocessing.Queue[Any],
) -> None:
    """Handles the result of a command execution, including uploading data if necessary."""
    if not isinstance(data, str) and isinstance(command, FeatureGroupStep):
        # uploaded if requested
        if command.features.get_initial_requested_features():
            if location is None:
                raise ValueError(
                    internal_invariant_error(
                        "FlightServer location is None during multiprocessing result handling.",
                        f"command={command}",
                        "The FlightServer location must be set before multiprocessing workers can upload results.",
                    )
                )
            cfw.upload_finished_data(location)

    if result_queue:
        result_queue.put(str(command.uuid), block=False)


def _report_failure(cfw_register: CfwManager, command_queue: multiprocessing.Queue[Any], e: BaseException) -> None:
    try:
        msg, exc_info = failure_report(e)
        if cfw_register:
            try:
                cfw_register.set_error(msg, exc_info, exception=e)
            except Exception:
                # Unpicklable exception: fall back to string-only (surfaces as MlodaRunError), never skip the STOP.
                cfw_register.set_error(msg, exc_info)
    finally:
        _handle_stop_command(command_queue)


def worker(
    command_queue: multiprocessing.Queue[Any],
    result_queue: multiprocessing.Queue[Any],
    cfw_register: CfwManager,
    cfw: ComputeFramework,
    from_cfw: UUID | None,
    registered_secrets: frozenset[str] = frozenset(),
    worker_index: int = 0,
) -> None:
    with registered_secrets_scope(registered_secrets):
        _run_worker(command_queue, result_queue, cfw_register, cfw, from_cfw, worker_index)


def _run_worker(
    command_queue: multiprocessing.Queue[Any],
    result_queue: multiprocessing.Queue[Any],
    cfw_register: CfwManager,
    cfw: ComputeFramework,
    from_cfw: UUID | None,
    worker_index: int,
) -> None:
    data = None
    location = cfw_register.get_location()

    if location is None:
        error_out(cfw_register, command_queue)
        return

    object.__setattr__(cfw, "worker_index", worker_index)
    run_context = RunContext()
    reason: CloseReason = "error"

    try:
        run_context = cfw_register.get_run_context()
        bootstrap = run_context.child_bootstrap
        if bootstrap is not None:
            bootstrap()

        while True:
            try:
                command = command_queue.get(block=False)
            except Empty:
                # Lets an orphaned worker exit on its own if its parent dies (e.g. SIGKILL).
                # Only checked here at poll time, so a command already in progress runs to
                # completion before this loop is reached again (best-effort, not preemptive).
                if _parent_gone():
                    reason = "parent_gone"
                    break
                time.sleep(0.01)
                continue

            if command == "STOP":
                reason = "stop"
                break

            if isinstance(command, set):
                if _handle_data_dropping(command_queue, cfw, command, location):
                    reason = "stop"
                    break
                continue

            data = _execute_command(command, cfw_register, cfw, data, from_cfw)
            _handle_command_result(command, cfw, location, data, result_queue)

            time.sleep(0.0001)
    except BaseException as e:
        _report_failure(cfw_register, command_queue, e)
    finally:
        context = CloseContext(
            deadline=time.monotonic() + run_context.graceful_shutdown_timeout,
            reason=reason,
            run_id=run_context.run_id,
            plan_id=run_context.plan_id,
            worker_index=worker_index,
            carrier=run_context.carrier,
            tenant_id=run_context.tenant_id,
            project_id=run_context.project_id,
            principal=run_context.principal,
        )
        _close_extenders(cfw, context)


def error_out(cfw_register: CfwManager, command_queue: multiprocessing.Queue[Any]) -> None:
    msg = """This is a critical error, the location should not be None."""
    logger.error(msg)
    exc_info = traceback.format_exc()
    if cfw_register:
        cfw_register.set_error(msg, exc_info)
    _handle_stop_command(command_queue)
