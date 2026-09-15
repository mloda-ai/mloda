from __future__ import annotations

import multiprocessing
import queue
import threading
import time
import logging
from multiprocessing.process import BaseProcess
from typing import Any, Callable
from uuid import UUID

from mloda.core.runtime.mp_context import mp_spawn_context, spawn_daemon_process

logger = logging.getLogger(__name__)


class WorkerManager:
    """Manages thread/process lifecycle for parallel execution."""

    def __init__(self) -> None:
        """Initialize empty state."""
        self.tasks: list[threading.Thread | BaseProcess] = []
        self.process_register: dict[UUID, tuple[Any, Any, Any]] = {}
        self.result_queues_collection: set[Any] = set()
        self.result_uuids_collection: set[UUID] = set()
        # cfw_uuid -> resolved flag, for DROP_COMPLETE tuples drained by poll_result_queues
        # before wait_for_drop_completion reads them. resolved is True once the worker's own
        # children_if_root is fully satisfied (it drops its data and exits).
        self.completed_drops: dict[UUID, bool] = {}
        # cfw_uuid -> step uuids dispatched to that worker. Needed because a worker that
        # exits cleanly is invisible to find_dead_workers, so the only way to notice the
        # loss is that steps were assigned to it and no result ever arrived.
        self.assigned_steps: dict[UUID, set[UUID]] = {}

    def add_thread_task(self, task: threading.Thread) -> None:
        """Add task to list and call task.start()."""
        self.tasks.append(task)
        task.start()

    def create_worker_process(
        self, cfw_uuid: UUID, target: Callable[..., None], args: tuple[Any, ...]
    ) -> tuple[Any, Any, Any]:
        """Create worker process with command and result queues.

        Appends a zero-based worker_index as a trailing positional arg to the target.
        """
        ctx = mp_spawn_context()
        command_queue: multiprocessing.Queue[Any] = ctx.Queue()
        result_queue: multiprocessing.Queue[Any] = ctx.Queue()

        worker_index = len(self.process_register)
        # As a side effect of daemon=True, code running inside a worker cannot itself
        # spawn multiprocessing children (Python raises on that).
        process = spawn_daemon_process(ctx, target, (command_queue, result_queue, *args, worker_index))

        self.process_register[cfw_uuid] = (process, command_queue, result_queue)
        self.result_queues_collection.add(result_queue)
        self.tasks.append(process)
        process.start()

        return process, command_queue, result_queue

    def get_process_queues(self, cfw_uuid: UUID) -> tuple[Any, Any, Any] | None:
        """Return registered tuple or None."""
        return self.process_register.get(cfw_uuid)

    def send_command(self, cfw_uuid: UUID, command: Any) -> None:
        """Put command in command_queue, raise ValueError if not found."""
        result = self.process_register.get(cfw_uuid)
        if result is None:
            raise ValueError(f"No process found for CFW UUID: {cfw_uuid}")
        _, command_queue, _ = result
        command_queue.put(command)

    def poll_result_queues(self) -> None:
        """Non-blocking poll of all result queues; collects step-UUID strings and drains DROP_COMPLETE tuples into
        completed_drops."""
        for r_queue in self.result_queues_collection:
            try:
                msg = r_queue.get(block=False)
            except queue.Empty:
                continue
            if isinstance(msg, str):
                self.result_uuids_collection.add(UUID(msg))
            elif isinstance(msg, tuple) and len(msg) >= 2 and msg[0] == "DROP_COMPLETE":
                # A 2-tuple (no resolved flag) is treated as resolved, matching the meaning a
                # bare ack always had before the flag was added.
                self.completed_drops[msg[1]] = bool(msg[2]) if len(msg) >= 3 else True

    def record_assignment(self, cfw_uuid: UUID, step_uuids: set[UUID]) -> None:
        """Remember that these steps were dispatched to this worker."""
        self.assigned_steps.setdefault(cfw_uuid, set()).update(step_uuids)

    def find_dead_workers(self) -> list[tuple[UUID, int]]:
        """Return (cfw_uuid, exitcode) for workers that died abnormally (exitcode not in {None, 0})."""
        dead: list[tuple[UUID, int]] = []
        for cfw_uuid, (process, _, _) in self.process_register.items():
            exitcode = process.exitcode
            if exitcode is not None and exitcode != 0:
                dead.append((cfw_uuid, exitcode))
        return dead

    def find_orphaned_steps(self) -> list[tuple[UUID, int, list[UUID]]]:
        """Return (cfw_uuid, exitcode, orphaned step uuids) per exited worker still owing results.

        Complements ``find_dead_workers``, which only reports a non-zero exitcode. A worker
        that takes the data-drop path breaks its own loop and exits with code 0, so it is
        invisible there while the steps dispatched to it stay in ``currently_running_steps``
        forever and the run loop waits on a process that is gone.

        Any exitcode counts here, including 0: once a process has exited it will never
        produce a result, so an assigned step with no result is lost whatever the code.
        Results are checked against ``result_uuids_collection``, so a step whose result
        arrived before the exit is not reported.
        """
        orphaned: list[tuple[UUID, int, list[UUID]]] = []
        for cfw_uuid, (process, _, _) in self.process_register.items():
            exitcode = process.exitcode
            if exitcode is None:
                continue
            pending = self.assigned_steps.get(cfw_uuid, set()) - self.result_uuids_collection
            if pending:
                orphaned.append((cfw_uuid, exitcode, sorted(pending, key=str)))
        return orphaned

    def is_step_done(self, step_uuid: UUID) -> bool:
        """Return step_uuid in result_uuids_collection."""
        return step_uuid in self.result_uuids_collection

    def clear_completed_drop(self, cfw_uuid: UUID) -> None:
        """Discard a stale drop flag; a cfw goes through multiple drop cycles, and a late completion
        drained for an earlier cycle must not be mistaken for a later one."""
        self.completed_drops.pop(cfw_uuid, None)

    def wait_for_drop_completion(self, result_queue: Any, cfw_uuid: UUID, timeout: float = 5.0) -> bool | None:
        """Poll queue until ("DROP_COMPLETE", cfw_uuid, resolved) is received or timeout, checking
        completed_drops first. Returns the worker's own resolved flag, or None on timeout (no ack
        ever arrived)."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if cfw_uuid in self.completed_drops:
                return self.completed_drops.pop(cfw_uuid)
            try:
                msg = result_queue.get(block=False)
                if isinstance(msg, tuple) and len(msg) >= 2 and msg[0] == "DROP_COMPLETE" and msg[1] == cfw_uuid:
                    return bool(msg[2]) if len(msg) >= 3 else True
                result_queue.put(msg, block=False)
                time.sleep(0.001)
            except queue.Empty:
                time.sleep(0.001)
        logger.warning(f"Drop operation for CFW {cfw_uuid} timed out after {timeout}s")
        return None

    def join_all(self, graceful_timeout: float = 2.0) -> None:
        """Ask alive workers to STOP and give them graceful_timeout to exit on their own
        (so their close() teardown runs), then terminate/join all tasks as before."""
        for process, command_queue, _ in self.process_register.values():
            try:
                if process.is_alive():
                    command_queue.put("STOP", block=False)
            except Exception as e:
                logger.error(f"Error sending graceful STOP: {e}")

        deadline = time.time() + graceful_timeout
        for process, _, _ in self.process_register.values():
            try:
                process.join(timeout=max(0.0, deadline - time.time()))
            except Exception as e:
                logger.error(f"Error joining process during graceful shutdown: {e}")

        failures: list[str] = []
        for task in self.tasks:
            try:
                if isinstance(task, BaseProcess):
                    task.terminate()
                task.join()
            except Exception as e:
                logger.error(f"Error joining task: {e}")
                failures.append(f"{getattr(task, 'name', None) or task}: {e}")

        if failures:
            raise Exception(
                f"Error while joining tasks: {len(failures)} of {len(self.tasks)} failed ({'; '.join(failures)})"
            )
