from __future__ import annotations

import multiprocessing
import queue
import threading
import time
import logging
from multiprocessing.process import BaseProcess
from typing import Any, Callable, Optional
from uuid import UUID

from mloda.core.runtime.mp_context import mp_spawn_context

logger = logging.getLogger(__name__)


class WorkerManager:
    """Manages thread/process lifecycle for parallel execution."""

    def __init__(self) -> None:
        """Initialize empty state."""
        self.tasks: list[threading.Thread | BaseProcess] = []
        self.process_register: dict[UUID, tuple[Any, Any, Any]] = {}
        self.result_queues_collection: set[Any] = set()
        self.result_uuids_collection: set[UUID] = set()
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
        process = ctx.Process(target=target, args=(command_queue, result_queue, *args, worker_index))

        self.process_register[cfw_uuid] = (process, command_queue, result_queue)
        self.result_queues_collection.add(result_queue)
        self.tasks.append(process)
        process.start()

        return process, command_queue, result_queue

    def get_process_queues(self, cfw_uuid: UUID) -> Optional[tuple[Any, Any, Any]]:
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
        """Non-blocking poll all result queues, collect step-UUID strings.

        The result queue also carries ("DROP_COMPLETE", cfw_uuid) control tuples;
        skip non-str messages rather than pass them to UUID().
        """
        for r_queue in self.result_queues_collection:
            try:
                msg = r_queue.get(block=False)
            except queue.Empty:
                continue
            if isinstance(msg, str):
                self.result_uuids_collection.add(UUID(msg))

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

    def wait_for_drop_completion(self, result_queue: Any, cfw_uuid: UUID, timeout: float = 5.0) -> None:
        """Poll queue until ("DROP_COMPLETE", cfw_uuid) received or timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                msg = result_queue.get(block=False)
                if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "DROP_COMPLETE" and msg[1] == cfw_uuid:
                    return
                result_queue.put(msg, block=False)
            except queue.Empty:
                time.sleep(0.001)
        logger.warning(f"Drop operation for CFW {cfw_uuid} timed out after {timeout}s")

    def join_all(self) -> None:
        """Terminate processes (not threads), join all tasks, raise Exception if any fail."""
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
