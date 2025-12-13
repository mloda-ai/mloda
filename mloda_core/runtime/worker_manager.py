from __future__ import annotations

import multiprocessing
import queue
import threading
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID

logger = logging.getLogger(__name__)


class WorkerManager:
    """Manages thread/process lifecycle for parallel execution."""

    def __init__(self) -> None:
        """Initialize empty state."""
        self.tasks: List[Union[threading.Thread, multiprocessing.Process]] = []
        self.process_register: Dict[UUID, Tuple[Any, Any, Any]] = {}
        self.result_queues_collection: Set[Any] = set()
        self.result_uuids_collection: Set[UUID] = set()

    def add_thread_task(self, task: threading.Thread) -> None:
        """Add task to list and call task.start()."""
        self.tasks.append(task)
        task.start()

    def create_worker_process(
        self, cfw_uuid: UUID, target: Callable[..., None], args: Tuple[Any, ...]
    ) -> Tuple[Any, Any, Any]:
        """Create worker process with command and result queues."""
        command_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
        result_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()

        process = multiprocessing.Process(target=target, args=(command_queue, result_queue, *args))

        self.process_register[cfw_uuid] = (process, command_queue, result_queue)
        self.result_queues_collection.add(result_queue)
        self.tasks.append(process)
        process.start()

        return process, command_queue, result_queue

    def get_process_queues(self, cfw_uuid: UUID) -> Optional[Tuple[Any, Any, Any]]:
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
        """Non-blocking poll all result queues, add UUIDs to result_uuids_collection."""
        for r_queue in self.result_queues_collection:
            try:
                result_uuid = r_queue.get(block=False)
                self.result_uuids_collection.add(UUID(result_uuid))
            except queue.Empty:
                continue

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
