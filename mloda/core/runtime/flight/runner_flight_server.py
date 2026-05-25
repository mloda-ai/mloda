from __future__ import annotations

import multiprocessing
import queue
import time

from typing import Any

import logging

from mloda.core.abstract_plugins.components.error_utils import internal_invariant_error
from mloda.core.runtime.flight.flight_server import FlightServer, create_location

logger = logging.getLogger(__name__)

LOCATION_PUBLISH_TIMEOUT_SECONDS = 9.9
LOCATION_PUBLISH_POLL_SECONDS = 0.1


class ParallelRunnerFlightServer:
    def __init__(self) -> None:
        self.flight_server_process: Any = None
        self.location: Any = None
        self.final_tasks: list[Any] = []
        self.queue: Any

    def start_flight_server(self, location: Any, location_queue: Any) -> None:
        flight_server = FlightServer(location=location)
        location_queue.put(flight_server.location)
        flight_server.serve()

    def start_flight_server_process(self) -> None:
        if not self.flight_server_process:
            location = create_location()
            location_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
            self.flight_server_process = multiprocessing.Process(
                target=self.start_flight_server,
                args=(location, location_queue),
            )
            self.flight_server_process.start()
            try:
                self.location = self.wait_for_flight_server_location(location_queue)
            except Exception:
                self.end_flight_server_process()
                raise

    def wait_for_flight_server_location(self, location_queue: multiprocessing.Queue[Any]) -> Any:
        deadline = time.monotonic() + LOCATION_PUBLISH_TIMEOUT_SECONDS
        while True:
            try:
                return location_queue.get(timeout=LOCATION_PUBLISH_POLL_SECONDS)
            except queue.Empty as exc:
                if self.flight_server_process is not None and not self.flight_server_process.is_alive():
                    self.flight_server_process.join(timeout=1)
                    try:
                        return location_queue.get_nowait()
                    except queue.Empty:
                        raise RuntimeError(self.flight_server_start_error_message()) from exc

                if time.monotonic() >= deadline:
                    raise TimeoutError(self.flight_server_start_error_message()) from exc

    def flight_server_start_error_message(self) -> str:
        exitcode = None if self.flight_server_process is None else self.flight_server_process.exitcode
        is_alive = False if self.flight_server_process is None else self.flight_server_process.is_alive()
        return internal_invariant_error(
            "ParallelRunnerFlightServer child process did not publish its Flight location.",
            actual_values=f"exitcode={exitcode}, is_alive={is_alive}",
            hint="Check the child process logs for FlightServer startup failures before retrying.",
        )

    def end_flight_server_process(self) -> None:
        if self.flight_server_process:
            if self.flight_server_process.is_alive():
                self.flight_server_process.terminate()
            self.flight_server_process.join()
            self.flight_server_process = None
            self.location = None

    def get_location(self) -> Any:
        if self.location is None:
            raise ValueError(
                internal_invariant_error(
                    "FlightServer location is None in ParallelRunnerFlightServer.get_location().",
                    hint="Ensure start_flight_server_process() was called before get_location().",
                )
            )
        return self.location
