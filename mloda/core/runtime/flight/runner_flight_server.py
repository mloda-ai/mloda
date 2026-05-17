import multiprocessing

from typing import Any

import logging

from mloda.core.abstract_plugins.components.error_utils import internal_invariant_error
from mloda.core.runtime.flight.flight_server import FlightServer, create_location

logger = logging.getLogger(__name__)


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
            self.location = location_queue.get(timeout=10)

    def end_flight_server_process(self) -> None:
        if self.flight_server_process:
            self.flight_server_process.terminate()
            self.flight_server_process.join()

    def get_location(self) -> Any:
        if self.location is None:
            raise ValueError(
                internal_invariant_error(
                    "FlightServer location is None in ParallelRunnerFlightServer.get_location().",
                    hint="Ensure start_flight_server_process() was called before get_location().",
                )
            )
        return self.location
