import multiprocessing

from typing import Any, List

import logging

from mloda_core.runtime.flight.flight_server import FlightServer, create_location

logger = logging.getLogger(__name__)


class ParallelRunnerFlightServer:
    def __init__(self) -> None:
        self.flight_server_process: Any = None
        self.location: Any = None
        self.final_tasks: List[Any] = []
        self.queue: Any

    def start_flight_server(self, location: Any) -> None:
        flight_server = FlightServer(location=location)
        flight_server.serve()

    def start_flight_server_process(self) -> None:
        if not self.flight_server_process:
            self.location = create_location()
            self.flight_server_process = multiprocessing.Process(
                target=self.start_flight_server,
                args=(self.location,),
            )
            self.flight_server_process.start()

    def end_flight_server_process(self) -> None:
        if self.flight_server_process:
            self.flight_server_process.terminate()
            self.flight_server_process.join()

    def get_location(self) -> Any:
        if self.location is None:
            raise ValueError("Location is not set. This should not happen.")
        return self.location
