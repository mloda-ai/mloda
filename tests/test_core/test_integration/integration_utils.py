import time
from typing import Any
from mloda.core.runtime.flight.flight_server import FlightServer


def assert_flight_infos(flight_server: Any) -> None:
    """
    These settings depend on the performance of the test env.

    And on wait_for_drop_data of Runner class.
    """
    attempt = 0
    retries = 50

    flight_infos = set()
    while attempt < retries:
        flight_infos = FlightServer.list_flight_infos(flight_server.location)

        if len(flight_infos) == 0:
            break
        print(f"Attempt {attempt}")
        attempt += 1

        time.sleep(0.02)

    assert len(flight_infos) == 0
