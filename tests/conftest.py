import os
from typing import Any
import pytest

from mloda_core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer


@pytest.fixture(autouse=True)
def set_acero_alignment_handling() -> Any:
    """Modern hardware does not care about this. https://arrow.apache.org/docs/cpp/env_vars.html"""
    os.environ["ACERO_ALIGNMENT_HANDLING"] = "ignore"
    yield
    # Optionally, unset the variable or reset it after the test
    del os.environ["ACERO_ALIGNMENT_HANDLING"]


@pytest.fixture(scope="session")
def flight_server_setup() -> ParallelRunnerFlightServer:
    return ParallelRunnerFlightServer()


@pytest.fixture(scope="session")
def flight_server(flight_server_setup: ParallelRunnerFlightServer) -> Any:
    yield flight_server_setup
    flight_server_setup.end_flight_server_process()
