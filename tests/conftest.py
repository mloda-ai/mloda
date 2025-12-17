import os
from typing import Any
import pytest

from mloda.core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer


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


CHECK_SKIP_COUNT_ENV_VAR = "CHECK_SKIP_COUNT"
EXPECTED_SKIP_COUNT_ENV_VAR = "EXPECTED_SKIP_COUNT"

if os.getenv(CHECK_SKIP_COUNT_ENV_VAR) == "1":

    @pytest.hookimpl(trylast=True)
    def pytest_terminal_summary(terminalreporter: Any, exitstatus: Any, config: Any) -> None:
        expected_skips = os.getenv(EXPECTED_SKIP_COUNT_ENV_VAR)
        if expected_skips is None:
            raise SystemExit(f"ERROR: {EXPECTED_SKIP_COUNT_ENV_VAR} is not set.")

        try:
            int_expected_skips = int(expected_skips)
        except ValueError:
            raise SystemExit(f"ERROR: {EXPECTED_SKIP_COUNT_ENV_VAR} must be an integer.")

        skipped = len(terminalreporter.stats.get("skipped", []))
        if skipped != int_expected_skips:
            raise SystemExit(
                f"""ERROR: Expected {expected_skips} skipped tests, but got {skipped}. Somehow the number of skipped tests does not match the expected value. Please check your test setup.
                    If this expected adjust the var EXPECTED_SKIP_COUNT in the tox.ini. 
                    If this just during development, you can adjust CHECK_SKIP_COUNT to something else than 1.
                    """
            )
