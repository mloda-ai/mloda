"""Test that start_flight_server_process raises ImportError mentioning pyarrow when pyarrow is absent."""

from __future__ import annotations

import pytest

from tests.test_core.test_optional_pyarrow._pyarrow_blocker import run_blocked

_BODY: str = """
import sys

from mloda.core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer

server = ParallelRunnerFlightServer()

try:
    server.start_flight_server_process()
    print("NO_RAISE")
    if hasattr(server, "end_flight_server_process"):
        server.end_flight_server_process()
    sys.exit(0)
except ImportError as e:
    if "pyarrow" in str(e).lower():
        print("IMPORTERROR")
    else:
        print("WRONGMSG:" + str(e))
except Exception as e:
    print("WRONG:" + type(e).__name__)
"""


@pytest.mark.timeout(30)
def test_start_flight_server_process_raises_import_error_without_pyarrow() -> None:
    """start_flight_server_process must raise ImportError mentioning pyarrow when pyarrow is absent."""
    result = run_blocked(_BODY)
    assert result.returncode == 0, f"Body crashed unexpectedly.\nstdout: {result.stdout}\nstderr:\n{result.stderr}"
    assert "IMPORTERROR" in result.stdout, (
        f"Expected IMPORTERROR sentinel. Got stdout: {result.stdout!r}\nstderr: {result.stderr}"
    )
