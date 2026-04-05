"""Tests for improved error messages in ParallelRunnerFlightServer."""

import pytest

from mloda.core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer


class TestFlightServerLocationNoneError:
    """Tests that get_location() with None location produces an actionable error."""

    def test_error_mentions_internal_error(self) -> None:
        server = ParallelRunnerFlightServer()
        with pytest.raises(ValueError, match="Internal error"):
            server.get_location()

    def test_error_mentions_start_method(self) -> None:
        server = ParallelRunnerFlightServer()
        with pytest.raises(ValueError, match="start_flight_server_process"):
            server.get_location()

    def test_error_contains_report_url(self) -> None:
        server = ParallelRunnerFlightServer()
        with pytest.raises(ValueError, match="mloda-ai/mloda/issues"):
            server.get_location()
