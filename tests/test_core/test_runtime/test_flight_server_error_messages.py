"""Tests for improved error messages in ParallelRunnerFlightServer."""

import multiprocessing
from typing import Any

import pytest

from mloda.core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer


class TestFlightServerProcessLocation:
    def test_start_flight_server_process_publishes_child_bound_location(self, monkeypatch: pytest.MonkeyPatch) -> None:
        published_location = "grpc://127.0.0.1:39123"

        class BoundFlightServer:
            def __init__(self, location: str) -> None:
                assert location == "grpc://127.0.0.1:0"
                self.location = published_location

            def serve(self) -> None:
                return None

        class InlineProcess:
            def __init__(self, target: Any, args: tuple[Any, ...]) -> None:
                self.target = target
                self.args = args
                self.started = False

            def start(self) -> None:
                self.started = True
                self.target(*self.args)

        monkeypatch.setattr(
            "mloda.core.runtime.flight.runner_flight_server.create_location", lambda: "grpc://127.0.0.1:0"
        )
        monkeypatch.setattr("mloda.core.runtime.flight.runner_flight_server.FlightServer", BoundFlightServer)

        class FakeCtx:
            Process = InlineProcess
            Queue = staticmethod(multiprocessing.Queue)

        monkeypatch.setattr(
            "mloda.core.runtime.flight.runner_flight_server.mp_spawn_context",
            lambda: FakeCtx(),
        )

        server = ParallelRunnerFlightServer()
        server.start_flight_server_process()

        assert server.location == published_location

    def test_start_flight_server_process_reaps_child_when_location_not_published(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class DeadProcess:
            def __init__(self, target: Any, args: tuple[Any, ...]) -> None:
                self.target = target
                self.args = args
                self.exitcode: int | None = None
                self.join_called = False
                self.terminate_called = False

            def start(self) -> None:
                self.exitcode = 1

            def is_alive(self) -> bool:
                return False

            def join(self, timeout: float | None = None) -> None:
                self.join_called = True

            def terminate(self) -> None:
                self.terminate_called = True

        monkeypatch.setattr(
            "mloda.core.runtime.flight.runner_flight_server.create_location", lambda: "grpc://127.0.0.1:0"
        )

        class FakeCtx:
            Process = DeadProcess
            Queue = staticmethod(multiprocessing.Queue)

        monkeypatch.setattr(
            "mloda.core.runtime.flight.runner_flight_server.mp_spawn_context",
            lambda: FakeCtx(),
        )

        server = ParallelRunnerFlightServer()

        with pytest.raises(RuntimeError, match="did not publish its Flight location") as exc_info:
            server.start_flight_server_process()

        assert "exitcode" in str(exc_info.value)
        assert "1" in str(exc_info.value)
        assert server.flight_server_process is None
        assert server.location is None


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
