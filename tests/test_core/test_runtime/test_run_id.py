"""Tests for the vendored UUIDv7 run-id generator (mloda/core/runtime/run_id.py)."""

import sys
import uuid

import pytest

from mloda.core.runtime.run_id import generate_run_id
from tests.helpers.uuid7_assertions import assert_valid_uuid7


class TestGenerateRunIdReturnsAValidUuid7:
    def test_returns_a_string(self) -> None:
        run_id = generate_run_id()

        assert isinstance(run_id, str)

    def test_parses_as_a_valid_uuid7(self) -> None:
        run_id = generate_run_id()

        assert_valid_uuid7(run_id)


class TestGenerateRunIdUniqueness:
    def test_two_consecutive_calls_differ(self) -> None:
        first = generate_run_id()
        second = generate_run_id()

        assert first != second


class TestGenerateRunIdTimestampMonotonicity:
    def test_millisecond_timestamp_is_non_decreasing_across_a_tight_loop(self) -> None:
        run_ids = [generate_run_id() for _ in range(500)]

        timestamps_ms = [uuid.UUID(run_id).int >> 80 for run_id in run_ids]

        for earlier, later in zip(timestamps_ms, timestamps_ms[1:]):
            assert later >= earlier


class TestGenerateRunIdBranches:
    """Both the stdlib uuid7 branch and the hand-rolled branch work on every interpreter."""

    def test_uses_stdlib_uuid7_on_python_314_and_later(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = uuid.UUID("01909a3b-1234-7abc-8def-0123456789ab")
        calls: list[int] = []

        def fake_uuid7() -> uuid.UUID:
            calls.append(1)
            return sentinel

        monkeypatch.setattr(uuid, "uuid7", fake_uuid7, raising=False)
        monkeypatch.setattr(sys, "version_info", (3, 14, 0, "final", 0))

        assert generate_run_id() == str(sentinel)
        assert calls == [1]

    def test_hand_rolled_branch_below_python_314_never_calls_uuid7(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom() -> uuid.UUID:
            raise AssertionError("uuid.uuid7 must not be used below 3.14")

        monkeypatch.setattr(uuid, "uuid7", boom, raising=False)
        monkeypatch.setattr(sys, "version_info", (3, 10, 0, "final", 0))

        assert_valid_uuid7(generate_run_id())

    def test_hand_rolled_branch_embeds_the_current_millisecond_timestamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "version_info", (3, 10, 0, "final", 0))
        monkeypatch.setattr("time.time_ns", lambda: 1_700_000_000_123 * 1_000_000)

        assert uuid.UUID(generate_run_id()).int >> 80 == 1_700_000_000_123
