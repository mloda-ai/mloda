"""Tests for the vendored UUIDv7 run-id generator (mloda/core/runtime/run_id.py)."""

import uuid

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
