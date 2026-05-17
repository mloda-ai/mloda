"""
Shared test mixin for tz-aware datetime bounds on BaseFilterEngine implementations.

Regression guard for #435: GlobalFilter time-range / min / max bounds reach
engines as tz-aware ``datetime`` objects. Engines must be able to compare them
against the framework's native temporal column type without re-parsing.

Designed to compose with ``FilterEngineTestMixin``: the ``filter_engine``
fixture is reused, so each consumer only adds:

- ``sample_time_data`` fixture: framework-native data with columns
  ``id`` (1..6) and ``ts`` (Jan 1..Jan 6, 2023 at midnight UTC, tz-aware).
- ``get_id_column_values`` method: extract the ``id`` column from a result.
"""

from abc import abstractmethod
from datetime import datetime, timezone
from typing import Any

import pytest

from mloda.user import Feature
from mloda.user import FilterType
from mloda.user import SingleFilter


SAMPLE_IDS: list[int] = [1, 2, 3, 4, 5, 6]
SAMPLE_TIMESTAMPS: list[datetime] = [datetime(2023, 1, day, tzinfo=timezone.utc) for day in range(1, 7)]

RANGE_MIN = datetime(2023, 1, 2, tzinfo=timezone.utc)
RANGE_MAX = datetime(2023, 1, 5, tzinfo=timezone.utc)
EXPECTED_IDS_RANGE_EXCLUSIVE: list[int] = [2, 3, 4]

MIN_BOUND = datetime(2023, 1, 4, tzinfo=timezone.utc)
EXPECTED_IDS_MIN: list[int] = [4, 5, 6]

MAX_BOUND = datetime(2023, 1, 3, tzinfo=timezone.utc)
EXPECTED_IDS_MAX: list[int] = [1, 2, 3]


class TimeRangeFilterEngineTestMixin:
    """Shared tz-aware datetime-bound tests for BaseFilterEngine implementations."""

    @pytest.fixture
    @abstractmethod
    def sample_time_data(self) -> Any:
        """Return framework-native data with columns ``id`` and tz-aware ``ts``."""
        raise NotImplementedError

    @abstractmethod
    def get_id_column_values(self, result: Any) -> list[int]:
        """Extract the ``id`` column from a filter result as a list of ints."""
        raise NotImplementedError

    def test_do_range_filter_tz_aware_datetime_bounds(self, filter_engine: Any, sample_time_data: Any) -> None:
        """Range filter with tz-aware UTC datetime bounds against a tz-aware temporal column."""
        single_filter = SingleFilter(
            Feature("ts"),
            FilterType.RANGE,
            {"min": RANGE_MIN, "max": RANGE_MAX, "max_exclusive": True},
        )

        result = filter_engine.do_range_filter(sample_time_data, single_filter)

        ids = sorted(self.get_id_column_values(result))
        assert ids == EXPECTED_IDS_RANGE_EXCLUSIVE

    def test_do_min_filter_tz_aware_datetime_bound(self, filter_engine: Any, sample_time_data: Any) -> None:
        """Min filter with a tz-aware UTC datetime bound against a tz-aware temporal column."""
        single_filter = SingleFilter(Feature("ts"), FilterType.MIN, {"value": MIN_BOUND})

        result = filter_engine.do_min_filter(sample_time_data, single_filter)

        ids = sorted(self.get_id_column_values(result))
        assert ids == EXPECTED_IDS_MIN

    def test_do_max_filter_tz_aware_datetime_bound(self, filter_engine: Any, sample_time_data: Any) -> None:
        """Max filter with a tz-aware UTC datetime bound against a tz-aware temporal column."""
        single_filter = SingleFilter(Feature("ts"), FilterType.MAX, {"value": MAX_BOUND})

        result = filter_engine.do_max_filter(sample_time_data, single_filter)

        ids = sorted(self.get_id_column_values(result))
        assert ids == EXPECTED_IDS_MAX
