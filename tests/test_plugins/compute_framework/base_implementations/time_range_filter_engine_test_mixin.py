"""
Shared test mixin for tz-aware datetime bounds on BaseFilterEngine range filters.

Regression guard for #435: GlobalFilter time-range bounds reach engines as
tz-aware ``datetime`` objects. Engines must be able to compare them against the
framework's native temporal column type without re-parsing.

Each framework-specific test class that mixes this in provides:
- ``time_filter_engine`` fixture: the engine class under test.
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
EXPECTED_IDS_EXCLUSIVE: list[int] = [2, 3, 4]


class TimeRangeFilterEngineTestMixin:
    """Shared tz-aware range filter test for BaseFilterEngine implementations."""

    @pytest.fixture
    @abstractmethod
    def time_filter_engine(self) -> Any:
        """Return the filter engine class to test."""
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def sample_time_data(self) -> Any:
        """Return framework-native data with columns ``id`` and tz-aware ``ts``."""
        raise NotImplementedError

    @abstractmethod
    def get_id_column_values(self, result: Any) -> list[int]:
        """Extract the ``id`` column from a filter result as a list of ints."""
        raise NotImplementedError

    def test_do_range_filter_tz_aware_datetime_bounds(self, time_filter_engine: Any, sample_time_data: Any) -> None:
        """Range filter with tz-aware UTC datetime bounds against a tz-aware temporal column."""
        single_filter = SingleFilter(
            Feature("ts"),
            FilterType.RANGE,
            {"min": RANGE_MIN, "max": RANGE_MAX, "max_exclusive": True},
        )

        result = time_filter_engine.do_range_filter(sample_time_data, single_filter)

        ids = sorted(self.get_id_column_values(result))
        assert ids == EXPECTED_IDS_EXCLUSIVE
