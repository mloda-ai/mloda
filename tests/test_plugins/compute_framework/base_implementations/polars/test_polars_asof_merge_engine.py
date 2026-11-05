"""
Tests for PolarsMergeEngine / PolarsLazyMergeEngine merge_asof (as-of join).

Two hosts: one for the eager PolarsMergeEngine (pl.DataFrame), one for the
lazy PolarsLazyMergeEngine (pl.LazyFrame).
"""

from datetime import UTC, datetime
from typing import Any

import pytest

from mloda._testing.compute_framework.merge_suite import AsofMergeEngineSuite
from mloda._testing.framework_adapter import FrameworkAdapter, PolarsAdapter, PolarsLazyAdapter
from mloda.user import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore


def _tz_aware_left_naive_right() -> tuple[Any, Any]:
    """(tz-AWARE left, tz-NAIVE right) polars frames for the cross-side as-of tz guard."""
    left = pl.DataFrame(
        {
            "k": [1, 1],
            "t": [
                datetime(2021, 1, 1, 0, 0, 10, tzinfo=UTC),
                datetime(2021, 1, 1, 0, 0, 20, tzinfo=UTC),
            ],
            "lv": [1, 2],
        }
    )
    right = pl.DataFrame({"k": [1], "t": [datetime(2021, 1, 1, 0, 0, 8)], "rv": [7.0]})
    return left, right


def _both_tz_aware() -> tuple[Any, Any]:
    """Both-aware (same tz) polars frames for the false-positive guard."""
    left = pl.DataFrame(
        {
            "k": [1, 1],
            "t": [
                datetime(2021, 1, 1, 0, 0, 10, tzinfo=UTC),
                datetime(2021, 1, 1, 0, 0, 20, tzinfo=UTC),
            ],
            "lv": [1, 2],
        }
    )
    right = pl.DataFrame({"k": [1], "t": [datetime(2021, 1, 1, 0, 0, 8, tzinfo=UTC)], "rv": [7.0]})
    return left, right


class _PolarsAsofTzChecks:
    """Cross-side timezone as-of guard tests shared by the eager and lazy polars hosts.

    Two ordered Datetime columns whose timezone-awareness differs must be
    rejected by ComparisonContract.require_compatible with a ValueError mentioning timezone,
    instead of the silent/low-level failure polars' join_asof gives today.
    """

    @staticmethod
    def _as_framework(adapter: FrameworkAdapter, left: Any, right: Any) -> tuple[Any, Any]:
        return adapter.from_arrow(left.to_arrow()), adapter.from_arrow(right.to_arrow())

    def test_asof_tz_aware_left_naive_right_raises(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        left, right = self._as_framework(adapter, *_tz_aware_left_naive_right())
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        with pytest.raises(ValueError, match=r"(?i)time[ -]?zone"):
            result = merge_engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)
            adapter.rows(result)

    def test_asof_both_tz_aware_succeeds(self, adapter: FrameworkAdapter, merge_engine: Any) -> None:
        left, right = self._as_framework(adapter, *_both_tz_aware())
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        result = merge_engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)
        rows = {row["lv"]: row["rv"] for row in adapter.rows(result)}
        assert rows[1] == 7.0
        assert rows[2] == 7.0


class TestPolarsAsofMergeEngine(_PolarsAsofTzChecks, AsofMergeEngineSuite):
    """Unit tests for the eager PolarsMergeEngine.merge_asof."""

    adapter_class = PolarsAdapter

    @classmethod
    def coercion_error_types(cls) -> tuple[type[BaseException], ...]:
        """Polars raises its own exception types on strict str.to_datetime failures."""
        return (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError, ValueError)


class TestPolarsLazyAsofMergeEngine(_PolarsAsofTzChecks, AsofMergeEngineSuite):
    """Unit tests for the lazy PolarsLazyMergeEngine.merge_asof."""

    adapter_class = PolarsLazyAdapter

    @classmethod
    def coercion_error_types(cls) -> tuple[type[BaseException], ...]:
        """Lazy polars surfaces strict str.to_datetime failures at collect time."""
        return (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError, ValueError)
