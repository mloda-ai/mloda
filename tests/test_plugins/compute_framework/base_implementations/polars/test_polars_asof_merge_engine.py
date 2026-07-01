"""
Tests for PolarsMergeEngine / PolarsLazyMergeEngine merge_asof (as-of join).

Two classes: one for the eager PolarsMergeEngine (pl.DataFrame), one for the
lazy PolarsLazyMergeEngine (pl.LazyFrame).
"""

from datetime import datetime, timezone
from typing import Any, Optional

import pytest

from mloda.user import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_merge_engine import PolarsMergeEngine
from mloda_plugins.compute_framework.base_implementations.polars.polars_lazy_merge_engine import PolarsLazyMergeEngine
from tests.test_plugins.compute_framework.test_tooling.asof.asof_merge_engine_test_base import AsofMergeEngineTestBase

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
                datetime(2021, 1, 1, 0, 0, 10, tzinfo=timezone.utc),
                datetime(2021, 1, 1, 0, 0, 20, tzinfo=timezone.utc),
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
                datetime(2021, 1, 1, 0, 0, 10, tzinfo=timezone.utc),
                datetime(2021, 1, 1, 0, 0, 20, tzinfo=timezone.utc),
            ],
            "lv": [1, 2],
        }
    )
    right = pl.DataFrame({"k": [1], "t": [datetime(2021, 1, 1, 0, 0, 8, tzinfo=timezone.utc)], "rv": [7.0]})
    return left, right


class _PolarsAsofTzChecks(AsofMergeEngineTestBase):
    """Cross-side timezone as-of guard tests shared by the eager and lazy polars engines.

    Epic #518, Phase 2: two ordered Datetime columns whose timezone-awareness differs must be
    rejected by ComparisonContract.require_compatible with a ValueError mentioning timezone,
    instead of the silent/low-level failure polars' join_asof gives today.
    """

    def _as_framework(self, left: Any, right: Any) -> tuple[Any, Any]:
        if self.framework_type() is pl.LazyFrame:
            return left.lazy(), right.lazy()
        return left, right

    def test_asof_tz_aware_left_naive_right_raises(self) -> None:
        left, right = self._as_framework(*_tz_aware_left_naive_right())
        engine = self.merge_engine_class()()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        with pytest.raises(ValueError, match=r"(?i)time[ -]?zone"):
            result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)
            if hasattr(result, "collect"):
                result.collect()

    def test_asof_both_tz_aware_succeeds(self) -> None:
        left, right = self._as_framework(*_both_tz_aware())
        engine = self.merge_engine_class()()
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        result = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg)
        result_dicts = self.convert_framework_to_dict(result)
        rows = {row["lv"]: self._normalize_value(row["rv"]) for row in result_dicts}
        assert rows[1] == 7.0
        assert rows[2] == 7.0


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsAsofMergeEngine(_PolarsAsofTzChecks):
    """Unit tests for the eager PolarsMergeEngine.merge_asof."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        return PolarsMergeEngine

    @classmethod
    def framework_type(cls) -> type[Any]:
        if pl is None:
            raise ImportError("Polars is not installed")
        return pl.DataFrame

    def get_connection(self) -> Optional[Any]:
        return None

    @classmethod
    def coercion_error_types(cls) -> tuple[type[BaseException], ...]:
        """Polars raises its own exception types on strict str.to_datetime failures."""
        return (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError, ValueError)


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsLazyAsofMergeEngine(_PolarsAsofTzChecks):
    """Unit tests for the lazy PolarsLazyMergeEngine.merge_asof."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        return PolarsLazyMergeEngine

    @classmethod
    def framework_type(cls) -> type[Any]:
        if pl is None:
            raise ImportError("Polars is not installed")
        return pl.LazyFrame

    def get_connection(self) -> Optional[Any]:
        return None

    @classmethod
    def coercion_error_types(cls) -> tuple[type[BaseException], ...]:
        """Lazy polars surfaces strict str.to_datetime failures at collect time."""
        return (pl.exceptions.ComputeError, pl.exceptions.InvalidOperationError, ValueError)
