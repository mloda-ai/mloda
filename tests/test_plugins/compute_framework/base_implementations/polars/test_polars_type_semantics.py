"""Failing tests for the polars column-semantics introspector (epic #518, Phase 1).

Defines the not-yet-implemented module-level function ``column_semantics(df, column)``
in ``mloda_plugins.compute_framework.base_implementations.polars.polars_type_semantics``.
The function must work for both ``pl.DataFrame`` and ``pl.LazyFrame`` (via collect_schema).
Expected to fail at import time (ModuleNotFoundError) until Green implements it.
"""

import pytest
from datetime import datetime
from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda_plugins.compute_framework.base_implementations.polars.polars_type_semantics import column_semantics

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsColumnSemantics:
    @pytest.fixture
    def df(self) -> Any:
        base = pl.DataFrame(
            {
                "ts_naive": [datetime(2021, 1, 1), datetime(2021, 1, 2)],
                "num": [1, 2],
                "s": ["a", "b"],
            }
        )
        return base.with_columns(pl.col("ts_naive").dt.replace_time_zone("UTC").alias("ts_aware"))

    def test_ts_naive(self, df: Any) -> None:
        sem = column_semantics(df, "ts_naive")
        assert isinstance(sem, ColumnSemantics)
        assert sem.is_ordered is True
        assert sem.is_temporal is True
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit in {"ms", "us", "ns"}

    def test_ts_aware(self, df: Any) -> None:
        sem = column_semantics(df, "ts_aware")
        assert sem.is_ordered is True
        assert sem.is_temporal is True
        assert sem.is_numeric is False
        assert sem.is_tz_aware is True
        assert sem.unit in {"ms", "us", "ns"}

    def test_num(self, df: Any) -> None:
        sem = column_semantics(df, "num")
        assert sem.is_ordered is True
        assert sem.is_temporal is False
        assert sem.is_numeric is True
        assert sem.is_tz_aware is False
        assert sem.unit is None

    def test_s(self, df: Any) -> None:
        sem = column_semantics(df, "s")
        assert sem.is_ordered is False
        assert sem.is_temporal is False
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit is None

    def test_lazyframe_ts_aware(self, df: Any) -> None:
        sem = column_semantics(df.lazy(), "ts_aware")
        assert sem.is_temporal is True
        assert sem.is_tz_aware is True
        assert sem.unit in {"ms", "us", "ns"}

    def test_lazyframe_num(self, df: Any) -> None:
        sem = column_semantics(df.lazy(), "num")
        assert sem.is_numeric is True
        assert sem.is_temporal is False
        assert sem.unit is None
