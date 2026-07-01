"""Failing tests for the pandas column-semantics introspector (epic #518, Phase 1).

These tests define the behaviour of the not-yet-implemented module-level function
``column_semantics(df, column)`` living in
``mloda_plugins.compute_framework.base_implementations.pandas.pandas_type_semantics``.
They are expected to fail at import time (ModuleNotFoundError) until Green implements it.
"""

import pytest
from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_type_semantics import column_semantics

import logging

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    logger.warning("Pandas is not installed. Some tests will be skipped.")
    pd = None


@pytest.mark.skipif(pd is None, reason="Pandas is not installed. Skipping this test.")
class TestPandasColumnSemantics:
    @pytest.fixture
    def df(self) -> Any:
        naive = pd.to_datetime(["2021-01-01 00:00:00", "2021-01-02 00:00:00"])
        return pd.DataFrame(
            {
                "ts_naive": naive,
                "ts_aware": naive.tz_localize("UTC"),
                "num": [1, 2],
                "s": ["a", "b"],
            }
        )

    def test_ts_naive(self, df: Any) -> None:
        sem = column_semantics(df, "ts_naive")
        assert isinstance(sem, ColumnSemantics)
        assert sem.is_ordered is True
        assert sem.is_temporal is True
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit in {"s", "ms", "us", "ns"}

    def test_ts_aware(self, df: Any) -> None:
        sem = column_semantics(df, "ts_aware")
        assert sem.is_ordered is True
        assert sem.is_temporal is True
        assert sem.is_numeric is False
        assert sem.is_tz_aware is True
        assert sem.unit in {"s", "ms", "us", "ns"}

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
