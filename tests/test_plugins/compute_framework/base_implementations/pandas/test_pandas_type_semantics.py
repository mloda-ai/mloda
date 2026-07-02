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
    import pyarrow as pa
except ImportError:
    logger.warning("Pandas or PyArrow is not installed. Some tests will be skipped.")
    pd = None
    pa = None  # type: ignore[assignment]


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


@pytest.mark.skipif(pd is None or pa is None, reason="Pandas or PyArrow is not installed. Skipping this test.")
class TestPandasArrowDtypeColumnSemantics:
    """ArrowDtype-backed columns must be introspected via the pyarrow type (Fix 2).

    The pandas reader currently relies on ``isinstance(dtype, pd.DatetimeTZDtype)``,
    which is False for ``pd.ArrowDtype``. So a tz-aware Arrow timestamp is wrongly
    reported as naive. These tests pin the correct expectations.
    """

    def test_arrow_timestamp_aware(self) -> None:
        df = pd.DataFrame({"ts": pd.array([0, 1], dtype=pd.ArrowDtype(pa.timestamp("us", tz="UTC")))})
        sem = column_semantics(df, "ts")
        assert isinstance(sem, ColumnSemantics)
        assert sem.is_temporal is True
        assert sem.is_ordered is True
        assert sem.is_tz_aware is True
        assert sem.unit == "us"

    def test_arrow_timestamp_naive(self) -> None:
        df = pd.DataFrame({"ts": pd.array([0, 1], dtype=pd.ArrowDtype(pa.timestamp("us")))})
        sem = column_semantics(df, "ts")
        assert sem.is_temporal is True
        assert sem.is_tz_aware is False
        assert sem.unit == "us"

    def test_arrow_int64_is_numeric(self) -> None:
        df = pd.DataFrame({"n": pd.array([1, 2], dtype=pd.ArrowDtype(pa.int64()))})
        sem = column_semantics(df, "n")
        assert sem.is_numeric is True
        assert sem.is_temporal is False
