"""Failing tests for the pyarrow column-semantics introspector (epic #518, Phase 1).

Defines the not-yet-implemented module-level function ``column_semantics(table, column)``
in ``mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_type_semantics``.
Expected to fail at import time (ModuleNotFoundError) until Green implements it.
"""

import pytest
from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_type_semantics import column_semantics

import logging

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None  # type: ignore[assignment]


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestPyArrowColumnSemantics:
    @pytest.fixture
    def table(self) -> Any:
        return pa.table(
            {
                "ts_naive": pa.array([0, 1], pa.timestamp("us")),
                "ts_aware": pa.array([0, 1], pa.timestamp("us", "UTC")),
                "num": pa.array([1, 2], pa.int64()),
                "s": pa.array(["a", "b"], pa.string()),
            }
        )

    def test_ts_naive(self, table: Any) -> None:
        sem = column_semantics(table, "ts_naive")
        assert isinstance(sem, ColumnSemantics)
        assert sem.is_ordered is True
        assert sem.is_temporal is True
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit in {"s", "ms", "us", "ns"}

    def test_ts_aware(self, table: Any) -> None:
        sem = column_semantics(table, "ts_aware")
        assert sem.is_ordered is True
        assert sem.is_temporal is True
        assert sem.is_numeric is False
        assert sem.is_tz_aware is True
        assert sem.unit in {"s", "ms", "us", "ns"}

    def test_num(self, table: Any) -> None:
        sem = column_semantics(table, "num")
        assert sem.is_ordered is True
        assert sem.is_temporal is False
        assert sem.is_numeric is True
        assert sem.is_tz_aware is False
        assert sem.unit is None

    def test_s(self, table: Any) -> None:
        sem = column_semantics(table, "s")
        assert sem.is_ordered is False
        assert sem.is_temporal is False
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit is None
