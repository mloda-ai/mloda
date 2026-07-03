"""Failing tests for the SQL-family column-semantics helper (epic #518, Phase 1).

Defines the not-yet-implemented module-level function
``column_semantics_from_arrow(arrow_type, is_string_storage=False)`` in
``mloda_plugins.compute_framework.base_implementations.sql.sql_type_semantics``.
This helper backs both duckdb and sqlite (which store datetimes as ISO TEXT).
Expected to fail at import time (ModuleNotFoundError) until Green implements it.
"""

import pytest

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda_plugins.compute_framework.base_implementations.sql.sql_type_semantics import column_semantics_from_arrow

import logging

logger = logging.getLogger(__name__)

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None  # type: ignore[assignment]


@pytest.mark.skipif(pa is None, reason="PyArrow is not installed. Skipping this test.")
class TestSqlColumnSemanticsFromArrow:
    def test_timestamp_naive(self) -> None:
        sem = column_semantics_from_arrow(pa.timestamp("us"))
        assert isinstance(sem, ColumnSemantics)
        assert sem.is_ordered is True
        assert sem.is_temporal is True
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit == "us"

    def test_timestamp_aware(self) -> None:
        sem = column_semantics_from_arrow(pa.timestamp("us", "UTC"))
        assert sem.is_ordered is True
        assert sem.is_temporal is True
        assert sem.is_numeric is False
        assert sem.is_tz_aware is True
        assert sem.unit == "us"

    def test_int(self) -> None:
        sem = column_semantics_from_arrow(pa.int64())
        assert sem.is_ordered is True
        assert sem.is_temporal is False
        assert sem.is_numeric is True
        assert sem.is_tz_aware is False
        assert sem.unit is None

    def test_string(self) -> None:
        sem = column_semantics_from_arrow(pa.string())
        assert sem.is_ordered is False
        assert sem.is_temporal is False
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit is None

    def test_string_storage_flag_does_not_value_scan(self) -> None:
        # sqlite stores datetimes as ISO TEXT: with is_string_storage=True and a string
        # type, we do NOT value-scan, so it stays a plain non-ordered string.
        sem = column_semantics_from_arrow(pa.string(), is_string_storage=True)
        assert sem.is_ordered is False
        assert sem.is_temporal is False
        assert sem.is_numeric is False
        assert sem.is_tz_aware is False
        assert sem.unit is None

    def test_large_string_value_sample_classifies_temporal(self) -> None:
        # pa.types.is_string() is False for large_string, so value-inspection must also
        # cover large_string (and string_view) column types.
        sem = column_semantics_from_arrow(pa.large_string(), value_sample=["2024-01-01T00:00:00"])
        assert sem.is_temporal is True
        assert sem.is_ordered is True
