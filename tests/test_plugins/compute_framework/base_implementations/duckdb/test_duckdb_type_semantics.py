"""Column-semantics introspection for duckdb relations (epic #518, Fix 3).

A duckdb ``INTERVAL`` type must be treated as temporal but NOT numeric. The
reader currently reports ``is_numeric=True`` because the substring ``"int"``
appears inside ``"interval"``. This test pins the correct expectation and
fails until the numeric-token matching is tightened.
"""

import logging
from typing import Any

import pyarrow as pa
import pytest

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_type_semantics import (
    column_semantics,
    nan_condition,
)

logger = logging.getLogger(__name__)

try:
    import duckdb
except ImportError:
    logger.warning("DuckDB is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]


@pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed. Skipping this test.")
class TestDuckdbIntervalSemantics:
    def test_interval_is_temporal_not_numeric(self, connection: Any) -> None:
        relation = connection.sql("SELECT INTERVAL 1 HOUR AS c")
        sem = column_semantics(relation, "c")
        assert sem.is_temporal is True
        assert sem.is_numeric is False


@pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed. Skipping this test.")
class TestDuckdbNanConditionExactLookup:
    def test_case_mismatched_name_raises(self, connection: Any) -> None:
        relation = connection.sql("SELECT 1.5::DOUBLE AS Ratio")
        with pytest.raises(ValueError, match="ratio"):
            nan_condition(relation, "ratio")

    def test_exact_name_resolves(self, connection: Any) -> None:
        relation = connection.sql("SELECT 1.5::DOUBLE AS Ratio")
        assert nan_condition(relation, "Ratio") == 'isnan("Ratio")'


@pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed. Skipping this test.")
class TestDuckdbTimestampUnit:
    def test_timestamptz_ns_reports_nanosecond_unit(self, connection: Any) -> None:
        # A nanosecond timestamp with a time zone is duckdb's TIMESTAMPTZ_NS. Its
        # type id "timestamptz_ns" does not contain the substring "timestamp_ns",
        # so the unit used to fall through to the microsecond default. Built via
        # Arrow so it does not depend on the SQL cast parsing on duckdb 1.x.
        table = pa.table({"c": pa.array([None], type=pa.timestamp("ns", tz="UTC"))})
        relation = connection.from_arrow(table)
        assert column_semantics(relation, "c").unit == "ns"

    @pytest.mark.parametrize("arrow_unit", ["ns", "us", "ms", "s"])
    def test_non_tz_timestamp_units_are_unchanged(self, connection: Any, arrow_unit: str) -> None:
        table = pa.table({"c": pa.array([None], type=pa.timestamp(arrow_unit))})
        relation = connection.from_arrow(table)
        assert column_semantics(relation, "c").unit == arrow_unit
