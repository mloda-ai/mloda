"""Column-semantics introspection for duckdb relations (epic #518, Fix 3).

A duckdb ``INTERVAL`` type must be treated as temporal but NOT numeric. The
reader currently reports ``is_numeric=True`` because the substring ``"int"``
appears inside ``"interval"``. This test pins the correct expectation and
fails until the numeric-token matching is tightened.
"""

from typing import Any
import logging

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
class TestDuckdbZoneSemantics:
    def test_recorded_zone_is_reported(self, connection: Any) -> None:
        import pyarrow as pa

        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        table = pa.table({"ts": pa.array([1, 2], type=pa.timestamp("us", tz="America/New_York"))})
        sem = column_semantics(DuckdbRelation.from_arrow(connection, table), "ts")
        assert sem.is_tz_aware is True
        assert sem.tz == "America/New_York"

    @pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
    def test_unit_matches_stored_duckdb_type(self, connection: Any, unit: str) -> None:
        import pyarrow as pa

        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        table = pa.table({"ts": pa.array([1, 2], type=pa.timestamp(unit, tz="UTC"))})
        relation = DuckdbRelation.from_arrow(connection, table)
        expected = "ns" if str(relation.types[0]).lower().endswith("_ns") else "us"
        assert column_semantics(relation, "ts").unit == expected

    def test_native_timestamptz_without_zone_is_aware_with_unknown_zone(self, connection: Any) -> None:
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        relation = DuckdbRelation(connection, connection.sql("SELECT TIMESTAMPTZ '2021-01-01 01:30:00+00' AS ts"))
        sem = column_semantics(relation, "ts")
        assert sem.is_tz_aware is True
        assert sem.tz is None

    def test_naive_timestamp_has_no_zone(self, connection: Any) -> None:
        sem = column_semantics(connection.sql("SELECT TIMESTAMP '2021-01-01 01:30:00' AS ts"), "ts")
        assert sem.is_tz_aware is False
        assert sem.tz is None

    def test_column_named_time_zone_does_not_shadow_relation_attribute(self, connection: Any) -> None:
        rel = connection.sql("SELECT TIMESTAMPTZ '2021-01-01 01:30:00+00' AS ts, 'x' AS time_zone")
        sem = column_semantics(rel, "ts")
        assert sem.is_tz_aware is True
        assert sem.tz is None
