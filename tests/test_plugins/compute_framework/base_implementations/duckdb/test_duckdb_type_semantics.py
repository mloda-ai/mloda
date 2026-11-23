"""Column-semantics introspection for duckdb relations (epic #518, Fix 3).

A duckdb ``INTERVAL`` type must be treated as temporal but NOT numeric. The
reader currently reports ``is_numeric=True`` because the substring ``"int"``
appears inside ``"interval"``. This test pins the correct expectation and
fails until the numeric-token matching is tightened.
"""

from typing import Any
from unittest.mock import patch
import logging

import pytest

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
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
class TestDuckdbTimezoneIdentity:
    @pytest.mark.parametrize("session_zone", ["UTC", "Asia/Kolkata"])
    @pytest.mark.parametrize("wrapped", [False, True], ids=["bare", "wrapper"])
    def test_timestamptz_without_source_zone_stays_unknown(
        self, connection: Any, session_zone: str, wrapped: bool
    ) -> None:
        connection.execute(f"SET TimeZone = '{session_zone}'")
        native = connection.sql("SELECT TIMESTAMPTZ '2024-01-01 00:00:00+05:30' AS ts")
        sem = column_semantics(DuckdbRelation(connection, native) if wrapped else native, "ts")
        assert sem.is_temporal is True
        assert sem.is_tz_aware is True
        assert getattr(sem, "tz") is None

    @pytest.mark.parametrize("zone", ["UTC", "Europe/Berlin", "Asia/Kolkata"])
    @pytest.mark.parametrize("values", [[0, None, 1], [None, None]], ids=["populated", "all-null"])
    def test_recorded_arrow_zone_is_read_without_conversion(
        self, connection: Any, zone: str, values: list[int | None]
    ) -> None:
        pa = pytest.importorskip("pyarrow")
        table = pa.table({"ts": pa.array(values, type=pa.timestamp("us", tz=zone))})
        relation = DuckdbRelation.from_arrow(connection, table)
        assert relation.time_zone("ts") == zone
        with (
            patch.object(DuckdbRelation, "to_arrow_table", side_effect=AssertionError("Must not convert to Arrow")),
            patch.object(DuckdbRelation, "df", side_effect=AssertionError("Must not convert to pandas")),
        ):
            sem = column_semantics(relation, "ts")
        assert sem.is_temporal is True
        assert sem.is_tz_aware is True
        assert getattr(sem, "tz") == zone

    def test_explicit_zone_is_read_without_executing_the_relation(self, connection: Any) -> None:
        native = connection.sql("SELECT error('Timezone inspection must not execute rows')::TIMESTAMPTZ AS ts")
        relation = DuckdbRelation(connection, native).with_time_zone("ts", "Europe/Berlin")
        sem = column_semantics(relation, "ts")
        assert sem.is_tz_aware is True
        assert getattr(sem, "tz") == "Europe/Berlin"

    @pytest.mark.parametrize("column", ["ts", "d", "duration", "n"])
    def test_naive_or_non_timestamp_type_has_no_zone(self, connection: Any, column: str) -> None:
        relation = connection.sql(
            "SELECT NULL::TIMESTAMP AS ts, DATE '2024-01-01' AS d, INTERVAL 1 HOUR AS duration, 1 AS n"
        )
        sem = column_semantics(relation, column)
        assert sem.is_tz_aware is False
        assert getattr(sem, "tz") is None
