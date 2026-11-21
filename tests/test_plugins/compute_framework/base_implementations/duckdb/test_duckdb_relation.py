"""Tests for the DuckdbRelation class."""

from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

import pyarrow as pa
import pytest

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import (
    DuckdbRelation,
)
from mloda._testing.compute_framework.relation_suite import SqlRelationWindowSuite
from mloda._testing.framework_adapter import DuckDBAdapter, FrameworkAdapter


def _naive_ts(*hours: int) -> list[datetime]:
    return [datetime(2021, 1, 1, hour, 30) for hour in hours]


def _tz_table(unit: str = "us", zone: str = "+05:00") -> pa.Table:
    return pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "ts": pa.array(_naive_ts(1, 2, 3), type=pa.timestamp(unit, tz=zone)),
        }
    )


_ZONES = ["+05:00", "-03:30", "UTC", "Europe/Berlin"]

_ZONE_PRESERVING_OPS: list[tuple[str, Callable[[DuckdbRelation], DuckdbRelation], str, str | None]] = [
    ("filter", lambda r: r.filter("id > 1"), "ts", "+05:00"),
    ("select", lambda r: r.select("ts", "id"), "ts", "+05:00"),
    ("order", lambda r: r.order("id"), "ts", "+05:00"),
    ("limit", lambda r: r.limit(2), "ts", "+05:00"),
    ("set_alias", lambda r: r.set_alias("a"), "ts", "+05:00"),
    ("project_same_name", lambda r: r.project('"ts" AS "ts", "id"'), "ts", "+05:00"),
    ("aggregate", lambda r: r.aggregate('max("ts") AS "ts"'), "ts", "+05:00"),
    ("query", lambda r: r.query("v", 'SELECT "ts", "id" FROM v'), "ts", "+05:00"),
    ("window", lambda r: r.window("COUNT(*)", "w"), "ts", "+05:00"),
    ("project_rename", lambda r: r.project('"ts" AS "ts2"'), "ts2", None),
    ("project_varchar_cast", lambda r: r.project('CAST("ts" AS VARCHAR) AS "ts"'), "ts", None),
]


class TestDuckdbRelation(SqlRelationWindowSuite):
    adapter_class = DuckDBAdapter

    def raw_projection(self, relation: Any, sql: str) -> Any:
        return relation.project(sql)

    def test_types_exposes_relation_types_aligned_with_columns(self, connection: Any) -> None:
        arrow = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "score": pa.array([1.5, 2.5], type=pa.float64()),
                "name": pa.array(["Alice", "Bob"], type=pa.string()),
                "flag": pa.array([True, False], type=pa.bool_()),
            }
        )
        rel = DuckdbRelation.from_arrow(connection, arrow)

        assert list(zip(rel.columns, (str(dtype) for dtype in rel.types), strict=True)) == [
            ("id", "BIGINT"),
            ("score", "DOUBLE"),
            ("name", "VARCHAR"),
            ("flag", "BOOLEAN"),
        ]

    def test_types_aligned_with_columns_after_filter(self, sample_relation: "DuckdbRelation") -> None:
        filtered = sample_relation.filter("age >= ?", (35,))

        assert list(zip(filtered.columns, (str(dtype) for dtype in filtered.types), strict=True)) == [
            ("id", "BIGINT"),
            ("age", "BIGINT"),
            ("name", "VARCHAR"),
            ("category", "VARCHAR"),
        ]

    def test_types_reordered_after_select_projection(self, connection: Any) -> None:
        arrow = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "score": pa.array([1.5, 2.5], type=pa.float64()),
                "name": pa.array(["Alice", "Bob"], type=pa.string()),
            }
        )
        rel = DuckdbRelation.from_arrow(connection, arrow)

        projected = rel.select("score", "id")

        assert list(zip(projected.columns, (str(dtype) for dtype in projected.types), strict=True)) == [
            ("score", "DOUBLE"),
            ("id", "BIGINT"),
        ]

    def test_types_after_append_column(self, connection: Any) -> None:
        base = DuckdbRelation.from_arrow(connection, pa.table({"id": pa.array([1, 2], type=pa.int64())}))

        result = base.append_column("flag", [True, False])

        assert list(zip(result.columns, (str(dtype) for dtype in result.types), strict=True)) == [
            ("id", "BIGINT"),
            ("flag", "BOOLEAN"),
        ]

    def test_types_after_inner_join(self, connection: Any) -> None:
        left = DuckdbRelation.from_dict(connection, {"idx": [1, 2, 3], "val": ["a", "b", "c"]})
        right = DuckdbRelation.from_dict(connection, {"idx": [2, 3, 4], "score": [10, 20, 30]})

        left_aliased = left.set_alias("left_rel")
        right_aliased = right.set_alias("right_rel")

        joined = left_aliased.join(right_aliased, "left_rel.idx = right_rel.idx", how="inner")

        assert list(zip(joined.columns, (str(dtype) for dtype in joined.types), strict=True)) == [
            ("idx", "BIGINT"),
            ("val", "VARCHAR"),
            ("score", "BIGINT"),
        ]

    # DuckDB-specific tests

    def test_stores_connection_and_relation(self, connection: Any) -> None:
        native_rel = connection.from_arrow(pa.table({"x": [1, 2]}))
        rel = DuckdbRelation(connection, native_rel)
        assert rel.connection is connection
        assert rel.columns == ["x"]

    def test_drop_is_noop(self, sample_relation: "DuckdbRelation") -> None:
        sample_relation.drop()
        assert len(sample_relation) == 5

    def test_right_join(self, connection: Any) -> None:
        left = DuckdbRelation.from_dict(connection, {"idx": [1, 2, 3], "val": ["a", "b", "c"]})
        right = DuckdbRelation.from_dict(connection, {"idx": [2, 3, 4], "score": [10, 20, 30]})

        left_aliased = left.set_alias("left_rel")
        right_aliased = right.set_alias("right_rel")

        result = left_aliased.join(right_aliased, "left_rel.idx = right_rel.idx", how="right")
        assert len(result) == 3

    def test_outer_join(self, connection: Any) -> None:
        left = DuckdbRelation.from_dict(connection, {"idx": [1, 2, 3], "val": ["a", "b", "c"]})
        right = DuckdbRelation.from_dict(connection, {"idx": [2, 3, 4], "score": [10, 20, 30]})

        left_aliased = left.set_alias("left_rel")
        right_aliased = right.set_alias("right_rel")

        result = left_aliased.join(right_aliased, "left_rel.idx = right_rel.idx", how="outer")
        assert len(result) == 4

    def test_join_with_sql_injection_alias(self, connection: Any) -> None:
        """A crafted alias must not be interpreted as SQL."""
        left = DuckdbRelation.from_dict(connection, {"idx": [1, 2], "val": ["a", "b"]})
        right = DuckdbRelation.from_dict(connection, {"idx": [1, 2], "score": [10, 20]})

        malicious_alias = 'a" JOIN duckdb_tables() --'
        left_aliased = left.set_alias(malicious_alias)
        right_aliased = right.set_alias("right_rel")

        result = left_aliased.join(right_aliased, f"{malicious_alias}.idx = right_rel.idx", how="inner")
        assert len(result) == 2
        assert "val" in result.columns
        assert "score" in result.columns

    # --- Order ---

    def test_order_single_column(self, sample_relation: "DuckdbRelation", adapter: FrameworkAdapter) -> None:
        ordered = sample_relation.order("age")
        ids = self.column_values(adapter, ordered, "id")
        assert ids == [1, 2, 3, 4, 5]

    def test_order_single_column_desc(self, sample_relation: "DuckdbRelation", adapter: FrameworkAdapter) -> None:
        ordered = sample_relation.order("age DESC")
        ids = self.column_values(adapter, ordered, "id")
        assert ids == [5, 4, 3, 2, 1]

    def test_order_multiple_columns(self, sample_relation: "DuckdbRelation", adapter: FrameworkAdapter) -> None:
        ordered = sample_relation.order("category", "age")
        categories = self.column_values(adapter, ordered, "category")
        ages = self.column_values(adapter, ordered, "age")
        assert categories == ["A", "A", "B", "B", "C"]
        assert ages == [25, 35, 30, 45, 40]

    def test_order_preserves_all_columns(self, sample_relation: "DuckdbRelation") -> None:
        ordered = sample_relation.order("age")
        assert ordered.columns == ["id", "age", "name", "category"]

    def test_order_preserves_row_count(self, sample_relation: "DuckdbRelation") -> None:
        ordered = sample_relation.order("age")
        assert len(ordered) == 5

    def test_order_does_not_mutate_original(self, sample_relation: "DuckdbRelation", adapter: FrameworkAdapter) -> None:
        sample_relation.order("age DESC")
        ids = self.column_values(adapter, sample_relation, "id")
        assert ids == [1, 2, 3, 4, 5]

    def test_order_returns_duckdb_relation(self, sample_relation: "DuckdbRelation") -> None:
        ordered = sample_relation.order("age")
        assert isinstance(ordered, DuckdbRelation)

    # --- Project (public raw projection) ---

    def test_project_basic_expression(self, sample_relation: "DuckdbRelation", adapter: FrameworkAdapter) -> None:
        result = sample_relation.project("id, age * 2 AS double_age")
        ages = self.column_values(adapter, result, "double_age")
        assert ages == [50, 60, 70, 80, 90]

    def test_project_returns_duckdb_relation(self, sample_relation: "DuckdbRelation") -> None:
        result = sample_relation.project("id")
        assert isinstance(result, DuckdbRelation)

    def test_project_preserves_row_count(self, sample_relation: "DuckdbRelation") -> None:
        result = sample_relation.project("id, age")
        assert len(result) == 5

    # --- Aggregate ---

    def test_aggregate_with_group_by(self, sample_relation: "DuckdbRelation", adapter: FrameworkAdapter) -> None:
        result = sample_relation.aggregate("category, COUNT(*) AS n", "category").order("category")
        cats = self.column_values(adapter, result, "category")
        counts = self.column_values(adapter, result, "n")
        assert cats == ["A", "B", "C"]
        assert counts == [2, 2, 1]

    def test_aggregate_without_group_by(self, sample_relation: "DuckdbRelation", adapter: FrameworkAdapter) -> None:
        result = sample_relation.aggregate("COUNT(*) AS total")
        assert self.column_values(adapter, result, "total") == [5]

    def test_aggregate_returns_duckdb_relation(self, sample_relation: "DuckdbRelation") -> None:
        result = sample_relation.aggregate("COUNT(*) AS total")
        assert isinstance(result, DuckdbRelation)

    # --- Query ---

    def test_query_basic(self, sample_relation: "DuckdbRelation", adapter: FrameworkAdapter) -> None:
        result = sample_relation.query("data", "SELECT id, age FROM data WHERE age >= 35 ORDER BY id")
        ids = self.column_values(adapter, result, "id")
        assert ids == [3, 4, 5]

    def test_query_raises_value_error_when_no_relation_returned(self, connection: Any) -> None:
        """A statement that yields no relation must raise, not be wrapped."""

        class NoRelation:
            def query(self, view_name: str, sql: str) -> None:
                return None

        rel = DuckdbRelation(connection, cast(Any, NoRelation()))
        with pytest.raises(ValueError, match="SELECT"):
            rel.query("data", "SET threads = 1")

    def test_query_with_window_function(self, sample_relation: "DuckdbRelation", adapter: FrameworkAdapter) -> None:
        result = sample_relation.query(
            "data",
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY category ORDER BY age) AS rn FROM data ORDER BY id",
        )
        rns = self.column_values(adapter, result, "rn")
        assert rns == [1, 1, 2, 1, 2]

    def test_query_returns_duckdb_relation(self, sample_relation: "DuckdbRelation") -> None:
        result = sample_relation.query("data", "SELECT * FROM data LIMIT 1")
        assert isinstance(result, DuckdbRelation)

    # --- Injection-contract (verbatim passthrough) ---

    def test_project_passes_expression_verbatim(
        self, sample_relation: "DuckdbRelation", adapter: FrameworkAdapter
    ) -> None:
        """project() must pass the caller's expression unchanged; quoting would turn function calls into string literals."""
        result = sample_relation.project("UPPER(name) AS upper_name")
        assert self.column_values(adapter, result, "upper_name") == ["ALICE", "BOB", "CHARLIE", "DAVID", "EVE"]

    def test_aggregate_passes_expression_verbatim(
        self, sample_relation: "DuckdbRelation", adapter: FrameworkAdapter
    ) -> None:
        """aggregate() must pass the caller's SQL fragment unchanged; quoting would break FILTER (WHERE ...) syntax."""
        result = sample_relation.aggregate("COUNT(*) FILTER (WHERE age > 30) AS over_30")
        assert self.column_values(adapter, result, "over_30") == [3]

    def test_query_passes_sql_verbatim(self, sample_relation: "DuckdbRelation", adapter: FrameworkAdapter) -> None:
        """query() must pass the caller's SQL string unchanged; quoting would prevent execution of the SELECT statement."""
        result = sample_relation.query("t", "SELECT UPPER(name) AS upper_name FROM t ORDER BY upper_name")
        assert self.column_values(adapter, result, "upper_name") == ["ALICE", "BOB", "CHARLIE", "DAVID", "EVE"]

    # --- append_column: helper-name collision (issue #405 subtask 2) ---

    def test_append_column_when_existing_column_named_mloda_rn_zero(self, connection: Any) -> None:
        """Helper picker must skip __mloda_rn0__ if already present and use __mloda_rn1__."""
        rel = DuckdbRelation.from_dict(connection, {"__mloda_rn0__": [1, 2, 3], "b": [4, 5, 6]})
        result = rel.append_column("c", [7, 8, 9])
        assert set(result.columns) == {"__mloda_rn0__", "b", "c"}
        arrow = result.to_arrow_table()
        assert arrow.column("__mloda_rn0__").to_pylist() == [1, 2, 3]
        assert arrow.column("c").to_pylist() == [7, 8, 9]

    def test_append_column_when_existing_column_named_mloda_rn_legacy(self, connection: Any) -> None:
        """Helper picker must not collide with a pre-existing __mloda_rn__ column (the OLD hardcoded name)."""
        rel = DuckdbRelation.from_dict(connection, {"__mloda_rn__": ["x", "y", "z"], "b": [4, 5, 6]})
        result = rel.append_column("c", [7, 8, 9])
        assert set(result.columns) == {"__mloda_rn__", "b", "c"}
        arrow = result.to_arrow_table()
        assert arrow.column("__mloda_rn__").to_pylist() == ["x", "y", "z"]
        assert arrow.column("c").to_pylist() == [7, 8, 9]

    def test_from_arrow_records_each_zone_verbatim(self, connection: Any) -> None:
        columns: dict[str, Any] = {
            f"z{i}": pa.array(_naive_ts(1), type=pa.timestamp("us", tz=zone)) for i, zone in enumerate(_ZONES)
        }
        columns["naive"] = pa.array(_naive_ts(1), type=pa.timestamp("us"))
        columns["num"] = pa.array([1], type=pa.int64())
        rel = DuckdbRelation.from_arrow(connection, pa.table(columns))

        assert [rel.time_zone(f"z{i}") for i in range(len(_ZONES))] == _ZONES
        assert rel.time_zone("naive") is None
        assert rel.time_zone("num") is None

    @pytest.mark.parametrize("unit", ["us", "ns"])
    def test_to_arrow_table_reattaches_zone_with_same_instants(self, connection: Any, unit: str) -> None:
        source = _tz_table(unit)
        rel = DuckdbRelation.from_arrow(connection, source)
        duckdb_unit = rel._relation.arrow().read_all().schema.field("ts").type.unit
        result = rel.to_arrow_table()

        assert result.schema.field("ts").type == pa.timestamp(duckdb_unit, tz="+05:00")
        assert result.column("ts").cast(source.schema.field("ts").type).equals(source.column("ts"))

    def test_to_arrow_table_zero_rows_keeps_zone(self, connection: Any) -> None:
        rel = DuckdbRelation.from_arrow(connection, _tz_table().slice(0, 0))

        assert rel.to_arrow_table().schema.field("ts").type == pa.timestamp("us", tz="+05:00")

    @pytest.mark.parametrize(
        ("op", "column", "expected"),
        [pytest.param(op, column, expected, id=name) for name, op, column, expected in _ZONE_PRESERVING_OPS],
    )
    def test_zone_survival_across_operations(
        self,
        connection: Any,
        op: Callable[[DuckdbRelation], DuckdbRelation],
        column: str,
        expected: str | None,
    ) -> None:
        result = op(DuckdbRelation.from_arrow(connection, _tz_table()))

        assert result.time_zone(column) == expected

    @pytest.mark.parametrize("how", ["inner", "left", "outer", "right"])
    def test_join_keeps_both_sides_zones_and_shared_column_follows_side(self, connection: Any, how: str) -> None:
        def side(zone: str, own: str) -> DuckdbRelation:
            table = pa.table(
                {
                    "k": pa.array([1, 2], type=pa.int64()),
                    "ts": pa.array(_naive_ts(1, 2), type=pa.timestamp("us", tz=zone)),
                    own: pa.array(_naive_ts(3, 4), type=pa.timestamp("us", tz=zone)),
                }
            )
            return DuckdbRelation.from_arrow(connection, table)

        left = side("+05:00", "lts").set_alias("l")
        right = side("-03:30", "rts").set_alias("r")

        joined = left.join(right, "l.k = r.k", how=how)

        assert joined.time_zone("lts") == "+05:00"
        assert joined.time_zone("rts") == "-03:30"
        assert joined.time_zone("ts") == ("-03:30" if how == "right" else "+05:00")

    @pytest.mark.parametrize("how", ["inner", "left", "outer", "right"])
    def test_join_shared_column_without_kept_side_zone_does_not_borrow_other_zone(
        self, connection: Any, how: str
    ) -> None:
        native = connection.sql("SELECT 1::BIGINT AS k, TIMESTAMPTZ '2021-01-01 01:30:00+00' AS ts")
        left = DuckdbRelation(connection, native).set_alias("l")
        right = DuckdbRelation.from_arrow(
            connection,
            pa.table(
                {
                    "k": pa.array([1], type=pa.int64()),
                    "ts": pa.array(_naive_ts(1), type=pa.timestamp("us", tz="+05:00")),
                }
            ),
        ).set_alias("r")

        joined = left.join(right, "l.k = r.k", how=how)

        assert joined.time_zone("ts") == ("+05:00" if how == "right" else None)

    @pytest.mark.parametrize("projection", ['"a", "b" AS "a"', '"b" AS "a", "a"'], ids=["ts_first", "ts_last"])
    def test_duplicate_output_names_with_zoned_column_materialize(self, connection: Any, projection: str) -> None:
        table = pa.table(
            {
                "a": pa.array(_naive_ts(1, 2), type=pa.timestamp("us", tz="+05:00")),
                "b": pa.array([1, 2], type=pa.int64()),
            }
        )
        rel = DuckdbRelation.from_arrow(connection, table).project(projection)

        result = rel.to_arrow_table()

        assert result.num_columns == 2
        for field in result.schema:
            if pa.types.is_timestamp(field.type):
                assert field.type == pa.timestamp("us", tz="+05:00")

    def test_append_columns_keeps_existing_zones(self, connection: Any) -> None:
        rel = DuckdbRelation.from_arrow(connection, _tz_table())

        result = rel.append_columns({"extra": [7, 8, 9]})

        assert result.time_zone("ts") == "+05:00"
        assert result.to_arrow_table().schema.field("ts").type == pa.timestamp("us", tz="+05:00")

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda r: r.time_zone("missing"), id="time_zone"),
            pytest.param(lambda r: r.with_time_zone("missing", "UTC"), id="with_time_zone"),
        ],
    )
    def test_zone_calls_on_unknown_column_raise(self, connection: Any, call: Callable[[DuckdbRelation], Any]) -> None:
        rel = DuckdbRelation.from_arrow(connection, _tz_table())

        with pytest.raises(ValueError):
            call(rel)

    def test_with_time_zone_on_non_tz_column_raises(self, connection: Any) -> None:
        rel = DuckdbRelation.from_arrow(connection, _tz_table())

        with pytest.raises(ValueError):
            rel.with_time_zone("id", "UTC")

    def test_with_time_zone_sets_zone_on_derived_column(self, connection: Any) -> None:
        rel = DuckdbRelation.from_arrow(connection, _tz_table(zone="UTC"))
        derived = rel.project('"id", "ts" + INTERVAL 1 HOUR AS "ts2"')

        assert derived.time_zone("ts2") is None
        result = derived.with_time_zone("ts2", "Europe/Berlin")

        assert result.time_zone("ts2") == "Europe/Berlin"
        out = result.to_arrow_table()
        assert out.schema.field("ts2").type == pa.timestamp("us", tz="Europe/Berlin")
        expected = _tz_table(zone="UTC").column("ts").cast(pa.int64()).to_pylist()
        assert [v + 3_600_000_000 for v in expected] == out.column("ts2").cast(pa.int64()).to_pylist()

    def test_native_timestamptz_has_no_recorded_zone_and_reads_back_unchanged(self, connection: Any) -> None:
        native = connection.sql("SELECT TIMESTAMPTZ '2021-01-01 01:30:00+00' AS ts")
        rel = DuckdbRelation(connection, native)

        assert rel.time_zone("ts") is None
        assert rel.to_arrow_table().schema == native.arrow().read_all().schema
