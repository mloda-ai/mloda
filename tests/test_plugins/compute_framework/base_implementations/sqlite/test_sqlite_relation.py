import datetime
import sqlite3
import warnings
from typing import Any

import pyarrow as pa
import pytest

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import (
    SqliteRelation,
    _infer_sqlite_type_from_values,
)
from tests.test_plugins.compute_framework.base_implementations.relation_test_mixin import (
    RelationTestMixin,
)


class TestSqliteRelation(RelationTestMixin):
    @pytest.fixture
    def sample_relation(self, connection: sqlite3.Connection) -> SqliteRelation:
        arrow = pa.Table.from_pydict(
            {
                "id": [1, 2, 3, 4, 5],
                "age": [25, 30, 35, 40, 45],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "category": ["A", "B", "A", "C", "B"],
            }
        )
        return SqliteRelation.from_arrow(connection, arrow)

    @pytest.fixture
    def relation_class(self) -> Any:
        return SqliteRelation

    def get_column_values(self, result: Any, column: str) -> list[Any]:
        values: list[Any] = result.df()[column].tolist()
        return values

    def test_types_exposes_arrow_schema_from_arrow_aligned_with_columns(self, connection: sqlite3.Connection) -> None:
        arrow = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "score": pa.array([1.5, 2.5], type=pa.float64()),
                "name": pa.array(["Alice", "Bob"], type=pa.string()),
                "flag": pa.array([True, False], type=pa.bool_()),
            }
        )
        rel = SqliteRelation.from_arrow(connection, arrow)

        assert list(zip(rel.columns, rel.types, strict=True)) == [
            ("id", pa.int64()),
            ("score", pa.float64()),
            ("name", pa.string()),
            ("flag", pa.bool_()),
        ]

    def test_to_arrow_table_preserves_arrow_schema_for_non_empty_from_arrow(
        self, connection: sqlite3.Connection
    ) -> None:
        arrow = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int32()),
                "score": pa.array([1.5, 2.5], type=pa.float32()),
                "flag": pa.array([True, False], type=pa.bool_()),
            }
        )
        rel = SqliteRelation.from_arrow(connection, arrow)

        result = rel.to_arrow_table()
        flag_values = result.column("flag").to_pylist()

        assert {
            "schema": result.schema,
            "flag_value_types": [type(value) for value in flag_values],
            "flag_values": flag_values,
        } == {
            "schema": arrow.schema,
            "flag_value_types": [bool, bool],
            "flag_values": [True, False],
        }

    def test_raw_sql_star_projection_preserves_passthrough_arrow_types(self, connection: sqlite3.Connection) -> None:
        timestamps = [
            datetime.datetime(2024, 1, 1, 12, 0, 0, 123456),
            datetime.datetime(2024, 1, 2, 13, 30, 0, 654321),
        ]
        arrow = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int32()),
                "flag": pa.array([True, False], type=pa.bool_()),
                "ts": pa.array(timestamps, type=pa.timestamp("us")),
            }
        )
        rel = SqliteRelation.from_arrow(connection, arrow)

        projected = rel.select(_raw_sql="*, id + 1 AS next_id")
        result = projected.to_arrow_table()

        projected_types_by_name = dict(zip(projected.columns, projected.types, strict=True))
        next_id_relation_type = projected_types_by_name["next_id"]
        next_id_materialized_type = result.schema.field("next_id").type
        flag_values = result.column("flag").to_pylist()

        assert {
            "projected_passthrough_types": [projected_types_by_name[column] for column in arrow.column_names],
            "materialized_passthrough_types": [result.schema.field(column).type for column in arrow.column_names],
            "next_id_relation_type": next_id_relation_type,
            "next_id_materialized_type": next_id_materialized_type,
            "next_id_relation_is_string": pa.types.is_string(next_id_relation_type),
            "next_id_relation_is_numeric": pa.types.is_integer(next_id_relation_type)
            or pa.types.is_floating(next_id_relation_type),
            "flag_values": flag_values,
            "flag_value_types": [type(value) for value in flag_values],
            "ts_values": result.column("ts").to_pylist(),
        } == {
            "projected_passthrough_types": [pa.int32(), pa.bool_(), pa.timestamp("us")],
            "materialized_passthrough_types": [pa.int32(), pa.bool_(), pa.timestamp("us")],
            "next_id_relation_type": next_id_materialized_type,
            "next_id_materialized_type": pa.int64(),
            "next_id_relation_is_string": False,
            "next_id_relation_is_numeric": True,
            "flag_values": [True, False],
            "flag_value_types": [bool, bool],
            "ts_values": timestamps,
        }

    def test_raw_sql_expression_type_survives_empty_filter(self, connection: sqlite3.Connection) -> None:
        rel = SqliteRelation.from_arrow(
            connection,
            pa.table({"id": pa.array([1, 2], type=pa.int32())}),
        )

        projected = rel.select(_raw_sql="*, id + 1 AS next_id")
        empty = projected.filter("id > ?", (10,))
        types_by_name = dict(zip(empty.columns, empty.types, strict=True))
        result = empty.to_arrow_table()

        assert {
            "row_count": result.num_rows,
            "relation_next_id_type": types_by_name["next_id"],
            "arrow_next_id_type": result.schema.field("next_id").type,
        } == {
            "row_count": 0,
            "relation_next_id_type": pa.int64(),
            "arrow_next_id_type": pa.int64(),
        }

    def test_append_column_preserves_bool_arrow_type(self, connection: sqlite3.Connection) -> None:
        base = SqliteRelation.from_arrow(connection, pa.table({"id": pa.array([1, 2], type=pa.int32())}))

        result = base.append_column("flag", [True, False])
        arrow = result.to_arrow_table()
        types_by_name = dict(zip(result.columns, result.types, strict=True))
        flag_values = arrow.column("flag").to_pylist()

        assert {
            "relation_flag_type": types_by_name["flag"],
            "arrow_flag_type": arrow.schema.field("flag").type,
            "flag_values": flag_values,
            "flag_value_types": [type(value) for value in flag_values],
        } == {
            "relation_flag_type": pa.bool_(),
            "arrow_flag_type": pa.bool_(),
            "flag_values": [True, False],
            "flag_value_types": [bool, bool],
        }

    def test_from_dict_preserves_bytes_as_large_binary(self, connection: sqlite3.Connection) -> None:
        rel = SqliteRelation.from_dict(connection, {"payload": [b"abc", b"\xff\x00"]})
        expected_type = pa.large_binary()

        types_by_name = dict(zip(rel.columns, rel.types, strict=True))
        assert types_by_name["payload"] == expected_type

        arrow = rel.to_arrow_table()
        assert arrow.schema.field("payload").type == expected_type
        assert arrow.column("payload").to_pylist() == [b"abc", b"\xff\x00"]

    def test_raw_sql_all_null_expression_type_is_order_independent(self, connection: sqlite3.Connection) -> None:
        rel = SqliteRelation.from_arrow(connection, pa.table({"id": pa.array([1, 2], type=pa.int32())}))

        materialized_first = rel.select(_raw_sql="*, NULL AS empty_value")
        materialized_first_arrow = materialized_first.to_arrow_table()
        materialized_first_types = dict(zip(materialized_first.columns, materialized_first.types, strict=True))

        types_first = rel.select(_raw_sql="*, NULL AS empty_value")
        types_first_types = dict(zip(types_first.columns, types_first.types, strict=True))
        types_first_arrow = types_first.to_arrow_table()

        assert {
            "materialized_first_relation_type": materialized_first_types["empty_value"],
            "materialized_first_arrow_type": materialized_first_arrow.schema.field("empty_value").type,
            "types_first_relation_type": types_first_types["empty_value"],
            "types_first_arrow_type": types_first_arrow.schema.field("empty_value").type,
            "materialized_first_values": materialized_first_arrow.column("empty_value").to_pylist(),
            "types_first_values": types_first_arrow.column("empty_value").to_pylist(),
        } == {
            "materialized_first_relation_type": pa.string(),
            "materialized_first_arrow_type": pa.string(),
            "types_first_relation_type": pa.string(),
            "types_first_arrow_type": pa.string(),
            "materialized_first_values": [None, None],
            "types_first_values": [None, None],
        }

    def test_drop_removes_table(self, sample_relation: SqliteRelation) -> None:
        """drop() should drop the underlying temp table so it can no longer be queried."""
        table_name = sample_relation.table_name
        sample_relation.drop()
        with pytest.raises(sqlite3.OperationalError):
            sample_relation._connection.execute(f'SELECT * FROM "{table_name}"').fetchall()  # nosec

    def test_empty_table_preserves_column_types(self, connection: sqlite3.Connection) -> None:
        """Empty table converted to Arrow must preserve the original column types, not default to string."""
        arrow = pa.Table.from_pydict(
            {
                "id": pa.array([], type=pa.int64()),
                "score": pa.array([], type=pa.float64()),
                "name": pa.array([], type=pa.string()),
            }
        )
        rel = SqliteRelation.from_arrow(connection, arrow)
        result = rel.to_arrow_table()
        assert result.num_rows == 0
        assert pa.types.is_integer(result.schema.field("id").type), (
            f"Expected int type for 'id', got {result.schema.field('id').type}"
        )
        assert pa.types.is_floating(result.schema.field("score").type), (
            f"Expected float type for 'score', got {result.schema.field('score').type}"
        )
        assert pa.types.is_string(result.schema.field("name").type) or pa.types.is_large_string(
            result.schema.field("name").type
        ), f"Expected string type for 'name', got {result.schema.field('name').type}"

    def test_outer_join_shared_key_uses_wider_right_dtype(self, connection: sqlite3.Connection) -> None:
        right_only_idx = 2**40
        left = SqliteRelation.from_arrow(connection, pa.table({"idx": pa.array([1], type=pa.int32())}))
        right = SqliteRelation.from_arrow(connection, pa.table({"idx": pa.array([right_only_idx], type=pa.int64())}))

        result = left.join(right, "idx", how="outer")
        arrow = result.to_arrow_table()
        types_by_name = dict(zip(result.columns, result.types, strict=True))

        assert {
            "relation_idx_type": types_by_name["idx"],
            "arrow_idx_type": arrow.schema.field("idx").type,
            "idx_values": sorted(arrow.column("idx").to_pylist()),
        } == {
            "relation_idx_type": pa.int64(),
            "arrow_idx_type": pa.int64(),
            "idx_values": [1, right_only_idx],
        }

    def test_join_with_sql_injection_alias(self, connection: sqlite3.Connection) -> None:
        """A crafted alias must not be interpreted as SQL."""
        left = SqliteRelation.from_dict(connection, {"idx": [1, 2], "val": ["a", "b"]})
        right = SqliteRelation.from_dict(connection, {"idx": [1, 2], "score": [10, 20]})

        malicious_alias = 'a" JOIN sqlite_master --'
        left_aliased = left.set_alias(malicious_alias)
        right_aliased = right.set_alias("right_rel")

        result = left_aliased.join(right_aliased, f"{malicious_alias}.idx = right_rel.idx", how="inner")
        assert len(result) == 2
        assert "val" in result.columns
        assert "score" in result.columns


class TestSqliteRelationTableNameQuoting:
    def test_table_name_with_embedded_quote(self, connection: sqlite3.Connection) -> None:
        """A table name with an embedded double-quote must still be handled correctly."""
        tricky_name = 'table"name'
        conn = connection
        conn.execute(f"CREATE TEMP TABLE {quote_ident(tricky_name)} (x INTEGER)")
        conn.execute(f"INSERT INTO {quote_ident(tricky_name)} VALUES (42)")  # nosec
        rel = SqliteRelation(conn, tricky_name)
        cols = rel.columns
        assert cols == ["x"]
        assert len(rel) == 1

    def test_quote_ident_prevents_sql_injection(self, connection: sqlite3.Connection) -> None:
        """quote_ident neutralizes a SQL injection payload used as a table name."""
        conn = connection
        injection_name = 'x"; DROP TABLE secret; --'

        conn.execute("CREATE TABLE secret (val TEXT)")
        conn.execute("INSERT INTO secret VALUES ('safe')")

        conn.execute(f"CREATE TABLE {quote_ident(injection_name)} (id INTEGER)")
        conn.execute(f"INSERT INTO {quote_ident(injection_name)} VALUES (1)")  # nosec
        rows = conn.execute("SELECT val FROM secret").fetchall()
        assert rows == [("safe",)]

        result = conn.execute(f"SELECT id FROM {quote_ident(injection_name)}").fetchall()  # nosec
        assert result == [(1,)]


class TestInferSqliteType:
    def test_int_before_float_returns_real(self) -> None:
        """When a float follows an int in the list, widest type (REAL) should win."""
        result = _infer_sqlite_type_from_values([1, 1.5, 2])
        assert result == "REAL", f"Expected REAL but got {result}"

    def test_none_then_float_returns_real(self) -> None:
        """None values should be skipped; float after None must win."""
        result = _infer_sqlite_type_from_values([None, 1, 2.5])
        assert result == "REAL", f"Expected REAL but got {result}"

    def test_all_ints_returns_integer(self) -> None:
        result = _infer_sqlite_type_from_values([1, 2, 3])
        assert result == "INTEGER"

    def test_string_after_int_returns_text(self) -> None:
        result = _infer_sqlite_type_from_values([1, "hello"])
        assert result == "TEXT"

    def test_all_none_returns_text(self) -> None:
        result = _infer_sqlite_type_from_values([None, None])
        assert result == "TEXT"

    def test_string_before_float_returns_text(self) -> None:
        """TEXT dominates: a string before a float must not be overridden by the float."""
        result = _infer_sqlite_type_from_values(["hello", 1.5])
        assert result == "TEXT", f"Expected TEXT but got {result}"


class TestSqliteDatetimeAdapter:
    def test_inserting_datetime_does_not_emit_default_adapter_warning(self, connection: sqlite3.Connection) -> None:
        """Inserting datetime/date values must not trigger Python 3.12's default-adapter DeprecationWarning."""
        data: dict[str, list[Any]] = {
            "ts": [datetime.datetime(2024, 1, 1, 12, 0), datetime.datetime(2024, 1, 2, 12, 0)],
            "d": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)],
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            SqliteRelation.from_dict(connection, data)

    def test_inserting_datetime_via_from_arrow_does_not_emit_default_adapter_warning(
        self, connection: sqlite3.Connection
    ) -> None:
        """from_arrow's executemany path must also be free of the default-adapter DeprecationWarning."""
        arrow_table = pa.table(
            {
                "ts": pa.array(
                    [datetime.datetime(2024, 1, 1, 12, 0), datetime.datetime(2024, 1, 2, 12, 0)],
                    type=pa.timestamp("us"),
                ),
                "d": pa.array([datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)], type=pa.date32()),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            SqliteRelation.from_arrow(connection, arrow_table)

    def test_direct_executemany_with_datetime_does_not_emit_default_adapter_warning(self) -> None:
        """Module-global adapter registration must cover any sqlite3 connection, not only SqliteRelation."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (ts TEXT)")
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            conn.executemany(
                "INSERT INTO t VALUES (?)",
                [(datetime.datetime(2024, 1, 1, 12, 0),), (datetime.datetime(2024, 1, 2, 12, 0),)],
            )
        conn.close()
