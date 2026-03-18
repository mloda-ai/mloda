import sqlite3
from typing import Any, List

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

    def get_column_values(self, result: Any, column: str) -> List[Any]:
        values: List[Any] = result.df()[column].tolist()
        return values

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
