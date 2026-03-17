import sqlite3
from typing import Any

import pyarrow as pa
import pytest

from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import (
    SqliteRelation,
    _infer_sqlite_type_from_values,
)


@pytest.fixture
def connection() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


@pytest.fixture
def sample_relation(connection: sqlite3.Connection) -> SqliteRelation:
    arrow = pa.Table.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "age": [25, 30, 35, 40, 45],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "category": ["A", "B", "A", "C", "B"],
        }
    )
    return SqliteRelation.from_arrow(connection, arrow)


class TestSqliteRelationBasics:
    def test_columns(self, sample_relation: SqliteRelation) -> None:
        assert sample_relation.columns == ["id", "age", "name", "category"]

    def test_len(self, sample_relation: SqliteRelation) -> None:
        assert len(sample_relation) == 5

    def test_to_arrow_table(self, sample_relation: SqliteRelation) -> None:
        result = sample_relation.to_arrow_table()
        assert isinstance(result, pa.Table)
        assert result.num_rows == 5
        assert set(result.column_names) == {"id", "age", "name", "category"}

    def test_df(self, sample_relation: SqliteRelation) -> None:
        df = sample_relation.df()
        assert len(df) == 5
        assert set(df.columns) == {"id", "age", "name", "category"}

    def test_filter(self, sample_relation: SqliteRelation) -> None:
        filtered = sample_relation.filter('"age" >= 35')
        assert len(filtered) == 3
        assert isinstance(filtered, SqliteRelation)
        # Original is unchanged
        assert len(sample_relation) == 5

    def test_select(self, sample_relation: SqliteRelation) -> None:
        selected = sample_relation.select("id", "name")
        assert selected.columns == ["id", "name"]
        assert len(selected) == 5

    def test_select_star_expression(self, sample_relation: SqliteRelation) -> None:
        selected = sample_relation.select(raw_sql="*, age * 2 AS doubled_age")
        assert "doubled_age" in selected.columns
        assert len(selected) == 5

    def test_limit(self, sample_relation: SqliteRelation) -> None:
        limited = sample_relation.limit(2)
        assert len(limited) == 2

    def test_set_alias(self, sample_relation: SqliteRelation) -> None:
        aliased = sample_relation.set_alias("my_alias")
        assert aliased.get_alias() == "my_alias"
        # Original does not have alias
        assert sample_relation.get_alias() is None
        # Data is still accessible
        assert len(aliased) == 5

    def test_drop_removes_table(self, sample_relation: SqliteRelation) -> None:
        """drop() should drop the underlying temp table so it can no longer be queried."""
        table_name = sample_relation.table_name
        sample_relation.drop()
        # After drop, querying the table should raise an error
        with pytest.raises(Exception):
            sample_relation._connection.execute(f'SELECT * FROM "{table_name}"').fetchall()  # nosec


class TestSqliteRelationSelectRawSql:
    """Tests for the explicit raw_sql keyword parameter on SqliteRelation.select()."""

    def test_select_raw_sql_window_function(self, sample_relation: SqliteRelation) -> None:
        """select(raw_sql=...) passes the expression through unchanged and the derived column is accessible."""
        selected = sample_relation.select(raw_sql="*, age * 2 AS doubled_age")
        assert "doubled_age" in selected.columns
        assert len(selected) == 5

    def test_select_raw_sql_aggregate_window(self, sample_relation: SqliteRelation) -> None:
        """select(raw_sql=...) supports window-function syntax such as AVG() OVER ()."""
        selected = sample_relation.select(raw_sql="*, AVG(age) OVER () AS avg_age")
        assert "avg_age" in selected.columns
        assert len(selected) == 5

    def test_select_columns_still_quoted(self, sample_relation: SqliteRelation) -> None:
        """Positional column names continue to be quoted via quote_ident() when raw_sql is not used."""
        selected = sample_relation.select("id", "name")
        assert selected.columns == ["id", "name"]
        assert len(selected) == 5


class TestSqliteRelationFromArrow:
    def test_empty_table(self, connection: sqlite3.Connection) -> None:
        arrow = pa.Table.from_pydict({"id": pa.array([], type=pa.int64()), "name": pa.array([], type=pa.string())})
        rel = SqliteRelation.from_arrow(connection, arrow)
        assert len(rel) == 0
        assert set(rel.columns) == {"id", "name"}

    def test_roundtrip(self, connection: sqlite3.Connection) -> None:
        original = pa.Table.from_pydict({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        rel = SqliteRelation.from_arrow(connection, original)
        result = rel.to_arrow_table()
        assert result.num_rows == 3
        assert set(result.column_names) == {"a", "b"}

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


class TestSqliteRelationFromDict:
    def test_basic(self, connection: sqlite3.Connection) -> None:
        data: dict[str, list[Any]] = {"x": [10, 20], "y": ["a", "b"]}
        rel = SqliteRelation.from_dict(connection, data)
        assert len(rel) == 2
        assert rel.columns == ["x", "y"]

    def test_empty_dict_raises(self, connection: sqlite3.Connection) -> None:
        empty: dict[str, list[Any]] = {}
        with pytest.raises(ValueError, match="Cannot create relation from empty dictionary"):
            SqliteRelation.from_dict(connection, empty)

    def test_select_raw_sql_executes_safely(self, connection: sqlite3.Connection) -> None:
        """select(raw_sql=...) with a window function executes without injection."""
        arrow = pa.Table.from_pydict({"id": [1, 2, 3], "val": [10, 20, 30]})
        rel = SqliteRelation.from_arrow(connection, arrow)
        result = rel.select(raw_sql="*, val * 2 AS doubled")
        assert "doubled" in result.columns
        assert sorted(result.df()["doubled"].tolist()) == [20, 40, 60]


class TestSqliteRelationJoin:
    def test_inner_join(self, connection: sqlite3.Connection) -> None:
        left = SqliteRelation.from_dict(connection, {"idx": [1, 2, 3], "val": ["a", "b", "c"]})
        right = SqliteRelation.from_dict(connection, {"idx": [2, 3, 4], "score": [10, 20, 30]})

        left_aliased = left.set_alias("left_rel")
        right_aliased = right.set_alias("right_rel")

        result = left_aliased.join(right_aliased, "left_rel.idx = right_rel.idx", how="inner")
        assert len(result) == 2

    def test_left_join(self, connection: sqlite3.Connection) -> None:
        left = SqliteRelation.from_dict(connection, {"idx": [1, 2, 3], "val": ["a", "b", "c"]})
        right = SqliteRelation.from_dict(connection, {"idx": [2, 3, 4], "score": [10, 20, 30]})

        left_aliased = left.set_alias("left_rel")
        right_aliased = right.set_alias("right_rel")

        result = left_aliased.join(right_aliased, "left_rel.idx = right_rel.idx", how="left")
        assert len(result) == 3

    def test_unsupported_join_type(self, connection: sqlite3.Connection) -> None:
        left = SqliteRelation.from_dict(connection, {"idx": [1]})
        right = SqliteRelation.from_dict(connection, {"idx": [1]})

        with pytest.raises(ValueError, match="Unsupported join type"):
            left.join(right, "1=1", how="cross")


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
