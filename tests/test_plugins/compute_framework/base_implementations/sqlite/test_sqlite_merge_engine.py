import sqlite3
from typing import Any, Optional, Type

import pyarrow as pa
import pytest

from mloda.user import JoinType
from mloda.user import Index
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import _regexp
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_merge_engine import SqliteMergeEngine
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation
from tests.test_plugins.compute_framework.test_tooling.multi_index.multi_index_test_base import (
    MultiIndexMergeEngineTestBase,
)


@pytest.fixture
def index_obj() -> Index:
    return Index(("idx",))


@pytest.fixture
def left_data(connection: sqlite3.Connection) -> SqliteRelation:
    arrow = pa.Table.from_pydict({"idx": [1, 3], "col1": ["a", "b"]})
    return SqliteRelation.from_arrow(connection, arrow)


@pytest.fixture
def right_data(connection: sqlite3.Connection) -> SqliteRelation:
    arrow = pa.Table.from_pydict({"idx": [1, 2], "col2": ["x", "z"]})
    return SqliteRelation.from_arrow(connection, arrow)


class TestSqliteMergeEngine:
    def test_merge_inner(
        self, connection: sqlite3.Connection, left_data: SqliteRelation, right_data: SqliteRelation, index_obj: Index
    ) -> None:
        engine = SqliteMergeEngine(connection)
        result = engine.merge_inner(left_data, right_data, index_obj, index_obj)
        result_df = result.df()
        assert len(result_df) == 1
        assert result_df["idx"].tolist()[0] == 1

    def test_merge_left(
        self, connection: sqlite3.Connection, left_data: SqliteRelation, right_data: SqliteRelation, index_obj: Index
    ) -> None:
        engine = SqliteMergeEngine(connection)
        result = engine.merge_left(left_data, right_data, index_obj, index_obj)
        result_df = result.df()
        assert len(result_df) == 2
        result_df = result_df.sort_values("idx").reset_index(drop=True)
        assert result_df["idx"].tolist() == [1, 3]

    def test_merge_right(
        self, connection: sqlite3.Connection, left_data: SqliteRelation, right_data: SqliteRelation, index_obj: Index
    ) -> None:
        engine = SqliteMergeEngine(connection)
        result = engine.merge_right(left_data, right_data, index_obj, index_obj)
        result_df = result.df()
        assert len(result_df) == 2
        result_df = result_df.sort_values("idx").reset_index(drop=True)
        assert result_df["idx"].tolist() == [1, 2]

    def test_merge_append(
        self, connection: sqlite3.Connection, left_data: SqliteRelation, right_data: SqliteRelation, index_obj: Index
    ) -> None:
        engine = SqliteMergeEngine(connection)
        result = engine.merge_append(left_data, right_data, index_obj, index_obj)
        result_df = result.df()
        assert len(result_df) == 4
        assert "idx" in result_df.columns
        assert "col1" in result_df.columns
        assert "col2" in result_df.columns

    def test_merge_union(
        self, connection: sqlite3.Connection, left_data: SqliteRelation, right_data: SqliteRelation, index_obj: Index
    ) -> None:
        engine = SqliteMergeEngine(connection)
        result = engine.merge_union(left_data, right_data, index_obj, index_obj)
        result_df = result.df()
        assert len(result_df) <= 4
        assert "idx" in result_df.columns

    def test_merge_with_different_join_columns(self, connection: sqlite3.Connection) -> None:
        left = SqliteRelation.from_arrow(connection, pa.Table.from_pydict({"left_id": [1, 2], "col1": ["a", "b"]}))
        right = SqliteRelation.from_arrow(connection, pa.Table.from_pydict({"right_id": [1, 3], "col2": ["x", "z"]}))

        left_index = Index(("left_id",))
        right_index = Index(("right_id",))

        engine = SqliteMergeEngine(connection)
        result = engine.merge_inner(left, right, left_index, right_index)
        result_df = result.df()
        assert len(result_df) == 1

    def test_merge_with_empty_datasets(self, connection: sqlite3.Connection, index_obj: Index) -> None:
        empty = SqliteRelation.from_arrow(
            connection,
            pa.Table.from_pydict({"idx": pa.array([], type=pa.int64()), "col": pa.array([], type=pa.string())}),
        )
        non_empty = SqliteRelation.from_arrow(connection, pa.Table.from_pydict({"idx": [1], "col": ["a"]}))

        engine = SqliteMergeEngine(connection)

        result = engine.merge_inner(empty, non_empty, index_obj, index_obj)
        assert len(result.df()) == 0

        result = engine.merge_inner(non_empty, empty, index_obj, index_obj)
        assert len(result.df()) == 0

    def test_framework_connection_error(self) -> None:
        engine = SqliteMergeEngine()

        conn = sqlite3.connect(":memory:")
        data = SqliteRelation.from_dict(conn, {"idx": [1], "col": ["a"]})
        index_obj = Index(("idx",))

        with pytest.raises(ValueError, match="Framework connection not set"):
            engine.merge_inner(data, data, index_obj, index_obj)

    def test_merge_method_integration(
        self, connection: sqlite3.Connection, left_data: SqliteRelation, right_data: SqliteRelation, index_obj: Index
    ) -> None:
        engine = SqliteMergeEngine(connection)
        result = engine.merge(left_data, right_data, JoinType.INNER, index_obj, index_obj)
        assert len(result.df()) == 1

    def test_get_column_names(self, connection: sqlite3.Connection) -> None:
        engine = SqliteMergeEngine(connection)
        data = SqliteRelation.from_arrow(connection, pa.Table.from_pydict({"a": [1], "b": ["x"], "c": [1.0]}))
        cols = engine.get_column_names(data)
        assert set(cols) == {"a", "b", "c"}

    def test_is_empty_data(self, connection: sqlite3.Connection) -> None:
        engine = SqliteMergeEngine(connection)

        empty = SqliteRelation.from_arrow(connection, pa.Table.from_pydict({"col": pa.array([], type=pa.int64())}))
        assert engine.is_empty_data(empty) is True

        non_empty = SqliteRelation.from_arrow(connection, pa.Table.from_pydict({"col": [1]}))
        assert engine.is_empty_data(non_empty) is False

    def test_merge_union_sequential_calls_dont_conflict(
        self, connection: sqlite3.Connection, left_data: SqliteRelation, right_data: SqliteRelation, index_obj: Index
    ) -> None:
        """Two sequential merge_union calls must not silently use stale data."""
        engine = SqliteMergeEngine(connection)
        result1 = engine.merge_union(left_data, right_data, index_obj, index_obj)
        # Second merge with different data
        left2 = SqliteRelation.from_arrow(connection, pa.Table.from_pydict({"idx": [10, 20], "col1": ["x", "y"]}))
        right2 = SqliteRelation.from_arrow(connection, pa.Table.from_pydict({"idx": [30], "col2": ["z"]}))
        result2 = engine.merge_union(left2, right2, index_obj, index_obj)
        # result2 must contain idx=10,20,30 not the original idx=1,3,1,2
        idx_values2 = sorted(result2.df()["idx"].tolist())
        assert idx_values2 == [10, 20, 30], f"Expected [10, 20, 30] but got {idx_values2}"

    def test_left_join_with_empty_right_preserves_all_left_rows(
        self, connection: sqlite3.Connection, index_obj: Index
    ) -> None:
        """LEFT JOIN with empty right side should return ALL left rows (not empty)."""
        left = SqliteRelation.from_arrow(connection, pa.Table.from_pydict({"idx": [1, 2, 3], "val": ["a", "b", "c"]}))
        empty_right = SqliteRelation.from_arrow(
            connection,
            pa.Table.from_pydict({"idx": pa.array([], type=pa.int64()), "score": pa.array([], type=pa.float64())}),
        )
        engine = SqliteMergeEngine(connection)
        result = engine.merge_left(left, empty_right, index_obj, index_obj)
        result_df = result.df()
        assert len(result_df) == 3, f"Expected 3 rows (all left rows) but got {len(result_df)}"

    def test_right_join_with_empty_left_preserves_all_right_rows(
        self, connection: sqlite3.Connection, index_obj: Index
    ) -> None:
        """RIGHT JOIN with empty left side should return ALL right rows (not empty)."""
        empty_left = SqliteRelation.from_arrow(
            connection,
            pa.Table.from_pydict({"idx": pa.array([], type=pa.int64()), "val": pa.array([], type=pa.string())}),
        )
        right = SqliteRelation.from_arrow(connection, pa.Table.from_pydict({"idx": [1, 2], "score": [10.0, 20.0]}))
        engine = SqliteMergeEngine(connection)
        result = engine.merge_right(empty_left, right, index_obj, index_obj)
        result_df = result.df()
        assert len(result_df) == 2, f"Expected 2 rows (all right rows) but got {len(result_df)}"

    def test_outer_join_with_one_empty_side_preserves_non_empty(
        self, connection: sqlite3.Connection, index_obj: Index
    ) -> None:
        """FULL OUTER JOIN with one empty side should return all rows from the non-empty side."""
        left = SqliteRelation.from_arrow(connection, pa.Table.from_pydict({"idx": [1, 2], "val": ["a", "b"]}))
        empty_right = SqliteRelation.from_arrow(
            connection,
            pa.Table.from_pydict({"idx": pa.array([], type=pa.int64()), "score": pa.array([], type=pa.float64())}),
        )
        engine = SqliteMergeEngine(connection)
        result = engine.merge_full_outer(left, empty_right, index_obj, index_obj)
        result_df = result.df()
        assert len(result_df) == 2, f"Expected 2 rows from the non-empty side but got {len(result_df)}"

    def test_merge_full_outer(self, connection: sqlite3.Connection) -> None:
        """FULL OUTER JOIN: left=[1,2,3], right=[2,3,4] -> 4 rows with NULLs for unmatched sides."""
        left = SqliteRelation.from_arrow(
            connection, pa.Table.from_pydict({"idx": [1, 2, 3], "left_val": ["a", "b", "c"]})
        )
        right = SqliteRelation.from_arrow(
            connection, pa.Table.from_pydict({"idx": [2, 3, 4], "right_val": ["x", "y", "z"]})
        )
        index_obj = Index(("idx",))
        engine = SqliteMergeEngine(connection)
        result = engine.merge_full_outer(left, right, index_obj, index_obj)
        result_df = result.df().sort_values("idx").reset_index(drop=True)

        assert len(result_df) == 4, f"Expected 4 rows but got {len(result_df)}"
        assert set(result_df["idx"].tolist()) == {1, 2, 3, 4}

        # idx=1 has no right match -> right_val should be None/NaN
        row_1 = result_df[result_df["idx"] == 1].iloc[0]
        assert row_1["left_val"] == "a"
        assert row_1["right_val"] is None or str(row_1["right_val"]) in ("None", "nan", "")

        # idx=4 has no left match -> left_val should be None/NaN
        row_4 = result_df[result_df["idx"] == 4].iloc[0]
        assert row_4["right_val"] == "z"
        assert row_4["left_val"] is None or str(row_4["left_val"]) in ("None", "nan", "")

        # idx=2 and idx=3 should have both values
        row_2 = result_df[result_df["idx"] == 2].iloc[0]
        assert row_2["left_val"] == "b"
        assert row_2["right_val"] == "x"

    def test_full_outer_join_left_is_view(self, connection: sqlite3.Connection) -> None:
        """FULL OUTER JOIN where the left side is a view (from .select()) must not crash on rowid."""
        left_table = SqliteRelation.from_arrow(
            connection, pa.Table.from_pydict({"idx": [1, 2, 3], "col1": ["a", "b", "c"]})
        )
        # .select() produces a TEMP VIEW (_is_view=True), which has no rowid
        left_view = left_table.select("idx", "col1")
        right = SqliteRelation.from_arrow(
            connection, pa.Table.from_pydict({"idx": [2, 3, 4], "right_val": ["x", "y", "z"]})
        )
        index_obj = Index(("idx",))
        engine = SqliteMergeEngine(connection)
        result = engine.merge_full_outer(left_view, right, index_obj, index_obj)
        result_df = result.df().sort_values("idx").reset_index(drop=True)

        assert len(result_df) == 4, f"Expected 4 rows but got {len(result_df)}"
        assert set(result_df["idx"].tolist()) == {1, 2, 3, 4}

    def test_full_outer_join_both_views(self, connection: sqlite3.Connection) -> None:
        """FULL OUTER JOIN where both sides are views must not crash on rowid."""
        left_table = SqliteRelation.from_arrow(
            connection, pa.Table.from_pydict({"idx": [1, 2, 3], "col1": ["a", "b", "c"]})
        )
        right_table = SqliteRelation.from_arrow(
            connection, pa.Table.from_pydict({"idx": [2, 3, 4], "right_val": ["x", "y", "z"]})
        )
        # Both sides become views via .select()
        left_view = left_table.select("idx", "col1")
        right_view = right_table.select("idx", "right_val")
        index_obj = Index(("idx",))
        engine = SqliteMergeEngine(connection)
        result = engine.merge_full_outer(left_view, right_view, index_obj, index_obj)
        result_df = result.df().sort_values("idx").reset_index(drop=True)

        assert len(result_df) == 4, f"Expected 4 rows but got {len(result_df)}"
        assert set(result_df["idx"].tolist()) == {1, 2, 3, 4}

    def test_full_outer_join_left_is_prior_join_result(self, connection: sqlite3.Connection) -> None:
        """FULL OUTER JOIN where the left side is a prior inner join result (a view) must not crash."""
        # Create three base tables
        table_a = SqliteRelation.from_arrow(
            connection, pa.Table.from_pydict({"idx": [1, 2, 3], "a_val": ["a1", "a2", "a3"]})
        )
        table_b = SqliteRelation.from_arrow(connection, pa.Table.from_pydict({"idx": [2, 3], "b_val": ["b2", "b3"]}))
        table_c = SqliteRelation.from_arrow(connection, pa.Table.from_pydict({"idx": [3, 4], "c_val": ["c3", "c4"]}))

        index_obj = Index(("idx",))
        engine = SqliteMergeEngine(connection)

        # Inner join of A and B produces a view (idx=[2,3])
        inner_result = engine.merge_inner(table_a, table_b, index_obj, index_obj)
        assert inner_result._is_view is True

        # Full outer join of the inner join result with C
        # Left side is a view from the prior join; this exercises the rowid bug path
        result = engine.merge_full_outer(inner_result, table_c, index_obj, index_obj)
        result_df = result.df().sort_values("idx").reset_index(drop=True)

        # inner_result has idx=[2,3], table_c has idx=[3,4]
        # Full outer: idx=2 (left only), idx=3 (both), idx=4 (right only) -> 3 rows
        assert len(result_df) == 3, f"Expected 3 rows but got {len(result_df)}"
        assert set(result_df["idx"].tolist()) == {2, 3, 4}


def test_execute_sql_raises_value_error_when_no_connection() -> None:
    engine = SqliteMergeEngine()
    with pytest.raises(ValueError, match="Framework connection is not set"):
        engine._execute_sql("SELECT 1")


def test_register_table_raises_value_error_when_no_connection() -> None:
    engine = SqliteMergeEngine()
    with pytest.raises(ValueError, match="Framework connection is not set"):
        engine._register_table("some_table", None)


class TestSqliteMergeEngineMultiIndex(MultiIndexMergeEngineTestBase):
    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        return SqliteMergeEngine

    @classmethod
    def framework_type(cls) -> type[Any]:
        return SqliteRelation

    def get_connection(self) -> Optional[Any]:
        if not hasattr(self, "_connection"):
            self._connection = sqlite3.connect(":memory:")
            self._connection.create_function("REGEXP", 2, _regexp)
        return self._connection
