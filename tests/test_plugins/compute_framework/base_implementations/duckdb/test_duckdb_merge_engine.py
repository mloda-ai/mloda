from typing import Any, Optional
import pytest

from mloda.user import JoinType
from mloda.user import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_merge_engine import DuckDBMergeEngine
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from tests.test_plugins.compute_framework.test_tooling.merge_conformance.merge_conformance_test_base import (
    MergeConformanceTestBase,
)
from tests.test_plugins.compute_framework.test_tooling.merge_link import make_merge_link
from tests.test_plugins.compute_framework.test_tooling.multi_index.multi_index_test_base import (
    MultiIndexMergeEngineTestBase,
)

import logging

logger = logging.getLogger(__name__)

try:
    import duckdb
    import pyarrow as pa
except ImportError:
    logger.warning("DuckDB or PyArrow is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]
    pa = None  # type: ignore[assignment, unused-ignore]


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed. Skipping this test.")
class TestDuckDBMergeEngine:
    """Unit tests for the DuckDBMergeEngine class."""

    @pytest.fixture
    def left_data(self, connection: Any) -> Any:
        """Create sample left dataset."""
        arrow_table = pa.Table.from_pydict({"idx": [1, 3], "col1": ["a", "b"]})
        return DuckdbRelation.from_arrow(connection, arrow_table)

    @pytest.fixture
    def right_data(self, connection: Any) -> Any:
        """Create sample right dataset."""
        arrow_table = pa.Table.from_pydict({"idx": [1, 2], "col2": ["x", "z"]})
        return DuckdbRelation.from_arrow(connection, arrow_table)

    def test_check_import(self) -> None:
        """Test that check_import passes without errors."""
        engine = DuckDBMergeEngine()
        engine.check_import()  # Should not raise any exception

    def test_merge_inner(self, connection: Any, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test inner join."""
        engine = DuckDBMergeEngine(connection)
        result = engine.merge_inner(left_data, right_data, index_obj, index_obj)

        result_df = result.df()
        assert len(result_df) == 1
        assert result_df["idx"].tolist()[0] == 1
        assert result_df["col1"].tolist()[0] == "a"
        assert result_df["col2"].tolist()[0] == "x"

    def test_merge_left(self, connection: Any, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test left join."""
        engine = DuckDBMergeEngine(connection)
        result = engine.merge_left(left_data, right_data, index_obj, index_obj)

        result_df = result.df()
        assert len(result_df) == 2
        # Sort by idx for consistent comparison
        result_df = result_df.sort_values("idx").reset_index(drop=True)
        assert result_df["idx"].tolist() == [1, 3]
        assert result_df["col1"].tolist() == ["a", "b"]
        assert result_df["col2"].tolist()[0] == "x"
        assert result_df["col2"].isna().tolist()[1]  # Should be null for idx=3

    def test_merge_right(self, connection: Any, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test right join."""
        engine = DuckDBMergeEngine(connection)
        result = engine.merge_right(left_data, right_data, index_obj, index_obj)

        result_df = result.df()
        assert len(result_df) == 2
        # Sort by idx for consistent comparison
        result_df = result_df.sort_values("idx").reset_index(drop=True)
        assert result_df["idx"].tolist() == [1, 2]
        assert result_df["col1"].tolist()[0] == "a"
        assert result_df["col1"].isna().tolist()[1]  # Should be null for idx=2
        assert result_df["col2"].tolist() == ["x", "z"]

    def test_merge_full_outer(self, connection: Any, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test outer join."""
        engine = DuckDBMergeEngine(connection)
        result = engine.merge_full_outer(left_data, right_data, index_obj, index_obj)

        result_df = result.df()
        assert len(result_df) == 3
        # Sort by idx for consistent comparison
        result_df = result_df.sort_values("idx").reset_index(drop=True)
        assert result_df["idx"].tolist() == [1, 2, 3]

    def test_merge_append(self, connection: Any, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test append operation (UNION ALL)."""
        engine = DuckDBMergeEngine(connection)
        result = engine.merge_append(left_data, right_data, index_obj, index_obj)

        result_df = result.df()
        assert len(result_df) == 4  # Should have 2 + 2 rows
        # Check that all columns are present
        assert "idx" in result_df.columns
        assert "col1" in result_df.columns
        assert "col2" in result_df.columns

    def test_merge_union(self, connection: Any, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test union operation (removes duplicates)."""
        engine = DuckDBMergeEngine(connection)
        result = engine.merge_union(left_data, right_data, index_obj, index_obj)

        result_df = result.df()
        assert len(result_df) <= 4  # Should be <= 4 due to duplicate removal
        # Check that all columns are present
        assert "idx" in result_df.columns
        assert "col1" in result_df.columns
        assert "col2" in result_df.columns

    def test_merge_with_different_join_columns(self, connection: Any) -> None:
        """Test merge with different column names for join."""
        left_arrow = pa.Table.from_pydict({"left_id": [1, 2], "col1": ["a", "b"]})
        right_arrow = pa.Table.from_pydict({"right_id": [1, 3], "col2": ["x", "z"]})

        left_data = DuckdbRelation.from_arrow(connection, left_arrow)
        right_data = DuckdbRelation.from_arrow(connection, right_arrow)

        left_index = Index(("left_id",))
        right_index = Index(("right_id",))

        engine = DuckDBMergeEngine(connection)
        result = engine.merge_inner(left_data, right_data, left_index, right_index)

        result_df = result.df()
        assert len(result_df) == 1
        assert result_df["left_id"].tolist()[0] == 1
        assert result_df["col1"].tolist()[0] == "a"
        assert result_df["col2"].tolist()[0] == "x"

    def test_merge_with_empty_datasets(self, connection: Any, index_obj: Any) -> None:
        """Test merge operations with empty datasets."""
        empty_arrow = pa.Table.from_pydict({"idx": [], "col": []})
        non_empty_arrow = pa.Table.from_pydict({"idx": [1], "col": ["a"]})

        empty_df = DuckdbRelation.from_arrow(connection, empty_arrow)
        non_empty_df = DuckdbRelation.from_arrow(connection, non_empty_arrow)

        engine = DuckDBMergeEngine(connection)

        # Empty left
        result = engine.merge_inner(empty_df, non_empty_df, index_obj, index_obj)
        assert len(result.df()) == 0

        # Empty right
        result = engine.merge_inner(non_empty_df, empty_df, index_obj, index_obj)
        assert len(result.df()) == 0

        # Both empty
        result = engine.merge_inner(empty_df, empty_df, index_obj, index_obj)
        assert len(result.df()) == 0

    def test_merge_with_complex_data(self, connection: Any) -> None:
        """Test merge with more complex data structures."""
        left_arrow = pa.Table.from_pydict({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})
        right_arrow = pa.Table.from_pydict(
            {"id": [1, 2, 4], "city": ["New York", "London", "Tokyo"], "country": ["USA", "UK", "Japan"]}
        )

        left_data = DuckdbRelation.from_arrow(connection, left_arrow)
        right_data = DuckdbRelation.from_arrow(connection, right_arrow)

        index_obj = Index(("id",))
        engine = DuckDBMergeEngine(connection)

        # Test inner join
        result = engine.merge_inner(left_data, right_data, index_obj, index_obj)
        result_df = result.df()
        assert len(result_df) == 2
        assert set(result_df["id"].tolist()) == {1, 2}
        assert set(result_df["name"].tolist()) == {"Alice", "Bob"}

    def test_merge_with_null_values(self, connection: Any) -> None:
        """Test merge operations with null values in join columns."""
        left_arrow = pa.Table.from_pydict({"id": [1, None], "col1": ["a", "b"]})
        right_arrow = pa.Table.from_pydict({"id": [1, None], "col2": ["x", "y"]})

        left_data = DuckdbRelation.from_arrow(connection, left_arrow)
        right_data = DuckdbRelation.from_arrow(connection, right_arrow)

        index_obj = Index(("id",))
        engine = DuckDBMergeEngine(connection)

        result = engine.merge_inner(left_data, right_data, index_obj, index_obj)
        result_df = result.df()
        # DuckDB should handle null joins appropriately
        assert len(result_df) >= 1

    def test_join_logic_integration(self, connection: Any, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test that join_logic method works correctly."""
        engine = DuckDBMergeEngine(connection)

        # Test inner join through join_logic
        result = engine.join_logic(JoinType.INNER, left_data, right_data, index_obj, index_obj)
        result_df = result.df()
        assert len(result_df) == 1
        assert result_df["idx"].tolist()[0] == 1

    def test_merge_method_integration(self, connection: Any, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test the main merge method that dispatches to specific join types."""
        engine = DuckDBMergeEngine(connection)

        # Test all join types through the main merge method
        result = engine.merge(left_data, right_data, make_merge_link(JoinType.INNER, index_obj, index_obj))
        result_df = result.df()
        assert len(result_df) == 1
        assert result_df["idx"].tolist()[0] == 1

        result = engine.merge(left_data, right_data, make_merge_link(JoinType.LEFT, index_obj, index_obj))
        result_df = result.df()
        assert len(result_df) == 2

    def test_merge_with_duplicate_data(self, connection: Any) -> None:
        """Test merge operations with duplicate data."""
        # Create data with duplicates
        left_arrow = pa.Table.from_pydict({"idx": [1, 1, 2], "col1": ["a", "a2", "b"]})
        right_arrow = pa.Table.from_pydict({"idx": [1, 1, 3], "col2": ["x", "x2", "z"]})

        left_data = DuckdbRelation.from_arrow(connection, left_arrow)
        right_data = DuckdbRelation.from_arrow(connection, right_arrow)

        index_obj = Index(("idx",))
        engine = DuckDBMergeEngine(connection)

        # Test inner join with duplicates
        result = engine.merge_inner(left_data, right_data, index_obj, index_obj)
        result_df = result.df()
        # Should have 4 rows (2x2 combinations for idx=1)
        assert len(result_df) == 4

    def test_merge_with_different_data_types(self, connection: Any) -> None:
        """Test merge with different data types."""
        left_arrow = pa.Table.from_pydict({"id": [1, 2, 3], "score": [85.5, 92.0, 78.5], "active": [True, False, True]})
        right_arrow = pa.Table.from_pydict({"id": [1, 2, 4], "grade": ["A", "A+", "B"], "count": [10, 15, 8]})

        left_data = DuckdbRelation.from_arrow(connection, left_arrow)
        right_data = DuckdbRelation.from_arrow(connection, right_arrow)

        index_obj = Index(("id",))
        engine = DuckDBMergeEngine(connection)

        result = engine.merge_inner(left_data, right_data, index_obj, index_obj)
        result_df = result.df()
        assert len(result_df) == 2
        assert set(result_df["id"].tolist()) == {1, 2}

    def test_framework_connection_error(self) -> None:
        """Test that merge engine raises error when no connection is provided."""
        engine = DuckDBMergeEngine()  # No connection provided

        # Create dummy data
        conn = duckdb.connect()
        arrow_table = pa.Table.from_pydict({"idx": [1], "col": ["a"]})
        data = DuckdbRelation.from_arrow(conn, arrow_table)
        index_obj = Index(("idx",))

        with pytest.raises(ValueError, match="Framework connection not set"):
            engine.merge_inner(data, data, index_obj, index_obj)

    def test_column_names_method(self, connection: Any) -> None:
        """Test the get_column_names method."""
        engine = DuckDBMergeEngine(connection)

        arrow_table = pa.Table.from_pydict({"col1": [1, 2], "col2": ["a", "b"], "col3": [1.1, 2.2]})
        data = DuckdbRelation.from_arrow(connection, arrow_table)

        columns = engine.get_column_names(data)
        assert set(columns) == {"col1", "col2", "col3"}

    def test_is_empty_data_method(self, connection: Any) -> None:
        """Test the is_empty_data method."""
        engine = DuckDBMergeEngine(connection)

        # Test with empty data
        empty_arrow = pa.Table.from_pydict({"col": []})
        empty_data = DuckdbRelation.from_arrow(connection, empty_arrow)
        assert engine.is_empty_data(empty_data) is True

        # Test with non-empty data
        non_empty_arrow = pa.Table.from_pydict({"col": [1, 2]})
        non_empty_data = DuckdbRelation.from_arrow(connection, non_empty_arrow)
        assert engine.is_empty_data(non_empty_data) is False

    def test_column_exists_in_result_method(self, connection: Any) -> None:
        """Test the column_exists_in_result method."""
        engine = DuckDBMergeEngine(connection)

        arrow_table = pa.Table.from_pydict({"col1": [1, 2], "col2": ["a", "b"]})
        data = DuckdbRelation.from_arrow(connection, arrow_table)

        assert engine.column_exists_in_result(data, "col1") is True
        assert engine.column_exists_in_result(data, "col2") is True
        assert engine.column_exists_in_result(data, "nonexistent") is False


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed. Skipping this test.")
class TestDuckDBMergeEngineViewLeak:
    """Regression tests for issue #475: SQL merge engines must not leak per-merge temp views.

    merge_union, merge_append and merge_asof register uniquely-named `_left_<uuid>` /
    `_right_<uuid>` views on every call. On a long-lived connection these accumulate and
    are never dropped. These tests run many merges through ONE connection and assert the
    catalog does not retain the temp registration views, while results stay correct.
    """

    @staticmethod
    def _leaked_view_count(connection: Any) -> int:
        rows = connection.sql(
            "SELECT count(*) FROM duckdb_views() WHERE view_name LIKE '_left_%' OR view_name LIKE '_right_%'"
        ).fetchall()
        return int(rows[0][0])

    def test_merge_union_does_not_leak_views(self, connection: Any, index_obj: Any) -> None:
        engine = DuckDBMergeEngine(connection)
        for _ in range(5):
            left = DuckdbRelation.from_arrow(connection, pa.Table.from_pydict({"idx": [1, 3], "col1": ["a", "b"]}))
            right = DuckdbRelation.from_arrow(connection, pa.Table.from_pydict({"idx": [1, 2], "col2": ["x", "z"]}))
            result_df = engine.merge_union(left, right, index_obj, index_obj).df()
            assert sorted(result_df["idx"].tolist()) == [1, 1, 2, 3]

        assert self._leaked_view_count(connection) == 0

    def test_merge_append_does_not_leak_views(self, connection: Any, index_obj: Any) -> None:
        engine = DuckDBMergeEngine(connection)
        for _ in range(5):
            left = DuckdbRelation.from_arrow(connection, pa.Table.from_pydict({"idx": [1, 3], "col1": ["a", "b"]}))
            right = DuckdbRelation.from_arrow(connection, pa.Table.from_pydict({"idx": [1, 2], "col2": ["x", "z"]}))
            result_df = engine.merge_append(left, right, index_obj, index_obj).df()
            assert len(result_df) == 4

        assert self._leaked_view_count(connection) == 0

    def test_merge_asof_does_not_leak_views(self, connection: Any) -> None:
        engine = DuckDBMergeEngine(connection)
        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        for _ in range(5):
            left = DuckdbRelation.from_arrow(connection, pa.Table.from_pydict({"k": [1], "t": [10], "lv": [100]}))
            right = DuckdbRelation.from_arrow(connection, pa.Table.from_pydict({"k": [1], "t": [8], "rv": [7]}))
            result_df = engine.merge_asof(left, right, Index(("k",)), Index(("k",)), cfg).df()
            assert len(result_df) == 1
            assert result_df["rv"].tolist()[0] == 7

        assert self._leaked_view_count(connection) == 0


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed. Skipping this test.")
class TestDuckDBEquiJoinTimezoneGuard:
    """Temporal cross-side timezone guard on duckdb equi-joins.

    A TIMESTAMP WITH TIME ZONE column (arrow timestamp with tz) on one side and a plain
    TIMESTAMP column (arrow timestamp without tz) on the other must be rejected by the
    ComparisonContract. DuckDBMergeEngine opts into the guard
    (`provides_column_semantics = True`) and implements _column_semantics.
    """

    @staticmethod
    def _rel(connection: Any, table: Any) -> Any:
        return DuckdbRelation.from_arrow(connection, table)

    def test_inner_equi_join_tz_aware_vs_naive_raises(self, connection: Any) -> None:
        """NEGATIVE: TIMESTAMPTZ key on the left, plain TIMESTAMP on the right must be rejected."""
        import datetime as dt

        left_arrow = pa.table(
            {
                "t": pa.array(
                    [dt.datetime(2021, 1, 1, 12, 0, tzinfo=dt.timezone.utc)], type=pa.timestamp("us", tz="UTC")
                ),
                "lv": ["a"],
            }
        )
        right_arrow = pa.table(
            {
                "t": pa.array([dt.datetime(2021, 1, 1, 12, 0)], type=pa.timestamp("us")),
                "rv": ["x"],
            }
        )
        left_data = self._rel(connection, left_arrow)
        right_data = self._rel(connection, right_arrow)
        idx = Index(("t",))
        engine = DuckDBMergeEngine(connection)

        with pytest.raises(ValueError, match=r"(?i)time[ -]?zone"):
            engine.merge(left_data, right_data, make_merge_link(JoinType.INNER, idx, idx))

    def test_inner_equi_join_both_plain_timestamp_succeeds(self, connection: Any) -> None:
        """POSITIVE: both-plain-TIMESTAMP keys are legal and join normally."""
        import datetime as dt

        left_arrow = pa.table({"t": pa.array([dt.datetime(2021, 1, 1, 12, 0)], type=pa.timestamp("us")), "lv": ["a"]})
        right_arrow = pa.table({"t": pa.array([dt.datetime(2021, 1, 1, 12, 0)], type=pa.timestamp("us")), "rv": ["x"]})
        left_data = self._rel(connection, left_arrow)
        right_data = self._rel(connection, right_arrow)
        idx = Index(("t",))
        engine = DuckDBMergeEngine(connection)

        result = engine.merge(left_data, right_data, make_merge_link(JoinType.INNER, idx, idx))
        result_df = result.df()
        assert len(result_df) == 1
        assert result_df["lv"].tolist()[0] == "a"
        assert result_df["rv"].tolist()[0] == "x"

    def test_inner_equi_join_string_key_is_legal(self, connection: Any) -> None:
        """POSITIVE: a non-temporal (string) equi-join key stays unaffected by the guard."""
        left_arrow = pa.table({"k": ["a"], "lv": [1]})
        right_arrow = pa.table({"k": ["a"], "rv": [2]})
        left_data = self._rel(connection, left_arrow)
        right_data = self._rel(connection, right_arrow)
        idx = Index(("k",))
        engine = DuckDBMergeEngine(connection)

        result = engine.merge(left_data, right_data, make_merge_link(JoinType.INNER, idx, idx))
        result_df = result.df()
        assert len(result_df) == 1
        assert result_df["lv"].tolist()[0] == 1
        assert result_df["rv"].tolist()[0] == 2


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed. Skipping this test.")
class TestDuckDBMergeEngineMultiIndex(MultiIndexMergeEngineTestBase):
    """Test DuckDBMergeEngine multi-index support using shared test scenarios."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        """Return the DuckDBMergeEngine class."""
        return DuckDBMergeEngine

    @classmethod
    def framework_type(cls) -> type[Any]:
        """Return DuckDB relation type."""
        if duckdb is None:
            raise ImportError("DuckDB is not installed")
        return DuckdbRelation

    def get_connection(self) -> Optional[Any]:
        """DuckDB requires a connection object."""
        if not hasattr(self, "_connection"):
            self._connection = duckdb.connect()
        return self._connection


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed. Skipping this test.")
class TestDuckDBMergeConformance(MergeConformanceTestBase):
    """Cross-framework merge conformance for DuckDBMergeEngine."""

    @classmethod
    def merge_engine_class(cls) -> type[BaseMergeEngine]:
        """Return the DuckDBMergeEngine class."""
        return DuckDBMergeEngine

    @classmethod
    def framework_type(cls) -> type[Any]:
        """Return DuckDB relation type."""
        if duckdb is None:
            raise ImportError("DuckDB is not installed")
        return DuckdbRelation

    def get_connection(self) -> Optional[Any]:
        """DuckDB requires a connection object."""
        if not hasattr(self, "_connection"):
            self._connection = duckdb.connect()
        return self._connection
