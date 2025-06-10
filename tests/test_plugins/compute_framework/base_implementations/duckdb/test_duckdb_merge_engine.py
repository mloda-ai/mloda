from typing import Any
import pytest

from mloda_core.abstract_plugins.components.link import JoinType
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_merge_engine import DuckDBMergeEngine

import logging

logger = logging.getLogger(__name__)

try:
    import duckdb
    import pyarrow as pa
except ImportError:
    logger.warning("DuckDB or PyArrow is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore
    pa = None


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed. Skipping this test.")
class TestDuckDBMergeEngine:
    """Unit tests for the DuckDBMergeEngine class."""

    @pytest.fixture
    def connection(self) -> Any:
        """Create a DuckDB connection for testing."""
        return duckdb.connect()

    @pytest.fixture
    def left_data(self, connection: Any) -> Any:
        """Create sample left dataset."""
        arrow_table = pa.Table.from_pydict({"idx": [1, 3], "col1": ["a", "b"]})
        return connection.from_arrow(arrow_table)

    @pytest.fixture
    def right_data(self, connection: Any) -> Any:
        """Create sample right dataset."""
        arrow_table = pa.Table.from_pydict({"idx": [1, 2], "col2": ["x", "z"]})
        return connection.from_arrow(arrow_table)

    @pytest.fixture
    def index_obj(self) -> Any:
        """Create index object for joins."""
        return Index(("idx",))

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

        left_data = connection.from_arrow(left_arrow)
        right_data = connection.from_arrow(right_arrow)

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

        empty_df = connection.from_arrow(empty_arrow)
        non_empty_df = connection.from_arrow(non_empty_arrow)

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

    def test_merge_with_multi_index_error(self, connection: Any, left_data: Any, right_data: Any) -> None:
        """Test that multi-index raises an error."""
        multi_index = Index(("col1", "col2"))
        engine = DuckDBMergeEngine(connection)

        with pytest.raises(ValueError, match="MultiIndex is not yet implemented"):
            engine.merge_inner(left_data, right_data, multi_index, multi_index)

    def test_merge_with_complex_data(self, connection: Any) -> None:
        """Test merge with more complex data structures."""
        left_arrow = pa.Table.from_pydict({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})
        right_arrow = pa.Table.from_pydict(
            {"id": [1, 2, 4], "city": ["New York", "London", "Tokyo"], "country": ["USA", "UK", "Japan"]}
        )

        left_data = connection.from_arrow(left_arrow)
        right_data = connection.from_arrow(right_arrow)

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

        left_data = connection.from_arrow(left_arrow)
        right_data = connection.from_arrow(right_arrow)

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
        result = engine.join_logic("inner", left_data, right_data, index_obj, index_obj, JoinType.INNER)
        result_df = result.df()
        assert len(result_df) == 1
        assert result_df["idx"].tolist()[0] == 1

    def test_merge_method_integration(self, connection: Any, left_data: Any, right_data: Any, index_obj: Any) -> None:
        """Test the main merge method that dispatches to specific join types."""
        engine = DuckDBMergeEngine(connection)

        # Test all join types through the main merge method
        result = engine.merge(left_data, right_data, JoinType.INNER, index_obj, index_obj)
        result_df = result.df()
        assert len(result_df) == 1
        assert result_df["idx"].tolist()[0] == 1

        result = engine.merge(left_data, right_data, JoinType.LEFT, index_obj, index_obj)
        result_df = result.df()
        assert len(result_df) == 2

    def test_merge_with_duplicate_data(self, connection: Any) -> None:
        """Test merge operations with duplicate data."""
        # Create data with duplicates
        left_arrow = pa.Table.from_pydict({"idx": [1, 1, 2], "col1": ["a", "a2", "b"]})
        right_arrow = pa.Table.from_pydict({"idx": [1, 1, 3], "col2": ["x", "x2", "z"]})

        left_data = connection.from_arrow(left_arrow)
        right_data = connection.from_arrow(right_arrow)

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

        left_data = connection.from_arrow(left_arrow)
        right_data = connection.from_arrow(right_arrow)

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
        data = conn.from_arrow(arrow_table)
        index_obj = Index(("idx",))

        with pytest.raises(ValueError, match="Framework connection not set"):
            engine.merge_inner(data, data, index_obj, index_obj)

    def test_column_names_method(self, connection: Any) -> None:
        """Test the get_column_names method."""
        engine = DuckDBMergeEngine(connection)

        arrow_table = pa.Table.from_pydict({"col1": [1, 2], "col2": ["a", "b"], "col3": [1.1, 2.2]})
        data = connection.from_arrow(arrow_table)

        columns = engine.get_column_names(data)
        assert set(columns) == {"col1", "col2", "col3"}

    def test_is_empty_data_method(self, connection: Any) -> None:
        """Test the is_empty_data method."""
        engine = DuckDBMergeEngine(connection)

        # Test with empty data
        empty_arrow = pa.Table.from_pydict({"col": []})
        empty_data = connection.from_arrow(empty_arrow)
        assert engine.is_empty_data(empty_data) is True

        # Test with non-empty data
        non_empty_arrow = pa.Table.from_pydict({"col": [1, 2]})
        non_empty_data = connection.from_arrow(non_empty_arrow)
        assert engine.is_empty_data(non_empty_data) is False

    def test_column_exists_in_result_method(self, connection: Any) -> None:
        """Test the column_exists_in_result method."""
        engine = DuckDBMergeEngine(connection)

        arrow_table = pa.Table.from_pydict({"col1": [1, 2], "col2": ["a", "b"]})
        data = connection.from_arrow(arrow_table)

        assert engine.column_exists_in_result(data, "col1") is True
        assert engine.column_exists_in_result(data, "col2") is True
        assert engine.column_exists_in_result(data, "nonexistent") is False
