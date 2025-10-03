from typing import Any
import pytest
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_pyarrow_transformer import (
    DuckDBPyarrowTransformer,
)

import logging

logger = logging.getLogger(__name__)

try:
    import duckdb
except ImportError:
    logger.warning("DuckDB is not installed. Some tests will be skipped.")
    duckdb = None

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed. Skipping this test.")
class TestDuckDBPyarrowTransformer:
    @pytest.fixture
    def connection(self) -> Any:
        """Create a DuckDB connection for testing."""
        conn = duckdb.connect()
        yield conn
        try:
            conn.close()
        except Exception as e:
            logger.debug(f"Error closing DuckDB connection: {e}")

    def test_framework_types(self) -> None:
        """Test that the transformer returns the correct framework types."""
        assert DuckDBPyarrowTransformer.framework() == duckdb.DuckDBPyRelation
        assert DuckDBPyarrowTransformer.other_framework() == pa.Table

    def test_check_imports(self) -> None:
        """Test that import checks work correctly."""
        assert DuckDBPyarrowTransformer.check_imports() is True

    def test_duckdb_to_pyarrow_transformation(self) -> None:
        """Test transformation from DuckDB relation to PyArrow Table."""
        # Create test data
        conn = duckdb.connect()
        arrow_table = pa.Table.from_pydict(
            {"int_col": [1, 2, 3], "str_col": ["a", "b", "c"], "float_col": [1.1, 2.2, 3.3]}
        )
        duckdb_relation = conn.from_arrow(arrow_table)

        # Transform to PyArrow
        result_arrow_table = DuckDBPyarrowTransformer.transform_fw_to_other_fw(duckdb_relation)

        # Verify it's a PyArrow Table
        assert isinstance(result_arrow_table, pa.Table)

        # Verify data integrity
        assert result_arrow_table.num_rows == 3
        assert result_arrow_table.num_columns == 3
        assert set(result_arrow_table.column_names) == {"int_col", "str_col", "float_col"}

    def test_pyarrow_to_duckdb_transformation(self, connection: Any) -> None:
        """Test transformation from PyArrow Table to DuckDB relation."""
        # Create test data
        arrow_table = pa.Table.from_pydict(
            {"int_col": [1, 2, 3], "str_col": ["a", "b", "c"], "float_col": [1.1, 2.2, 3.3]}
        )

        # Transform to DuckDB
        duckdb_relation = DuckDBPyarrowTransformer.transform_other_fw_to_fw(arrow_table, connection)

        # Verify it's a DuckDB relation
        assert isinstance(duckdb_relation, duckdb.DuckDBPyRelation)

        # Verify data integrity
        result_df = duckdb_relation.df()
        assert len(result_df) == 3
        assert len(result_df.columns) == 3
        assert set(result_df.columns) == {"int_col", "str_col", "float_col"}

    def test_roundtrip_transformation(self, connection: Any) -> None:
        """Test that data survives a roundtrip transformation."""
        # Create original data
        original_arrow = pa.Table.from_pydict(
            {
                "int_col": [1, 2, 3, 4],
                "str_col": ["hello", "world", "test", "data"],
                "float_col": [1.1, 2.2, 3.3, 4.4],
                "bool_col": [True, False, True, False],
            }
        )
        original_relation = connection.from_arrow(original_arrow)

        # Roundtrip: DuckDB -> PyArrow -> DuckDB
        arrow_table = DuckDBPyarrowTransformer.transform_fw_to_other_fw(original_relation)
        restored_relation = DuckDBPyarrowTransformer.transform_other_fw_to_fw(arrow_table, connection)

        # Verify data integrity
        original_df = original_relation.df()
        restored_df = restored_relation.df()

        # Compare column names
        assert set(original_df.columns) == set(restored_df.columns)

        # Compare data (sort both for consistent comparison)
        original_sorted = original_df.sort_values(list(original_df.columns)).reset_index(drop=True)
        restored_sorted = restored_df.sort_values(list(restored_df.columns)).reset_index(drop=True)

        for col in original_df.columns:
            assert original_sorted[col].tolist() == restored_sorted[col].tolist()

    def test_empty_relation_transformation(self, connection: Any) -> None:
        """Test transformation of empty relations."""
        # Create empty relation with schema
        empty_arrow = pa.Table.from_pydict(
            {"int_col": pa.array([], type=pa.int64()), "str_col": pa.array([], type=pa.string())}
        )
        empty_relation = connection.from_arrow(empty_arrow)

        # Transform to PyArrow and back
        arrow_table = DuckDBPyarrowTransformer.transform_fw_to_other_fw(empty_relation)
        restored_relation = DuckDBPyarrowTransformer.transform_other_fw_to_fw(arrow_table, connection)

        # Verify structure is preserved
        restored_df = restored_relation.df()
        assert len(restored_df) == 0
        assert set(restored_df.columns) == {"int_col", "str_col"}

    def test_null_values_transformation(self, connection: Any) -> None:
        """Test transformation of relations with null values."""
        # Create relation with null values
        arrow_with_nulls = pa.Table.from_pydict(
            {"int_col": [1, None, 3], "str_col": ["a", None, "c"], "float_col": [1.1, 2.2, None]}
        )
        relation_with_nulls = connection.from_arrow(arrow_with_nulls)

        # Roundtrip transformation
        arrow_table = DuckDBPyarrowTransformer.transform_fw_to_other_fw(relation_with_nulls)
        restored_relation = DuckDBPyarrowTransformer.transform_other_fw_to_fw(arrow_table, connection)

        # Verify null values are preserved
        original_df = relation_with_nulls.df()
        restored_df = restored_relation.df()

        # Check that null patterns are preserved
        for col in original_df.columns:
            original_nulls = original_df[col].isna()
            restored_nulls = restored_df[col].isna()
            assert original_nulls.tolist() == restored_nulls.tolist()

    def test_transform_method_with_correct_orientation(self, connection: Any) -> None:
        """Test the generic transform method with correct framework orientation."""
        # Create test data
        arrow_table = pa.Table.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]})
        duckdb_relation = connection.from_arrow(arrow_table)

        # Test DuckDB -> PyArrow transformation
        arrow_result = DuckDBPyarrowTransformer.transform(duckdb.DuckDBPyRelation, pa.Table, duckdb_relation, None)
        assert isinstance(arrow_result, pa.Table)

        # Test PyArrow -> DuckDB transformation
        duckdb_result = DuckDBPyarrowTransformer.transform(pa.Table, duckdb.DuckDBPyRelation, arrow_result, connection)
        assert isinstance(duckdb_result, duckdb.DuckDBPyRelation)

        # Verify data integrity
        original_df = duckdb_relation.df()
        result_df = duckdb_result.df()
        assert original_df.equals(result_df)

    def test_transform_method_with_unsupported_frameworks(self, connection: Any) -> None:
        """Test that transform method raises error for unsupported frameworks."""
        arrow_table = pa.Table.from_pydict({"a": [1, 2, 3]})
        duckdb_relation = connection.from_arrow(arrow_table)

        with pytest.raises(ValueError):
            DuckDBPyarrowTransformer.transform(list, dict, duckdb_relation, None)

    def test_large_dataset_transformation(self, connection: Any) -> None:
        """Test transformation with larger datasets."""
        # Create larger test data
        size = 1000
        large_arrow = pa.Table.from_pydict(
            {
                "id": list(range(size)),
                "value": [f"value_{i}" for i in range(size)],
                "score": [i * 0.1 for i in range(size)],
            }
        )
        large_relation = connection.from_arrow(large_arrow)

        # Transform to PyArrow and back
        arrow_result = DuckDBPyarrowTransformer.transform_fw_to_other_fw(large_relation)
        restored_relation = DuckDBPyarrowTransformer.transform_other_fw_to_fw(arrow_result, connection)

        # Verify data integrity
        original_df = large_relation.df()
        restored_df = restored_relation.df()
        assert len(original_df) == len(restored_df) == size
        assert original_df.equals(restored_df)

    def test_different_data_types_transformation(self, connection: Any) -> None:
        """Test transformation with various data types."""
        # Create data with different types
        mixed_arrow = pa.Table.from_pydict(
            {
                "int8_col": pa.array([1, 2, 3], type=pa.int8()),
                "int32_col": pa.array([100, 200, 300], type=pa.int32()),
                "int64_col": pa.array([1000, 2000, 3000], type=pa.int64()),
                "float32_col": pa.array([1.1, 2.2, 3.3], type=pa.float32()),
                "float64_col": pa.array([10.1, 20.2, 30.3], type=pa.float64()),
                "string_col": pa.array(["a", "b", "c"], type=pa.string()),
                "bool_col": pa.array([True, False, True], type=pa.bool_()),
            }
        )
        mixed_relation = connection.from_arrow(mixed_arrow)

        # Transform to PyArrow and back
        arrow_result = DuckDBPyarrowTransformer.transform_fw_to_other_fw(mixed_relation)
        restored_relation = DuckDBPyarrowTransformer.transform_other_fw_to_fw(arrow_result, connection)

        # Verify data integrity
        original_df = mixed_relation.df()
        restored_df = restored_relation.df()

        # Check that all columns are preserved
        assert set(original_df.columns) == set(restored_df.columns)
        assert len(original_df) == len(restored_df)

    def test_connection_independence(self, connection: Any) -> None:
        """Test that transformations work with different connections."""
        # Create data with one connection
        arrow_table = pa.Table.from_pydict({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        relation1 = connection.from_arrow(arrow_table)

        # Transform to PyArrow
        arrow_result = DuckDBPyarrowTransformer.transform_fw_to_other_fw(relation1)

        # Transform back with a different connection (should work)
        conn2 = duckdb.connect()
        relation2 = DuckDBPyarrowTransformer.transform_other_fw_to_fw(arrow_result, conn2)

        # Verify data integrity
        df1 = relation1.df()
        df2 = relation2.df()
        assert df1.equals(df2)

    def test_schema_preservation(self, connection: Any) -> None:
        """Test that schema information is preserved during transformation."""
        # Create data with specific schema
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("name", pa.string()),
                pa.field("score", pa.float64()),
                pa.field("active", pa.bool_()),
            ]
        )

        arrow_table = pa.Table.from_arrays(
            [
                pa.array([1, 2, 3], type=pa.int64()),
                pa.array(["Alice", "Bob", "Charlie"], type=pa.string()),
                pa.array([85.5, 92.0, 78.5], type=pa.float64()),
                pa.array([True, False, True], type=pa.bool_()),
            ],
            schema=schema,
        )

        relation = connection.from_arrow(arrow_table)

        # Transform to PyArrow and back
        arrow_result = DuckDBPyarrowTransformer.transform_fw_to_other_fw(relation)
        restored_relation = DuckDBPyarrowTransformer.transform_other_fw_to_fw(arrow_result, connection)

        # Verify schema is preserved
        original_df = relation.df()
        restored_df = restored_relation.df()

        assert list(original_df.columns) == list(restored_df.columns)
        assert len(original_df) == len(restored_df)
