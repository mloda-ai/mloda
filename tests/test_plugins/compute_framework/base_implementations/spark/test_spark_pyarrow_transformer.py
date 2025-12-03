"""
Spark PyArrow Transformer Tests

This module contains comprehensive tests for the Spark PyArrow transformer implementation.

Requirements:
- PySpark must be installed (pip install pyspark)
- PyArrow must be installed (pip install pyarrow)
- Java 8+ must be installed and JAVA_HOME environment variable must be set

Environment Setup:
- JAVA_HOME: Must point to a valid Java installation

Test Coverage:
- Bidirectional transformation between Spark DataFrame and PyArrow Table
- Data integrity verification
- Error handling for missing dependencies
- Framework connection object handling

The tests use a shared SparkSession fixture to avoid Java gateway conflicts and
ensure proper resource management across all test methods.
"""

import pytest
from typing import Any

from mloda_plugins.compute_framework.base_implementations.spark.spark_pyarrow_transformer import SparkPyArrowTransformer

# Import shared fixtures and availability flags from conftest.py
try:
    from tests.test_plugins.compute_framework.base_implementations.spark.conftest import PYSPARK_AVAILABLE, SKIP_REASON
except ImportError:
    # Fallback for when running tests directly
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import PYSPARK_AVAILABLE, SKIP_REASON  # type: ignore

import logging

logger = logging.getLogger(__name__)

# Import PySpark types for schema creation (only if available)
if PYSPARK_AVAILABLE:
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
else:
    StructType = None
    StructField = None
    StringType = None
    IntegerType = None
    DoubleType = None

try:
    import pyarrow as pa

    PYARROW_AVAILABLE = True
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None
    PYARROW_AVAILABLE = False


@pytest.fixture
def sample_spark_dataframe(spark_session: Any) -> Any:
    """Create a sample Spark DataFrame for testing."""
    if not PYSPARK_AVAILABLE:
        pytest.skip(SKIP_REASON or "PySpark is not available")

    data = [
        {"id": 1, "name": "Alice", "age": 25, "score": 85.5},
        {"id": 2, "name": "Bob", "age": 30, "score": 92.0},
        {"id": 3, "name": "Charlie", "age": 35, "score": 78.5},
        {"id": 4, "name": "David", "age": 28, "score": 88.0},
        {"id": 5, "name": "Eve", "age": 42, "score": 95.5},
    ]
    return spark_session.createDataFrame(data)


@pytest.fixture
def sample_pyarrow_table() -> Any:
    """Create a sample PyArrow Table for testing."""
    if not PYARROW_AVAILABLE:
        pytest.skip("PyArrow is not available")

    data = {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 28, 42],
        "score": [85.5, 92.0, 78.5, 88.0, 95.5],
    }
    return pa.table(data)


@pytest.mark.skipif(
    not PYSPARK_AVAILABLE or not PYARROW_AVAILABLE, reason="PySpark or PyArrow is not installed. Skipping this test."
)
class TestSparkPyArrowTransformer:
    """Tests for SparkPyArrowTransformer."""

    def test_framework_types(self) -> None:
        """Test that framework types are correctly identified."""
        from pyspark.sql import DataFrame

        assert SparkPyArrowTransformer.framework() == DataFrame
        assert SparkPyArrowTransformer.other_framework() == pa.Table

    def test_import_methods(self) -> None:
        """Test that import methods work correctly."""
        # These should not raise exceptions
        SparkPyArrowTransformer.import_fw()
        SparkPyArrowTransformer.import_other_fw()

    def test_spark_to_pyarrow_transformation(self, sample_spark_dataframe: Any) -> None:
        """Test transformation from Spark DataFrame to PyArrow Table."""
        # Transform Spark DataFrame to PyArrow Table
        pyarrow_table = SparkPyArrowTransformer.transform_fw_to_other_fw(sample_spark_dataframe)

        # Verify the result is a PyArrow Table
        assert isinstance(pyarrow_table, pa.Table)

        # Verify data integrity
        assert pyarrow_table.num_rows == 5
        assert pyarrow_table.num_columns == 4

        # Check column names
        expected_columns = {"id", "name", "age", "score"}
        actual_columns = set(pyarrow_table.column_names)
        assert actual_columns == expected_columns

        # Check some data values
        id_column = pyarrow_table.column("id").to_pylist()
        assert id_column == [1, 2, 3, 4, 5]

        name_column = pyarrow_table.column("name").to_pylist()
        assert name_column == ["Alice", "Bob", "Charlie", "David", "Eve"]

    def test_pyarrow_to_spark_transformation(self, sample_pyarrow_table: Any, spark_session: Any) -> None:
        """Test transformation from PyArrow Table to Spark DataFrame."""
        # Transform PyArrow Table to Spark DataFrame
        spark_dataframe = SparkPyArrowTransformer.transform_other_fw_to_fw(
            sample_pyarrow_table, framework_connection_object=spark_session
        )

        # Verify the result is a Spark DataFrame
        from pyspark.sql import DataFrame

        assert isinstance(spark_dataframe, DataFrame)

        # Verify data integrity
        assert spark_dataframe.count() == 5
        assert len(spark_dataframe.columns) == 4

        # Check column names
        expected_columns = {"id", "name", "age", "score"}
        actual_columns = set(spark_dataframe.columns)
        assert actual_columns == expected_columns

        # Check some data values
        collected_data = spark_dataframe.collect()
        ids = [row["id"] for row in collected_data]
        assert sorted(ids) == [1, 2, 3, 4, 5]

        names = [row["name"] for row in collected_data]
        assert set(names) == {"Alice", "Bob", "Charlie", "David", "Eve"}

    def test_bidirectional_transformation(self, sample_spark_dataframe: Any, spark_session: Any) -> None:
        """Test bidirectional transformation preserves data integrity."""
        # Original Spark DataFrame
        original_data = sample_spark_dataframe.collect()
        original_count = sample_spark_dataframe.count()

        # Transform to PyArrow and back to Spark
        pyarrow_table = SparkPyArrowTransformer.transform_fw_to_other_fw(sample_spark_dataframe)
        restored_spark_df = SparkPyArrowTransformer.transform_other_fw_to_fw(
            pyarrow_table, framework_connection_object=spark_session
        )

        # Verify data integrity
        assert restored_spark_df.count() == original_count
        assert len(restored_spark_df.columns) == len(sample_spark_dataframe.columns)

        # Check that column names are preserved
        original_columns = set(sample_spark_dataframe.columns)
        restored_columns = set(restored_spark_df.columns)
        assert original_columns == restored_columns

        # Check that data values are preserved (order might be different)
        restored_data = restored_spark_df.collect()

        # Compare by converting to dictionaries and sorting by id
        original_dicts = sorted([row.asDict() for row in original_data], key=lambda x: x["id"])
        restored_dicts = sorted([row.asDict() for row in restored_data], key=lambda x: x["id"])

        assert original_dicts == restored_dicts

    def test_empty_dataframe_transformation(self, spark_session: Any) -> None:
        """Test transformation of empty Spark DataFrame."""
        # Create empty DataFrame with schema
        schema = StructType(
            [
                StructField("id", IntegerType(), True),
                StructField("name", StringType(), True),
                StructField("age", IntegerType(), True),
                StructField("score", DoubleType(), True),
            ]
        )
        empty_df = spark_session.createDataFrame([], schema)

        # Transform to PyArrow
        pyarrow_table = SparkPyArrowTransformer.transform_fw_to_other_fw(empty_df)

        # Verify empty table
        assert isinstance(pyarrow_table, pa.Table)
        assert pyarrow_table.num_rows == 0
        assert pyarrow_table.num_columns == 4

        # Transform back to Spark should raise an error for empty data
        # because Spark cannot infer schema from empty datasets
        with pytest.raises(Exception):  # Could be PySparkValueError or similar
            SparkPyArrowTransformer.transform_other_fw_to_fw(pyarrow_table, framework_connection_object=spark_session)

    def test_pyarrow_to_spark_without_connection_object(self, sample_pyarrow_table: Any, spark_session: Any) -> None:
        """Test PyArrow to Spark transformation without explicit connection object."""
        # This should work because we have an active SparkSession
        spark_dataframe = SparkPyArrowTransformer.transform_other_fw_to_fw(sample_pyarrow_table)

        # Verify the result
        from pyspark.sql import DataFrame

        assert isinstance(spark_dataframe, DataFrame)
        assert spark_dataframe.count() == 5

    def test_pyarrow_to_spark_invalid_connection_object(self, sample_pyarrow_table: Any) -> None:
        """Test PyArrow to Spark transformation with invalid connection object."""
        with pytest.raises(ValueError, match="Expected a SparkSession object"):
            SparkPyArrowTransformer.transform_other_fw_to_fw(
                sample_pyarrow_table, framework_connection_object="invalid"
            )

    def test_transformation_with_different_data_types(self, spark_session: Any) -> None:
        """Test transformation with various Spark data types."""
        # Create DataFrame with different data types
        data = [
            {"int_col": 1, "str_col": "test", "float_col": 1.5, "bool_col": True},
            {"int_col": 2, "str_col": "data", "float_col": 2.5, "bool_col": False},
            {"int_col": 3, "str_col": "value", "float_col": 3.5, "bool_col": True},
        ]
        spark_df = spark_session.createDataFrame(data)

        # Transform to PyArrow and back
        pyarrow_table = SparkPyArrowTransformer.transform_fw_to_other_fw(spark_df)
        restored_df = SparkPyArrowTransformer.transform_other_fw_to_fw(
            pyarrow_table, framework_connection_object=spark_session
        )

        # Verify data types are preserved (approximately)
        assert restored_df.count() == 3
        assert len(restored_df.columns) == 4

        # Check that data is preserved
        original_data = spark_df.collect()
        restored_data = restored_df.collect()

        # Sort by int_col for comparison
        original_sorted = sorted([row.asDict() for row in original_data], key=lambda x: x["int_col"])
        restored_sorted = sorted([row.asDict() for row in restored_data], key=lambda x: x["int_col"])

        assert original_sorted == restored_sorted

    def test_large_dataframe_transformation(self, spark_session: Any) -> None:
        """Test transformation with a larger dataset."""
        # Create a larger dataset
        data = [{"id": i, "value": i * 2, "category": f"cat_{i % 3}"} for i in range(1000)]
        large_df = spark_session.createDataFrame(data)

        # Transform to PyArrow
        pyarrow_table = SparkPyArrowTransformer.transform_fw_to_other_fw(large_df)

        # Verify large table
        assert isinstance(pyarrow_table, pa.Table)
        assert pyarrow_table.num_rows == 1000
        assert pyarrow_table.num_columns == 3

        # Transform back to Spark
        restored_df = SparkPyArrowTransformer.transform_other_fw_to_fw(
            pyarrow_table, framework_connection_object=spark_session
        )

        # Verify large DataFrame
        from pyspark.sql import DataFrame

        assert isinstance(restored_df, DataFrame)
        assert restored_df.count() == 1000
        assert len(restored_df.columns) == 3

    def test_transformation_preserves_null_values(self, spark_session: Any) -> None:
        """Test that null values are preserved during transformation."""
        # Create DataFrame with null values
        data = [
            {"id": 1, "name": "Alice", "age": 25},
            {"id": 2, "name": None, "age": 30},
            {"id": 3, "name": "Charlie", "age": None},
            {"id": 4, "name": None, "age": None},
        ]
        df_with_nulls = spark_session.createDataFrame(data)

        # Transform to PyArrow and back
        pyarrow_table = SparkPyArrowTransformer.transform_fw_to_other_fw(df_with_nulls)
        restored_df = SparkPyArrowTransformer.transform_other_fw_to_fw(
            pyarrow_table, framework_connection_object=spark_session
        )

        # Verify null values are preserved
        assert restored_df.count() == 4

        # Check specific null values
        collected_data = restored_df.collect()
        sorted_data = sorted([row.asDict() for row in collected_data], key=lambda x: x["id"])

        assert sorted_data[1]["name"] is None  # id=2
        assert sorted_data[2]["age"] is None  # id=3
        assert sorted_data[3]["name"] is None and sorted_data[3]["age"] is None  # id=4
