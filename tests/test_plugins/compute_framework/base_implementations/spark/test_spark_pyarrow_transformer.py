"""
Spark PyArrow Transformer Tests

This module contains tests for the Spark PyArrow transformer implementation using TransformerTestBase.

Requirements:
- PySpark must be installed (pip install pyspark)
- PyArrow must be installed (pip install pyarrow)
- Java 8+ must be installed and JAVA_HOME environment variable must be set

Environment Setup:
- JAVA_HOME: Must point to a valid Java installation

Test Coverage:
- All standard transformer tests from TransformerTestBase
- Spark-specific connection object handling tests
- Spark-specific error handling tests
"""

import pytest
from typing import Any, Optional, Type

from mloda_plugins.compute_framework.base_implementations.spark.spark_pyarrow_transformer import SparkPyarrowTransformer
from tests.test_plugins.compute_framework.test_tooling.transformer.transformer_test_base import TransformerTestBase

try:
    from tests.test_plugins.compute_framework.base_implementations.spark.conftest import PYSPARK_AVAILABLE, SKIP_REASON
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import PYSPARK_AVAILABLE, SKIP_REASON  # type: ignore

import logging

logger = logging.getLogger(__name__)

if PYSPARK_AVAILABLE:
    from pyspark.sql import DataFrame
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
else:
    DataFrame = None
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


@pytest.mark.skipif(
    not PYSPARK_AVAILABLE or not PYARROW_AVAILABLE, reason="PySpark or PyArrow is not installed. Skipping this test."
)
class TestSparkPyarrowTransformer(TransformerTestBase):
    """Tests for SparkPyarrowTransformer using TransformerTestBase."""

    @pytest.fixture(autouse=True)
    def setup_spark_session(self, spark_session: Any) -> None:
        """Setup spark session for all tests."""
        self._spark_session = spark_session

    @classmethod
    def transformer_class(cls) -> Type[Any]:
        """Return the SparkPyarrowTransformer class."""
        return SparkPyarrowTransformer

    @classmethod
    def framework_type(cls) -> Type[Any]:
        """Return the Spark DataFrame type."""
        return DataFrame  # type: ignore[no-any-return]

    def get_connection(self) -> Optional[Any]:
        """Return the SparkSession connection object."""
        return self._spark_session

    def test_pyarrow_to_spark_without_connection_object(self) -> None:
        """Test PyArrow to Spark transformation without explicit connection object."""
        scenario = {"data": {"id": [1, 2, 3], "value": ["a", "b", "c"]}}
        source_table = pa.Table.from_pydict(scenario["data"])

        transformer = self.transformer_class()
        spark_dataframe = transformer.transform_other_fw_to_fw(source_table)

        assert isinstance(spark_dataframe, DataFrame)
        assert spark_dataframe.count() == 3

    def test_pyarrow_to_spark_invalid_connection_object(self) -> None:
        """Test PyArrow to Spark transformation with invalid connection object."""
        scenario = {"data": {"id": [1, 2, 3], "value": ["a", "b", "c"]}}
        source_table = pa.Table.from_pydict(scenario["data"])

        transformer = self.transformer_class()
        with pytest.raises(ValueError, match="Expected a SparkSession object"):
            transformer.transform_other_fw_to_fw(source_table, framework_connection_object="invalid")

    def test_empty_dataframe_transformation_raises_error(self) -> None:
        """Test that transforming empty PyArrow table back to Spark raises an error."""
        schema = StructType(
            [
                StructField("id", IntegerType(), True),
                StructField("name", StringType(), True),
                StructField("age", IntegerType(), True),
                StructField("score", DoubleType(), True),
            ]
        )
        empty_df = self._spark_session.createDataFrame([], schema)

        transformer = self.transformer_class()
        pyarrow_table = transformer.transform_fw_to_other_fw(empty_df)

        assert isinstance(pyarrow_table, pa.Table)
        assert pyarrow_table.num_rows == 0
        assert pyarrow_table.num_columns == 4

        with pytest.raises(Exception):
            transformer.transform_other_fw_to_fw(pyarrow_table, framework_connection_object=self._spark_session)
