"""
Spark Framework Tests

This module contains comprehensive tests for the Spark compute framework implementation.

Requirements:
- PySpark must be installed (pip install pyspark)
- Java 8+ must be installed and JAVA_HOME environment variable must be set

Environment Setup:
- JAVA_HOME: Must point to a valid Java installation
- SKIP_SPARK_INSTALLATION_TEST: Set to "true" to skip installation validation tests

Test Structure:
- TestSparkFrameworkAvailability: Tests framework availability detection
- TestSparkInstallation: Validates PySpark and Java setup
- TestSparkFrameworkComputeFramework: Core framework functionality tests

The tests use a shared SparkSession fixture to avoid Java gateway conflicts and
ensure proper resource management across all test methods.
"""

import os
from typing import Any
from mloda_core.abstract_plugins.components.link import JoinType
import pytest
from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import SparkFramework
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.components.index.index import Index
from tests.test_plugins.compute_framework.test_tooling.availability_test_helper import (
    assert_unavailable_when_import_blocked,
)

# Import shared fixtures and availability flags from conftest.py
from tests.test_plugins.compute_framework.base_implementations.spark.conftest import (
    PYSPARK_AVAILABLE,
    SKIP_REASON,
)

import logging

logger = logging.getLogger(__name__)

# Import PySpark types for type checking (only if available)
if PYSPARK_AVAILABLE:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType
    import pyspark
else:
    SparkSession = None
    StructType = None
    StructField = None
    StringType = None
    IntegerType = None
    pyspark = None


class TestSparkFrameworkAvailability:
    def test_is_available_when_pyspark_not_installed(self) -> None:
        """Test that is_available() returns False when pyspark import fails."""
        assert_unavailable_when_import_blocked(SparkFramework, ["pyspark.sql"])


class TestSparkInstallation:
    @pytest.mark.skipif(
        os.getenv("SKIP_SPARK_INSTALLATION_TEST", "false").lower() == "true",
        reason="Spark installation test is disabled by environment variable",
    )
    def test_spark_is_installed(self, spark_session: Any) -> None:
        """Test that PySpark is properly installed and can be imported."""
        try:
            import pyarrow as pa

            # Test basic functionality using the shared spark_session fixture
            data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
            df = spark_session.createDataFrame(data)
            result = df.collect()
            assert len(result) == 2
        except ImportError:
            pytest.fail("PySpark is not installed but is required for this test environment")


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason=SKIP_REASON or "PySpark is not available")
class TestSparkFrameworkComputeFramework:
    def test_expected_data_framework(self, spark_session: Any) -> None:
        from pyspark.sql import DataFrame

        spark_framework = SparkFramework(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        assert spark_framework.expected_data_framework() == DataFrame

    def test_transform_dict_to_dataframe(self, spark_session: Any) -> None:
        spark_framework = SparkFramework(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        spark_framework.set_framework_connection_object(spark_session)

        dict_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}
        result = spark_framework.transform(dict_data, set())
        result_data = result.collect()

        expected_data = spark_session.createDataFrame(
            [{"column1": 1, "column2": 4}, {"column1": 2, "column2": 5}, {"column1": 3, "column2": 6}]
        )
        expected_data_collected = expected_data.collect()

        # Compare the data (order might be different)
        assert len(result_data) == len(expected_data_collected)
        assert set(result.columns) == set(expected_data.columns)

    def test_transform_invalid_data(self, spark_session: Any) -> None:
        spark_framework = SparkFramework(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        spark_framework.set_framework_connection_object(spark_session)

        with pytest.raises(ValueError):
            spark_framework.transform(data=["a"], feature_names=set())

    def test_select_data_by_column_names(self, spark_session: Any) -> None:
        spark_framework = SparkFramework(mode=ParallelizationModes.SYNC, children_if_root=frozenset())

        expected_data = spark_session.createDataFrame(
            [{"column1": 1, "column2": 4}, {"column1": 2, "column2": 5}, {"column1": 3, "column2": 6}]
        )
        data = spark_framework.select_data_by_column_names(expected_data, {FeatureName("column1")})
        assert data.columns == ["column1"]

    def test_set_column_names(self, spark_session: Any) -> None:
        spark_framework = SparkFramework(mode=ParallelizationModes.SYNC, children_if_root=frozenset())

        expected_data = spark_session.createDataFrame(
            [{"column1": 1, "column2": 4}, {"column1": 2, "column2": 5}, {"column1": 3, "column2": 6}]
        )
        spark_framework.data = expected_data
        spark_framework.set_column_names()
        assert spark_framework.column_names == {"column1", "column2"}

    def test_merge_inner(self, spark_session: Any) -> None:
        spark_framework = SparkFramework(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        spark_framework.set_framework_connection_object(spark_session)

        left_data = spark_session.createDataFrame([{"idx": 1, "col1": "a"}, {"idx": 3, "col1": "b"}])
        right_data = spark_session.createDataFrame([{"idx": 1, "col2": "x"}, {"idx": 2, "col2": "z"}])
        idx = Index(("idx",))

        spark_framework.data = left_data
        merge_engine_class = spark_framework.merge_engine()
        framework_connection = spark_framework.get_framework_connection_object()
        merge_engine = merge_engine_class(framework_connection)
        result = merge_engine.merge(left_data, right_data, JoinType.INNER, idx, idx)

        # Check that we got a result and it has the expected structure
        assert result is not None
        result_count = result.count()
        assert result_count == 1  # Should have 1 matching row

    def test_merge_left(self, spark_session: Any) -> None:
        spark_framework = SparkFramework(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        spark_framework.set_framework_connection_object(spark_session)

        left_data = spark_session.createDataFrame([{"idx": 1, "col1": "a"}, {"idx": 3, "col1": "b"}])
        right_data = spark_session.createDataFrame([{"idx": 1, "col2": "x"}, {"idx": 2, "col2": "z"}])
        idx = Index(("idx",))

        spark_framework.data = left_data
        framework_connection = spark_framework.get_framework_connection_object()
        merge_engine_class = spark_framework.merge_engine()
        merge_engine = merge_engine_class(framework_connection)
        result = merge_engine.merge(left_data, right_data, JoinType.LEFT, idx, idx)

        # Check that we got a result with all left rows
        assert result is not None
        result_count = result.count()
        assert result_count == 2  # Should have 2 rows (all from left)

    def test_merge_append(self, spark_session: Any) -> None:
        spark_framework = SparkFramework(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        spark_framework.set_framework_connection_object(spark_session)

        left_data = spark_session.createDataFrame([{"idx": 1, "col1": "a"}, {"idx": 3, "col1": "b"}])
        right_data = spark_session.createDataFrame([{"idx": 1, "col2": "x"}, {"idx": 2, "col2": "z"}])
        idx = Index(("idx",))

        spark_framework.data = left_data
        framework_connection = spark_framework.get_framework_connection_object()
        merge_engine_class = spark_framework.merge_engine()
        merge_engine = merge_engine_class(framework_connection)
        result = merge_engine.merge(left_data, right_data, JoinType.APPEND, idx, idx)

        # Check that we got a result with combined rows
        assert result is not None
        result_count = result.count()
        assert result_count == 4  # Should have 2 + 2 rows

    def test_merge_union(self, spark_session: Any) -> None:
        spark_framework = SparkFramework(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        spark_framework.set_framework_connection_object(spark_session)

        left_data = spark_session.createDataFrame([{"idx": 1, "col1": "a"}, {"idx": 3, "col1": "b"}])
        right_data = spark_session.createDataFrame([{"idx": 1, "col2": "x"}, {"idx": 2, "col2": "z"}])
        idx = Index(("idx",))

        spark_framework.data = left_data
        framework_connection = spark_framework.get_framework_connection_object()
        merge_engine_class = spark_framework.merge_engine()
        merge_engine = merge_engine_class(framework_connection)
        result = merge_engine.merge(left_data, right_data, JoinType.UNION, idx, idx)

        # Check that we got a result (union removes duplicates)
        assert result is not None
        result_count = result.count()
        # The exact count depends on duplicate handling, but should be <= 4
        assert result_count <= 4

    def test_framework_connection_object(self, spark_session: Any) -> None:
        """Test that framework connection object is properly set and retrieved."""
        framework = SparkFramework(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        framework.set_framework_connection_object(spark_session)

        connection = framework.get_framework_connection_object()
        assert connection is not None
        if PYSPARK_AVAILABLE:
            assert isinstance(connection, SparkSession)

    def test_framework_connection_object_invalid_type(self, spark_session: Any) -> None:
        """Test that setting invalid connection object raises error."""
        framework = SparkFramework(mode=ParallelizationModes.SYNC, children_if_root=frozenset())

        with pytest.raises(ValueError, match="Expected a SparkSession object"):
            framework.set_framework_connection_object("invalid")

    def test_transform_empty_dict(self, spark_session: Any) -> None:
        """Test transformation of empty dictionary."""
        spark_framework = SparkFramework(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        spark_framework.set_framework_connection_object(spark_session)

        result = spark_framework.transform({}, set())
        assert result is not None
        assert result.count() == 0

    def test_infer_spark_type(self, spark_session: Any) -> None:
        """Test Spark type inference."""
        from pyspark.sql.types import BooleanType, IntegerType, DoubleType, StringType

        spark_framework = SparkFramework(mode=ParallelizationModes.SYNC, children_if_root=frozenset())

        assert isinstance(spark_framework._infer_spark_type(True), BooleanType)
        assert isinstance(spark_framework._infer_spark_type(42), IntegerType)
        assert isinstance(spark_framework._infer_spark_type(3.14), DoubleType)
        assert isinstance(spark_framework._infer_spark_type("test"), StringType)
