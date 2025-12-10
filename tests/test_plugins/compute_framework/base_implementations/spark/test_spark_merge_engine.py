"""
Spark Merge Engine Tests

This module contains comprehensive tests for the Spark merge engine implementation.

Requirements:
- PySpark must be installed (pip install pyspark)
- Java 8+ must be installed and JAVA_HOME environment variable must be set

Environment Setup:
- JAVA_HOME: Must point to a valid Java installation

Test Coverage:
- All join types (inner, left, right, full outer, append, union)
- Column conflict handling
- Multi-index validation
- Join logic with same and different column names

The tests use a shared SparkSession fixture to avoid Java gateway conflicts and
ensure proper resource management across all test methods.
"""

from typing import Any
import pytest
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_plugins.compute_framework.base_implementations.spark.spark_merge_engine import SparkMergeEngine

# Import shared fixtures and availability flags from conftest.py
from tests.test_plugins.compute_framework.base_implementations.spark.conftest import (
    PYSPARK_AVAILABLE,
    SKIP_REASON,
)

import logging

logger = logging.getLogger(__name__)


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason=SKIP_REASON or "PySpark is not available")
class TestSparkMergeEngine:
    def test_check_import(self) -> None:
        """Test that check_import works correctly."""
        merge_engine = SparkMergeEngine()
        # Should not raise an exception if PySpark is available
        merge_engine.check_import()

    def test_merge_inner(self, spark_session: Any) -> None:
        """Test inner join functionality."""
        left_data = spark_session.createDataFrame([{"idx": 1, "col1": "a"}, {"idx": 3, "col1": "b"}])
        right_data = spark_session.createDataFrame([{"idx": 1, "col2": "x"}, {"idx": 2, "col2": "z"}])
        idx = Index(("idx",))

        merge_engine = SparkMergeEngine(spark_session)
        result = merge_engine.merge_inner(left_data, right_data, idx, idx)

        assert result is not None
        result_count = result.count()
        assert result_count == 1  # Only one matching row (idx=1)

        # Check that both columns are present
        columns = set(result.columns)
        assert "col1" in columns
        assert "col2" in columns

    def test_merge_left(self, spark_session: Any) -> None:
        """Test left join functionality."""
        left_data = spark_session.createDataFrame([{"idx": 1, "col1": "a"}, {"idx": 3, "col1": "b"}])
        right_data = spark_session.createDataFrame([{"idx": 1, "col2": "x"}, {"idx": 2, "col2": "z"}])
        idx = Index(("idx",))

        merge_engine = SparkMergeEngine(spark_session)
        result = merge_engine.merge_left(left_data, right_data, idx, idx)

        assert result is not None
        result_count = result.count()
        assert result_count == 2  # All rows from left table

    def test_merge_right(self, spark_session: Any) -> None:
        """Test right join functionality."""
        left_data = spark_session.createDataFrame([{"idx": 1, "col1": "a"}, {"idx": 3, "col1": "b"}])
        right_data = spark_session.createDataFrame([{"idx": 1, "col2": "x"}, {"idx": 2, "col2": "z"}])
        idx = Index(("idx",))

        merge_engine = SparkMergeEngine(spark_session)
        result = merge_engine.merge_right(left_data, right_data, idx, idx)

        assert result is not None
        result_count = result.count()
        assert result_count == 2  # All rows from right table

    def test_merge_full_outer(self, spark_session: Any) -> None:
        """Test full outer join functionality."""
        left_data = spark_session.createDataFrame([{"idx": 1, "col1": "a"}, {"idx": 3, "col1": "b"}])
        right_data = spark_session.createDataFrame([{"idx": 1, "col2": "x"}, {"idx": 2, "col2": "z"}])
        idx = Index(("idx",))

        merge_engine = SparkMergeEngine(spark_session)
        result = merge_engine.merge_full_outer(left_data, right_data, idx, idx)

        assert result is not None
        result_count = result.count()
        assert result_count == 3  # All unique rows from both tables

    def test_merge_append(self, spark_session: Any) -> None:
        """Test append (union all) functionality."""
        left_data = spark_session.createDataFrame([{"idx": 1, "col1": "a"}, {"idx": 3, "col1": "b"}])
        right_data = spark_session.createDataFrame([{"idx": 1, "col2": "x"}, {"idx": 2, "col2": "z"}])
        idx = Index(("idx",))

        merge_engine = SparkMergeEngine(spark_session)
        result = merge_engine.merge_append(left_data, right_data, idx, idx)

        assert result is not None
        result_count = result.count()
        assert result_count == 4  # 2 + 2 rows

    def test_merge_union(self, spark_session: Any) -> None:
        """Test union (with deduplication) functionality."""
        left_data = spark_session.createDataFrame([{"idx": 1, "col1": "a"}, {"idx": 3, "col1": "b"}])
        right_data = spark_session.createDataFrame([{"idx": 1, "col2": "x"}, {"idx": 2, "col2": "z"}])
        idx = Index(("idx",))

        merge_engine = SparkMergeEngine(spark_session)
        result = merge_engine.merge_union(left_data, right_data, idx, idx)

        assert result is not None
        result_count = result.count()
        # Should be <= 4 due to deduplication
        assert result_count <= 4

    def test_join_logic_same_column_names(self, spark_session: Any) -> None:
        """Test join logic when both tables have the same column name."""
        left_data = spark_session.createDataFrame([{"idx": 1, "col1": "a"}, {"idx": 3, "col1": "b"}])
        right_data = spark_session.createDataFrame([{"idx": 1, "col2": "x"}, {"idx": 2, "col2": "z"}])
        idx = Index(("idx",))

        merge_engine = SparkMergeEngine(spark_session)
        result = merge_engine._join_logic("inner", left_data, right_data, idx, idx)

        assert result is not None
        assert result.count() == 1

    def test_join_logic_different_column_names(self, spark_session: Any) -> None:
        """Test join logic when tables have different column names."""
        # Create data with different index column names
        left_data_diff = spark_session.createDataFrame([{"left_idx": 1, "col1": "a"}, {"left_idx": 3, "col1": "b"}])
        right_data_diff = spark_session.createDataFrame([{"right_idx": 1, "col2": "x"}, {"right_idx": 2, "col2": "z"}])

        left_idx = Index(("left_idx",))
        right_idx = Index(("right_idx",))

        merge_engine = SparkMergeEngine(spark_session)
        result = merge_engine._join_logic("inner", left_data_diff, right_data_diff, left_idx, right_idx)

        assert result is not None
        assert result.count() == 1

    def test_multi_index_not_supported(self, spark_session: Any) -> None:
        """Test that multi-index raises appropriate error."""
        left_data = spark_session.createDataFrame([{"idx": 1, "col1": "a"}, {"idx": 3, "col1": "b"}])
        right_data = spark_session.createDataFrame([{"idx": 1, "col2": "x"}, {"idx": 2, "col2": "z"}])
        idx = Index(("idx",))
        multi_idx = Index(("col1", "col2"))

        merge_engine = SparkMergeEngine(spark_session)

        with pytest.raises(ValueError, match="MultiIndex is not yet implemented"):
            merge_engine._join_logic("inner", left_data, right_data, multi_idx, idx)

    def test_handle_column_conflicts(self, spark_session: Any) -> None:
        """Test column conflict handling."""
        # Create data with conflicting column names
        left_data_conflict = spark_session.createDataFrame(
            [{"idx": 1, "common_col": "left_a"}, {"idx": 3, "common_col": "left_b"}]
        )
        right_data_conflict = spark_session.createDataFrame(
            [{"idx": 1, "common_col": "right_x"}, {"idx": 2, "common_col": "right_z"}]
        )
        idx = Index(("idx",))

        merge_engine = SparkMergeEngine(spark_session)
        left_result, right_result = merge_engine._handle_column_conflicts(
            left_data_conflict, right_data_conflict, idx, idx
        )

        # Check that conflicting columns in right DataFrame are renamed
        assert "common_col_right" in right_result.columns
        assert "common_col" in left_result.columns
