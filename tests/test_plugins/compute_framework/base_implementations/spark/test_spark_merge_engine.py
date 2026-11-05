"""
Spark Merge Engine Tests

This module contains comprehensive tests for the Spark merge engine implementation.

Requirements:
- PySpark must be installed (pip install pyspark)
- Java 17+ must be installed and JAVA_HOME environment variable must be set

Environment Setup:
- JAVA_HOME: Must point to a valid Java installation

Test Coverage:
- Join types and multi-index through the shared MergeEngineSuite host
- Column conflict handling
- Multi-index validation
- Join logic with same and different column names

The tests use a shared SparkSession fixture to avoid Java gateway conflicts and
ensure proper resource management across all test methods.
"""

from collections.abc import Iterator
from typing import Any
import pytest
from mloda._testing.compute_framework.merge_suite import MergeEngineSuite
from mloda._testing.declarations import Cause
from mloda._testing.framework_adapter import FrameworkAdapter, SparkAdapter
from mloda.user import Index
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


class TestSparkMergeEngineSuite(MergeEngineSuite):
    adapter_class = SparkAdapter
    switched_off = {
        "test_merge_append": Cause.BACKEND,
        "test_merge_union": Cause.BACKEND,
        "test_merge_inner_with_multi_index": Cause.NOT_IMPLEMENTED,
        "test_merge_left_with_multi_index": Cause.NOT_IMPLEMENTED,
        "test_merge_full_outer_with_multi_index": Cause.NOT_IMPLEMENTED,
    }

    @pytest.fixture
    def adapter(self, spark_session: Any) -> Iterator[FrameworkAdapter]:
        """SparkAdapter borrows the active session, so spark_session must exist first."""
        spark_adapter = self.adapter_class()
        with spark_adapter:
            yield spark_adapter

    @pytest.mark.skip(reason="Spark unions by position, so frames with differently named columns do not align")
    def test_merge_append(self, adapter: FrameworkAdapter, merge_engine: Any) -> None: ...

    @pytest.mark.skip(reason="Spark unions by position, so frames with differently named columns do not align")
    def test_merge_union(self, adapter: FrameworkAdapter, merge_engine: Any) -> None: ...

    @pytest.mark.skip(reason="SparkMergeEngine rejects multi-column indexes, see test_multi_index_not_supported")
    def test_merge_inner_with_multi_index(self, adapter: FrameworkAdapter, merge_engine: Any) -> None: ...

    @pytest.mark.skip(reason="SparkMergeEngine rejects multi-column indexes, see test_multi_index_not_supported")
    def test_merge_left_with_multi_index(self, adapter: FrameworkAdapter, merge_engine: Any) -> None: ...

    @pytest.mark.skip(reason="SparkMergeEngine rejects multi-column indexes, see test_multi_index_not_supported")
    def test_merge_full_outer_with_multi_index(self, adapter: FrameworkAdapter, merge_engine: Any) -> None: ...
