"""
Spark Filter Engine Tests

This module contains comprehensive tests for the Spark filter engine implementation.

Requirements:
- PySpark must be installed (pip install pyspark)
- Java 8+ must be installed and JAVA_HOME environment variable must be set

Environment Setup:
- JAVA_HOME: Must point to a valid Java installation

Test Coverage:
- All filter types (range, min, max, equal, regex, categorical)
- Parameter validation and error handling
- Edge cases and boundary conditions
- Integration with Spark DataFrames

The tests use a shared SparkSession fixture to avoid Java gateway conflicts and
ensure proper resource management across all test methods.
"""

from typing import Any
import pytest
from mloda import Feature
from mloda.user import SingleFilter
from mloda.user import FilterType
from mloda_plugins.compute_framework.base_implementations.spark.spark_filter_engine import SparkFilterEngine

# Import shared fixtures and availability flags from conftest.py
from tests.test_plugins.compute_framework.base_implementations.spark.conftest import (
    PYSPARK_AVAILABLE,
    SKIP_REASON,
)

import logging

logger = logging.getLogger(__name__)

# Import PySpark types for schema creation (only if available)
if PYSPARK_AVAILABLE:
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType
else:
    StructType = None
    StructField = None
    StringType = None
    IntegerType = None
    DoubleType = None
    BooleanType = None


@pytest.fixture
def sample_data(spark_session: Any) -> Any:
    """Create sample data for testing filters."""
    if not PYSPARK_AVAILABLE:
        pytest.skip(SKIP_REASON or "PySpark is not available")

    data = [
        {"id": 1, "age": 25, "name": "Alice", "score": 85.5, "category": "A", "is_active": True},
        {"id": 2, "age": 30, "name": "Bob", "score": 92.0, "category": "B", "is_active": False},
        {"id": 3, "age": 35, "name": "Charlie", "score": 78.5, "category": "A", "is_active": True},
        {"id": 4, "age": 28, "name": "David", "score": 88.0, "category": "C", "is_active": False},
        {"id": 5, "age": 42, "name": "Eve", "score": 95.5, "category": "B", "is_active": True},
        {"id": 6, "age": 22, "name": "Frank", "score": 72.0, "category": "A", "is_active": False},
    ]
    return spark_session.createDataFrame(data)


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason=SKIP_REASON or "PySpark is not available")
class TestSparkFilterEngine:
    def test_final_filters(self) -> None:
        """Test that final_filters returns True."""
        assert SparkFilterEngine.final_filters() is True

    def test_range_filter_inclusive(self, sample_data: Any) -> None:
        """Test range filter with inclusive bounds."""
        feature = Feature("age")
        filter_type = FilterType.range
        parameter = {"min": 25, "max": 35, "max_exclusive": False}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_range_filter(sample_data, single_filter)
        result_data = result.collect()

        # Should include ages 25, 28, 30, 35 (inclusive)
        assert len(result_data) == 4
        ages = sorted([row["age"] for row in result_data])
        assert ages == [25, 28, 30, 35]

    def test_range_filter_exclusive_max(self, sample_data: Any) -> None:
        """Test range filter with exclusive max bound."""
        feature = Feature("age")
        filter_type = FilterType.range
        parameter = {"min": 25, "max": 35, "max_exclusive": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_range_filter(sample_data, single_filter)
        result_data = result.collect()

        # Should include ages 25, 28, 30 but not 35 (exclusive)
        assert len(result_data) == 3
        ages = sorted([row["age"] for row in result_data])
        assert ages == [25, 28, 30]

    def test_range_filter_missing_parameters(self, sample_data: Any) -> None:
        """Test range filter with missing parameters."""
        feature = Feature("age")
        filter_type = FilterType.range
        parameter = {"min": 25}  # Missing max parameter
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter .* not supported"):
            SparkFilterEngine.do_range_filter(sample_data, single_filter)

    def test_min_filter(self, sample_data: Any) -> None:
        """Test minimum value filter."""
        feature = Feature("age")
        filter_type = FilterType.min
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_min_filter(sample_data, single_filter)
        result_data = result.collect()

        # Should include ages >= 30
        assert len(result_data) == 3
        ages = [row["age"] for row in result_data]
        assert all(age >= 30 for age in ages)
        assert sorted(ages) == [30, 35, 42]

    def test_min_filter_missing_value(self, sample_data: Any) -> None:
        """Test min filter with missing value parameter."""
        feature = Feature("age")
        filter_type = FilterType.min
        parameter = {"invalid": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
            SparkFilterEngine.do_min_filter(sample_data, single_filter)

    def test_max_filter_simple(self, sample_data: Any) -> None:
        """Test maximum value filter with simple value parameter."""
        feature = Feature("age")
        filter_type = FilterType.max
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_max_filter(sample_data, single_filter)
        result_data = result.collect()

        # Should include ages <= 30
        assert len(result_data) == 4
        ages = sorted([row["age"] for row in result_data])
        assert ages == [22, 25, 28, 30]

    def test_max_filter_complex_inclusive(self, sample_data: Any) -> None:
        """Test maximum value filter with complex max parameter (inclusive)."""
        feature = Feature("age")
        filter_type = FilterType.max
        parameter = {"max": 30, "max_exclusive": False}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_max_filter(sample_data, single_filter)
        result_data = result.collect()

        # Should include ages <= 30
        assert len(result_data) == 4
        ages = sorted([row["age"] for row in result_data])
        assert ages == [22, 25, 28, 30]

    def test_max_filter_complex_exclusive(self, sample_data: Any) -> None:
        """Test maximum value filter with complex max_exclusive parameter."""
        feature = Feature("age")
        filter_type = FilterType.max
        parameter = {"max": 30, "max_exclusive": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_max_filter(sample_data, single_filter)
        result_data = result.collect()

        # Should include ages < 30
        assert len(result_data) == 3
        ages = sorted([row["age"] for row in result_data])
        assert ages == [22, 25, 28]

    def test_max_filter_invalid_parameters(self, sample_data: Any) -> None:
        """Test max filter with invalid parameters."""
        feature = Feature("age")
        filter_type = FilterType.max
        parameter = {"invalid": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="No valid filter parameter found"):
            SparkFilterEngine.do_max_filter(sample_data, single_filter)

    def test_max_filter_with_min_parameter(self, sample_data: Any) -> None:
        """Test max filter with min parameter (should raise error)."""
        feature = Feature("age")
        filter_type = FilterType.max
        parameter = {"min": 20, "max": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter .* not supported as max filter"):
            SparkFilterEngine.do_max_filter(sample_data, single_filter)

    def test_equal_filter(self, sample_data: Any) -> None:
        """Test equality filter."""
        feature = Feature("age")
        filter_type = FilterType.equal
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_equal_filter(sample_data, single_filter)
        result_data = result.collect()

        # Should include only age == 30
        assert len(result_data) == 1
        assert result_data[0]["age"] == 30
        assert result_data[0]["name"] == "Bob"

    def test_equal_filter_string(self, sample_data: Any) -> None:
        """Test equality filter on string column."""
        feature = Feature("name")
        filter_type = FilterType.equal
        parameter = {"value": "Alice"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_equal_filter(sample_data, single_filter)
        result_data = result.collect()

        # Should include only name == "Alice"
        assert len(result_data) == 1
        assert result_data[0]["name"] == "Alice"
        assert result_data[0]["age"] == 25

    def test_equal_filter_boolean(self, sample_data: Any) -> None:
        """Test equality filter on boolean column."""
        feature = Feature("is_active")
        filter_type = FilterType.equal
        parameter = {"value": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_equal_filter(sample_data, single_filter)
        result_data = result.collect()

        # Should include only is_active == True
        assert len(result_data) == 3
        for row in result_data:
            assert row["is_active"] is True

    def test_equal_filter_missing_value(self, sample_data: Any) -> None:
        """Test equal filter with missing value parameter."""
        feature = Feature("age")
        filter_type = FilterType.equal
        parameter = {"invalid": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
            SparkFilterEngine.do_equal_filter(sample_data, single_filter)

    def test_regex_filter(self, sample_data: Any) -> None:
        """Test regex filter."""
        feature = Feature("name")
        filter_type = FilterType.regex
        parameter = {"value": "^A.*"}  # Names starting with 'A'
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_regex_filter(sample_data, single_filter)
        result_data = result.collect()

        # Should include only "Alice"
        assert len(result_data) == 1
        assert result_data[0]["name"] == "Alice"

    def test_regex_filter_multiple_matches(self, sample_data: Any) -> None:
        """Test regex filter with multiple matches."""
        feature = Feature("name")
        filter_type = FilterType.regex
        parameter = {"value": ".*e$"}  # Names ending with 'e'
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_regex_filter(sample_data, single_filter)
        result_data = result.collect()

        # Should include "Alice", "Charlie", and "Eve"
        assert len(result_data) == 3
        names = sorted([row["name"] for row in result_data])
        assert names == ["Alice", "Charlie", "Eve"]

    def test_regex_filter_missing_value(self, sample_data: Any) -> None:
        """Test regex filter with missing value parameter."""
        feature = Feature("name")
        filter_type = FilterType.regex
        parameter = {"invalid": "^A.*"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
            SparkFilterEngine.do_regex_filter(sample_data, single_filter)

    def test_categorical_inclusion_filter(self, sample_data: Any) -> None:
        """Test categorical inclusion filter."""
        feature = Feature("category")
        filter_type = FilterType.categorical_inclusion
        parameter = {"values": ["A", "B"]}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_categorical_inclusion_filter(sample_data, single_filter)
        result_data = result.collect()

        # Should include categories A and B
        assert len(result_data) == 5  # 3 A's + 2 B's
        categories = [row["category"] for row in result_data]
        assert all(cat in ["A", "B"] for cat in categories)

    def test_categorical_inclusion_filter_single_value(self, sample_data: Any) -> None:
        """Test categorical inclusion filter with single value."""
        feature = Feature("category")
        filter_type = FilterType.categorical_inclusion
        parameter = {"values": ["C"]}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_categorical_inclusion_filter(sample_data, single_filter)
        result_data = result.collect()

        # Should include only category C
        assert len(result_data) == 1
        assert result_data[0]["category"] == "C"
        assert result_data[0]["name"] == "David"

    def test_categorical_inclusion_filter_missing_values(self, sample_data: Any) -> None:
        """Test categorical inclusion filter with missing values parameter."""
        feature = Feature("category")
        filter_type = FilterType.categorical_inclusion
        parameter = {"invalid": ["A", "B"]}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'values' not found"):
            SparkFilterEngine.do_categorical_inclusion_filter(sample_data, single_filter)

    def test_filter_on_float_column(self, sample_data: Any) -> None:
        """Test filters on float/double columns."""
        feature = Feature("score")
        filter_type = FilterType.range
        parameter = {"min": 80.0, "max": 90.0, "max_exclusive": False}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_range_filter(sample_data, single_filter)
        result_data = result.collect()

        # Should include scores between 80.0 and 90.0
        assert len(result_data) == 2
        scores = [row["score"] for row in result_data]
        assert all(80.0 <= score <= 90.0 for score in scores)
        assert sorted(scores) == [85.5, 88.0]

    def test_filter_empty_result(self, sample_data: Any) -> None:
        """Test filter that returns empty result."""
        feature = Feature("age")
        filter_type = FilterType.equal
        parameter = {"value": 100}  # No one is 100 years old
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_equal_filter(sample_data, single_filter)
        result_data = result.collect()

        # Should return empty result
        assert len(result_data) == 0

    def test_filter_nonexistent_column(self, sample_data: Any) -> None:
        """Test filter on nonexistent column."""
        feature = Feature("nonexistent")
        filter_type = FilterType.equal
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # This should raise an exception when Spark tries to access the column
        with pytest.raises(Exception):  # Could be AnalysisException or similar
            result = SparkFilterEngine.do_equal_filter(sample_data, single_filter)
            result.collect()  # Force evaluation

    def test_complex_regex_patterns(self, spark_session: Any) -> None:
        """Test complex regex patterns."""
        data = [
            {"id": 1, "email": "alice@test.com"},
            {"id": 2, "email": "bob@example.org"},
            {"id": 3, "email": "charlie@test.com"},
            {"id": 4, "email": "david@company.net"},
            {"id": 5, "email": "eve@test.org"},
        ]
        email_data = spark_session.createDataFrame(data)

        # Test regex filter for emails ending with .com
        feature = Feature("email")
        filter_type = FilterType.regex
        parameter = {"value": r"\.com$"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_regex_filter(email_data, single_filter)
        result_data = result.collect()

        assert len(result_data) == 2
        emails = [row["email"] for row in result_data]
        assert all(email.endswith(".com") for email in emails)

    def test_filter_with_null_values(self, spark_session: Any) -> None:
        """Test filtering with null values in data."""
        data = [
            {"id": 1, "age": 25, "name": "Alice"},
            {"id": 2, "age": 30, "name": "Bob"},
            {"id": 3, "age": None, "name": "Charlie"},
            {"id": 4, "age": 35, "name": "David"},
        ]
        null_data = spark_session.createDataFrame(data)

        # Test min filter - should exclude null values
        feature = Feature("age")
        filter_type = FilterType.min
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_min_filter(null_data, single_filter)
        result_data = result.collect()

        # Should not include the row with null age
        assert len(result_data) == 2  # Bob and David
        ages = [row["age"] for row in result_data]
        assert None not in ages
        assert sorted(ages) == [30, 35]

    def test_filter_with_empty_data(self, spark_session: Any) -> None:
        """Test filtering with empty Spark DataFrame."""
        schema = StructType(
            [
                StructField("id", IntegerType(), True),
                StructField("age", IntegerType(), True),
                StructField("name", StringType(), True),
            ]
        )
        empty_data = spark_session.createDataFrame([], schema)

        # Test min filter on empty data
        feature = Feature("age")
        filter_type = FilterType.min
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_min_filter(empty_data, single_filter)
        result_data = result.collect()
        assert len(result_data) == 0
