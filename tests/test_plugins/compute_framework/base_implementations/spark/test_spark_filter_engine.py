"""Unit tests for the SparkFilterEngine class.

Requires PySpark installed and JAVA_HOME set; see conftest.py for the shared SparkSession.
"""

from typing import Any
import pytest
from mloda.user import Feature
from mloda.user import SingleFilter
from mloda.user import FilterType
from mloda_plugins.compute_framework.base_implementations.spark.spark_filter_engine import SparkFilterEngine

# Import shared fixtures and availability flags from conftest.py
from tests.test_plugins.compute_framework.base_implementations.spark.conftest import (
    PYSPARK_AVAILABLE,
    SKIP_REASON,
)
from tests.test_plugins.compute_framework.base_implementations.filter_engine_test_mixin import (
    FilterEngineTestMixin,
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
def spark_sample_data(spark_session: Any) -> Any:
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
class TestSparkFilterEngine(FilterEngineTestMixin):
    filter_engine_class = SparkFilterEngine

    @pytest.fixture
    def sample_data(self, spark_session: Any) -> Any:
        return spark_session.createDataFrame(
            [
                (1, 25, "Alice", "A"),
                (2, 30, "Bob", "B"),
                (3, 35, "Charlie", "A"),
                (4, 40, "David", "C"),
                (5, 45, "Eve", "B"),
            ],
            "id bigint, age bigint, name string, category string",
        )

    @pytest.fixture
    def nullable_category_sample_data(self, spark_session: Any) -> Any:
        return spark_session.createDataFrame(
            [
                (1, "A", 1),
                (2, None, None),
                (3, "B", 2),
                (4, None, None),
                (5, "C", 3),
            ],
            "id bigint, category string, score bigint",
        )

    @pytest.fixture
    def decimal_sample_data(self, spark_session: Any) -> Any:
        from decimal import Decimal

        rows = [(Decimal("12.34"),), (Decimal("5.50"),), (Decimal("99.99"),), (None,)]
        return spark_session.createDataFrame(rows, "d decimal(10,2)")

    def get_column_values(self, result: Any, column: str) -> list[Any]:
        return [row[column] for row in result.collect()]

    def get_decimal_column_dtype(self, data: Any) -> Any:
        return data.schema["d"].dataType

    def result_row_count(self, result: Any) -> int:
        """Spark DataFrames have no ``__len__``."""
        count: int = result.count()
        return count

    def test_min_filter_missing_value(self, spark_sample_data: Any) -> None:
        """Test min filter with missing value parameter."""
        feature = Feature("age")
        filter_type = FilterType.MIN
        parameter = {"invalid": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
            SparkFilterEngine.do_min_filter(spark_sample_data, single_filter)

    def test_max_filter_complex_inclusive(self, spark_sample_data: Any) -> None:
        """Test maximum value filter with complex max parameter (inclusive)."""
        feature = Feature("age")
        filter_type = FilterType.MAX
        parameter = {"max": 30, "max_exclusive": False}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_max_filter(spark_sample_data, single_filter)
        result_data = result.collect()

        # Should include ages <= 30
        assert len(result_data) == 4
        ages = sorted([row["age"] for row in result_data])
        assert ages == [22, 25, 28, 30]

    def test_max_filter_invalid_parameters(self, spark_sample_data: Any) -> None:
        """Test max filter with invalid parameters."""
        feature = Feature("age")
        filter_type = FilterType.MAX
        parameter = {"invalid": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="No valid filter parameter found"):
            SparkFilterEngine.do_max_filter(spark_sample_data, single_filter)

    def test_max_filter_with_min_parameter(self, spark_sample_data: Any) -> None:
        """Test max filter with min parameter (should raise error)."""
        feature = Feature("age")
        filter_type = FilterType.MAX
        parameter = {"min": 20, "max": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter .* not supported as max filter"):
            SparkFilterEngine.do_max_filter(spark_sample_data, single_filter)

    def test_equal_filter_string(self, spark_sample_data: Any) -> None:
        """Test equality filter on string column."""
        feature = Feature("name")
        filter_type = FilterType.EQUAL
        parameter = {"value": "Alice"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_equal_filter(spark_sample_data, single_filter)
        result_data = result.collect()

        # Should include only name == "Alice"
        assert len(result_data) == 1
        assert result_data[0]["name"] == "Alice"
        assert result_data[0]["age"] == 25

    def test_equal_filter_boolean(self, spark_sample_data: Any) -> None:
        """Test equality filter on boolean column."""
        feature = Feature("is_active")
        filter_type = FilterType.EQUAL
        parameter = {"value": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_equal_filter(spark_sample_data, single_filter)
        result_data = result.collect()

        # Should include only is_active == True
        assert len(result_data) == 3
        for row in result_data:
            assert row["is_active"] is True

    def test_equal_filter_missing_value(self, spark_sample_data: Any) -> None:
        """Test equal filter with missing value parameter."""
        feature = Feature("age")
        filter_type = FilterType.EQUAL
        parameter = {"invalid": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
            SparkFilterEngine.do_equal_filter(spark_sample_data, single_filter)

    def test_regex_filter_multiple_matches(self, spark_sample_data: Any) -> None:
        """Test regex filter with multiple matches."""
        feature = Feature("name")
        filter_type = FilterType.REGEX
        parameter = {"value": ".*e$"}  # Names ending with 'e'
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_regex_filter(spark_sample_data, single_filter)
        result_data = result.collect()

        # Should include "Alice", "Charlie", and "Eve"
        assert len(result_data) == 3
        names = sorted([row["name"] for row in result_data])
        assert names == ["Alice", "Charlie", "Eve"]

    def test_regex_filter_missing_value(self, spark_sample_data: Any) -> None:
        """Test regex filter with missing value parameter."""
        feature = Feature("name")
        filter_type = FilterType.REGEX
        parameter = {"invalid": "^A.*"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
            SparkFilterEngine.do_regex_filter(spark_sample_data, single_filter)

    def test_categorical_inclusion_filter_single_value(self, spark_sample_data: Any) -> None:
        """Test categorical inclusion filter with single value."""
        feature = Feature("category")
        filter_type = FilterType.CATEGORICAL_INCLUSION
        parameter = {"values": ["C"]}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_categorical_inclusion_filter(spark_sample_data, single_filter)
        result_data = result.collect()

        # Should include only category C
        assert len(result_data) == 1
        assert result_data[0]["category"] == "C"
        assert result_data[0]["name"] == "David"

    def test_categorical_inclusion_filter_missing_values(self, spark_sample_data: Any) -> None:
        """Test categorical inclusion filter with missing values parameter."""
        feature = Feature("category")
        filter_type = FilterType.CATEGORICAL_INCLUSION
        parameter = {"invalid": ["A", "B"]}
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'values' not found"):
            SparkFilterEngine.do_categorical_inclusion_filter(spark_sample_data, single_filter)

    def test_filter_on_float_column(self, spark_sample_data: Any) -> None:
        """Test filters on float/double columns."""
        feature = Feature("score")
        filter_type = FilterType.RANGE
        parameter = {"min": 80.0, "max": 90.0, "max_exclusive": False}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_range_filter(spark_sample_data, single_filter)
        result_data = result.collect()

        # Should include scores between 80.0 and 90.0
        assert len(result_data) == 2
        scores = [row["score"] for row in result_data]
        assert all(80.0 <= score <= 90.0 for score in scores)
        assert sorted(scores) == [85.5, 88.0]

    def test_filter_empty_result(self, spark_sample_data: Any) -> None:
        """Test filter that returns empty result."""
        feature = Feature("age")
        filter_type = FilterType.EQUAL
        parameter = {"value": 100}  # No one is 100 years old
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_equal_filter(spark_sample_data, single_filter)
        result_data = result.collect()

        # Should return empty result
        assert len(result_data) == 0

    def test_filter_nonexistent_column(self, spark_sample_data: Any) -> None:
        """Test filter on nonexistent column."""
        feature = Feature("nonexistent")
        filter_type = FilterType.EQUAL
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # This should raise an exception when Spark tries to access the column
        with pytest.raises(Exception):  # Could be AnalysisException or similar
            result = SparkFilterEngine.do_equal_filter(spark_sample_data, single_filter)
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
        filter_type = FilterType.REGEX
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
        filter_type = FilterType.MIN
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
        filter_type = FilterType.MIN
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = SparkFilterEngine.do_min_filter(empty_data, single_filter)
        result_data = result.collect()
        assert len(result_data) == 0
