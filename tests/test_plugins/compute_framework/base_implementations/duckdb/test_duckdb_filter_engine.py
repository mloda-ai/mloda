from typing import Any
import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.filter.single_filter import SingleFilter
from mloda_core.filter.filter_type_enum import FilterTypeEnum
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_filter_engine import DuckDBFilterEngine

import logging

logger = logging.getLogger(__name__)

try:
    import duckdb
    import pyarrow as pa
except ImportError:
    logger.warning("DuckDB or PyArrow is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]
    pa = None


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed. Skipping this test.")
class TestDuckDBFilterEngine:
    """Unit tests for the DuckDBFilterEngine class."""

    @pytest.fixture
    def sample_data(self, connection: Any) -> Any:
        """Create a sample DuckDB relation for testing."""
        arrow_table = pa.Table.from_pydict(
            {
                "id": [1, 2, 3, 4, 5],
                "age": [25, 30, 35, 40, 45],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "category": ["A", "B", "A", "C", "B"],
            }
        )
        return connection.from_arrow(arrow_table)

    def test_do_range_filter(self, sample_data: Any) -> None:
        """Test range filter with min and max values."""
        # Create a filter for age between 30 and 40
        feature = Feature("age")
        filter_type = FilterTypeEnum.range
        parameter = {"min": 30, "max": 40, "max_exclusive": False}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = DuckDBFilterEngine.do_range_filter(sample_data, single_filter)

        # Check the result
        result_df = result.df()
        assert len(result_df) == 3
        ages = sorted(result_df["age"].tolist())
        ids = sorted(result_df["id"].tolist())
        assert ages == [30, 35, 40]
        assert ids == [2, 3, 4]

    def test_do_range_filter_exclusive(self, sample_data: Any) -> None:
        """Test range filter with exclusive max value."""
        # Create a filter for age between 30 and 40 (exclusive)
        feature = Feature("age")
        filter_type = FilterTypeEnum.range
        parameter = {"min": 30, "max": 40, "max_exclusive": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = DuckDBFilterEngine.do_range_filter(sample_data, single_filter)

        # Check the result
        result_df = result.df()
        assert len(result_df) == 2
        ages = sorted(result_df["age"].tolist())
        ids = sorted(result_df["id"].tolist())
        assert ages == [30, 35]
        assert ids == [2, 3]

    def test_do_min_filter(self, sample_data: Any) -> None:
        """Test min filter."""
        # Create a filter for age >= 40
        feature = Feature("age")
        filter_type = FilterTypeEnum.min
        parameter = {"value": 40}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = DuckDBFilterEngine.do_min_filter(sample_data, single_filter)

        # Check the result
        result_df = result.df()
        assert len(result_df) == 2
        ages = sorted(result_df["age"].tolist())
        ids = sorted(result_df["id"].tolist())
        assert ages == [40, 45]
        assert ids == [4, 5]

    def test_do_max_filter(self, sample_data: Any) -> None:
        """Test max filter."""
        # Create a filter for age <= 30
        feature = Feature("age")
        filter_type = FilterTypeEnum.max
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = DuckDBFilterEngine.do_max_filter(sample_data, single_filter)

        # Check the result
        result_df = result.df()
        assert len(result_df) == 2
        ages = sorted(result_df["age"].tolist())
        ids = sorted(result_df["id"].tolist())
        assert ages == [25, 30]
        assert ids == [1, 2]

    def test_do_max_filter_with_tuple(self, sample_data: Any) -> None:
        """Test max filter with tuple parameter."""
        # Create a filter for age < 35
        feature = Feature("age")
        filter_type = FilterTypeEnum.max
        parameter = {"max": 35, "max_exclusive": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = DuckDBFilterEngine.do_max_filter(sample_data, single_filter)

        # Check the result
        result_df = result.df()
        assert len(result_df) == 2
        ages = sorted(result_df["age"].tolist())
        ids = sorted(result_df["id"].tolist())
        assert ages == [25, 30]
        assert ids == [1, 2]

    def test_do_equal_filter(self, sample_data: Any) -> None:
        """Test equal filter."""
        # Create a filter for age == 35
        feature = Feature("age")
        filter_type = FilterTypeEnum.equal
        parameter = {"value": 35}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = DuckDBFilterEngine.do_equal_filter(sample_data, single_filter)

        # Check the result
        result_df = result.df()
        assert len(result_df) == 1
        assert result_df["age"].tolist()[0] == 35
        assert result_df["id"].tolist()[0] == 3

    def test_do_regex_filter(self, sample_data: Any) -> None:
        """Test regex filter."""
        # Create a filter for names starting with 'A'
        feature = Feature("name")
        filter_type = FilterTypeEnum.regex
        parameter = {"value": "^A"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = DuckDBFilterEngine.do_regex_filter(sample_data, single_filter)

        # Check the result
        result_df = result.df()
        assert len(result_df) == 1
        assert result_df["name"].tolist()[0] == "Alice"
        assert result_df["id"].tolist()[0] == 1

    def test_do_categorical_inclusion_filter(self, sample_data: Any) -> None:
        """Test categorical inclusion filter."""
        # Create a filter for category in ['A', 'B']
        feature = Feature("category")
        filter_type = FilterTypeEnum.categorical_inclusion
        parameter = {"values": ["A", "B"]}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = DuckDBFilterEngine.do_categorical_inclusion_filter(sample_data, single_filter)

        # Check the result
        result_df = result.df()
        assert len(result_df) == 4
        categories = set(result_df["category"].tolist())
        ids = set(result_df["id"].tolist())
        assert categories == {"A", "B"}
        assert ids == {1, 2, 3, 5}

    def test_apply_filters(self, sample_data: Any) -> None:
        """Test applying multiple filters."""
        # Create a feature set with filters
        feature = Feature("age")
        filters = [
            SingleFilter(feature, FilterTypeEnum.min, {"value": 30}),
            SingleFilter(Feature("category"), FilterTypeEnum.equal, {"value": "A"}),
        ]

        # Create a mock feature set
        class MockFeatureSet:
            def __init__(self, filters: Any) -> None:
                self.filters = filters

            def get_all_names(self) -> Any:
                return ["age", "category"]

        feature_set = MockFeatureSet(filters)

        # Apply the filters
        result = DuckDBFilterEngine.apply_filters(sample_data, feature_set)

        # Check the result
        result_df = result.df()
        assert len(result_df) == 1
        assert result_df["age"].tolist()[0] == 35
        assert result_df["category"].tolist()[0] == "A"
        assert result_df["id"].tolist()[0] == 3

    def test_final_filters(self) -> None:
        """Test that final_filters returns True."""
        assert DuckDBFilterEngine.final_filters() is True

    def test_do_range_filter_missing_parameters(self, sample_data: Any) -> None:
        """Test range filter with missing parameters."""
        feature = Feature("age")
        filter_type = FilterTypeEnum.range
        parameter = {"min": 30}  # Missing max parameter
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter .* not supported"):
            DuckDBFilterEngine.do_range_filter(sample_data, single_filter)

    def test_filter_with_null_values(self, sample_data: Any) -> None:
        """Test filtering with null values in data."""
        # Create extended data with null values
        conn = duckdb.connect()
        arrow_table = pa.Table.from_pydict(
            {
                "id": [1, 2, 3, 4, 5, 6],
                "age": [25, 30, 35, 40, 45, None],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
                "category": ["A", "B", "A", "C", "B", "A"],
            }
        )
        extended_data = conn.from_arrow(arrow_table)

        # Test min filter - should exclude null values
        feature = Feature("age")
        filter_type = FilterTypeEnum.min
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = DuckDBFilterEngine.do_min_filter(extended_data, single_filter)

        # Should not include the row with null age
        result_df = result.df()
        assert len(result_df) == 4  # All rows with age >= 30, excluding null
        ages = result_df["age"].tolist()
        assert None not in ages
        assert sorted(ages) == [30, 35, 40, 45]

    def test_filter_with_empty_data(self) -> None:
        """Test filtering with empty DuckDB relation."""
        conn = duckdb.connect()
        arrow_table = pa.Table.from_pydict(
            {
                "id": [],
                "age": [],
                "name": [],
                "category": [],
            }
        )
        empty_data = conn.from_arrow(arrow_table)

        # Test min filter on empty data
        feature = Feature("age")
        filter_type = FilterTypeEnum.min
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = DuckDBFilterEngine.do_min_filter(empty_data, single_filter)
        result_df = result.df()
        assert len(result_df) == 0

    def test_filter_with_string_data(self) -> None:
        """Test filtering with string data types."""
        conn = duckdb.connect()
        arrow_table = pa.Table.from_pydict(
            {
                "id": [1, 2, 3, 4],
                "status": ["active", "inactive", "pending", "active"],
                "priority": ["high", "low", "medium", "high"],
            }
        )
        data = conn.from_arrow(arrow_table)

        # Test equal filter on string
        feature = Feature("status")
        filter_type = FilterTypeEnum.equal
        parameter = {"value": "active"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = DuckDBFilterEngine.do_equal_filter(data, single_filter)
        result_df = result.df()
        assert len(result_df) == 2
        statuses = result_df["status"].tolist()
        assert all(status == "active" for status in statuses)

    def test_filter_with_boolean_data(self) -> None:
        """Test filtering with boolean data types."""
        conn = duckdb.connect()
        arrow_table = pa.Table.from_pydict(
            {
                "id": [1, 2, 3, 4],
                "is_active": [True, False, True, False],
                "is_premium": [False, True, True, False],
            }
        )
        data = conn.from_arrow(arrow_table)

        # Test equal filter on boolean
        feature = Feature("is_active")
        filter_type = FilterTypeEnum.equal
        parameter = {"value": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = DuckDBFilterEngine.do_equal_filter(data, single_filter)
        result_df = result.df()
        assert len(result_df) == 2
        active_flags = result_df["is_active"].tolist()
        assert all(flag is True for flag in active_flags)

    def test_complex_regex_patterns(self) -> None:
        """Test complex regex patterns."""
        conn = duckdb.connect()
        arrow_table = pa.Table.from_pydict(
            {
                "id": [1, 2, 3, 4, 5],
                "email": ["alice@test.com", "bob@example.org", "charlie@test.com", "david@company.net", "eve@test.org"],
            }
        )
        data = conn.from_arrow(arrow_table)

        # Test regex filter for emails ending with .com
        feature = Feature("email")
        filter_type = FilterTypeEnum.regex
        parameter = {"value": r"\.com$"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = DuckDBFilterEngine.do_regex_filter(data, single_filter)
        result_df = result.df()
        assert len(result_df) == 2
        emails = result_df["email"].tolist()
        assert all(email.endswith(".com") for email in emails)
