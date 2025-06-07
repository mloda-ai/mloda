from typing import Any
import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.filter.single_filter import SingleFilter
from mloda_core.filter.filter_type_enum import FilterTypeEnum
from mloda_plugins.compute_framework.base_implementations.polars.polars_filter_engine import PolarsFilterEngine

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsFilterEngine:
    """Unit tests for the PolarsFilterEngine class."""

    @pytest.fixture
    def sample_data(self) -> Any:
        """Create a sample Polars DataFrame for testing."""
        return pl.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "age": [25, 30, 35, 40, 45],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "category": ["A", "B", "A", "C", "B"],
            }
        )

    def test_do_range_filter(self, sample_data: Any) -> None:
        """Test range filter with min and max values."""
        # Create a filter for age between 30 and 40
        feature = Feature("age")
        filter_type = FilterTypeEnum.range
        parameter = {"min": 30, "max": 40, "max_exclusive": False}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = PolarsFilterEngine.do_range_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 3
        ages = result["age"].to_list()
        ids = result["id"].to_list()
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
        result = PolarsFilterEngine.do_range_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 2
        ages = result["age"].to_list()
        ids = result["id"].to_list()
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
        result = PolarsFilterEngine.do_min_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 2
        ages = result["age"].to_list()
        ids = result["id"].to_list()
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
        result = PolarsFilterEngine.do_max_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 2
        ages = result["age"].to_list()
        ids = result["id"].to_list()
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
        result = PolarsFilterEngine.do_max_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 2
        ages = result["age"].to_list()
        ids = result["id"].to_list()
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
        result = PolarsFilterEngine.do_equal_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 1
        assert result["age"].to_list()[0] == 35
        assert result["id"].to_list()[0] == 3

    def test_do_regex_filter(self, sample_data: Any) -> None:
        """Test regex filter."""
        # Create a filter for names starting with 'A'
        feature = Feature("name")
        filter_type = FilterTypeEnum.regex
        parameter = {"value": "^A"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = PolarsFilterEngine.do_regex_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 1
        assert result["name"].to_list()[0] == "Alice"
        assert result["id"].to_list()[0] == 1

    def test_do_categorical_inclusion_filter(self, sample_data: Any) -> None:
        """Test categorical inclusion filter."""
        # Create a filter for category in ['A', 'B']
        feature = Feature("category")
        filter_type = FilterTypeEnum.categorical_inclusion
        parameter = {"values": ["A", "B"]}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = PolarsFilterEngine.do_categorical_inclusion_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 4
        categories = set(result["category"].to_list())
        ids = set(result["id"].to_list())
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
        result = PolarsFilterEngine.apply_filters(sample_data, feature_set)

        # Check the result
        assert len(result) == 1
        assert result["age"].to_list()[0] == 35
        assert result["category"].to_list()[0] == "A"
        assert result["id"].to_list()[0] == 3

    def test_final_filters(self) -> None:
        """Test that final_filters returns True."""
        assert PolarsFilterEngine.final_filters() is True

    def test_do_range_filter_missing_parameters(self, sample_data: Any) -> None:
        """Test range filter with missing parameters."""
        feature = Feature("age")
        filter_type = FilterTypeEnum.range
        parameter = {"min": 30}  # Missing max parameter
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter .* not supported"):
            PolarsFilterEngine.do_range_filter(sample_data, single_filter)

    def test_filter_with_null_values(self, sample_data: Any) -> None:
        """Test filtering with null values in data."""
        # Create a simple extension with one more row
        extended_data = pl.concat(
            [sample_data, pl.DataFrame({"id": [6], "age": [None], "name": ["Frank"], "category": ["A"]})]
        )

        # Test min filter - should exclude null values
        feature = Feature("age")
        filter_type = FilterTypeEnum.min
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = PolarsFilterEngine.do_min_filter(extended_data, single_filter)

        # Should not include the row with null age
        assert len(result) == 4  # All rows with age >= 30, excluding null
        ages = result["age"].to_list()
        assert None not in ages
        assert ages == [30, 35, 40, 45]
