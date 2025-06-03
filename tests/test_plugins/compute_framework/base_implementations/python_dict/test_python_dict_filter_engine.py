from typing import Any
import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.filter.single_filter import SingleFilter
from mloda_core.filter.filter_type_enum import FilterTypeEnum
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_filter_engine import (
    PythonDictFilterEngine,
)


class TestPythonDictFilterEngine:
    """Unit tests for the PythonDictFilterEngine class."""

    @pytest.fixture
    def sample_data(self) -> Any:
        """Create a sample list of dicts for testing."""
        return [
            {"id": 1, "age": 25, "name": "Alice", "category": "A"},
            {"id": 2, "age": 30, "name": "Bob", "category": "B"},
            {"id": 3, "age": 35, "name": "Charlie", "category": "A"},
            {"id": 4, "age": 40, "name": "David", "category": "C"},
            {"id": 5, "age": 45, "name": "Eve", "category": "B"},
        ]

    def test_do_range_filter(self, sample_data: Any) -> None:
        """Test range filter with min and max values."""
        # Create a filter for age between 30 and 40
        feature = Feature("age")
        filter_type = FilterTypeEnum.range
        parameter = {"min": 30, "max": 40, "max_exclusive": False}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = PythonDictFilterEngine.do_range_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 3
        ages = [row["age"] for row in result]
        ids = [row["id"] for row in result]
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
        result = PythonDictFilterEngine.do_range_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 2
        ages = [row["age"] for row in result]
        ids = [row["id"] for row in result]
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
        result = PythonDictFilterEngine.do_min_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 2
        ages = [row["age"] for row in result]
        ids = [row["id"] for row in result]
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
        result = PythonDictFilterEngine.do_max_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 2
        ages = [row["age"] for row in result]
        ids = [row["id"] for row in result]
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
        result = PythonDictFilterEngine.do_max_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 2
        ages = [row["age"] for row in result]
        ids = [row["id"] for row in result]
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
        result = PythonDictFilterEngine.do_equal_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 1
        assert result[0]["age"] == 35
        assert result[0]["id"] == 3

    def test_do_regex_filter(self, sample_data: Any) -> None:
        """Test regex filter."""
        # Create a filter for names starting with 'A'
        feature = Feature("name")
        filter_type = FilterTypeEnum.regex
        parameter = {"value": "^A"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = PythonDictFilterEngine.do_regex_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 1
        assert result[0]["name"] == "Alice"
        assert result[0]["id"] == 1

    def test_do_categorical_inclusion_filter(self, sample_data: Any) -> None:
        """Test categorical inclusion filter."""
        # Create a filter for category in ['A', 'B']
        feature = Feature("category")
        filter_type = FilterTypeEnum.categorical_inclusion
        parameter = {"values": ["A", "B"]}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = PythonDictFilterEngine.do_categorical_inclusion_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 4
        categories = {row["category"] for row in result}
        ids = {row["id"] for row in result}
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
        result = PythonDictFilterEngine.apply_filters(sample_data, feature_set)

        # Check the result
        assert len(result) == 1
        assert result[0]["age"] == 35
        assert result[0]["category"] == "A"
        assert result[0]["id"] == 3

    def test_final_filters(self) -> None:
        """Test that final_filters returns True."""
        assert PythonDictFilterEngine.final_filters() is True

    def test_do_range_filter_missing_parameters(self, sample_data: Any) -> None:
        """Test range filter with missing parameters."""
        feature = Feature("age")
        filter_type = FilterTypeEnum.range
        parameter = {"min": 30}  # Missing max parameter
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter .* not supported"):
            PythonDictFilterEngine.do_range_filter(sample_data, single_filter)

    def test_do_min_filter_missing_value(self, sample_data: Any) -> None:
        """Test min filter with missing value parameter."""
        feature = Feature("age")
        filter_type = FilterTypeEnum.min
        parameter = {"invalid": 30}  # Wrong parameter name
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
            PythonDictFilterEngine.do_min_filter(sample_data, single_filter)

    def test_do_equal_filter_missing_value(self, sample_data: Any) -> None:
        """Test equal filter with missing value parameter."""
        feature = Feature("age")
        filter_type = FilterTypeEnum.equal
        parameter = {"invalid": 30}  # Wrong parameter name
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
            PythonDictFilterEngine.do_equal_filter(sample_data, single_filter)

    def test_do_regex_filter_missing_value(self, sample_data: Any) -> None:
        """Test regex filter with missing value parameter."""
        feature = Feature("name")
        filter_type = FilterTypeEnum.regex
        parameter = {"invalid": "^A"}  # Wrong parameter name
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
            PythonDictFilterEngine.do_regex_filter(sample_data, single_filter)

    def test_do_categorical_inclusion_filter_missing_values(self, sample_data: Any) -> None:
        """Test categorical inclusion filter with missing values parameter."""
        feature = Feature("category")
        filter_type = FilterTypeEnum.categorical_inclusion
        parameter = {"invalid": ["A", "B"]}  # Wrong parameter name
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'values' not found"):
            PythonDictFilterEngine.do_categorical_inclusion_filter(sample_data, single_filter)

    def test_filter_with_none_values(self, sample_data: Any) -> None:
        """Test filtering with None values in data."""
        # Add some None values to the data
        data_with_none = sample_data + [{"id": 6, "age": None, "name": "Frank", "category": "A"}]

        # Test min filter - should exclude None values
        feature = Feature("age")
        filter_type = FilterTypeEnum.min
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = PythonDictFilterEngine.do_min_filter(data_with_none, single_filter)

        # Should not include the row with None age
        assert len(result) == 4  # All rows with age >= 30, excluding None
        ages = [row["age"] for row in result]
        assert None not in ages
        assert ages == [30, 35, 40, 45]
