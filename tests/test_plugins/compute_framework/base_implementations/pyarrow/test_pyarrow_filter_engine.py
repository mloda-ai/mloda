from typing import Any
import pytest
import pyarrow as pa

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.filter.single_filter import SingleFilter
from mloda_core.filter.filter_type_enum import FilterTypeEnum
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_filter_engine import PyArrowFilterEngine


class TestPyArrowFilterEngine:
    """Unit tests for the PyArrowFilterEngine class."""

    @pytest.fixture
    def sample_data(self) -> Any:
        """Create a sample PyArrow table for testing."""
        return pa.table(
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
        result = PyArrowFilterEngine.do_range_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 3
        assert result["age"].to_pylist() == [30, 35, 40]
        assert result["id"].to_pylist() == [2, 3, 4]

    def test_do_range_filter_exclusive(self, sample_data: Any) -> None:
        """Test range filter with exclusive max value."""
        # Create a filter for age between 30 and 40 (exclusive)
        feature = Feature("age")
        filter_type = FilterTypeEnum.range
        parameter = {"min": 30, "max": 40, "max_exclusive": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = PyArrowFilterEngine.do_range_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 2
        assert result["age"].to_pylist() == [30, 35]
        assert result["id"].to_pylist() == [2, 3]

    def test_do_min_filter(self, sample_data: Any) -> None:
        """Test min filter."""
        # Create a filter for age >= 40
        feature = Feature("age")
        filter_type = FilterTypeEnum.min
        parameter = {"value": 40}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = PyArrowFilterEngine.do_min_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 2
        assert result["age"].to_pylist() == [40, 45]
        assert result["id"].to_pylist() == [4, 5]

    def test_do_max_filter(self, sample_data: Any) -> None:
        """Test max filter."""
        # Create a filter for age <= 30
        feature = Feature("age")
        filter_type = FilterTypeEnum.max
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = PyArrowFilterEngine.do_max_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 2
        assert result["age"].to_pylist() == [25, 30]
        assert result["id"].to_pylist() == [1, 2]

    def test_do_max_filter_with_tuple(self, sample_data: Any) -> None:
        """Test max filter with tuple parameter."""
        # Create a filter for age < 35
        feature = Feature("age")
        filter_type = FilterTypeEnum.max
        parameter = {"max": 35, "max_exclusive": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = PyArrowFilterEngine.do_max_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 2
        assert result["age"].to_pylist() == [25, 30]
        assert result["id"].to_pylist() == [1, 2]

    def test_do_equal_filter(self, sample_data: Any) -> None:
        """Test equal filter."""
        # Create a filter for age == 35
        feature = Feature("age")
        filter_type = FilterTypeEnum.equal
        parameter = {"value": 35}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = PyArrowFilterEngine.do_equal_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 1
        assert result["age"].to_pylist() == [35]
        assert result["id"].to_pylist() == [3]

    def test_do_regex_filter(self, sample_data: Any) -> None:
        """Test regex filter."""
        # Create a filter for names starting with 'A'
        feature = Feature("name")
        filter_type = FilterTypeEnum.regex
        parameter = {"value": "^A"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = PyArrowFilterEngine.do_regex_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 1
        assert result["name"].to_pylist() == ["Alice"]
        assert result["id"].to_pylist() == [1]

    def test_do_categorical_inclusion_filter(self, sample_data: Any) -> None:
        """Test categorical inclusion filter."""
        # Create a filter for category in ['A', 'B']
        feature = Feature("category")
        filter_type = FilterTypeEnum.categorical_inclusion
        parameter = {"values": ["A", "B"]}
        single_filter = SingleFilter(feature, filter_type, parameter)

        # Apply the filter
        result = PyArrowFilterEngine.do_categorical_inclusion_filter(sample_data, single_filter)

        # Check the result
        assert len(result) == 4
        assert set(result["category"].to_pylist()) == {"A", "B"}
        assert set(result["id"].to_pylist()) == {1, 2, 3, 5}

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
        result = PyArrowFilterEngine.apply_filters(sample_data, feature_set)

        # Check the result
        assert len(result) == 1
        assert result["age"].to_pylist() == [35]
        assert result["category"].to_pylist() == ["A"]
        assert result["id"].to_pylist() == [3]

    def test_final_filters(self) -> None:
        """Test that final_filters returns True."""
        assert PyArrowFilterEngine.final_filters() is True
