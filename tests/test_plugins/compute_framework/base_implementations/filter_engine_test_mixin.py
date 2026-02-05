"""
Shared test mixin for all BaseFilterEngine implementations.

This mixin provides common test methods that verify the filter engine contract.
Each framework-specific test class should inherit from this mixin and provide:
- filter_engine fixture: Returns the filter engine class
- sample_data fixture: Returns framework-specific test data
- get_column_values method: Extracts column values as a list from results
"""

from abc import abstractmethod
from typing import Any, List

import pytest

from mloda.user import Feature
from mloda.user import SingleFilter
from mloda.user import FilterType


class FilterEngineTestMixin:
    """Shared tests for all BaseFilterEngine implementations."""

    @pytest.fixture
    @abstractmethod
    def filter_engine(self) -> Any:
        """Return the filter engine class to test.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def sample_data(self) -> Any:
        """Return framework-specific sample data.

        Override in framework-specific test class.
        Data should contain columns: id, age, name, category
        with values:
            id: [1, 2, 3, 4, 5]
            age: [25, 30, 35, 40, 45]
            name: ["Alice", "Bob", "Charlie", "David", "Eve"]
            category: ["A", "B", "A", "C", "B"]
        """
        raise NotImplementedError

    @abstractmethod
    def get_column_values(self, result: Any, column: str) -> List[Any]:
        """Extract column values as a list from the result.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    def _assert_values_equal(self, actual: List[Any], expected: List[Any]) -> None:
        """Assert values are equal regardless of order."""
        assert sorted(actual) == sorted(expected)

    def test_do_range_filter(self, filter_engine: Any, sample_data: Any) -> None:
        """Test range filter with min and max values."""
        feature = Feature("age")
        filter_type = FilterType.range
        parameter = {"min": 30, "max": 40, "max_exclusive": False}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_range_filter(sample_data, single_filter)

        assert len(result) == 3
        self._assert_values_equal(self.get_column_values(result, "age"), [30, 35, 40])
        self._assert_values_equal(self.get_column_values(result, "id"), [2, 3, 4])

    def test_do_range_filter_exclusive(self, filter_engine: Any, sample_data: Any) -> None:
        """Test range filter with exclusive max value."""
        feature = Feature("age")
        filter_type = FilterType.range
        parameter = {"min": 30, "max": 40, "max_exclusive": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_range_filter(sample_data, single_filter)

        assert len(result) == 2
        self._assert_values_equal(self.get_column_values(result, "age"), [30, 35])
        self._assert_values_equal(self.get_column_values(result, "id"), [2, 3])

    def test_do_min_filter(self, filter_engine: Any, sample_data: Any) -> None:
        """Test min filter."""
        feature = Feature("age")
        filter_type = FilterType.min
        parameter = {"value": 40}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_min_filter(sample_data, single_filter)

        assert len(result) == 2
        self._assert_values_equal(self.get_column_values(result, "age"), [40, 45])
        self._assert_values_equal(self.get_column_values(result, "id"), [4, 5])

    def test_do_max_filter(self, filter_engine: Any, sample_data: Any) -> None:
        """Test max filter."""
        feature = Feature("age")
        filter_type = FilterType.max
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_max_filter(sample_data, single_filter)

        assert len(result) == 2
        self._assert_values_equal(self.get_column_values(result, "age"), [25, 30])
        self._assert_values_equal(self.get_column_values(result, "id"), [1, 2])

    def test_do_max_filter_with_tuple(self, filter_engine: Any, sample_data: Any) -> None:
        """Test max filter with tuple parameter."""
        feature = Feature("age")
        filter_type = FilterType.max
        parameter = {"max": 35, "max_exclusive": True}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_max_filter(sample_data, single_filter)

        assert len(result) == 2
        self._assert_values_equal(self.get_column_values(result, "age"), [25, 30])
        self._assert_values_equal(self.get_column_values(result, "id"), [1, 2])

    def test_do_equal_filter(self, filter_engine: Any, sample_data: Any) -> None:
        """Test equal filter."""
        feature = Feature("age")
        filter_type = FilterType.equal
        parameter = {"value": 35}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_equal_filter(sample_data, single_filter)

        assert len(result) == 1
        assert self.get_column_values(result, "age")[0] == 35
        assert self.get_column_values(result, "id")[0] == 3

    def test_do_regex_filter(self, filter_engine: Any, sample_data: Any) -> None:
        """Test regex filter."""
        feature = Feature("name")
        filter_type = FilterType.regex
        parameter = {"value": "^A"}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_regex_filter(sample_data, single_filter)

        assert len(result) == 1
        assert self.get_column_values(result, "name")[0] == "Alice"
        assert self.get_column_values(result, "id")[0] == 1

    def test_do_categorical_inclusion_filter(self, filter_engine: Any, sample_data: Any) -> None:
        """Test categorical inclusion filter."""
        feature = Feature("category")
        filter_type = FilterType.categorical_inclusion
        parameter = {"values": ["A", "B"]}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = filter_engine.do_categorical_inclusion_filter(sample_data, single_filter)

        assert len(result) == 4
        assert set(self.get_column_values(result, "category")) == {"A", "B"}
        assert set(self.get_column_values(result, "id")) == {1, 2, 3, 5}

    def test_apply_filters(self, filter_engine: Any, sample_data: Any) -> None:
        """Test applying multiple filters."""
        feature = Feature("age")
        filters = [
            SingleFilter(feature, FilterType.min, {"value": 30}),
            SingleFilter(Feature("category"), FilterType.equal, {"value": "A"}),
        ]

        class MockFeatureSet:
            def __init__(self, filters: Any) -> None:
                self.filters = filters

            def get_all_names(self) -> Any:
                return ["age", "category"]

        feature_set = MockFeatureSet(filters)

        result = filter_engine.apply_filters(sample_data, feature_set)

        assert len(result) == 1
        assert self.get_column_values(result, "age")[0] == 35
        assert self.get_column_values(result, "category")[0] == "A"
        assert self.get_column_values(result, "id")[0] == 3

    def test_final_filters(self, filter_engine: Any) -> None:
        """Test that final_filters returns True."""
        assert filter_engine.final_filters() is True

    def test_do_range_filter_missing_parameters(self, filter_engine: Any, sample_data: Any) -> None:
        """Test range filter with missing parameters."""
        feature = Feature("age")
        filter_type = FilterType.range
        parameter = {"min": 30}  # Missing max parameter
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter .* not supported"):
            filter_engine.do_range_filter(sample_data, single_filter)
