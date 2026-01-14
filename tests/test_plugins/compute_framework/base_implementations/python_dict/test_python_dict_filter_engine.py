"""Unit tests for the PythonDictFilterEngine class."""

from typing import Any, List

import pytest

from mloda.user import Feature
from mloda.user import SingleFilter
from mloda.user import FilterType
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_filter_engine import (
    PythonDictFilterEngine,
)

from tests.test_plugins.compute_framework.base_implementations.filter_engine_test_mixin import (
    FilterEngineTestMixin,
)


class TestPythonDictFilterEngine(FilterEngineTestMixin):
    """Unit tests for the PythonDictFilterEngine class using shared mixin."""

    @pytest.fixture
    def filter_engine(self) -> Any:
        """Return the PythonDictFilterEngine class."""
        return PythonDictFilterEngine

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

    def get_column_values(self, result: Any, column: str) -> List[Any]:
        """Extract column values from list of dicts."""
        return [row[column] for row in result]

    # Framework-specific tests below

    def test_do_min_filter_missing_value(self, sample_data: Any) -> None:
        """Test min filter with missing value parameter."""
        feature = Feature("age")
        filter_type = FilterType.min
        parameter = {"invalid": 30}  # Wrong parameter name
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
            PythonDictFilterEngine.do_min_filter(sample_data, single_filter)

    def test_do_equal_filter_missing_value(self, sample_data: Any) -> None:
        """Test equal filter with missing value parameter."""
        feature = Feature("age")
        filter_type = FilterType.equal
        parameter = {"invalid": 30}  # Wrong parameter name
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
            PythonDictFilterEngine.do_equal_filter(sample_data, single_filter)

    def test_do_regex_filter_missing_value(self, sample_data: Any) -> None:
        """Test regex filter with missing value parameter."""
        feature = Feature("name")
        filter_type = FilterType.regex
        parameter = {"invalid": "^A"}  # Wrong parameter name
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
            PythonDictFilterEngine.do_regex_filter(sample_data, single_filter)

    def test_do_categorical_inclusion_filter_missing_values(self, sample_data: Any) -> None:
        """Test categorical inclusion filter with missing values parameter."""
        feature = Feature("category")
        filter_type = FilterType.categorical_inclusion
        parameter = {"invalid": ["A", "B"]}  # Wrong parameter name
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'values' not found"):
            PythonDictFilterEngine.do_categorical_inclusion_filter(sample_data, single_filter)

    def test_filter_with_none_values(self, sample_data: Any) -> None:
        """Test filtering with None values in data."""
        data_with_none = sample_data + [{"id": 6, "age": None, "name": "Frank", "category": "A"}]

        feature = Feature("age")
        filter_type = FilterType.min
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = PythonDictFilterEngine.do_min_filter(data_with_none, single_filter)

        assert len(result) == 4
        ages = [row["age"] for row in result]
        assert None not in ages
        assert ages == [30, 35, 40, 45]
