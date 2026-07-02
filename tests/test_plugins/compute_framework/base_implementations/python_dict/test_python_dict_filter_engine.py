"""Unit tests for the PythonDictFilterEngine class."""

from typing import Any

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
from tests.test_plugins.compute_framework.base_implementations.time_range_filter_engine_test_mixin import (
    SAMPLE_IDS,
    SAMPLE_TIMESTAMPS,
    TimeRangeFilterEngineTestMixin,
)


class TestPythonDictFilterEngine(FilterEngineTestMixin, TimeRangeFilterEngineTestMixin):
    """Unit tests for the PythonDictFilterEngine class using shared mixins."""

    @pytest.fixture
    def filter_engine(self) -> Any:
        """Return the PythonDictFilterEngine class."""
        return PythonDictFilterEngine

    @pytest.fixture
    def sample_data(self) -> Any:
        """Create a sample columnar dict for testing."""
        return {
            "id": [1, 2, 3, 4, 5],
            "age": [25, 30, 35, 40, 45],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "category": ["A", "B", "A", "C", "B"],
        }

    def result_row_count(self, result: Any) -> int:
        """A columnar dict's row count is the length of any of its columns."""
        for column in result.values():
            return len(column)
        return 0

    def get_column_values(self, result: Any, column: str) -> list[Any]:
        """Extract column values from a columnar dict."""
        return list(result[column])

    @pytest.fixture
    def sample_time_data(self) -> Any:
        return {"id": list(SAMPLE_IDS), "ts": list(SAMPLE_TIMESTAMPS)}

    def get_id_column_values(self, result: Any) -> list[int]:
        return list(result["id"])

    # Framework-specific tests below

    def test_do_min_filter_missing_value(self, sample_data: Any) -> None:
        """Test min filter with missing value parameter."""
        feature = Feature("age")
        filter_type = FilterType.MIN
        parameter = {"invalid": 30}  # Wrong parameter name
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
            PythonDictFilterEngine.do_min_filter(sample_data, single_filter)

    def test_do_min_filter_rejects_min_key_fallback(self) -> None:
        """``do_min_filter`` must accept ONLY the canonical ``{"value": ...}`` parameter key.

        The pandas/pyarrow/polars filter engines raise when a min filter carries its
        threshold under ``{"min": ...}``; PythonDict must behave identically instead of
        silently falling back to ``parameter.min_value``.
        """
        data = {"x": [1, 2, 3]}
        single_filter = SingleFilter(Feature("x"), FilterType.MIN, {"min": 2})

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
            PythonDictFilterEngine.do_min_filter(data, single_filter)

    def test_do_equal_filter_missing_value(self, sample_data: Any) -> None:
        """Test equal filter with missing value parameter."""
        feature = Feature("age")
        filter_type = FilterType.EQUAL
        parameter = {"invalid": 30}  # Wrong parameter name
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
            PythonDictFilterEngine.do_equal_filter(sample_data, single_filter)

    def test_do_regex_filter_missing_value(self, sample_data: Any) -> None:
        """Test regex filter with missing value parameter."""
        feature = Feature("name")
        filter_type = FilterType.REGEX
        parameter = {"invalid": "^A"}  # Wrong parameter name
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'value' not found"):
            PythonDictFilterEngine.do_regex_filter(sample_data, single_filter)

    def test_do_categorical_inclusion_filter_missing_values(self, sample_data: Any) -> None:
        """Test categorical inclusion filter with missing values parameter."""
        feature = Feature("category")
        filter_type = FilterType.CATEGORICAL_INCLUSION
        parameter = {"invalid": ["A", "B"]}  # Wrong parameter name
        single_filter = SingleFilter(feature, filter_type, parameter)

        with pytest.raises(ValueError, match="Filter parameter 'values' not found"):
            PythonDictFilterEngine.do_categorical_inclusion_filter(sample_data, single_filter)

    def test_filter_with_none_values(self, sample_data: Any) -> None:
        """Test filtering with None values in data."""
        data_with_none = {
            "id": sample_data["id"] + [6],
            "age": sample_data["age"] + [None],
            "name": sample_data["name"] + ["Frank"],
            "category": sample_data["category"] + ["A"],
        }

        feature = Feature("age")
        filter_type = FilterType.MIN
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = PythonDictFilterEngine.do_min_filter(data_with_none, single_filter)

        assert self.result_row_count(result) == 4
        ages = list(result["age"])
        assert None not in ages
        assert ages == [30, 35, 40, 45]
