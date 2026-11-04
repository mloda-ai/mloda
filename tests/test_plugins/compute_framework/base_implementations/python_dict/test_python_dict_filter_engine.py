"""Unit tests for the PythonDictFilterEngine class."""

from decimal import Decimal
from typing import Any

from mloda._testing.compute_framework.filter_suite import FilterEngineSuite
from mloda._testing.framework_adapter import FrameworkAdapter, PythonDictAdapter
from mloda.user import Feature
from mloda.user import SingleFilter
from mloda.user import FilterType
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_filter_engine import (
    PythonDictFilterEngine,
)


class TestPythonDictFilterEngine(FilterEngineSuite):
    """Unit tests for the PythonDictFilterEngine class using the shared suite."""

    adapter_class = PythonDictAdapter

    def decimal_dtype(self, adapter: FrameworkAdapter, result: Any) -> Any:
        return type(next((value for value in result["d"] if value is not None), Decimal("0")))

    # Framework-specific tests below

    def test_filter_with_none_values(self, adapter: FrameworkAdapter, sample_data: Any) -> None:
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

        assert self.row_count(adapter, result) == 4
        ages = list(result["age"])
        assert None not in ages
        assert ages == [30, 35, 40, 45]
