"""Unit tests for the PolarsFilterEngine class."""

from typing import Any
import logging

import pytest

from mloda._testing.compute_framework.filter_suite import FilterEngineSuite
from mloda._testing.framework_adapter import FrameworkAdapter, PolarsAdapter, PolarsLazyAdapter
from mloda.user import Feature
from mloda.user import SingleFilter
from mloda.user import FilterType
from mloda_plugins.compute_framework.base_implementations.polars.polars_filter_engine import PolarsFilterEngine

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore


class TestPolarsFilterEngine(FilterEngineSuite):
    """Unit tests for the PolarsFilterEngine class using the shared suite."""

    adapter_class = PolarsAdapter

    # Framework-specific tests below

    def test_filter_with_null_values(self, sample_data: Any) -> None:
        """Test filtering with null values in data."""
        extended_data = pl.concat(
            [sample_data, pl.DataFrame({"id": [6], "age": [None], "name": ["Frank"], "category": ["A"]})]
        )

        feature = Feature("age")
        filter_type = FilterType.MIN
        parameter = {"value": 30}
        single_filter = SingleFilter(feature, filter_type, parameter)

        result = PolarsFilterEngine.do_min_filter(extended_data, single_filter)

        assert len(result) == 4
        ages = result["age"].to_list()
        assert None not in ages
        assert ages == [30, 35, 40, 45]

    def test_do_min_filter_missing_column_raises_column_not_found(self, adapter: FrameworkAdapter) -> None:
        """A missing column must surface polars' own error, not an AttributeError from the NaN check."""
        data = pl.DataFrame({"a": [1.0, 2.0]})
        single_filter = SingleFilter(Feature("missing"), FilterType.MIN, {"value": 1.0})

        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            result = PolarsFilterEngine.do_min_filter(data, single_filter)
            if hasattr(result, "collect"):
                result.collect()


class TestPolarsLazyFilterEngine(FilterEngineSuite):
    """Unit tests for the PolarsFilterEngine on LazyFrames using the shared suite."""

    adapter_class = PolarsLazyAdapter
