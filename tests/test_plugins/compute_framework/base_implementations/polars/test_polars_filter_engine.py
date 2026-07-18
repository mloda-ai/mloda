"""Unit tests for the PolarsFilterEngine class."""

from typing import Any
import logging

import pytest

from mloda.user import Feature
from mloda.user import SingleFilter
from mloda.user import FilterType
from mloda_plugins.compute_framework.base_implementations.polars.polars_filter_engine import PolarsFilterEngine

from tests.test_plugins.compute_framework.base_implementations.filter_engine_test_mixin import (
    FilterEngineTestMixin,
)
from tests.test_plugins.compute_framework.base_implementations.time_range_filter_engine_test_mixin import (
    SAMPLE_IDS,
    SAMPLE_TIMESTAMPS,
    TimeRangeFilterEngineTestMixin,
)

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsFilterEngine(FilterEngineTestMixin, TimeRangeFilterEngineTestMixin):
    """Unit tests for the PolarsFilterEngine class using shared mixins."""

    @pytest.fixture
    def filter_engine(self) -> Any:
        """Return the PolarsFilterEngine class."""
        return PolarsFilterEngine

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

    @pytest.fixture
    def nullable_category_sample_data(self) -> Any:
        """Create a sample Polars DataFrame with null categories for testing."""
        return pl.DataFrame({"id": [1, 2, 3, 4, 5], "category": ["A", None, "B", None, "C"]})

    def get_column_values(self, result: Any, column: str) -> list[Any]:
        """Extract column values from Polars DataFrame."""
        return result[column].to_list()  # type: ignore[no-any-return]

    @pytest.fixture
    def sample_time_data(self) -> Any:
        return pl.DataFrame(
            {"id": SAMPLE_IDS, "ts": SAMPLE_TIMESTAMPS},
            schema={"id": pl.Int64, "ts": pl.Datetime(time_unit="us", time_zone="UTC")},
        )

    def get_id_column_values(self, result: Any) -> list[int]:
        return list(result["id"].to_list())

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
