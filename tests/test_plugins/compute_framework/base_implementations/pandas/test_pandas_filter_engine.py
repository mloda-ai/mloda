"""Unit tests for the PandasFilterEngine class."""

from typing import Any

import pytest
import pandas as pd

from mloda_plugins.compute_framework.base_implementations.pandas.pandas_filter_engine import PandasFilterEngine

from tests.test_plugins.compute_framework.base_implementations.filter_engine_test_mixin import (
    FilterEngineTestMixin,
)
from tests.test_plugins.compute_framework.base_implementations.time_range_filter_engine_test_mixin import (
    SAMPLE_IDS,
    SAMPLE_TIMESTAMPS,
    TimeRangeFilterEngineTestMixin,
)


class TestPandasFilterEngine(FilterEngineTestMixin, TimeRangeFilterEngineTestMixin):
    """Unit tests for the PandasFilterEngine class using shared mixins."""

    @pytest.fixture
    def filter_engine(self) -> Any:
        """Return the PandasFilterEngine class."""
        return PandasFilterEngine

    @pytest.fixture
    def sample_data(self) -> Any:
        """Create a sample pandas DataFrame for testing."""
        return pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "age": [25, 30, 35, 40, 45],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "category": ["A", "B", "A", "C", "B"],
            }
        )

    def get_column_values(self, result: Any, column: str) -> list[Any]:
        """Extract column values from pandas DataFrame."""
        return result[column].tolist()  # type: ignore[no-any-return]

    @pytest.fixture
    def sample_time_data(self) -> Any:
        return pd.DataFrame({"id": SAMPLE_IDS, "ts": pd.to_datetime(SAMPLE_TIMESTAMPS, utc=True)})

    def get_id_column_values(self, result: Any) -> list[int]:
        return list(result["id"].tolist())
