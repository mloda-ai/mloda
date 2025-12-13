"""Unit tests for the PyArrowFilterEngine class."""

from typing import Any, List

import pytest
import pyarrow as pa

from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_filter_engine import PyArrowFilterEngine

from tests.test_plugins.compute_framework.base_implementations.filter_engine_test_mixin import (
    FilterEngineTestMixin,
)


class TestPyArrowFilterEngine(FilterEngineTestMixin):
    """Unit tests for the PyArrowFilterEngine class using shared mixin."""

    @pytest.fixture
    def filter_engine(self) -> Any:
        """Return the PyArrowFilterEngine class."""
        return PyArrowFilterEngine

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

    def get_column_values(self, result: Any, column: str) -> List[Any]:
        """Extract column values from PyArrow table."""
        return result[column].to_pylist()  # type: ignore[no-any-return]
