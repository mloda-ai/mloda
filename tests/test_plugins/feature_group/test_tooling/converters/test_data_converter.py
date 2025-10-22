"""
Unit tests for DataConverter class.
"""

from typing import Any
import pytest
from tests.test_plugins.feature_group.test_tooling.converters.data_converter import DataConverter

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyarrow as pa
except ImportError:
    pa = None


class TestDataConverterToFramework:
    """Test DataConverter.to_framework method."""

    @pytest.mark.skipif(pd is None, reason="pandas not installed")
    def test_to_framework_pandas_basic(self) -> None:
        """Test converting List[Dict] to pandas DataFrame with basic data."""
        # Arrange
        converter = DataConverter()
        test_data: list[dict[str, Any]] = [{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}]

        # Act
        result = converter.to_framework(test_data, pd.DataFrame)

        # Assert
        assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"
        assert result.shape == (2, 2), f"Expected shape (2, 2), got {result.shape}"
        assert list(result.columns) == ["col1", "col2"], (
            f"Expected columns ['col1', 'col2'], got {list(result.columns)}"
        )

    @pytest.mark.skipif(pd is None, reason="pandas not installed")
    def test_to_framework_pandas_empty_data(self) -> None:
        """Test converting empty List[Dict] to pandas DataFrame."""
        # Arrange
        converter = DataConverter()
        test_data: list[dict[str, Any]] = []

        # Act
        result = converter.to_framework(test_data, pd.DataFrame)

        # Assert
        assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"
        assert len(result) == 0, f"Expected 0 rows, got {len(result)}"

    @pytest.mark.skipif(pa is None, reason="pyarrow not installed")
    def test_to_framework_pyarrow_basic(self) -> None:
        """Test converting List[Dict] to PyArrow Table with basic data."""
        # Arrange
        converter = DataConverter()
        test_data = [{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}]

        # Act
        result = converter.to_framework(test_data, pa.Table)

        # Assert
        assert isinstance(result, pa.Table), "Result should be a PyArrow Table"
        assert result.num_rows == 2, f"Expected 2 rows, got {result.num_rows}"
        assert result.column_names == ["col1", "col2"], f"Expected columns ['col1', 'col2'], got {result.column_names}"

    @pytest.mark.skipif(pd is None, reason="pandas not installed")
    def test_from_framework_pandas_basic(self) -> None:
        """Test converting pandas DataFrame to List[Dict] with basic data."""
        # Arrange
        converter = DataConverter()
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})

        # Act
        result = converter.from_framework(df, pd.DataFrame)

        # Assert
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 2, f"Expected 2 elements, got {len(result)}"
        assert all(isinstance(row, dict) for row in result), "All elements should be dicts"
        assert list(result[0].keys()) == ["col1", "col2"], (
            f"Expected keys ['col1', 'col2'], got {list(result[0].keys())}"
        )
        assert list(result[1].keys()) == ["col1", "col2"], (
            f"Expected keys ['col1', 'col2'], got {list(result[1].keys())}"
        )
