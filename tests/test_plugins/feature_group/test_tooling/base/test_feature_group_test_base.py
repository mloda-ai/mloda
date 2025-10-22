"""
Unit tests for FeatureGroupTestBase class.
"""

import pytest
import pandas as pd
from tests.test_plugins.feature_group.test_tooling.base.feature_group_test_base import FeatureGroupTestBase


class ConcreteFeatureGroupTest(FeatureGroupTestBase):
    """Concrete test class for testing FeatureGroupTestBase functionality."""

    def feature_group_class(self):
        """Return None since this is just a test of the base class itself."""
        return None


class TestFeatureGroupTestBase:
    """Test FeatureGroupTestBase class."""

    def test_base_class_to_pandas(self) -> None:
        """Test that to_pandas converts dict data to pandas DataFrame."""
        # Arrange
        test_instance = ConcreteFeatureGroupTest()
        test_data = {
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9],
        }

        # Act
        result = test_instance.to_pandas(test_data)

        # Assert
        assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"
        assert result.shape == (3, 3), f"Expected shape (3, 3), got {result.shape}"
        assert list(result.columns) == ["col1", "col2", "col3"], "Column names should match input dict keys"

    def test_base_class_to_framework(self) -> None:
        """Test that to_framework converts dict data to a specific framework (pandas)."""
        # Arrange
        test_instance = ConcreteFeatureGroupTest()
        test_data = {
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9],
        }

        # Act
        result = test_instance.to_framework(test_data, pd.DataFrame)

        # Assert
        assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"
        assert result.shape == (3, 3), f"Expected shape (3, 3), got {result.shape}"

    def test_assert_columns_exist_success(self) -> None:
        """Test that assert_columns_exist succeeds when all columns exist."""
        # Arrange
        test_instance = ConcreteFeatureGroupTest()
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9],
        })

        # Act & Assert
        # Should not raise any exception
        test_instance.assert_columns_exist(df, ["col1", "col2"])

    def test_assert_row_count_success(self) -> None:
        """Test that assert_row_count succeeds when row count matches."""
        # Arrange
        test_instance = ConcreteFeatureGroupTest()
        df = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": [6, 7, 8, 9, 10],
        })

        # Act & Assert
        # Should not raise any exception
        test_instance.assert_row_count(df, 5)
