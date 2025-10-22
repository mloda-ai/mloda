"""
Tests for structural validators.
"""

import pandas as pd
import pytest

from tests.test_plugins.feature_group.test_tooling.validators.structural_validators import (
    validate_row_count,
    validate_columns_exist,
    validate_no_nulls,
    validate_shape,
)


class TestValidateRowCount:
    """Tests for validate_row_count function."""

    def test_validate_row_count_success(self) -> None:
        """Test that validate_row_count succeeds when row count matches."""
        # Arrange
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]})

        # Act & Assert - should not raise
        validate_row_count(df, 5)

    def test_validate_row_count_failure(self) -> None:
        """Test that validate_row_count fails when row count does not match."""
        # Arrange
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": ["a", "b", "c", "d", "e"]})

        # Act & Assert - should raise AssertionError
        with pytest.raises(AssertionError):
            validate_row_count(df, 3)


class TestValidateColumnsExist:
    """Tests for validate_columns_exist function."""

    def test_validate_columns_exist_success(self) -> None:
        """Test that validate_columns_exist succeeds when all columns exist."""
        # Arrange
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [4.0, 5.0, 6.0]})

        # Act & Assert - should not raise
        validate_columns_exist(df, ["col1", "col2"])


class TestValidateNoNulls:
    """Tests for validate_no_nulls function."""

    def test_validate_no_nulls_success(self) -> None:
        """Test that validate_no_nulls succeeds when no nulls present."""
        # Arrange
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        # Act & Assert - should not raise
        validate_no_nulls(df, ["col1", "col2"])


class TestValidateShape:
    """Tests for validate_shape function."""

    def test_validate_shape_success(self) -> None:
        """Test that validate_shape succeeds when shape matches."""
        # Arrange
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        # Act & Assert - should not raise
        validate_shape(df, (3, 2))
