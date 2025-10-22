"""
Unit tests for data generators.
"""

import pytest
from tests.test_plugins.feature_group.test_tooling.data_generation.generators import DataGenerator, EdgeCaseGenerator


class TestDataGenerator:
    """Test DataGenerator class."""

    def test_generate_numeric_columns(self) -> None:
        """Test generating numeric columns."""
        data = DataGenerator.generate_numeric_columns(n_rows=100, column_names=["col1", "col2", "col3"])

        assert len(data) == 3
        assert "col1" in data
        assert "col2" in data
        assert "col3" in data
        assert len(data["col1"]) == 100
        assert len(data["col2"]) == 100
        assert len(data["col3"]) == 100

        # Check values are in range
        assert all(0 <= v <= 100 for v in data["col1"])
        assert all(0 <= v <= 100 for v in data["col2"])

    def test_generate_categorical_columns(self) -> None:
        """Test generating categorical columns."""
        data = DataGenerator.generate_categorical_columns(n_rows=100, column_names=["cat1", "cat2"], n_categories=3)

        assert len(data) == 2
        assert "cat1" in data
        assert "cat2" in data
        assert len(data["cat1"]) == 100
        assert len(data["cat2"]) == 100

        # Check categories are from {A, B, C}
        assert set(data["cat1"]) <= {"A", "B", "C"}
        assert set(data["cat2"]) <= {"A", "B", "C"}

    def test_generate_temporal_column(self) -> None:
        """Test generating temporal column."""
        data = DataGenerator.generate_temporal_column(n_rows=10, column_name="timestamp")

        assert "timestamp" in data
        assert len(data["timestamp"]) == 10

    def test_generate_data(self) -> None:
        """Test generating complete dataset."""
        data = DataGenerator.generate_data(
            n_rows=100, numeric_cols=["val1", "val2"], categorical_cols=["cat1"], temporal_col="timestamp"
        )

        assert "val1" in data
        assert "val2" in data
        assert "cat1" in data
        assert "timestamp" in data
        assert len(data["val1"]) == 100
        assert len(data["cat1"]) == 100
        assert len(data["timestamp"]) == 100

    def test_generate_data_empty(self) -> None:
        """Test generating data with no columns returns empty dict."""
        data = DataGenerator.generate_data(n_rows=100)
        assert data == {}


class TestEdgeCaseGenerator:
    """Test EdgeCaseGenerator class."""

    def test_with_nulls(self) -> None:
        """Test adding nulls to data."""
        base_data = {"col1": [1, 2, 3, 4, 5]}
        data_with_nulls = EdgeCaseGenerator.with_nulls(base_data, columns=["col1"], null_percentage=0.4)

        assert "col1" in data_with_nulls
        assert len(data_with_nulls["col1"]) == 5
        assert None in data_with_nulls["col1"]

        # Count nulls
        null_count = sum(1 for v in data_with_nulls["col1"] if v is None)
        assert null_count >= 1  # At least some nulls

    def test_empty_data(self) -> None:
        """Test generating empty data."""
        data = EdgeCaseGenerator.empty_data(["col1", "col2", "col3"])

        assert len(data) == 3
        assert "col1" in data
        assert "col2" in data
        assert "col3" in data
        assert len(data["col1"]) == 0
        assert len(data["col2"]) == 0

    def test_single_row(self) -> None:
        """Test generating single row."""
        data = EdgeCaseGenerator.single_row(["col1", "col2"], value=42)

        assert len(data) == 2
        assert data["col1"] == [42]
        assert data["col2"] == [42]
        assert len(data["col1"]) == 1

    def test_all_nulls(self) -> None:
        """Test generating all null values."""
        data = EdgeCaseGenerator.all_nulls(["col1", "col2"], n_rows=10)

        assert len(data) == 2
        assert len(data["col1"]) == 10
        assert len(data["col2"]) == 10
        assert all(v is None for v in data["col1"])
        assert all(v is None for v in data["col2"])

    def test_duplicate_rows(self) -> None:
        """Test duplicating rows."""
        base_data = {"col1": [1, 2], "col2": [3, 4]}
        dup_data = EdgeCaseGenerator.duplicate_rows(base_data, n_duplicates=2)

        assert len(dup_data["col1"]) == 4
        assert len(dup_data["col2"]) == 4
        assert dup_data["col1"] == [1, 2, 1, 2]
        assert dup_data["col2"] == [3, 4, 3, 4]
