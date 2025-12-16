"""
Tests for the DataFrameTestBase abstract base class.

This test file defines the contract for the base test class that consolidates
duplicated merge tests across all framework test files (Pandas, Polars, PyArrow, DuckDB).
"""

import pytest
from abc import ABC
from typing import Any, Optional, Type
from unittest.mock import Mock, MagicMock

from mloda.user import Index
from mloda.user import ParallelizationMode
from tests.test_plugins.compute_framework.test_tooling.dataframe_test_base import DataFrameTestBase


class ConcreteTestClass(DataFrameTestBase):
    """Minimal concrete implementation for testing the base class."""

    @classmethod
    def framework_class(cls) -> Type[Any]:
        return Mock

    def create_dataframe(self, data: dict[str, Any]) -> Any:
        return {"mocked_df": data}

    def get_connection(self) -> Optional[Any]:
        return None


class TestDataFrameTestBaseStructure:
    """Test that DataFrameTestBase has the correct structure."""

    def test_is_abstract_base_class(self) -> None:
        """Test that DataFrameTestBase is an ABC that cannot be instantiated."""
        assert issubclass(DataFrameTestBase, ABC)
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            DataFrameTestBase()  # type: ignore[abstract]

    def test_required_methods_exist(self) -> None:
        """Test that all required methods exist on the base class."""
        required_methods = [
            "framework_class",
            "create_dataframe",
            "get_connection",
            "_create_test_framework",
            "_get_merge_engine",
            "_assert_row_count",
            "_assert_result_equals",
            "setup_method",
            "test_merge_inner",
            "test_merge_left",
            "test_merge_right",
            "test_merge_full_outer",
            "test_merge_append",
            "test_merge_union",
        ]
        for method in required_methods:
            assert hasattr(DataFrameTestBase, method), f"Missing method: {method}"

    def test_concrete_subclass_can_be_instantiated(self) -> None:
        """Test that implementing all abstract methods allows instantiation."""
        concrete = ConcreteTestClass()
        assert concrete is not None


class TestDataFrameTestBaseFixtures:
    """Test that DataFrameTestBase provides correct test data fixtures."""

    def test_class_attributes(self) -> None:
        """Test that class attributes have correct values."""
        assert DataFrameTestBase.left_data_dict == {"idx": [1, 3], "col1": ["a", "b"]}
        assert DataFrameTestBase.right_data_dict == {"idx": [1, 2], "col2": ["x", "z"]}
        assert DataFrameTestBase.idx == Index(("idx",))

    def test_setup_method_creates_dataframes(self) -> None:
        """Test that setup_method creates left_data and right_data."""
        concrete = ConcreteTestClass()
        concrete.setup_method()

        assert concrete.left_data == {"mocked_df": DataFrameTestBase.left_data_dict}
        assert concrete.right_data == {"mocked_df": DataFrameTestBase.right_data_dict}


class TestDataFrameTestBaseHelpers:
    """Test that helper methods work correctly."""

    def test_create_test_framework(self) -> None:
        """Test that _create_test_framework creates a framework instance."""
        mock_framework_class = MagicMock()
        mock_framework_instance = MagicMock()
        mock_framework_class.return_value = mock_framework_instance

        class MockFrameworkTestClass(DataFrameTestBase):
            @classmethod
            def framework_class(cls) -> Type[Any]:
                return mock_framework_class

            def create_dataframe(self, data: dict[str, Any]) -> Any:
                return data

            def get_connection(self) -> Optional[Any]:
                return None

        concrete = MockFrameworkTestClass()
        framework = concrete._create_test_framework()

        mock_framework_class.assert_called_once_with(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        assert framework == mock_framework_instance

    def test_get_merge_engine(self) -> None:
        """Test that _get_merge_engine retrieves merge engine from framework."""
        mock_merge_engine_class = MagicMock()
        mock_framework = MagicMock()
        mock_framework.merge_engine.return_value = mock_merge_engine_class

        concrete = ConcreteTestClass()
        engine_class = concrete._get_merge_engine(mock_framework)

        mock_framework.merge_engine.assert_called_once()
        assert engine_class == mock_merge_engine_class

    def test_assert_row_count_passes(self) -> None:
        """Test that _assert_row_count passes with correct count."""
        concrete = ConcreteTestClass()
        mock_result = MagicMock()
        mock_result.__len__.return_value = 5
        concrete._assert_row_count(mock_result, 5)  # Should not raise

    def test_assert_row_count_fails(self) -> None:
        """Test that _assert_row_count fails with incorrect count."""
        concrete = ConcreteTestClass()
        mock_result = MagicMock()
        mock_result.__len__.return_value = 3
        with pytest.raises(AssertionError, match="Expected 5 rows, got 3"):
            concrete._assert_row_count(mock_result, 5)

    def test_assert_result_equals(self) -> None:
        """Test that _assert_result_equals calls equals() method."""
        concrete = ConcreteTestClass()
        mock_result = MagicMock()
        mock_expected = MagicMock()
        mock_result.equals.return_value = True

        concrete._assert_result_equals(mock_result, mock_expected)
        mock_result.equals.assert_called_once_with(mock_expected)

    def test_assert_result_equals_with_sorting(self) -> None:
        """Test that _assert_result_equals handles sort_columns parameter."""
        concrete = ConcreteTestClass()
        mock_result = MagicMock()
        mock_expected = MagicMock()
        mock_sorted_result = MagicMock()
        mock_sorted_expected = MagicMock()

        mock_result.sort.return_value = mock_sorted_result
        mock_expected.sort.return_value = mock_sorted_expected
        mock_sorted_result.equals.return_value = True

        concrete._assert_result_equals(mock_result, mock_expected, sort_columns=["col1", "col2"])

        mock_result.sort.assert_called_once_with(["col1", "col2"])
        mock_expected.sort.assert_called_once_with(["col1", "col2"])
        mock_sorted_result.equals.assert_called_once_with(mock_sorted_expected)
