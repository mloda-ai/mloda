"""
Shared test mixin for _extract_column_dtype implementations.

This mixin verifies that each compute framework's _extract_column_dtype method
returns dtype strings that are correctly classified by the _is_string_dtype and
_is_numeric_dtype static methods. Each framework-specific test class should
inherit from this mixin and provide:
- framework_instance fixture: Returns a compute framework instance
- dtype_sample_data fixture: Returns framework-specific data with int_col, str_col, float_col
"""

from abc import abstractmethod
from typing import Any

import pytest

from mloda.core.abstract_plugins.compute_framework import ComputeFramework


class DtypeExtractionTestMixin:
    """Shared tests for _extract_column_dtype across all compute frameworks."""

    @pytest.fixture
    @abstractmethod
    def framework_instance(self) -> Any:
        """Return a compute framework instance.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def dtype_sample_data(self) -> Any:
        """Return framework-specific sample data.

        Override in framework-specific test class.
        Data should contain columns:
            int_col: integer values [1, 2, 3]
            str_col: string values ["a", "b", "c"]
            float_col: float values [1.0, 2.0, 3.0]
        """
        raise NotImplementedError

    def test_extract_int_column_dtype_is_numeric(self, framework_instance: Any, dtype_sample_data: Any) -> None:
        """Integer column dtype must be classified as numeric."""
        dtype = framework_instance._extract_column_dtype(dtype_sample_data, "int_col")
        assert dtype is not None, "_extract_column_dtype returned None for existing int column"
        assert ComputeFramework._is_numeric_dtype(str(dtype).lower()), f"int dtype '{dtype}' not classified as numeric"

    def test_extract_string_column_dtype_is_string(self, framework_instance: Any, dtype_sample_data: Any) -> None:
        """String column dtype must be classified as string."""
        dtype = framework_instance._extract_column_dtype(dtype_sample_data, "str_col")
        assert dtype is not None, "_extract_column_dtype returned None for existing string column"
        assert ComputeFramework._is_string_dtype(str(dtype).lower()), f"string dtype '{dtype}' not classified as string"

    def test_extract_float_column_dtype_is_numeric(self, framework_instance: Any, dtype_sample_data: Any) -> None:
        """Float column dtype must be classified as numeric."""
        dtype = framework_instance._extract_column_dtype(dtype_sample_data, "float_col")
        assert dtype is not None, "_extract_column_dtype returned None for existing float column"
        assert ComputeFramework._is_numeric_dtype(str(dtype).lower()), (
            f"float dtype '{dtype}' not classified as numeric"
        )

    def test_extract_missing_column_returns_none(self, framework_instance: Any, dtype_sample_data: Any) -> None:
        """Missing column must return None."""
        dtype = framework_instance._extract_column_dtype(dtype_sample_data, "nonexistent")
        assert dtype is None

    def test_int_dtype_is_not_string(self, framework_instance: Any, dtype_sample_data: Any) -> None:
        """Integer column dtype must not be classified as string."""
        dtype = framework_instance._extract_column_dtype(dtype_sample_data, "int_col")
        assert dtype is not None
        assert not ComputeFramework._is_string_dtype(str(dtype).lower()), (
            f"int dtype '{dtype}' incorrectly classified as string"
        )

    def test_string_dtype_is_not_numeric(self, framework_instance: Any, dtype_sample_data: Any) -> None:
        """String column dtype must not be classified as numeric."""
        dtype = framework_instance._extract_column_dtype(dtype_sample_data, "str_col")
        assert dtype is not None
        assert not ComputeFramework._is_numeric_dtype(str(dtype).lower()), (
            f"string dtype '{dtype}' incorrectly classified as numeric"
        )
