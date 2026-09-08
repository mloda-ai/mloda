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
from mloda.user import DataType


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


class DuplicateColumnDtypeExtractionTestMixin:
    """Shared tests for _extract_column_dtype/_extract_column_data_type on duplicate-column data.

    Opt-in, not part of DtypeExtractionTestMixin's abstract fixture contract: only frameworks
    that can represent two columns sharing a name (PyArrow, and Iceberg's post-transform
    PyArrow branch) mix this in. Pandas/Polars/DuckDB/Sqlite/Spark cannot construct this shape.

    The mixin is intentionally named without a ``Test`` prefix so pytest does not collect it
    standalone. Framework subclasses pick up the test methods by inheritance.
    """

    @pytest.fixture
    def framework_instance(self) -> Any:
        """Return a compute framework instance.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    @pytest.fixture
    def dtype_duplicate_column_data(self) -> Any:
        """Return a pa.Table with two columns named "dup_col": first int, second string.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    @pytest.fixture
    def dtype_duplicate_column_data_reversed(self) -> Any:
        """Return a pa.Table with two columns named "dup_col": first string, second int.

        Override in framework-specific test class. The reversed order pins first-occurrence
        semantics: an implementation that merely prefers the numeric column would pass the
        int-first fixture but fail this one.
        """
        raise NotImplementedError

    def test_extract_duplicate_column_dtype_returns_first_occurrence(
        self, framework_instance: Any, dtype_duplicate_column_data: Any
    ) -> None:
        """A duplicate column name must not raise; the first occurrence's dtype wins."""
        dtype = framework_instance._extract_column_dtype(dtype_duplicate_column_data, "dup_col")
        assert dtype is not None, "_extract_column_dtype returned None for a duplicate column"
        assert ComputeFramework._is_numeric_dtype(str(dtype).lower()), (
            f"duplicate column dtype '{dtype}' should match the first (int) occurrence"
        )

    def test_extract_duplicate_column_data_type_returns_first_occurrence(
        self, framework_instance: Any, dtype_duplicate_column_data: Any
    ) -> None:
        """A duplicate column name must not raise; _extract_column_data_type returns the first occurrence."""
        data_type = framework_instance._extract_column_data_type(dtype_duplicate_column_data, "dup_col")
        assert data_type is not None, "_extract_column_data_type returned None for a duplicate column"
        assert data_type == DataType.INT64, (
            f"duplicate column data type '{data_type}' should match the first (int) occurrence"
        )

    def test_extract_duplicate_column_dtype_reversed_order_returns_first_occurrence(
        self, framework_instance: Any, dtype_duplicate_column_data_reversed: Any
    ) -> None:
        """Reversing which column comes first proves first-occurrence wins, not "prefers numeric"."""
        dtype = framework_instance._extract_column_dtype(dtype_duplicate_column_data_reversed, "dup_col")
        assert dtype is not None, "_extract_column_dtype returned None for a duplicate column"
        assert ComputeFramework._is_string_dtype(str(dtype).lower()), (
            f"duplicate column dtype '{dtype}' should match the first (string) occurrence"
        )

    def test_extract_duplicate_column_data_type_reversed_order_returns_first_occurrence(
        self, framework_instance: Any, dtype_duplicate_column_data_reversed: Any
    ) -> None:
        """Reversing which column comes first proves first-occurrence wins, not "prefers numeric"."""
        data_type = framework_instance._extract_column_data_type(dtype_duplicate_column_data_reversed, "dup_col")
        assert data_type is not None, "_extract_column_data_type returned None for a duplicate column"
        assert data_type == DataType.STRING, (
            f"duplicate column data type '{data_type}' should match the first (string) occurrence"
        )

    def test_extract_column_dtype_consistent_with_output_schema(
        self, framework_instance: Any, dtype_duplicate_column_data: Any
    ) -> None:
        """_extract_column_dtype must agree with _output_schema's first-occurrence handling."""
        output_schema = framework_instance._output_schema(dtype_duplicate_column_data)
        assert output_schema is not None, "_output_schema returned None for a duplicate column table"
        schema_dtype = dict(output_schema)["dup_col"]
        extracted_dtype = framework_instance._extract_column_dtype(dtype_duplicate_column_data, "dup_col")
        assert schema_dtype == extracted_dtype, (
            f"_output_schema reports '{schema_dtype}' but _extract_column_dtype reports '{extracted_dtype}'"
        )
