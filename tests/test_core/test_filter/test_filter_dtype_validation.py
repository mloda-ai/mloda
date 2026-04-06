"""Unit tests for filter column dtype compatibility validation.

Tests the dtype extraction and compatibility checking logic added to
ComputeFramework._validate_filter_columns (issue #371).
"""

import pyarrow as pa

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from uuid import uuid4


def _make_pyarrow_fw() -> PyArrowTable:
    return PyArrowTable(
        mode=ParallelizationMode.SYNC,
        children_if_root=frozenset(),
        uuid=uuid4(),
    )


class TestIsStringDtype:
    def test_pyarrow_string(self) -> None:
        assert ComputeFramework._is_string_dtype("string") is True

    def test_pyarrow_utf8(self) -> None:
        assert ComputeFramework._is_string_dtype("utf8") is True

    def test_pyarrow_large_string(self) -> None:
        assert ComputeFramework._is_string_dtype("large_string") is True

    def test_pandas_object(self) -> None:
        assert ComputeFramework._is_string_dtype("object") is True

    def test_sql_varchar(self) -> None:
        assert ComputeFramework._is_string_dtype("varchar") is True

    def test_sql_text(self) -> None:
        assert ComputeFramework._is_string_dtype("text") is True

    def test_int64_is_not_string(self) -> None:
        assert ComputeFramework._is_string_dtype("int64") is False

    def test_float64_is_not_string(self) -> None:
        assert ComputeFramework._is_string_dtype("float64") is False

    def test_python_dict_str(self) -> None:
        assert ComputeFramework._is_string_dtype("str") is True

    def test_struct_is_not_string(self) -> None:
        assert ComputeFramework._is_string_dtype("structtype()") is False

    def test_bool_is_not_string(self) -> None:
        assert ComputeFramework._is_string_dtype("bool") is False


class TestIsNumericDtype:
    def test_int64(self) -> None:
        assert ComputeFramework._is_numeric_dtype("int64") is True

    def test_float64(self) -> None:
        assert ComputeFramework._is_numeric_dtype("float64") is True

    def test_double(self) -> None:
        assert ComputeFramework._is_numeric_dtype("double") is True

    def test_uint32(self) -> None:
        assert ComputeFramework._is_numeric_dtype("uint32") is True

    def test_decimal(self) -> None:
        assert ComputeFramework._is_numeric_dtype("decimal128(10, 2)") is True

    def test_string_is_not_numeric(self) -> None:
        assert ComputeFramework._is_numeric_dtype("string") is False

    def test_bool_is_not_numeric(self) -> None:
        assert ComputeFramework._is_numeric_dtype("bool") is False


class TestExtractColumnDtypePyArrow:
    def test_int_column(self) -> None:
        fw = _make_pyarrow_fw()
        data = pa.table({"col": [1, 2, 3]})
        assert fw._extract_column_dtype(data, "col") == "int64"

    def test_string_column(self) -> None:
        fw = _make_pyarrow_fw()
        data = pa.table({"col": ["a", "b", "c"]})
        assert fw._extract_column_dtype(data, "col") == "string"

    def test_float_column(self) -> None:
        fw = _make_pyarrow_fw()
        data = pa.table({"col": [1.0, 2.0, 3.0]})
        assert fw._extract_column_dtype(data, "col") == "double"

    def test_missing_column_returns_none(self) -> None:
        fw = _make_pyarrow_fw()
        data = pa.table({"col": [1, 2, 3]})
        assert fw._extract_column_dtype(data, "nonexistent") is None


class TestExtractColumnDtypeBaseReturnsNone:
    def test_base_class_returns_none(self) -> None:
        class MinimalFramework(ComputeFramework):
            def _extract_column_names(self, data: object) -> set[str]:
                return set()

        fw = MinimalFramework(
            mode=ParallelizationMode.SYNC,
            children_if_root=frozenset(),
            uuid=uuid4(),
        )
        assert fw._extract_column_dtype(None, "col") is None


class TestCollectFilterValues:
    def test_equal_filter(self) -> None:
        from mloda.core.filter.single_filter import SingleFilter

        sf = SingleFilter("col", "equal", {"value": "active"})
        values = ComputeFramework._collect_filter_values(sf)
        assert values == ["active"]

    def test_range_filter(self) -> None:
        from mloda.core.filter.single_filter import SingleFilter

        sf = SingleFilter("col", "range", {"min": 0, "max": 100})
        values = ComputeFramework._collect_filter_values(sf)
        assert 0 in values
        assert 100 in values

    def test_categorical_inclusion_filter(self) -> None:
        from mloda.core.filter.single_filter import SingleFilter

        sf = SingleFilter("col", "categorical_inclusion", {"values": ["a", "b", "c"]})
        values = ComputeFramework._collect_filter_values(sf)
        assert values == ["a", "b", "c"]

    def test_min_filter(self) -> None:
        from mloda.core.filter.single_filter import SingleFilter

        sf = SingleFilter("col", "min", {"min": 5})
        values = ComputeFramework._collect_filter_values(sf)
        assert values == [5]
