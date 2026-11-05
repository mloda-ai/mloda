"""Conformance suites for dtype extraction and DataTypeValidator enforcement on framework-native frames."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from mloda._testing.compute_framework.host import ComputeFrameworkSuite
from mloda._testing.framework_adapter import FrameworkAdapter
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.validators.datatype_validator import (
    DataTypeMismatchError,
    DataTypeValidator,
)
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.optional_dependency import require
from mloda.provider import DefaultOptionKeys, FeatureSet
from mloda.user import DataType, Feature

if TYPE_CHECKING:
    import pyarrow as pa

_TS: tuple[datetime, ...] = (datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3))
_TS_UTC: tuple[datetime, ...] = tuple(v.replace(tzinfo=UTC) for v in _TS)

_DTYPE_SAMPLE = TypedTestData(
    {
        "int_col": (DataType.INT64, (1, 2, 3)),
        "str_col": (DataType.STRING, ("a", "b", "c")),
        "float_col": (DataType.DOUBLE, (1.0, 2.0, 3.0)),
    }
)
_DECIMAL_SAMPLE = TypedTestData({"d": (DataType.DECIMAL, (Decimal("12.34"), Decimal("5.50"), Decimal("99.99"), None))})
_PRECISION_SAMPLE = TypedTestData(
    {
        "int32_col": (DataType.INT32, (1, 2, 3)),
        "int64_col": (DataType.INT64, (1, 2, 3)),
        "float32_col": (DataType.FLOAT, (1.0, 2.0, 3.0)),
        "float64_col": (DataType.DOUBLE, (1.0, 2.0, 3.0)),
        "timestamp_ms_col": (DataType.TIMESTAMP_MILLIS, _TS),
        "timestamp_us_col": (DataType.TIMESTAMP_MICROS, _TS),
    }
)

# Case id -> expected resolved DataType (not the Arrow physical type) and values; the case id is the column name.
_CASES: dict[str, tuple[DataType, tuple[Any, ...]]] = {
    "int8_col": (DataType.INT8, (1, 2, 3)),
    "uint8_col": (DataType.UINT8, (1, 2, 3)),
    "uint64_col": (DataType.UINT64, (1, 2, 3)),
    "int16_col": (DataType.INT32, (1, 2, 3)),
    "uint16_col": (DataType.INT32, (1, 2, 3)),
    "uint32_col": (DataType.INT64, (1, 2, 3)),
    "timestamp_us_tz_col": (DataType.TIMESTAMP_MICROS_TZ, _TS_UTC),
    "timestamp_s_col": (DataType.TIMESTAMP_MICROS, _TS),
    "timestamp_ns_col": (DataType.TIMESTAMP_MICROS, _TS),
    "timestamp_ns_tz_col": (DataType.TIMESTAMP_MICROS_TZ, _TS_UTC),
    "list_str_col": (DataType.LIST, (["a"], ["b", "c"], [])),
    "list_struct_col": (DataType.LIST, ([{"x": 1}], [{"x": 2}, {"x": 3}], [])),
    "list_fixed_col": (DataType.LIST, ([1, 2], [3, 4], [5, 6])),
}

# Cases whose Arrow type differs from the declared type's own; the others get a type readback.
_ARROW_OVERRIDES: dict[str, Callable[[Any], Any]] = {
    "int16_col": lambda pa: pa.int16(),
    "uint16_col": lambda pa: pa.uint16(),
    "uint32_col": lambda pa: pa.uint32(),
    "timestamp_s_col": lambda pa: pa.timestamp("s"),
    "timestamp_ns_col": lambda pa: pa.timestamp("ns"),
    "timestamp_ns_tz_col": lambda pa: pa.timestamp("ns", tz="UTC"),
    "list_str_col": lambda pa: pa.list_(pa.string()),
    "list_struct_col": lambda pa: pa.list_(pa.struct([("x", pa.int64())])),
    "list_fixed_col": lambda pa: pa.list_(pa.int64(), 2),
}


def _assert_readback(adapter: FrameworkAdapter, frame: Any, data: TypedTestData) -> None:
    """Every column of a test frame keeps its declared type (None means the adapter cannot tell)."""
    for column, declared in data.types.items():
        assert adapter.has_type(frame, column, declared) is not False, (
            f"test frame lost the declared type {declared} of column '{column}'"
        )


class DtypeExtractionSuite(ComputeFrameworkSuite, suite_base=True):
    """Contract tests of _extract_column_dtype and _extract_column_data_type; every available framework is served."""

    aspect = "dtype extraction"

    def _serves(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> bool:
        return True

    def arrow_types(self, name: str) -> Mapping[str, pa.DataType] | None:
        if name == "decimal":
            pa = require("pyarrow", "building Arrow test tables")
            return {"d": pa.decimal128(10, 2)}
        return super().arrow_types(name)

    def _sample(self, adapter: FrameworkAdapter) -> Any:
        frame = self.frame(adapter, "dtype_sample", _DTYPE_SAMPLE)
        _assert_readback(adapter, frame, _DTYPE_SAMPLE)
        return frame

    def test_extract_int_column_dtype_is_numeric(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        dtype = framework_instance._extract_column_dtype(self._sample(adapter), "int_col")
        assert dtype is not None, "_extract_column_dtype returned None for existing int column"
        assert ComputeFramework._is_numeric_dtype(str(dtype).lower()), f"int dtype '{dtype}' not classified as numeric"

    def test_extract_string_column_dtype_is_string(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        dtype = framework_instance._extract_column_dtype(self._sample(adapter), "str_col")
        assert dtype is not None, "_extract_column_dtype returned None for existing string column"
        assert ComputeFramework._is_string_dtype(str(dtype).lower()), f"string dtype '{dtype}' not classified as string"

    def test_extract_float_column_dtype_is_numeric(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        dtype = framework_instance._extract_column_dtype(self._sample(adapter), "float_col")
        assert dtype is not None, "_extract_column_dtype returned None for existing float column"
        assert ComputeFramework._is_numeric_dtype(str(dtype).lower()), (
            f"float dtype '{dtype}' not classified as numeric"
        )

    def test_extract_missing_column_returns_none(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        assert framework_instance._extract_column_dtype(self._sample(adapter), "nonexistent") is None

    def test_int_dtype_is_not_string(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        dtype = framework_instance._extract_column_dtype(self._sample(adapter), "int_col")
        assert dtype is not None
        assert not ComputeFramework._is_string_dtype(str(dtype).lower()), (
            f"int dtype '{dtype}' incorrectly classified as string"
        )

    def test_string_dtype_is_not_numeric(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        dtype = framework_instance._extract_column_dtype(self._sample(adapter), "str_col")
        assert dtype is not None
        assert not ComputeFramework._is_numeric_dtype(str(dtype).lower()), (
            f"string dtype '{dtype}' incorrectly classified as numeric"
        )

    def test_extract_decimal_column_data_type(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        frame = self.frame(adapter, "decimal", _DECIMAL_SAMPLE)
        assert framework_instance._extract_column_data_type(frame, "d") == DataType.DECIMAL

    def test_extract_duplicate_column_dtype_returns_first_occurrence(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None:
        frame = self.duplicate_column_frame(adapter, DataType.INT64, DataType.STRING)
        dtype = framework_instance._extract_column_dtype(frame, "a")
        assert dtype is not None, "_extract_column_dtype returned None for a duplicate column"
        assert ComputeFramework._is_numeric_dtype(str(dtype).lower()), (
            f"duplicate column dtype '{dtype}' should match the first (int) occurrence"
        )

    def test_extract_duplicate_column_data_type_returns_first_occurrence(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None:
        frame = self.duplicate_column_frame(adapter, DataType.INT64, DataType.STRING)
        data_type = framework_instance._extract_column_data_type(frame, "a")
        assert data_type is not None, "_extract_column_data_type returned None for a duplicate column"
        assert data_type == DataType.INT64, (
            f"duplicate column data type '{data_type}' should match the first (int) occurrence"
        )

    def test_extract_duplicate_column_dtype_reversed_order_returns_first_occurrence(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None:
        """Reversing the order proves first-occurrence wins, not "prefers numeric"."""
        frame = self.duplicate_column_frame(adapter, DataType.STRING, DataType.INT64)
        dtype = framework_instance._extract_column_dtype(frame, "a")
        assert dtype is not None, "_extract_column_dtype returned None for a duplicate column"
        assert ComputeFramework._is_string_dtype(str(dtype).lower()), (
            f"duplicate column dtype '{dtype}' should match the first (string) occurrence"
        )

    def test_extract_duplicate_column_data_type_reversed_order_returns_first_occurrence(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None:
        frame = self.duplicate_column_frame(adapter, DataType.STRING, DataType.INT64)
        data_type = framework_instance._extract_column_data_type(frame, "a")
        assert data_type is not None, "_extract_column_data_type returned None for a duplicate column"
        assert data_type == DataType.STRING, (
            f"duplicate column data type '{data_type}' should match the first (string) occurrence"
        )

    def test_extract_column_dtype_consistent_with_output_schema(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None:
        """_extract_column_dtype must agree with _output_schema's first-occurrence handling."""
        frame = self.duplicate_column_frame(adapter, DataType.INT64, DataType.STRING)
        output_schema = framework_instance._output_schema(frame)
        assert output_schema is not None, "_output_schema returned None for a duplicate column table"
        schema_dtype = dict(output_schema)["a"]
        extracted_dtype = framework_instance._extract_column_dtype(frame, "a")
        assert schema_dtype == extracted_dtype, (
            f"_output_schema reports '{schema_dtype}' but _extract_column_dtype reports '{extracted_dtype}'"
        )


class DataTypeValidatorSuite(ComputeFrameworkSuite, suite_base=True):
    """Contract tests of DataTypeValidator on framework-native frames; every available framework is served."""

    aspect = "datatype validator"
    case_ids = frozenset(_CASES)

    def _serves(self, adapter: FrameworkAdapter, framework: type[ComputeFramework]) -> bool:
        return True

    def arrow_types(self, name: str) -> Mapping[str, pa.DataType] | None:
        override = _ARROW_OVERRIDES.get(name)
        if override is not None:
            return {name: override(require("pyarrow", "building Arrow test tables"))}
        return super().arrow_types(name)

    def _validator_frame(self, adapter: FrameworkAdapter) -> Any:
        frame = self.frame(adapter, "validator", _DTYPE_SAMPLE)
        _assert_readback(adapter, frame, _DTYPE_SAMPLE)
        return frame

    def _precision_frame(self, adapter: FrameworkAdapter) -> Any:
        frame = self.frame(adapter, "precision", _PRECISION_SAMPLE)
        _assert_readback(adapter, frame, _PRECISION_SAMPLE)
        return frame

    @staticmethod
    def _validate(framework_instance: Any, frame: Any, feature: Feature, strict: bool = False) -> None:
        """Run the validator with the resolver closure run_validate_output_features binds per call."""
        if strict:
            feature.options.add_to_group(DefaultOptionKeys.strict_type_enforcement, True)
        DataTypeValidator.validate(
            FeatureSet([feature]), lambda col: framework_instance._extract_column_data_type(frame, col)
        )

    @pytest.mark.parametrize("case_id", list(_CASES))
    def test_extended_column_resolves_to_expected_data_type(
        self, adapter: FrameworkAdapter, framework_instance: Any, case_id: str
    ) -> None:
        self.skip_unsupported_case(case_id)
        declared, values = _CASES[case_id]
        data = TypedTestData({case_id: (declared, values)})
        frame = self.frame(adapter, case_id, data)
        if case_id not in _ARROW_OVERRIDES:
            _assert_readback(adapter, frame, data)
        assert framework_instance._extract_column_data_type(frame, case_id) == declared

    def test_int_column_int64_declaration_passes(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        """An int column declared as INT64 must pass (exact or widened match)."""
        self._validate(framework_instance, self._validator_frame(adapter), Feature.int64_of("int_col"))

    def test_str_column_string_declaration_passes(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        """A string column declared as STRING must pass (exact match)."""
        self._validate(framework_instance, self._validator_frame(adapter), Feature.str_of("str_col"))

    def test_str_column_int_declaration_raises_in_strict(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None:
        with pytest.raises(DataTypeMismatchError):
            self._validate(framework_instance, self._validator_frame(adapter), Feature.int64_of("str_col"), strict=True)

    def test_str_column_int_declaration_raises_in_lenient(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None:
        """STRING vs numeric is not loosely compatible: lenient mode only relaxes within numeric and timestamp."""
        with pytest.raises(DataTypeMismatchError):
            self._validate(framework_instance, self._validator_frame(adapter), Feature.int64_of("str_col"))

    def test_untyped_feature_skipped(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        """A feature with data_type=None must skip validation regardless of column content."""
        self._validate(framework_instance, self._validator_frame(adapter), Feature.not_typed("str_col"))

    def test_missing_column_skipped(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        """A feature naming a column not in the data must not raise (resolver returns None)."""
        self._validate(framework_instance, self._validator_frame(adapter), Feature.int64_of("nonexistent_col"))

    def test_int_column_double_declaration_lenient_passes(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None:
        """INT64 actual vs DOUBLE declared in lenient mode must pass (numeric widening)."""
        self._validate(framework_instance, self._validator_frame(adapter), Feature.double_of("int_col"))

    def test_int32_column_strict_int32_passes(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        self._validate(framework_instance, self._precision_frame(adapter), Feature.int32_of("int32_col"), strict=True)

    def test_int64_column_strict_int32_raises(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        """INT64 actual vs INT32 declared in strict mode must raise (narrowing forbidden)."""
        with pytest.raises(DataTypeMismatchError):
            self._validate(
                framework_instance, self._precision_frame(adapter), Feature.int32_of("int64_col"), strict=True
            )

    def test_int32_column_strict_int64_passes(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        """INT32 actual vs INT64 declared in strict mode must pass (widening allowed)."""
        self._validate(framework_instance, self._precision_frame(adapter), Feature.int64_of("int32_col"), strict=True)

    def test_float32_column_strict_float_passes(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        self._validate(framework_instance, self._precision_frame(adapter), Feature.float_of("float32_col"), strict=True)

    def test_float64_column_strict_float_raises(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        """DOUBLE actual vs FLOAT declared in strict mode must raise (narrowing forbidden)."""
        with pytest.raises(DataTypeMismatchError):
            self._validate(
                framework_instance, self._precision_frame(adapter), Feature.float_of("float64_col"), strict=True
            )

    def test_float32_column_strict_double_passes(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        """FLOAT actual vs DOUBLE declared in strict mode must pass (widening allowed)."""
        self._validate(
            framework_instance, self._precision_frame(adapter), Feature.double_of("float32_col"), strict=True
        )

    def test_timestamp_ms_column_strict_ms_passes(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        self._validate(
            framework_instance,
            self._precision_frame(adapter),
            Feature.timestamp_millis_of("timestamp_ms_col"),
            strict=True,
        )

    def test_timestamp_us_column_strict_ms_raises(self, adapter: FrameworkAdapter, framework_instance: Any) -> None:
        """TIMESTAMP_MICROS actual vs TIMESTAMP_MILLIS declared strict must raise (narrowing)."""
        with pytest.raises(DataTypeMismatchError):
            self._validate(
                framework_instance,
                self._precision_frame(adapter),
                Feature.timestamp_millis_of("timestamp_us_col"),
                strict=True,
            )
