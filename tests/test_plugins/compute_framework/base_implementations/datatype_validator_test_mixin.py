"""
Shared test mixin for DataTypeValidator enforcement across compute frameworks.

This mixin verifies that each compute framework correctly resolves a column's native dtype
to a unified DataType via ComputeFramework._extract_column_data_type, and that
DataTypeValidator.validate, given a resolver closure, enforces feature data_type declarations
on framework-native data (not just PyArrow).

The canonical test data lives here as class-level ``VALIDATOR_COLUMNS`` / ``PRECISION_COLUMNS``
``ColumnSpec`` tuples. The mixin materialises that spec into a canonical, typed ``pa.Table``
via ``_arrow_table`` (the only place ``c.values`` is read, validated through ``TypedTestData``). Framework subclasses implement
``from_arrow(table)`` to convert that table into the framework-native container, and only
override the data fixtures when they need an additional fixture such as ``connection`` or
``spark_session``.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

import pyarrow as pa
import pytest

from mloda._testing.framework_adapter import to_arrow_table
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.validators.datatype_validator import (
    DataTypeMismatchError,
    DataTypeValidator,
)
from mloda.provider import DefaultOptionKeys
from mloda.provider import FeatureSet
from mloda.user import DataType
from mloda.user import Feature


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    data_type: DataType
    values: tuple[Any, ...]
    arrow_type: pa.DataType | None = None


_TS_VALUES: tuple[datetime, ...] = (
    datetime(2024, 1, 1),
    datetime(2024, 1, 2),
    datetime(2024, 1, 3),
)
_TS_VALUES_UTC: tuple[datetime, ...] = tuple(v.replace(tzinfo=UTC) for v in _TS_VALUES)


EXTENDED_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec("int8_col", DataType.INT8, (1, 2, 3), pa.int8()),
    ColumnSpec("uint8_col", DataType.UINT8, (1, 2, 3), pa.uint8()),
    ColumnSpec("uint64_col", DataType.UINT64, (1, 2, 3), pa.uint64()),
    ColumnSpec("int16_col", DataType.INT32, (1, 2, 3), pa.int16()),
    ColumnSpec("uint16_col", DataType.INT32, (1, 2, 3), pa.uint16()),
    ColumnSpec("uint32_col", DataType.INT64, (1, 2, 3), pa.uint32()),
    ColumnSpec("timestamp_us_tz_col", DataType.TIMESTAMP_MICROS_TZ, _TS_VALUES_UTC, pa.timestamp("us", tz="UTC")),
    ColumnSpec("timestamp_s_col", DataType.TIMESTAMP_MICROS, _TS_VALUES, pa.timestamp("s")),
    ColumnSpec("timestamp_ns_col", DataType.TIMESTAMP_MICROS, _TS_VALUES, pa.timestamp("ns")),
    ColumnSpec("timestamp_ns_tz_col", DataType.TIMESTAMP_MICROS_TZ, _TS_VALUES_UTC, pa.timestamp("ns", tz="UTC")),
    ColumnSpec("list_str_col", DataType.LIST, (["a"], ["b", "c"], []), pa.list_(pa.string())),
    ColumnSpec(
        "list_struct_col",
        DataType.LIST,
        ([{"x": 1}], [{"x": 2}, {"x": 3}], []),
        pa.list_(pa.struct([("x", pa.int64())])),
    ),
    ColumnSpec("list_fixed_col", DataType.LIST, ([1, 2], [3, 4], [5, 6]), pa.list_(pa.int64(), 2)),
)


class DataTypeValidatorFrameworkTestMixin:
    """Shared tests for DataTypeValidator enforcement across all compute frameworks.

    The mixin is intentionally named without a `Test` prefix so pytest does not collect it
    standalone. Framework subclasses pick up the test methods by inheritance.
    ``EXTENDED_SKIPS`` maps an extended column name to the reason a framework cannot hold it.
    """

    EXTENDED_SKIPS: dict[str, str] = {}

    VALIDATOR_COLUMNS: tuple[ColumnSpec, ...] = (
        ColumnSpec("int_col", DataType.INT64, (1, 2, 3)),
        ColumnSpec("str_col", DataType.STRING, ("a", "b", "c")),
        ColumnSpec("float_col", DataType.DOUBLE, (1.0, 2.0, 3.0)),
    )

    PRECISION_COLUMNS: tuple[ColumnSpec, ...] = (
        ColumnSpec("int32_col", DataType.INT32, (1, 2, 3)),
        ColumnSpec("int64_col", DataType.INT64, (1, 2, 3)),
        ColumnSpec("float32_col", DataType.FLOAT, (1.0, 2.0, 3.0)),
        ColumnSpec("float64_col", DataType.DOUBLE, (1.0, 2.0, 3.0)),
        ColumnSpec("timestamp_ms_col", DataType.TIMESTAMP_MILLIS, _TS_VALUES),
        ColumnSpec("timestamp_us_col", DataType.TIMESTAMP_MICROS, _TS_VALUES),
    )

    @pytest.fixture
    def framework_instance(self) -> Any:
        """Return a compute framework instance.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    def _arrow_table(self, columns: tuple[ColumnSpec, ...]) -> pa.Table:
        """Build a canonical typed pa.Table from the ColumnSpec tuple.

        Reads ``c.values`` exactly once, here. Subclasses convert the resulting
        table into their framework-native container via ``from_arrow(...)``.
        """
        typed = TypedTestData({c.name: (c.data_type, c.values) for c in columns})
        return to_arrow_table(typed, {c.name: c.arrow_type for c in columns if c.arrow_type is not None})

    def _skip_if_unsupported(self, spec: ColumnSpec) -> None:
        """Skip the extended case before any data is built when the framework cannot hold the column."""
        reason = self.EXTENDED_SKIPS.get(spec.name)
        if reason is not None:
            pytest.skip(reason)

    @pytest.fixture
    def extended_sample_data(self, spec: ColumnSpec) -> Any:
        self._skip_if_unsupported(spec)
        return self.from_arrow(self._arrow_table((spec,)))

    @pytest.mark.parametrize("spec", EXTENDED_COLUMNS, ids=lambda s: s.name)
    def test_extended_column_resolves_to_expected_data_type(
        self, framework_instance: Any, extended_sample_data: Any, spec: ColumnSpec
    ) -> None:
        assert framework_instance._extract_column_data_type(extended_sample_data, spec.name) == spec.data_type

    def from_arrow(self, table: pa.Table) -> Any:
        """Convert a canonical pa.Table to framework-native data.

        Override in framework-specific test class. Frameworks needing extra
        fixtures (connection, spark_session) or non-Arrow construction (iceberg)
        override the ``validator_sample_data`` / ``precision_sample_data`` fixtures
        themselves and use ``self._arrow_table(...)`` as the typed-values source.
        """
        raise NotImplementedError

    @pytest.fixture
    def validator_sample_data(self) -> Any:
        return self.from_arrow(self._arrow_table(self.VALIDATOR_COLUMNS))

    @pytest.fixture
    def precision_sample_data(self) -> Any:
        return self.from_arrow(self._arrow_table(self.PRECISION_COLUMNS))

    def _resolver(self, fw: Any, data: Any) -> Callable[[str], DataType | None]:
        """Build the resolver closure DataTypeValidator.validate now expects.

        Mirrors how ComputeFramework.run_validate_output_features will bind the per-call
        resolver from _extract_column_data_type.
        """
        return lambda col: fw._extract_column_data_type(data, col)

    def test_int_column_int64_declaration_passes(self, framework_instance: Any, validator_sample_data: Any) -> None:
        """An int column declared as INT64 must pass (exact or widened match)."""
        feature = Feature.int64_of("int_col")
        feature_set = FeatureSet([feature])

        DataTypeValidator.validate(feature_set, self._resolver(framework_instance, validator_sample_data))

    def test_str_column_string_declaration_passes(self, framework_instance: Any, validator_sample_data: Any) -> None:
        """A string column declared as STRING must pass (exact match)."""
        feature = Feature.str_of("str_col")
        feature_set = FeatureSet([feature])

        DataTypeValidator.validate(feature_set, self._resolver(framework_instance, validator_sample_data))

    def test_str_column_int_declaration_raises_in_strict(
        self, framework_instance: Any, validator_sample_data: Any
    ) -> None:
        """STRING actual vs INT64 declared in strict mode must raise DataTypeMismatchError."""
        feature = Feature.int64_of("str_col")
        feature.options.add_to_group(DefaultOptionKeys.strict_type_enforcement, True)
        feature_set = FeatureSet([feature])

        with pytest.raises(DataTypeMismatchError):
            DataTypeValidator.validate(feature_set, self._resolver(framework_instance, validator_sample_data))

    def test_str_column_int_declaration_raises_in_lenient(
        self, framework_instance: Any, validator_sample_data: Any
    ) -> None:
        """STRING actual vs INT64 declared in lenient mode must still raise.

        STRING vs numeric is not loosely compatible: the lenient mode only relaxes within
        the numeric family and within the timestamp family.
        """
        feature = Feature.int64_of("str_col")
        feature_set = FeatureSet([feature])

        with pytest.raises(DataTypeMismatchError):
            DataTypeValidator.validate(feature_set, self._resolver(framework_instance, validator_sample_data))

    def test_untyped_feature_skipped(self, framework_instance: Any, validator_sample_data: Any) -> None:
        """A feature with data_type=None must skip validation regardless of column content."""
        feature = Feature.not_typed("str_col")
        feature_set = FeatureSet([feature])

        DataTypeValidator.validate(feature_set, self._resolver(framework_instance, validator_sample_data))

    def test_missing_column_skipped(self, framework_instance: Any, validator_sample_data: Any) -> None:
        """A feature naming a column not in the data must not raise (resolver returns None)."""
        feature = Feature.int64_of("nonexistent_col")
        feature_set = FeatureSet([feature])

        DataTypeValidator.validate(feature_set, self._resolver(framework_instance, validator_sample_data))

    def test_int_column_double_declaration_lenient_passes(
        self, framework_instance: Any, validator_sample_data: Any
    ) -> None:
        """INT64 actual vs DOUBLE declared in lenient mode must pass (numeric widening)."""
        feature = Feature.double_of("int_col")
        feature_set = FeatureSet([feature])

        DataTypeValidator.validate(feature_set, self._resolver(framework_instance, validator_sample_data))

    # ------------------------------------------------------------------
    # Precision tests: lock per-framework width/precision distinctions
    # that the current widest-wins string mapping cannot enforce.
    # ------------------------------------------------------------------

    def test_int32_column_strict_int32_passes(self, framework_instance: Any, precision_sample_data: Any) -> None:
        """INT32 actual declared INT32 in strict mode must pass (exact match)."""
        feature = Feature.int32_of("int32_col")
        feature.options.add_to_group(DefaultOptionKeys.strict_type_enforcement, True)
        feature_set = FeatureSet([feature])

        DataTypeValidator.validate(feature_set, self._resolver(framework_instance, precision_sample_data))

    def test_int64_column_strict_int32_raises(self, framework_instance: Any, precision_sample_data: Any) -> None:
        """INT64 actual vs INT32 declared in strict mode must raise (narrowing forbidden).

        INT32's _COMPATIBLE_TYPES entry does not exist, so only INT32==INT32 is accepted.
        INT64 actual is wider than INT32 declared, and strict mode rejects it.
        """
        feature = Feature.int32_of("int64_col")
        feature.options.add_to_group(DefaultOptionKeys.strict_type_enforcement, True)
        feature_set = FeatureSet([feature])

        with pytest.raises(DataTypeMismatchError):
            DataTypeValidator.validate(feature_set, self._resolver(framework_instance, precision_sample_data))

    def test_int32_column_strict_int64_passes(self, framework_instance: Any, precision_sample_data: Any) -> None:
        """INT32 actual vs INT64 declared in strict mode must pass (widening allowed).

        INT64's _COMPATIBLE_TYPES entry is {INT32, INT64}, so INT32 actual is accepted.
        """
        feature = Feature.int64_of("int32_col")
        feature.options.add_to_group(DefaultOptionKeys.strict_type_enforcement, True)
        feature_set = FeatureSet([feature])

        DataTypeValidator.validate(feature_set, self._resolver(framework_instance, precision_sample_data))

    def test_float32_column_strict_float_passes(self, framework_instance: Any, precision_sample_data: Any) -> None:
        """FLOAT actual declared FLOAT in strict mode must pass (exact match)."""
        feature = Feature.float_of("float32_col")
        feature.options.add_to_group(DefaultOptionKeys.strict_type_enforcement, True)
        feature_set = FeatureSet([feature])

        DataTypeValidator.validate(feature_set, self._resolver(framework_instance, precision_sample_data))

    def test_float64_column_strict_float_raises(self, framework_instance: Any, precision_sample_data: Any) -> None:
        """DOUBLE actual vs FLOAT declared in strict mode must raise (narrowing forbidden).

        FLOAT has no _COMPATIBLE_TYPES entry, so only FLOAT==FLOAT is accepted.
        """
        feature = Feature.float_of("float64_col")
        feature.options.add_to_group(DefaultOptionKeys.strict_type_enforcement, True)
        feature_set = FeatureSet([feature])

        with pytest.raises(DataTypeMismatchError):
            DataTypeValidator.validate(feature_set, self._resolver(framework_instance, precision_sample_data))

    def test_float32_column_strict_double_passes(self, framework_instance: Any, precision_sample_data: Any) -> None:
        """FLOAT actual vs DOUBLE declared in strict mode must pass (widening allowed).

        DOUBLE's _COMPATIBLE_TYPES entry includes FLOAT.
        """
        feature = Feature.double_of("float32_col")
        feature.options.add_to_group(DefaultOptionKeys.strict_type_enforcement, True)
        feature_set = FeatureSet([feature])

        DataTypeValidator.validate(feature_set, self._resolver(framework_instance, precision_sample_data))

    def test_timestamp_ms_column_strict_ms_passes(self, framework_instance: Any, precision_sample_data: Any) -> None:
        """TIMESTAMP_MILLIS actual declared TIMESTAMP_MILLIS strict must pass (exact match)."""
        feature = Feature.timestamp_millis_of("timestamp_ms_col")
        feature.options.add_to_group(DefaultOptionKeys.strict_type_enforcement, True)
        feature_set = FeatureSet([feature])

        DataTypeValidator.validate(feature_set, self._resolver(framework_instance, precision_sample_data))

    def test_timestamp_us_column_strict_ms_raises(self, framework_instance: Any, precision_sample_data: Any) -> None:
        """TIMESTAMP_MICROS actual vs TIMESTAMP_MILLIS declared strict must raise (narrowing).

        TIMESTAMP_MILLIS has no _COMPATIBLE_TYPES entry (only TIMESTAMP_MICROS lists
        TIMESTAMP_MILLIS as compatible, not the reverse), so MILLIS rejects MICROS.
        """
        feature = Feature.timestamp_millis_of("timestamp_us_col")
        feature.options.add_to_group(DefaultOptionKeys.strict_type_enforcement, True)
        feature_set = FeatureSet([feature])

        with pytest.raises(DataTypeMismatchError):
            DataTypeValidator.validate(feature_set, self._resolver(framework_instance, precision_sample_data))
