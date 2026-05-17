"""
Shared test mixin for DataTypeValidator enforcement across compute frameworks.

This mixin verifies that each compute framework correctly resolves a column's native dtype
to a unified DataType via ComputeFramework._extract_column_data_type, and that
DataTypeValidator.validate, given a resolver closure, enforces feature data_type declarations
on framework-native data (not just PyArrow).

The canonical test data lives here as class-level ``VALIDATOR_COLUMNS`` / ``PRECISION_COLUMNS``
``ColumnSpec`` tuples. Framework subclasses implement ``build_data(columns)`` to materialise
that spec into the framework-native container (and override the data fixtures only when they
need an additional fixture such as ``connection`` or ``spark_session``).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import pytest

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


_TS_VALUES: tuple[datetime, ...] = (
    datetime(2024, 1, 1),
    datetime(2024, 1, 2),
    datetime(2024, 1, 3),
)


class DataTypeValidatorFrameworkTestMixin:
    """Shared tests for DataTypeValidator enforcement across all compute frameworks.

    The mixin is intentionally named without a `Test` prefix so pytest does not collect it
    standalone. Framework subclasses pick up the test methods by inheritance.
    """

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

    def build_data(self, columns: tuple[ColumnSpec, ...]) -> Any:
        """Build framework-native data from a canonical column spec.

        Override in framework-specific test class. Frameworks needing extra fixtures
        (connection, spark_session) override the ``validator_sample_data`` /
        ``precision_sample_data`` fixtures themselves to pass the fixture through.
        """
        raise NotImplementedError

    @pytest.fixture
    def validator_sample_data(self) -> Any:
        return self.build_data(self.VALIDATOR_COLUMNS)

    @pytest.fixture
    def precision_sample_data(self) -> Any:
        return self.build_data(self.PRECISION_COLUMNS)

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
