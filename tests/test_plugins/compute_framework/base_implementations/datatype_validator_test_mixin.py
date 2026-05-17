"""
Shared test mixin for DataTypeValidator enforcement across compute frameworks.

This mixin verifies that each compute framework correctly resolves a column's native dtype
to a unified DataType via ComputeFramework._extract_column_data_type, and that
DataTypeValidator.validate, given a resolver closure, enforces feature data_type declarations
on framework-native data (not just PyArrow).

Each framework-specific test class should inherit from this mixin and provide:
- framework_instance fixture: Returns a compute framework instance
- validator_sample_data fixture: Returns framework-native data with columns
    int_col: integer values [1, 2, 3]
    str_col: string values ["a", "b", "c"]
    float_col: float values [1.0, 2.0, 3.0]
"""

from abc import abstractmethod
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


class DataTypeValidatorFrameworkTestMixin:
    """Shared tests for DataTypeValidator enforcement across all compute frameworks.

    The mixin is intentionally named without a `Test` prefix so pytest does not collect it
    standalone. Framework subclasses pick up the test methods by inheritance.
    """

    @pytest.fixture
    @abstractmethod
    def framework_instance(self) -> Any:
        """Return a compute framework instance.

        Override in framework-specific test class.
        """
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def validator_sample_data(self) -> Any:
        """Return framework-specific sample data.

        Override in framework-specific test class.
        Data should contain columns:
            int_col: integer values [1, 2, 3]
            str_col: string values ["a", "b", "c"]
            float_col: float values [1.0, 2.0, 3.0]
        """
        raise NotImplementedError

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
