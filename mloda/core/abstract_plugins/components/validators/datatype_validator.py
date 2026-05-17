from typing import Any, Callable, Optional


from mloda.core.abstract_plugins.components.data_types import DataType


class DataTypeMismatchError(ValueError):
    """Raised when feature data type doesn't match declared type."""

    def __init__(self, feature_name: str, declared: DataType, actual: DataType) -> None:
        self.feature_name = feature_name
        self.declared = declared
        self.actual = actual
        super().__init__(
            f"Feature '{feature_name}': declared {declared.name}, got {actual.name}, coercion not supported"
        )


class DataTypeValidator:
    """Validates feature data matches declared DataType."""

    _COMPATIBLE_TYPES = {
        DataType.INT64: {DataType.INT32, DataType.INT64},
        DataType.DOUBLE: {DataType.FLOAT, DataType.DOUBLE, DataType.INT32, DataType.INT64},
        DataType.TIMESTAMP_MICROS: {DataType.TIMESTAMP_MILLIS, DataType.TIMESTAMP_MICROS},
    }

    @classmethod
    def _types_compatible(cls, declared: DataType, actual: DataType) -> bool:
        """Check if actual type is compatible with declared (allows widening)."""
        if declared == actual:
            return True
        return actual in cls._COMPATIBLE_TYPES.get(declared, set())

    @classmethod
    def _types_loosely_compatible(cls, declared: DataType, actual: DataType) -> bool:
        """Check if types are loosely compatible (allows any numeric/timestamp pairing).

        Lenient mode allows data type mismatches within the same category:
        - All numeric types (INT32, INT64, FLOAT, DOUBLE) are interchangeable
        - All timestamp types are interchangeable
        - Other types must match exactly

        This fixes legacy FeatureGroups that declare INT32 but return DOUBLE.
        """
        if declared == actual:
            return True

        numeric_types = {DataType.INT32, DataType.INT64, DataType.FLOAT, DataType.DOUBLE}
        if declared in numeric_types and actual in numeric_types:
            return True

        timestamp_types = {DataType.TIMESTAMP_MILLIS, DataType.TIMESTAMP_MICROS}
        if declared in timestamp_types and actual in timestamp_types:
            return True

        return False

    @classmethod
    def validate(
        cls,
        features: Any,
        get_column_data_type: Callable[[str], Optional[DataType]],
    ) -> None:
        """Validate that data columns match declared feature types.

        Args:
            features: FeatureSet containing features to validate.
            get_column_data_type: Resolver that maps a column name to its actual DataType
                or None (column missing / unknown dtype). The caller (typically
                ComputeFramework.run_validate_output_features) binds this per-call so the
                validator stays decoupled from any specific data model.
        """
        from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys

        for feature in features.features:
            if feature.data_type is None:
                continue

            actual_type = get_column_data_type(feature.name)
            if actual_type is None:
                continue

            strict_mode = False
            if feature.options:
                strict_value = feature.options.get(DefaultOptionKeys.strict_type_enforcement)
                strict_mode = strict_value if strict_value is not None else False

            if strict_mode:
                if not cls._types_compatible(feature.data_type, actual_type):
                    raise DataTypeMismatchError(feature.name, feature.data_type, actual_type)
            else:
                if not cls._types_loosely_compatible(feature.data_type, actual_type):
                    raise DataTypeMismatchError(feature.name, feature.data_type, actual_type)
