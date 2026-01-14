import pytest
import pyarrow as pa

from mloda.core.abstract_plugins.components.validators.datatype_validator import (
    DataTypeMismatchError,
    DataTypeValidator,
)
from mloda.user import DataType
from mloda.user import Feature
from mloda.provider import FeatureSet


class TestDataTypeMismatchError:
    """Test DataTypeMismatchError exception class."""

    def test_datatype_mismatch_error_message(self) -> None:
        """Test DataTypeMismatchError contains useful info."""
        error = DataTypeMismatchError("age", DataType.INT32, DataType.STRING)
        msg = str(error)
        assert "age" in msg
        assert "INT32" in msg
        assert "STRING" in msg


class TestTypesCompatible:
    """Test the _types_compatible static method."""

    def test_types_compatible_exact_match(self) -> None:
        """Test exact type match is compatible."""
        assert DataTypeValidator._types_compatible(DataType.INT32, DataType.INT32) is True
        assert DataTypeValidator._types_compatible(DataType.STRING, DataType.STRING) is True
        assert DataTypeValidator._types_compatible(DataType.BOOLEAN, DataType.BOOLEAN) is True

    def test_types_compatible_widening(self) -> None:
        """Test type widening rules."""
        # INT32 -> INT64 is safe widening
        assert DataTypeValidator._types_compatible(DataType.INT64, DataType.INT32) is True
        # FLOAT -> DOUBLE is safe widening
        assert DataTypeValidator._types_compatible(DataType.DOUBLE, DataType.FLOAT) is True
        # INT32 -> DOUBLE is safe widening
        assert DataTypeValidator._types_compatible(DataType.DOUBLE, DataType.INT32) is True
        # TIMESTAMP_MILLIS -> TIMESTAMP_MICROS
        assert DataTypeValidator._types_compatible(DataType.TIMESTAMP_MICROS, DataType.TIMESTAMP_MILLIS) is True

    def test_types_incompatible(self) -> None:
        """Test incompatible types return False."""
        # STRING to INT is never compatible
        assert DataTypeValidator._types_compatible(DataType.INT32, DataType.STRING) is False
        # INT64 to INT32 (narrowing) is not allowed
        assert DataTypeValidator._types_compatible(DataType.INT32, DataType.INT64) is False
        # DOUBLE to FLOAT (narrowing) is not allowed
        assert DataTypeValidator._types_compatible(DataType.FLOAT, DataType.DOUBLE) is False
        # BOOLEAN to INT is not allowed
        assert DataTypeValidator._types_compatible(DataType.INT32, DataType.BOOLEAN) is False


class TestValidate:
    """Test the validate static method."""

    def test_validate_passes_on_match(self) -> None:
        """Test validation passes when types match."""
        # Create PyArrow table with INT32 column
        table = pa.table({"age": pa.array([25, 30], type=pa.int32())})

        feature = Feature.int32_of("age")
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Should not raise
        DataTypeValidator.validate(table, feature_set)

    def test_validate_raises_on_mismatch(self) -> None:
        """Test validation raises DataTypeMismatchError on type mismatch."""
        # Create PyArrow table with STRING column
        table = pa.table({"age": pa.array(["25", "30"], type=pa.string())})

        feature = Feature.int32_of("age")  # Declared as INT32
        feature_set = FeatureSet()
        feature_set.add(feature)

        with pytest.raises(DataTypeMismatchError):
            DataTypeValidator.validate(table, feature_set)

    def test_validate_skips_untyped_features(self) -> None:
        """Test validation skips features without declared data_type."""
        table = pa.table({"age": pa.array(["25", "30"], type=pa.string())})

        feature = Feature.not_typed("age")  # No data_type
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Should not raise - untyped features are skipped
        DataTypeValidator.validate(table, feature_set)

    def test_validate_passes_with_widening(self) -> None:
        """Test validation passes when actual type can be widened to declared type."""
        # Create PyArrow table with INT32 column
        table = pa.table({"age": pa.array([25, 30], type=pa.int32())})

        feature = Feature.int64_of("age")  # Declared as INT64
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Should not raise - INT32 can be widened to INT64
        DataTypeValidator.validate(table, feature_set)

    def test_validate_multiple_features(self) -> None:
        """Test validation works with multiple features."""
        table = pa.table(
            {
                "age": pa.array([25, 30], type=pa.int32()),
                "name": pa.array(["Alice", "Bob"], type=pa.string()),
                "score": pa.array([95.5, 87.3], type=pa.float64()),
            }
        )

        feature_set = FeatureSet()
        feature_set.add(Feature.int32_of("age"))
        feature_set.add(Feature.str_of("name"))
        feature_set.add(Feature.double_of("score"))

        # Should not raise - all types match
        DataTypeValidator.validate(table, feature_set)

    def test_validate_raises_on_first_mismatch(self) -> None:
        """Test validation raises on first type mismatch."""
        table = pa.table(
            {
                "age": pa.array([25, 30], type=pa.int32()),
                "name": pa.array([123, 456], type=pa.int32()),  # Wrong type
            }
        )

        feature_set = FeatureSet()
        feature_set.add(Feature.int32_of("age"))
        feature_set.add(Feature.str_of("name"))  # Expects STRING

        with pytest.raises(DataTypeMismatchError):
            DataTypeValidator.validate(table, feature_set)
