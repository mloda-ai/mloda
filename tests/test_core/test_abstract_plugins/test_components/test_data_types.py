"""Tests for DataType enum and its conversion methods."""

import datetime
import decimal

import pytest
import pyarrow as pa

from mloda.user import DataType


class TestFromArrowType:
    """Tests for DataType.from_arrow_type() method."""

    def test_from_arrow_type_basic_types(self) -> None:
        """Test conversion from PyArrow basic types to DataType enum."""
        # Integer types
        assert DataType.from_arrow_type(pa.int32()) == DataType.INT32
        assert DataType.from_arrow_type(pa.int64()) == DataType.INT64

        # Float types
        assert DataType.from_arrow_type(pa.float32()) == DataType.FLOAT
        assert DataType.from_arrow_type(pa.float64()) == DataType.DOUBLE

        # Boolean
        assert DataType.from_arrow_type(pa.bool_()) == DataType.BOOLEAN

        # String types
        assert DataType.from_arrow_type(pa.string()) == DataType.STRING

        # Binary
        assert DataType.from_arrow_type(pa.binary()) == DataType.BINARY

        # Date
        assert DataType.from_arrow_type(pa.date32()) == DataType.DATE

    def test_from_arrow_type_timestamps(self) -> None:
        """Test conversion from PyArrow timestamp types to DataType enum."""
        # Millisecond precision
        assert DataType.from_arrow_type(pa.timestamp("ms")) == DataType.TIMESTAMP_MILLIS

        # Microsecond precision
        assert DataType.from_arrow_type(pa.timestamp("us")) == DataType.TIMESTAMP_MICROS

    def test_from_arrow_type_decimal(self) -> None:
        """Test conversion from PyArrow decimal types to DataType enum."""
        # Standard decimal with 38 precision and 18 scale
        assert DataType.from_arrow_type(pa.decimal128(38, 18)) == DataType.DECIMAL

        # Different precision/scale should still map to DECIMAL
        assert DataType.from_arrow_type(pa.decimal128(10, 2)) == DataType.DECIMAL
        assert DataType.from_arrow_type(pa.decimal128(20, 5)) == DataType.DECIMAL

    def test_from_arrow_type_round_trip(self) -> None:
        """Test that converting to arrow and back preserves the DataType."""
        all_data_types = [
            DataType.INT32,
            DataType.INT64,
            DataType.FLOAT,
            DataType.DOUBLE,
            DataType.BOOLEAN,
            DataType.STRING,
            DataType.BINARY,
            DataType.DATE,
            DataType.TIMESTAMP_MILLIS,
            DataType.TIMESTAMP_MICROS,
            DataType.DECIMAL,
        ]

        for data_type in all_data_types:
            # Convert to arrow type and back
            arrow_type = DataType.to_arrow_type(data_type)
            result = DataType.from_arrow_type(arrow_type)
            assert result == data_type, f"Round trip failed for {data_type}"

    def test_from_arrow_type_unsupported_type(self) -> None:
        """Test that unsupported PyArrow types raise ValueError."""
        # Test with a type that shouldn't be supported
        with pytest.raises(ValueError, match="Unsupported.*"):
            DataType.from_arrow_type(pa.list_(pa.int32()))

        with pytest.raises(ValueError, match="Unsupported.*"):
            DataType.from_arrow_type(pa.struct([("field1", pa.int32())]))


class TestInferTypeFromPyType:
    """Tests for DataType.infer_type_from_py_type() method."""

    def test_infer_boolean(self) -> None:
        assert DataType.infer_type_from_py_type(True) == DataType.BOOLEAN
        assert DataType.infer_type_from_py_type(False) == DataType.BOOLEAN

    def test_infer_int32(self) -> None:
        assert DataType.infer_type_from_py_type(0) == DataType.INT32
        assert DataType.infer_type_from_py_type(42) == DataType.INT32
        assert DataType.infer_type_from_py_type(-1) == DataType.INT32
        assert DataType.infer_type_from_py_type(2**31 - 1) == DataType.INT32

    def test_infer_int64(self) -> None:
        assert DataType.infer_type_from_py_type(2**31) == DataType.INT64
        assert DataType.infer_type_from_py_type(-(2**31) - 1) == DataType.INT64

    def test_infer_double(self) -> None:
        assert DataType.infer_type_from_py_type(3.14) == DataType.DOUBLE
        assert DataType.infer_type_from_py_type(0.0) == DataType.DOUBLE

    def test_infer_string(self) -> None:
        assert DataType.infer_type_from_py_type("hello") == DataType.STRING
        assert DataType.infer_type_from_py_type("") == DataType.STRING

    def test_infer_binary(self) -> None:
        assert DataType.infer_type_from_py_type(b"data") == DataType.BINARY
        assert DataType.infer_type_from_py_type(b"") == DataType.BINARY

    def test_infer_date(self) -> None:
        assert DataType.infer_type_from_py_type(datetime.date(2026, 1, 1)) == DataType.DATE

    def test_infer_datetime_maps_to_timestamp(self) -> None:
        assert DataType.infer_type_from_py_type(datetime.datetime(2026, 1, 1, 12, 0)) == DataType.TIMESTAMP_MICROS

    def test_infer_decimal(self) -> None:
        assert DataType.infer_type_from_py_type(decimal.Decimal("3.14")) == DataType.DECIMAL
        assert DataType.infer_type_from_py_type(decimal.Decimal("0")) == DataType.DECIMAL

    def test_infer_arrow_date_scalar(self) -> None:
        assert DataType.infer_type_from_py_type(pa.scalar(1, type=pa.date32())) == DataType.DATE

    def test_infer_arrow_timestamp_scalar(self) -> None:
        assert DataType.infer_type_from_py_type(pa.scalar(1, type=pa.timestamp("us"))) == DataType.TIMESTAMP_MICROS

    def test_infer_unsupported_type(self) -> None:
        with pytest.raises(ValueError, match="Unsupported data type"):
            DataType.infer_type_from_py_type([1, 2, 3])

    def test_boolean_before_int(self) -> None:
        """bool is a subclass of int, so must be checked first."""
        assert DataType.infer_type_from_py_type(True) == DataType.BOOLEAN
        assert DataType.infer_type_from_py_type(True) != DataType.INT32
