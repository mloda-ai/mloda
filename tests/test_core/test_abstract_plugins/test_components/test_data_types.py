"""Tests for DataType enum and its conversion methods."""

import pytest
import pyarrow as pa

from mloda_core.abstract_plugins.components.data_types import DataType


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
