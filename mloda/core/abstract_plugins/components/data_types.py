from __future__ import annotations

import datetime
import decimal
import sys
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import pyarrow as pa


def _pyarrow() -> Any:
    """pyarrow is an optional backend: import it at the point of use, never at module import."""
    try:
        import pyarrow

        return pyarrow
    except ImportError:
        raise ImportError(
            "pyarrow is required for Arrow type conversions. Install it with: pip install 'mloda[pyarrow]'"
        )


class DataType(Enum):
    """
    These enums are based on arrow data types, which are found in the parquet file format,
    but also have direct or indirect conversions to a lot of dataframe libraries.

        pandas
        pyarrow
        dask
        vaex
        polas
        spark
        ...
    """

    INT32 = "INT32"  # pa.int32()
    INT64 = "INT64"  # pa.int64()
    FLOAT = "FLOAT"  # pa.float32()
    DOUBLE = "DOUBLE"  # pa.float64()
    BOOLEAN = "BOOLEAN"  # pa.bool_()
    STRING = "STRING"  # pa.string()
    BINARY = "BINARY"  # pa.binary()
    DATE = "DATE"  # pa.date32()
    TIMESTAMP_MILLIS = "TIMESTAMP_MILLIS"  # pa.timestamp("ms")
    TIMESTAMP_MICROS = "TIMESTAMP_MICROS"  # pa.timestamp("us")
    DECIMAL = "DECIMAL"  # pa.decimal128(38, 18)

    def __hash__(self) -> int:
        return hash(self.value)

    @classmethod
    def infer_type_from_py_type(cls, value: Any) -> "DataType":
        """
        Infers the Arrow DataType based on the Python value provided.

        Args:
            value (Any): The Python value to infer the type from.

        Returns:
            DataType: The inferred Arrow DataType.
        """
        if isinstance(value, bool):
            return cls.BOOLEAN
        elif isinstance(value, int):
            # Decide between INT32 and INT64 based on the value's size
            if -(2**31) <= value < 2**31:
                return cls.INT32
            else:
                return cls.INT64
        elif isinstance(value, float):
            return cls.DOUBLE  # Defaulting to DOUBLE for higher precision
        elif isinstance(value, str):
            return cls.STRING
        elif isinstance(value, bytes):
            return cls.BINARY
        elif isinstance(value, datetime.datetime):
            return cls.TIMESTAMP_MICROS
        elif isinstance(value, datetime.date):
            return cls.DATE
        elif isinstance(value, decimal.Decimal):
            return cls.DECIMAL

        # Hot path: never import pyarrow if it is absent. If it is not in sys.modules, value cannot be a pyarrow object.
        # The import is a dict hit once present, and unlike a raw sys.modules read it waits out a concurrent first import.
        if sys.modules.get("pyarrow") is not None:
            import pyarrow as pa

            if isinstance(value, pa.Date32Scalar) or isinstance(value, pa.Date32Array):
                return cls.DATE
            if isinstance(value, pa.TimestampScalar) or isinstance(value, pa.TimestampArray):
                return cls.TIMESTAMP_MICROS

        raise ValueError(f"Unsupported data type: {type(value)}")

    @classmethod
    def to_arrow_type(cls, data_type: "DataType") -> pa.DataType:
        """
        Converts the custom DataType enum to the corresponding PyArrow DataType.

        Args:
            data_type (DataType): The custom DataType enum member.

        Returns:
            pa.DataType: The corresponding PyArrow DataType.
        """
        pa = _pyarrow()
        mapping = {
            cls.INT32: pa.int32(),
            cls.INT64: pa.int64(),
            cls.FLOAT: pa.float32(),
            cls.DOUBLE: pa.float64(),
            cls.BOOLEAN: pa.bool_(),
            cls.STRING: pa.string(),
            cls.BINARY: pa.binary(),
            cls.DATE: pa.date32(),
            cls.TIMESTAMP_MILLIS: pa.timestamp("ms"),
            cls.TIMESTAMP_MICROS: pa.timestamp("us"),
            cls.DECIMAL: pa.decimal128(38, 18),
        }

        if data_type in mapping:
            return mapping[data_type]
        else:
            raise ValueError(f"Unsupported DataType: {data_type}")

    @classmethod
    def _arrow_type_to_dtype_or_none(cls, arrow_type: pa.DataType) -> Optional["DataType"]:
        pa = _pyarrow()
        if pa.types.is_int32(arrow_type):
            return cls.INT32
        elif pa.types.is_int64(arrow_type):
            return cls.INT64
        elif pa.types.is_float32(arrow_type):
            return cls.FLOAT
        elif pa.types.is_float64(arrow_type):
            return cls.DOUBLE
        elif pa.types.is_boolean(arrow_type):
            return cls.BOOLEAN
        elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            return cls.STRING
        elif pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
            return cls.BINARY
        elif pa.types.is_date32(arrow_type):
            return cls.DATE
        elif pa.types.is_timestamp(arrow_type):
            if arrow_type.unit == "ms":
                return cls.TIMESTAMP_MILLIS
            elif arrow_type.unit == "us":
                return cls.TIMESTAMP_MICROS
        elif pa.types.is_decimal(arrow_type):
            return cls.DECIMAL
        return None

    @classmethod
    def from_arrow_type(cls, arrow_type: pa.DataType) -> "DataType":
        """
        Converts a PyArrow DataType to the custom DataType enum.

        Args:
            arrow_type (pa.DataType): The PyArrow DataType to convert.

        Returns:
            DataType: The corresponding DataType enum member.

        Raises:
            ValueError: If the arrow_type is not supported.
        """
        result = cls._arrow_type_to_dtype_or_none(arrow_type)
        if result is None:
            raise ValueError(f"Unsupported PyArrow type: {arrow_type}")
        return result

    @classmethod
    def from_arrow_type_safe(cls, arrow_type: pa.DataType) -> Optional["DataType"]:
        """Non-raising counterpart of from_arrow_type: returns None for unsupported types."""
        return cls._arrow_type_to_dtype_or_none(arrow_type)

    @classmethod
    def infer_arrow_type(cls, value: Any) -> pa.DataType:
        """
        Infers the PyArrow DataType directly from a Python value.

        Args:
            value (Any): The Python value to infer the type from.

        Returns:
            pa.DataType: The inferred PyArrow DataType.
        """
        data_type = cls.infer_type_from_py_type(value)
        return cls.to_arrow_type(data_type)
