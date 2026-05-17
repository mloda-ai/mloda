import datetime
import decimal
from enum import Enum
from typing import Any, Optional
import pyarrow as pa


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
        elif isinstance(value, pa.Date32Scalar) or isinstance(value, pa.Date32Array):
            return cls.DATE
        elif isinstance(value, pa.TimestampScalar) or isinstance(value, pa.TimestampArray):
            return cls.TIMESTAMP_MICROS
        else:
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
    def from_dtype_string(cls, dtype_string: str) -> Optional["DataType"]:
        """Map a native dtype string from any compute framework to a DataType.

        The mapping is biased toward the widest type in each numeric / timestamp family so
        the existing loose-compat widening rules keep strict mode quiet on frameworks that
        collapse types (e.g. SQLite's `TEXT` for everything, PythonDict's `type.__name__`).

        Returns None for unmapped strings, including pandas `"object"` which is genuinely
        ambiguous (overloads STRING, BINARY, DATE, DECIMAL, arbitrary Python objects).
        """
        s_lower = dtype_string.lower()
        exact = _DTYPE_STRING_EXACT_MAP.get(s_lower)
        if exact is not None:
            return exact
        for prefix, data_type in _DTYPE_STRING_PREFIX_MAP:
            if s_lower.startswith(prefix):
                return data_type
        return None

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


_DTYPE_STRING_EXACT_MAP: dict[str, DataType] = {
    # INT64 (widest int wins; collapsed-type frameworks lean here)
    "int64": DataType.INT64,
    "bigint": DataType.INT64,
    "longtype()": DataType.INT64,
    "long": DataType.INT64,
    "int": DataType.INT64,
    "integer": DataType.INT64,
    # INT32
    "int32": DataType.INT32,
    "integertype()": DataType.INT32,
    # DOUBLE (widest float wins)
    "float64": DataType.DOUBLE,
    "double": DataType.DOUBLE,
    "doubletype()": DataType.DOUBLE,
    "real": DataType.DOUBLE,
    "float": DataType.DOUBLE,
    # FLOAT
    "float32": DataType.FLOAT,
    "floattype()": DataType.FLOAT,
    # BOOLEAN
    "bool": DataType.BOOLEAN,
    "boolean": DataType.BOOLEAN,
    "booleantype()": DataType.BOOLEAN,
    # STRING
    "str": DataType.STRING,
    "string": DataType.STRING,
    "varchar": DataType.STRING,
    "text": DataType.STRING,
    "stringtype()": DataType.STRING,
    "utf8": DataType.STRING,
    "large_string": DataType.STRING,
    "large_utf8": DataType.STRING,
    # BINARY
    "bytes": DataType.BINARY,
    "binary": DataType.BINARY,
    "blob": DataType.BINARY,
    "binarytype()": DataType.BINARY,
    "large_binary": DataType.BINARY,
    # DATE
    "date": DataType.DATE,
    "datetype()": DataType.DATE,
    "date32[day]": DataType.DATE,
    # TIMESTAMP_MICROS (widest ts wins)
    "datetime": DataType.TIMESTAMP_MICROS,
    "timestamp": DataType.TIMESTAMP_MICROS,
    "timestamptz": DataType.TIMESTAMP_MICROS,
    "timestamptype()": DataType.TIMESTAMP_MICROS,
    "timestampntztype()": DataType.TIMESTAMP_MICROS,
}

_DTYPE_STRING_PREFIX_MAP: tuple[tuple[str, DataType], ...] = (
    ("decimal128(", DataType.DECIMAL),
    ("decimaltype(", DataType.DECIMAL),
    ("decimal(", DataType.DECIMAL),
    ("datetime64[", DataType.TIMESTAMP_MICROS),
    ("datetime(", DataType.TIMESTAMP_MICROS),
    ("timestamp[", DataType.TIMESTAMP_MICROS),
)
