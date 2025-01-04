from enum import Enum
from typing import Any
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
        elif isinstance(value, pa.Date32Scalar) or isinstance(value, pa.Date32Array):
            return cls.DATE
        elif isinstance(value, pa.TimestampScalar) or isinstance(value, pa.TimestampArray):
            # Defaulting to TIMESTAMP_MICROS; adjust as needed
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
