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


class TestFromDtypeString:
    """Tests for DataType.from_dtype_string(): maps native dtype strings from any framework to DataType.

    Covers the union of dtype strings emitted by the 8 non-Arrow compute frameworks
    (PandasDataFrame, PolarsDataFrame, PolarsLazyDataFrame, DuckDBFramework, SparkFramework,
    SqliteFramework, PythonDictFramework, IcebergFramework). Mapping policy is biased toward
    the widest type in each numeric / timestamp family so the existing loose-compat widening
    rules keep strict mode quiet on collapsed-type frameworks (SQLite, PythonDict).
    """

    def test_method_exists(self) -> None:
        """from_dtype_string must exist as a classmethod on DataType.

        This is the foundational regression: if the method is missing, every other test in
        this class collapses to AttributeError. Asserting existence up front gives a clean
        signal before the parametrised body runs.
        """
        assert hasattr(DataType, "from_dtype_string")

    @pytest.mark.parametrize(
        ("dtype_string", "expected"),
        [
            # INT64 (widest int wins; collapsed-type frameworks lean here)
            pytest.param("int64", DataType.INT64, id="int64_lower"),
            pytest.param("Int64", DataType.INT64, id="Int64_pandas_nullable"),
            pytest.param("BIGINT", DataType.INT64, id="BIGINT_duckdb"),
            pytest.param("LongType()", DataType.INT64, id="LongType_spark"),
            pytest.param("long", DataType.INT64, id="long"),
            pytest.param("int", DataType.INT64, id="int_pydict_or_sqlite"),
            pytest.param("integer", DataType.INT64, id="integer_lower"),
            pytest.param("INTEGER", DataType.INT64, id="INTEGER_uppercase_lowercases_to_integer"),
            # INT32 (exact-match only; lowercased "integer" goes to INT64)
            pytest.param("int32", DataType.INT32, id="int32_lower"),
            pytest.param("Int32", DataType.INT32, id="Int32_pandas_nullable"),
            pytest.param("IntegerType()", DataType.INT32, id="IntegerType_spark"),
            # DOUBLE (widest float wins)
            pytest.param("float64", DataType.DOUBLE, id="float64_lower"),
            pytest.param("Float64", DataType.DOUBLE, id="Float64_polars"),
            pytest.param("DOUBLE", DataType.DOUBLE, id="DOUBLE_duckdb"),
            pytest.param("DoubleType()", DataType.DOUBLE, id="DoubleType_spark"),
            pytest.param("double", DataType.DOUBLE, id="double"),
            pytest.param("real", DataType.DOUBLE, id="real_sqlite"),
            pytest.param("float", DataType.DOUBLE, id="float_collapsed"),
            # FLOAT (exact-match only)
            pytest.param("float32", DataType.FLOAT, id="float32_lower"),
            pytest.param("Float32", DataType.FLOAT, id="Float32_polars"),
            pytest.param("FloatType()", DataType.FLOAT, id="FloatType_spark"),
            # BOOLEAN
            pytest.param("bool", DataType.BOOLEAN, id="bool"),
            pytest.param("Boolean", DataType.BOOLEAN, id="Boolean_polars"),
            pytest.param("BOOLEAN", DataType.BOOLEAN, id="BOOLEAN_duckdb"),
            pytest.param("BooleanType()", DataType.BOOLEAN, id="BooleanType_spark"),
            # STRING
            pytest.param("str", DataType.STRING, id="str_pydict"),
            pytest.param("string", DataType.STRING, id="string_lower"),
            pytest.param("String", DataType.STRING, id="String_polars"),
            pytest.param("VARCHAR", DataType.STRING, id="VARCHAR_duckdb"),
            pytest.param("varchar", DataType.STRING, id="varchar_lower"),
            pytest.param("text", DataType.STRING, id="text_lower"),
            pytest.param("TEXT", DataType.STRING, id="TEXT_sqlite"),
            pytest.param("StringType()", DataType.STRING, id="StringType_spark"),
            pytest.param("utf8", DataType.STRING, id="utf8_arrow_style"),
            pytest.param("large_string", DataType.STRING, id="large_string"),
            pytest.param("large_utf8", DataType.STRING, id="large_utf8"),
            # BINARY
            pytest.param("bytes", DataType.BINARY, id="bytes_pydict"),
            pytest.param("binary", DataType.BINARY, id="binary_lower"),
            pytest.param("Binary", DataType.BINARY, id="Binary_polars"),
            pytest.param("BLOB", DataType.BINARY, id="BLOB_sqlite"),
            pytest.param("blob", DataType.BINARY, id="blob_lower"),
            pytest.param("BinaryType()", DataType.BINARY, id="BinaryType_spark"),
            pytest.param("large_binary", DataType.BINARY, id="large_binary"),
            # DATE
            pytest.param("date", DataType.DATE, id="date_lower"),
            pytest.param("Date", DataType.DATE, id="Date_polars"),
            pytest.param("DATE", DataType.DATE, id="DATE_duckdb"),
            pytest.param("DateType()", DataType.DATE, id="DateType_spark"),
            pytest.param("date32[day]", DataType.DATE, id="date32_day_arrow_style"),
            pytest.param("date64[ms]", DataType.DATE, id="date64_ms_arrow_style"),
            # TIMESTAMP_MICROS (widest ts wins)
            pytest.param("datetime", DataType.TIMESTAMP_MICROS, id="datetime_lower"),
            pytest.param("timestamp", DataType.TIMESTAMP_MICROS, id="timestamp_lower"),
            pytest.param("Timestamp", DataType.TIMESTAMP_MICROS, id="Timestamp"),
            pytest.param("timestamptz", DataType.TIMESTAMP_MICROS, id="timestamptz_duckdb"),
            pytest.param("TIMESTAMP", DataType.TIMESTAMP_MICROS, id="TIMESTAMP_duckdb"),
            pytest.param("TimestampType()", DataType.TIMESTAMP_MICROS, id="TimestampType_spark"),
            pytest.param("TimestampNTZType()", DataType.TIMESTAMP_MICROS, id="TimestampNTZType_spark"),
            pytest.param(
                "datetime(time_unit='us', time_zone=None)", DataType.TIMESTAMP_MICROS, id="datetime_us_polars"
            ),
            pytest.param(
                "Datetime(time_unit='ms', time_zone=None)", DataType.TIMESTAMP_MICROS, id="Datetime_ms_polars"
            ),
            pytest.param(
                "Datetime(time_unit='ns', time_zone='UTC')",
                DataType.TIMESTAMP_MICROS,
                id="Datetime_ns_utc_polars",
            ),
            pytest.param("datetime64[ns]", DataType.TIMESTAMP_MICROS, id="datetime64_ns_pandas"),
            pytest.param("datetime64[ms]", DataType.TIMESTAMP_MICROS, id="datetime64_ms_pandas"),
            pytest.param("datetime64[us]", DataType.TIMESTAMP_MICROS, id="datetime64_us_pandas"),
            pytest.param("timestamp[us]", DataType.TIMESTAMP_MICROS, id="timestamp_us_arrow_style"),
            pytest.param("timestamp[ms]", DataType.TIMESTAMP_MICROS, id="timestamp_ms_arrow_style"),
            # DECIMAL (parameterised; prefix match)
            pytest.param("decimal(38, 18)", DataType.DECIMAL, id="decimal_38_18"),
            pytest.param("DECIMAL(38,18)", DataType.DECIMAL, id="DECIMAL_uppercase"),
            pytest.param("Decimal(precision=10, scale=2)", DataType.DECIMAL, id="Decimal_polars_style"),
            pytest.param("decimal128(38, 18)", DataType.DECIMAL, id="decimal128_arrow_style"),
            pytest.param("DecimalType(10,2)", DataType.DECIMAL, id="DecimalType_spark"),
            # None (unmapped; covers pandas "object" overload, unknown strings, empty)
            pytest.param("object", None, id="object_pandas_overloaded"),
            pytest.param("foo", None, id="foo_unknown"),
            pytest.param("", None, id="empty_string"),
            pytest.param("weird_dtype_xyz", None, id="weird_unknown"),
        ],
    )
    def test_from_dtype_string_mapping(self, dtype_string: str, expected: DataType | None) -> None:
        """from_dtype_string maps a native dtype string to a DataType, or None if unmapped."""
        assert DataType.from_dtype_string(dtype_string) == expected
