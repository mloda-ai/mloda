"""SparkAdapter: shared FrameworkAdapter tests plus Spark-specific type, time zone and lifecycle checks."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from mloda._testing.framework_adapter import SparkAdapter
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType

# Reuse the shared adapter tests; the local `adapter` fixture below swaps in SparkAdapter.
from tests.test_testing_kit.test_framework_adapter import (  # noqa: F401
    _reclaimed,
    test_adapter_reads_a_run_all_result,
    test_list_cells_read_back_as_lists,
    test_lost_type_false_for_numeric_columns,
    test_null_timestamp_reads_back_as_none,
    test_round_trip,
    test_rows_reads_the_result_as_a_list_of_dicts,
    test_rows_zero_row_result_is_empty_list,
    test_zero_row_keeps_names,
)


@pytest.fixture
def adapter(spark_session: Any) -> Any:
    from mloda._testing.framework_adapter import SparkAdapter

    return SparkAdapter()


def test_capabilities_typed_columns(adapter: Any) -> None:
    assert adapter.capabilities.typed_columns is True


def test_int32_column_type_is_integer_type(adapter: Any) -> None:
    from pyspark.sql.types import IntegerType

    with adapter:
        result = adapter.from_typed(TypedTestData({"c": (DataType.INT32, [1, 2])}))

        assert adapter.column_type(result, "c") == IntegerType()


def test_null_typed_column_is_lost(adapter: Any) -> None:
    pa = pytest.importorskip("pyarrow")
    with adapter:
        result = adapter.from_arrow(pa.table({"n": pa.array([None, None], type=pa.null())}))

        assert adapter.lost_type(result, "n") is True


def test_naive_timestamp_keeps_wall_clock_under_non_utc_session(adapter: Any, spark_non_utc_timezone: Any) -> None:
    value = datetime(2024, 1, 1, 12, 30)
    with adapter:
        result = adapter.from_typed(TypedTestData({"t": (DataType.TIMESTAMP_MICROS, [value])}))

        assert adapter.column_values(result, "t") == [value]


def test_connection_lifecycle_keeps_session_alive(adapter: Any, spark_session: Any) -> None:
    from pyspark.sql import SparkSession

    with adapter:
        assert adapter.connection is SparkSession.getActiveSession()
    assert adapter.connection is None
    assert spark_session.range(3).count() == 3


@pytest.mark.parametrize(
    "pair", [(DataType.INT64, DataType.STRING), (DataType.STRING, DataType.INT64)], ids=["int_str", "str_int"]
)
def test_duplicate_column_frame_keeps_both_columns_in_pair_order(adapter: Any, pair: tuple[DataType, DataType]) -> None:
    from pyspark.sql.types import LongType, StringType

    spark_types = {DataType.INT64: LongType(), DataType.STRING: StringType()}
    with adapter:
        frame = adapter.duplicate_column_frame(*pair)

        assert adapter.column_names(frame) == ["a", "a"]
        assert [f.dataType for f in frame.schema.fields] == [spark_types[pair[0]], spark_types[pair[1]]]


def test_zero_column_frame_has_no_column_names(adapter: Any) -> None:
    with adapter:
        assert adapter.column_names(adapter.zero_column_frame()) == []


def test_open_twice_raises(adapter: Any) -> None:
    with adapter:
        with pytest.raises(RuntimeError):
            adapter.open()


def test_from_typed_outside_session_raises(adapter: Any) -> None:
    with pytest.raises(RuntimeError):
        adapter.from_typed(TypedTestData({"i": (DataType.INT64, [1])}))


_D = DataType
_HAS_TYPE_CASES: list[tuple[Any, DataType, bool | None]] = [
    (lambda t: t.ByteType(), _D.INT8, True),
    (lambda t: t.ByteType(), _D.INT32, False),
    (lambda t: t.IntegerType(), _D.INT32, True),
    (lambda t: t.IntegerType(), _D.INT64, False),
    (lambda t: t.LongType(), _D.INT64, True),
    (lambda t: t.LongType(), _D.INT32, False),
    (lambda t: t.ShortType(), _D.INT32, False),
    (lambda t: t.FloatType(), _D.FLOAT, True),
    (lambda t: t.FloatType(), _D.DOUBLE, False),
    (lambda t: t.DoubleType(), _D.DOUBLE, True),
    (lambda t: t.DoubleType(), _D.FLOAT, False),
    (lambda t: t.BooleanType(), _D.BOOLEAN, True),
    (lambda t: t.BooleanType(), _D.INT64, False),
    (lambda t: t.StringType(), _D.STRING, True),
    (lambda t: t.StringType(), _D.BINARY, False),
    (lambda t: t.StringType("UTF8_LCASE"), _D.STRING, True),
    (lambda t: t.BinaryType(), _D.BINARY, True),
    (lambda t: t.BinaryType(), _D.STRING, False),
    (lambda t: t.DateType(), _D.DATE, True),
    (lambda t: t.DateType(), _D.TIMESTAMP_MICROS, False),
    (lambda t: t.TimestampNTZType(), _D.TIMESTAMP_MICROS, True),
    (lambda t: t.TimestampNTZType(), _D.TIMESTAMP_MICROS_TZ, False),
    (lambda t: t.TimestampType(), _D.TIMESTAMP_MICROS_TZ, True),
    (lambda t: t.TimestampType(), _D.TIMESTAMP_MICROS, None),
    (lambda t: t.DecimalType(38, 18), _D.DECIMAL, True),
    (lambda t: t.DecimalType(10, 2), _D.DECIMAL, False),
    (lambda t: t.ArrayType(t.LongType()), _D.LIST, True),
    (lambda t: t.ArrayType(t.LongType()), _D.INT64, False),
    (lambda t: t.LongType(), _D.UINT8, None),
    (lambda t: t.LongType(), _D.UINT64, None),
    (lambda t: t.TimestampNTZType(), _D.TIMESTAMP_MILLIS, None),
]


def _jvm_free_spark_frame(spark_type: Any, column: str = "c") -> Any:
    """A JVM-free stand-in for a one-column Spark DataFrame."""
    from pyspark.sql.types import StructField, StructType

    return SimpleNamespace(columns=[column], schema=StructType([StructField(column, spark_type)]))


@pytest.mark.parametrize("make_type, declared, expected", _HAS_TYPE_CASES)
def test_spark_has_type_is_exact(make_type: Any, declared: DataType, expected: bool | None) -> None:
    types = pytest.importorskip("pyspark.sql.types")

    assert SparkAdapter().has_type(_jvm_free_spark_frame(make_type(types)), "c", declared) is expected


def test_spark_has_type_matches_adapter_schema() -> None:
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("pyspark.sql.types")
    from decimal import Decimal

    from pyspark.sql.pandas.types import from_arrow_schema

    from mloda._testing.framework_adapter import to_arrow_table
    from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import SparkFramework

    data = TypedTestData(
        {
            "i": (DataType.INT64, [1]),
            "s": (DataType.STRING, ["a"]),
            "t": (DataType.TIMESTAMP_MICROS, [datetime(2024, 1, 1, 12, 30)]),
            "z": (DataType.TIMESTAMP_MICROS_TZ, [datetime(2024, 1, 1, 12, 30, tzinfo=UTC)]),
            "d": (DataType.DECIMAL, [Decimal("1.5")]),
            "l": (DataType.LIST, [[1, 2]]),
        }
    )
    arrow_types = {"l": pa.list_(pa.int64())}
    schema = from_arrow_schema(to_arrow_table(data, arrow_types).schema, prefer_timestamp_ntz=True)
    expected = {
        "i": DataType.INT64,
        "s": DataType.STRING,
        "t": DataType.TIMESTAMP_MICROS,
        "z": DataType.TIMESTAMP_MICROS_TZ,
        "d": DataType.DECIMAL,
        "l": DataType.LIST,
    }
    for name, dt in expected.items():
        assert SparkFramework._spark_type_to_exact_data_type(schema[name].dataType) == dt, name
        frame = _jvm_free_spark_frame(schema[name].dataType, name)
        assert SparkAdapter().has_type(frame, name, dt) is True, name


def test_spark_has_type_on_zero_rows(adapter: Any) -> None:
    with adapter:
        result = adapter.from_typed(TypedTestData({"i": (DataType.INT64, []), "s": (DataType.STRING, [])}))

        assert adapter.has_type(result, "i", DataType.INT64) is True
        assert adapter.has_type(result, "s", DataType.STRING) is True


def test_spark_has_type_distinguishes_aware_from_naive(adapter: Any) -> None:
    aware = datetime(2024, 1, 1, 12, 30, tzinfo=UTC)
    with adapter:
        zoned = adapter.from_typed(TypedTestData({"t": (DataType.TIMESTAMP_MICROS_TZ, [aware])}))
        naive = adapter.from_typed(TypedTestData({"t": (DataType.TIMESTAMP_MICROS, [datetime(2024, 1, 1, 12, 30)])}))

        assert adapter.has_type(zoned, "t", DataType.TIMESTAMP_MICROS_TZ) is True
        assert adapter.has_type(naive, "t", DataType.TIMESTAMP_MICROS_TZ) is False
