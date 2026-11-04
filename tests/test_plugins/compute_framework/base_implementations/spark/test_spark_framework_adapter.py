"""SparkAdapter: shared FrameworkAdapter tests plus Spark-specific type, time zone and lifecycle checks."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType

# Reuse the shared adapter tests; the local `adapter` fixture below swaps in SparkAdapter.
from tests.test_testing_kit.test_framework_adapter import (  # noqa: F401
    test_list_cells_read_back_as_lists,
    test_lost_type_false_for_numeric_columns,
    test_null_timestamp_reads_back_as_none,
    test_round_trip,
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


def test_tz_timestamp_reads_back_as_same_instant(adapter: Any, spark_non_utc_timezone: Any) -> None:
    value = datetime(2024, 1, 1, 12, 30, tzinfo=UTC)
    with adapter:
        result = adapter.from_typed(TypedTestData({"t": (DataType.TIMESTAMP_MICROS_TZ, [value])}))

        read = adapter.column_values(result, "t")

    assert read[0] == value


def test_connection_lifecycle_keeps_session_alive(adapter: Any, spark_session: Any) -> None:
    from pyspark.sql import SparkSession

    with adapter:
        assert adapter.connection is SparkSession.getActiveSession()
    assert adapter.connection is None
    assert spark_session.range(3).count() == 3


def test_open_twice_raises(adapter: Any) -> None:
    with adapter:
        with pytest.raises(RuntimeError):
            adapter.open()


def test_from_typed_outside_session_raises(adapter: Any) -> None:
    with pytest.raises(RuntimeError):
        adapter.from_typed(TypedTestData({"i": (DataType.INT64, [1])}))
