"""Real Spark transfers keep Arrow instants and zones, or name the rejected column and zone."""

import os
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pytest

from mloda._testing.framework_adapter import SparkAdapter
from mloda.user import ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import SparkFramework
from mloda_plugins.compute_framework.base_implementations.spark.spark_pyarrow_transformer import SparkPyArrowTransformer

_TIMEZONE = "spark.sql.session.timeZone"
_COLUMN = "event_time"
_ROUTES = ["transformer", "adapter"]
_TICKS = [
    None,
    -1,
    1678604399123456,
    1678604400654321,
    None,
    1699162200123456,
    1699165800654321,
    None,
]


@pytest.fixture
def zoned_spark(spark_session: Any, request: pytest.FixtureRequest) -> Iterator[Any]:
    previous = spark_session.conf.get(_TIMEZONE)
    spark_session.conf.set(_TIMEZONE, getattr(request, "param", "UTC"))
    try:
        yield spark_session
    finally:
        spark_session.conf.set(_TIMEZONE, previous)


def _table(zone: str, shape: str = "values") -> pa.Table:
    ticks = [] if shape == "empty" else [None, None, None] if shape == "all-null" else _TICKS
    # Construct from integer ticks so PYARROW_IGNORE_TIMEZONE cannot alter the fixture's instants.
    column = pa.array(ticks, type=pa.int64()).cast(pa.timestamp("us", tz=zone))
    return pa.table(
        {
            "row_id": pa.array(range(len(ticks)), type=pa.int64()),
            "payload": pa.array([f"row-{i}" if i != 2 else None for i in range(len(ticks))], type=pa.string()),
            _COLUMN: column,
        }
    )


def _ordered_frame(frame: Any, spark: Any, read_zone: str | None, expected_config: list[str]) -> Any:
    if read_zone is not None:
        spark.conf.set(_TIMEZONE, read_zone)
        expected_config[0] = read_zone
    return frame.orderBy("row_id")


def _round_trip(table: pa.Table, spark: Any, route: str, read_zone: str | None, expected_config: list[str]) -> pa.Table:
    if route == "adapter":
        with SparkAdapter() as adapter:
            frame = adapter.from_arrow(table)
            return adapter.to_table(_ordered_frame(frame, spark, read_zone, expected_config))
    if route == "framework":
        framework = SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        framework.set_framework_connection_object(spark)
        frame = framework.transform(table, table.column_names)
    else:
        assert route == "transformer"
        frame = SparkPyArrowTransformer.transform_other_fw_to_fw(table, framework_connection_object=spark)
    result = SparkPyArrowTransformer.transform_fw_to_other_fw(_ordered_frame(frame, spark, read_zone, expected_config))
    assert isinstance(result, pa.Table)
    return result


def _assert_transfer_contract(table: pa.Table, spark: Any, route: str, *, read_zone: str | None = None) -> str:
    expected_config = [spark.conf.get(_TIMEZONE)]
    before_ignore_timezone = os.environ.get("PYARROW_IGNORE_TIMEZONE")
    aware = [
        (field.name, field.type.tz) for field in table.schema if pa.types.is_timestamp(field.type) and field.type.tz
    ]
    try:
        result = _round_trip(table, spark, route, read_zone, expected_config)
    except ValueError as error:
        message = str(error)
        assert any(name in message and zone in message for name, zone in aware), (
            f"a timezone refusal must name its column and source zone, got: {message}"
        )
        return "rejected"
    finally:
        assert spark.conf.get(_TIMEZONE) == expected_config[0], "a transfer must not change the caller's session zone"
        assert os.environ.get("PYARROW_IGNORE_TIMEZONE") == before_ignore_timezone

    assert isinstance(result, pa.Table)
    assert set(result.column_names) == set(table.column_names)
    assert result.num_rows == table.num_rows
    for name in table.column_names:
        expected = table[name].combine_chunks()
        actual = result[name].combine_chunks()
        if pa.types.is_timestamp(expected.type):
            assert pa.types.is_timestamp(actual.type)
            assert actual.cast(pa.int64()).to_pylist() == expected.cast(pa.int64()).to_pylist(), (
                f"Spark changed timestamp instants for {name} in {expected.type.tz}"
            )
            assert actual.type == expected.type, (
                f"Spark replaced the source zone for {name}: {expected.type} -> {actual.type}"
            )
            assert [value.isoformat() if value is not None else None for value in actual.to_pylist()] == [
                value.isoformat() if value is not None else None for value in expected.to_pylist()
            ]
        else:
            assert actual.equals(expected), f"Spark changed the companion column {name}"
    return "preserved"


@pytest.mark.parametrize("zoned_spark", ["UTC", "America/Los_Angeles"], indirect=True)
@pytest.mark.parametrize("zone", ["America/New_York", "+05:45"], ids=["dst-zone", "sub-hour-offset"])
@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("ignore_timezone", [None, "1"], ids=["normal-arrow", "ignore-timezone-env"])
def test_zoned_arrow_transfers_keep_instants_and_original_zone_or_refuse_explicitly(
    zoned_spark: Any, zone: str, route: str, ignore_timezone: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if ignore_timezone is None:
        monkeypatch.delenv("PYARROW_IGNORE_TIMEZONE", raising=False)
    else:
        monkeypatch.setenv("PYARROW_IGNORE_TIMEZONE", ignore_timezone)
    source = _table(zone)
    assert source[_COLUMN].cast(pa.int64()).to_pylist() == _TICKS
    assert source.schema.field(_COLUMN).type == pa.timestamp("us", tz=zone)

    _assert_transfer_contract(source, zoned_spark, route)


@pytest.mark.parametrize("zone", ["America/New_York", "+05:45"])
def test_public_framework_transform_applies_the_same_timezone_contract(zoned_spark: Any, zone: str) -> None:
    _assert_transfer_contract(_table(zone), zoned_spark, "framework")


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("shape", ["all-null", "empty"])
def test_timezone_contract_uses_the_declared_schema_even_without_nonnull_values(
    zoned_spark: Any, route: str, shape: str
) -> None:
    _assert_transfer_contract(_table("America/New_York", shape), zoned_spark, route)


@pytest.mark.parametrize("route", _ROUTES)
def test_each_of_two_zoned_columns_keeps_its_own_zone_or_is_named_in_the_refusal(zoned_spark: Any, route: str) -> None:
    source = _table("America/New_York")
    source = source.append_column("offset_time", _table("+05:45")[_COLUMN])

    _assert_transfer_contract(source, zoned_spark, route)


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("zone", ["America/New_York", "UTC"])
def test_changing_the_session_zone_cannot_relabel_an_already_accepted_column(
    zoned_spark: Any, route: str, zone: str
) -> None:
    zoned_spark.conf.set(_TIMEZONE, zone)
    read_zone = "UTC" if zone == "America/New_York" else "America/New_York"

    _assert_transfer_contract(_table(zone), zoned_spark, route, read_zone=read_zone)


@pytest.mark.parametrize("route", _ROUTES)
def test_utc_timestamp_control_uses_the_same_contract(zoned_spark: Any, route: str) -> None:
    _assert_transfer_contract(_table("UTC"), zoned_spark, route)


@pytest.mark.parametrize("route", _ROUTES)
def test_unzoned_payload_columns_still_round_trip(zoned_spark: Any, route: str) -> None:
    source = _table("UTC").drop([_COLUMN])

    assert _assert_transfer_contract(source, zoned_spark, route) == "preserved"
