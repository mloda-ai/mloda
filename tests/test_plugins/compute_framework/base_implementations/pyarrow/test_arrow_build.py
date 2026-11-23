"""Tests for the shared Arrow builder and the dict entry points that use it.

PYARROW_IGNORE_TIMEZONE=1 makes plain pyarrow read an aware datetime's wall time as UTC; the
builder must give the same instants and zone with the variable unset or set.
"""

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import IcebergFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_value_set import value_set
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable, arrow_append_columns
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_pyarrow_transformer import (
    PythonDictPyArrowTransformer,
)
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

NY = ZoneInfo("America/New_York")
NOON_NY = datetime(2023, 3, 11, 12, 0, tzinfo=NY)
LATER_NY = datetime(2023, 3, 13, 9, 30, tzinfo=NY)
KOLKATA = timezone(timedelta(hours=5, minutes=30))

ENV_VALUES = pytest.mark.parametrize("ignore_timezone", [None, "1"], ids=["unset", "pyarrow_ignore_timezone"])


@pytest.fixture
def ignore_timezone_env(monkeypatch: pytest.MonkeyPatch, ignore_timezone: str | None) -> None:
    if ignore_timezone is None:
        monkeypatch.delenv("PYARROW_IGNORE_TIMEZONE", raising=False)
    else:
        monkeypatch.setenv("PYARROW_IGNORE_TIMEZONE", ignore_timezone)


def _utc(values: list[Any]) -> list[datetime | None]:
    return [None if v is None else v.astimezone(UTC) for v in values]


def _builders() -> Any:
    from mloda_plugins.compute_framework.base_implementations.pyarrow import arrow_build

    return arrow_build


@ENV_VALUES
@pytest.mark.usefixtures("ignore_timezone_env")
class TestArrowBuild:
    def test_array_keeps_instants_and_zone(self) -> None:
        arr = _builders().arrow_array([NOON_NY, None, LATER_NY])
        assert arr.type == pa.timestamp("us", tz="America/New_York")
        assert _utc(arr.to_pylist()) == _utc([NOON_NY, None, LATER_NY])

    def test_array_fixed_offset_zone(self) -> None:
        values = [datetime(2023, 3, 11, 12, 0, tzinfo=KOLKATA)]
        arr = _builders().arrow_array(values)
        assert arr.type == pa.timestamp("us", tz="+05:30")
        assert _utc(arr.to_pylist()) == _utc(values)

    def test_array_accepts_tuple(self) -> None:
        arr = _builders().arrow_array((NOON_NY,))
        assert arr.type == pa.timestamp("us", tz="America/New_York")
        assert _utc(arr.to_pylist()) == _utc([NOON_NY])

    def test_explicit_type_wins(self) -> None:
        target = pa.timestamp("us", tz="Asia/Tokyo")
        arr = _builders().arrow_array([NOON_NY], type=target)
        assert arr.type == target
        assert _utc(arr.to_pylist()) == _utc([NOON_NY])

    def test_table_keeps_instants_and_zone(self) -> None:
        table = _builders().arrow_table({"t": [NOON_NY, LATER_NY], "n": [1, 2]})
        assert table.schema.field("t").type == pa.timestamp("us", tz="America/New_York")
        assert _utc(table.column("t").to_pylist()) == _utc([NOON_NY, LATER_NY])
        assert table.column("n").to_pylist() == [1, 2]

    def test_table_passes_arrow_columns_through(self) -> None:
        column = pa.array([1, 2], type=pa.int8())
        table = _builders().arrow_table({"a": column})
        assert table.schema.field("a").type == pa.int8()


class TestArrowBuildUnchanged:
    """Naive, non-datetime, empty and all-null columns behave exactly like pa.array."""

    @pytest.mark.parametrize(
        "values",
        [
            [datetime(2023, 3, 11, 12, 0), None],
            [1, 2, None],
            ["a", None],
            [NOON_NY, datetime(2023, 3, 11, 12, 0)],
            [],
            [None, None],
        ],
        ids=["naive", "ints", "strings", "mixed_aware_naive", "empty", "all_none"],
    )
    @ENV_VALUES
    @pytest.mark.usefixtures("ignore_timezone_env")
    def test_array_matches_pa_array(self, values: list[Any]) -> None:
        got = _builders().arrow_array(values)
        want = pa.array(values)
        assert got.type == want.type
        assert got.to_pylist() == want.to_pylist()


def _arrow_of(result: Any) -> pa.Table:
    return result if isinstance(result, pa.Table) else result.to_arrow_table()


def _python_dict(data: dict[str, list[Any]]) -> Any:
    return PythonDictPyArrowTransformer.transform_fw_to_other_fw(data)


def _duckdb_from_dict(data: dict[str, list[Any]]) -> Any:
    import duckdb

    return DuckdbRelation.from_dict(duckdb.connect(), data)


def _duckdb_framework(data: dict[str, list[Any]]) -> Any:
    import duckdb

    framework = DuckDBFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    framework.set_framework_connection_object(duckdb.connect())
    return framework.transform(data, ["t"])


def _iceberg_framework(data: dict[str, list[Any]]) -> Any:
    framework = IcebergFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    return framework.transform(data, ["t"])


def _pyarrow_framework(data: dict[str, list[Any]]) -> Any:
    framework = PyArrowTable(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    return framework.transform(data, ["t"])


def _append_columns(data: dict[str, list[Any]]) -> Any:
    return arrow_append_columns(pa.table({"id": pa.array([1, 2], type=pa.int64())}), data)


def _sqlite_from_dict(data: dict[str, list[Any]]) -> Any:
    return SqliteRelation.from_dict(sqlite3.connect(":memory:"), data)


def _duckdb_append_columns(data: dict[str, list[Any]]) -> Any:
    import duckdb

    base = DuckdbRelation.from_dict(duckdb.connect(), {"id": [1, 2]})
    return base.append_columns(data)


_ENTRY_POINTS: list[Callable[[dict[str, list[Any]]], Any]] = [
    _python_dict,
    _duckdb_from_dict,
    _duckdb_framework,
    _iceberg_framework,
    _pyarrow_framework,
    _append_columns,
    _sqlite_from_dict,
    _duckdb_append_columns,
]


@ENV_VALUES
@pytest.mark.usefixtures("ignore_timezone_env")
class TestEntryPointsKeepNonUtcInstants:
    @pytest.mark.parametrize("build", _ENTRY_POINTS, ids=lambda f: f.__name__.lstrip("_"))
    def test_dict_build_keeps_instants_and_zone(self, build: Callable[[dict[str, list[Any]]], Any]) -> None:
        table = _arrow_of(build({"t": [NOON_NY, LATER_NY]}))
        field_type = table.schema.field("t").type
        assert pa.types.is_timestamp(field_type)
        assert field_type.tz == "America/New_York"
        assert _utc(table.column("t").to_pylist()) == _utc([NOON_NY, LATER_NY])

    def test_value_set_keeps_instants_and_zone(self) -> None:
        column = pa.array([NOON_NY.astimezone(UTC)], type=pa.timestamp("us", tz="America/New_York"))
        result = value_set(column, [NOON_NY, LATER_NY])
        assert result.type == pa.timestamp("us", tz="America/New_York")
        assert _utc(result.to_pylist()) == _utc([NOON_NY, LATER_NY])
