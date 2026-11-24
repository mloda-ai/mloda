"""Tests for the shared Arrow builder and the dict entry points that use it.

PYARROW_IGNORE_TIMEZONE=1 makes plain pyarrow read an aware datetime's wall time as UTC; the
builder must give the same instants and zone with the variable unset or set.
"""

import contextlib
import itertools
import time
import sqlite3
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.provider import FeatureSet
from mloda.user import DataType, Feature
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import IcebergFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_value_set import value_set
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable, arrow_append_columns
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_pyarrow_transformer import (
    PythonDictPyArrowTransformer,
)
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow import PyArrowAggregation
from mloda_plugins.feature_group.experimental.data_quality.missing_value.pyarrow import PyArrowMissingValueImputation
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pyarrow import PyArrowTimeWindow
from mloda_plugins.feature_group.input_data.db_sources.sqlite_fg import SqliteFG
from tests.helpers.time_zones import bad_zone_values
from tests.mixins.reader_feature_groups.format_file_writers import write_sqlite

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
    @pytest.mark.parametrize(
        "values, zone",
        [
            ([NOON_NY, None, LATER_NY], "America/New_York"),
            ((NOON_NY,), "America/New_York"),
            ([datetime(2023, 3, 11, 12, 0, tzinfo=KOLKATA)], "+05:30"),
        ],
        ids=["named_zone_list", "tuple", "fixed_offset"],
    )
    def test_array_keeps_instants_and_zone(self, values: Any, zone: str) -> None:
        arr = _builders().arrow_array(values)
        assert arr.type == pa.timestamp("us", tz=zone)
        assert _utc(arr.to_pylist()) == _utc(list(values))

    def test_explicit_type_wins(self) -> None:
        target = pa.timestamp("us", tz="Asia/Tokyo")
        arr = _builders().arrow_array([NOON_NY], type=target)
        assert arr.type == target
        assert _utc(arr.to_pylist()) == _utc([NOON_NY])

    def test_explicit_type_rejects_naive_aware_mix(self) -> None:
        target = pa.timestamp("us", tz="UTC")
        with pytest.raises(ValueError) as err:
            _builders().arrow_array([_NAIVE, _UTC_NOON], type=target, column="t")
        for text in ["t", "naive", "aware"]:
            assert text in str(err.value)

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
            [],
            [None, None],
        ],
        ids=["naive", "ints", "strings", "empty", "all_none"],
    )
    @ENV_VALUES
    @pytest.mark.usefixtures("ignore_timezone_env")
    def test_array_matches_pa_array(self, values: list[Any]) -> None:
        got = _builders().arrow_array(values)
        want = pa.array(values)
        assert got.type == want.type
        assert got.to_pylist() == want.to_pylist()


_NAIVE = datetime(2023, 3, 11, 12, 0)
_UTC_NOON = datetime(2023, 3, 11, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "values, expected",
    [
        pytest.param([None, NOON_NY, None], "America/New_York", id="ny"),
        pytest.param([datetime(2023, 3, 11, tzinfo=KOLKATA)], "+05:30", id="fixed_offset"),
        pytest.param([_UTC_NOON, None], "UTC", id="utc"),
        pytest.param(
            [
                datetime(2023, 1, 1, tzinfo=timezone(timedelta(hours=1), "CET")),
                datetime(2023, 1, 2, tzinfo=timezone(timedelta(hours=1))),
            ],
            "+01:00",
            id="named_and_unnamed_offset_are_one_zone",
        ),
        pytest.param(
            [
                datetime(2023, 1, 1, tzinfo=ZoneInfo("Europe/Berlin")),
                datetime(2023, 7, 1, tzinfo=ZoneInfo("Europe/Berlin")),
            ],
            "Europe/Berlin",
            id="winter_and_summer_one_zone",
        ),
        pytest.param([_NAIVE, _NAIVE], None, id="naive"),
        pytest.param([None, None], None, id="all_none"),
        pytest.param([], None, id="empty"),
        pytest.param([1, 2], None, id="ints"),
        pytest.param([_NAIVE, _UTC_NOON, "x"], None, id="datetimes_and_non_datetime"),
    ],
)
def test_aware_zone(values: list[Any], expected: str | None) -> None:
    assert _builders().aware_zone(values) == expected


@pytest.mark.parametrize(
    "values, zones",
    [
        pytest.param([NOON_NY, datetime(2023, 3, 11, tzinfo=KOLKATA)], ["America/New_York", "+05:30"], id="ny_kolkata"),
        pytest.param(
            [
                datetime(2023, 1, 1, tzinfo=timezone(timedelta(hours=1), "CET")),
                datetime(2023, 7, 1, tzinfo=timezone(timedelta(hours=2), "CEST")),
            ],
            ["+01:00", "+02:00"],
            id="cet_cest_fixed_offsets",
        ),
        pytest.param(
            [_UTC_NOON, datetime(2023, 3, 11, 12, 0, tzinfo=ZoneInfo("Etc/UTC"))],
            ["UTC", "Etc/UTC"],
            id="utc_and_etc_utc",
        ),
        pytest.param([_UTC_NOON, _NAIVE], ["naive", "aware"], id="aware_then_naive"),
        pytest.param([_NAIVE, None, _UTC_NOON], ["naive", "aware"], id="naive_then_aware"),
    ],
)
def test_aware_zone_rejects_mixed_zones(values: list[Any], zones: list[str]) -> None:
    with pytest.raises(ValueError) as named:
        _builders().aware_zone(values, "zoned_col")
    for text in ["zoned_col", *zones]:
        assert text in str(named.value)
    with pytest.raises(ValueError) as unnamed:
        _builders().aware_zone(values)
    assert "zoned_col" not in str(unnamed.value)
    assert "column" not in str(unnamed.value)
    for zone in zones:
        assert zone in str(unnamed.value)


@pytest.fixture
def new_york_local(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def _summer(tz: Any) -> list[Any]:
    return [datetime(2023, 7, 1, 12, 0, tzinfo=tz)]


@pytest.mark.parametrize("kind", ["keyless", "custom", "tzoffset_cet", "tzstr_est5edt", "tzlocal"])
def test_aware_zone_rejects_unnameable_zone(kind: str, request: pytest.FixtureRequest) -> None:
    from dateutil import tz

    if kind == "tzoffset_cet":
        values, texts = _summer(tz.tzoffset("CET", 3600)), ["CET"]
    elif kind == "tzstr_est5edt":
        values, texts = _summer(tz.tzstr("EST5EDT")), ["EST"]
    elif kind == "tzlocal":
        request.getfixturevalue("new_york_local")
        values, texts = _summer(tz.tzlocal()), []
    else:
        values, texts = bad_zone_values(kind)
    with pytest.raises(ValueError) as named:
        _builders().aware_zone(values, "zoned_col")
    for text in ["zoned_col", *texts]:
        assert text in str(named.value)


def _arrow_of(result: Any) -> pa.Table:
    return result if isinstance(result, pa.Table) else result.to_arrow_table()


def _python_dict(data: dict[str, list[Any]]) -> Any:
    return PythonDictPyArrowTransformer.transform_fw_to_other_fw(data)


def _duckdb_from_dict(data: dict[str, list[Any]]) -> Any:
    duckdb = pytest.importorskip("duckdb")

    return DuckdbRelation.from_dict(duckdb.connect(), data)


def _duckdb_framework(data: dict[str, list[Any]]) -> Any:
    duckdb = pytest.importorskip("duckdb")

    framework = DuckDBFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    framework.set_framework_connection_object(duckdb.connect())
    return framework.transform(data, ["t"])


def _iceberg_framework(data: dict[str, list[Any]]) -> Any:
    pytest.importorskip("pyiceberg")
    framework = IcebergFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    return framework.transform(data, ["t"])


def _pyarrow_framework(data: dict[str, list[Any]]) -> Any:
    framework = PyArrowTable(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    return framework.transform(data, ["t"])


def _polars_framework(data: dict[str, list[Any]]) -> Any:
    pytest.importorskip("polars")
    from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

    framework = PolarsDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    return framework.transform(data, list(data)).to_arrow()


def _polars_lazy_framework(data: dict[str, list[Any]]) -> Any:
    pytest.importorskip("polars")
    from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

    framework = PolarsLazyDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    return framework.transform(data, list(data)).collect().to_arrow()


def _append_columns(data: dict[str, list[Any]]) -> Any:
    return arrow_append_columns(pa.table({"id": pa.array([1, 2], type=pa.int64())}), data)


def _sqlite_from_dict(data: dict[str, list[Any]]) -> Any:
    return SqliteRelation.from_dict(sqlite3.connect(":memory:"), data)


def _duckdb_append_columns(data: dict[str, list[Any]]) -> Any:
    duckdb = pytest.importorskip("duckdb")

    base = DuckdbRelation.from_dict(duckdb.connect(), {"id": [1, 2]})
    return base.append_columns(data)


def _polars_append_columns(data: dict[str, list[Any]]) -> Any:
    pl = pytest.importorskip("polars")
    from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

    framework = PolarsDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    appended: Any = framework._append_columns(pl.DataFrame({"id": [1, 2]}), data)
    return appended.to_arrow()


def _polars_lazy_append_columns(data: dict[str, list[Any]]) -> Any:
    pl = pytest.importorskip("polars")
    from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

    framework = PolarsLazyDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    appended: Any = framework._append_columns(pl.LazyFrame({"id": [1, 2]}), data)
    return appended.collect().to_arrow()


_POLARS_ENTRY_POINTS: list[Callable[[dict[str, list[Any]]], Any]] = [
    _polars_framework,
    _polars_lazy_framework,
    _polars_append_columns,
    _polars_lazy_append_columns,
]

_ENTRY_POINTS: list[Callable[[dict[str, list[Any]]], Any]] = [
    _python_dict,
    _duckdb_from_dict,
    _duckdb_framework,
    _iceberg_framework,
    _pyarrow_framework,
    *_POLARS_ENTRY_POINTS,
    _append_columns,
    _sqlite_from_dict,
    _duckdb_append_columns,
]


@ENV_VALUES
@pytest.mark.usefixtures("ignore_timezone_env")
class TestEntryPointsKeepNonUtcInstants:
    @pytest.mark.parametrize("zone_source", ["zoneinfo", "dateutil"])
    @pytest.mark.parametrize("build", _ENTRY_POINTS, ids=lambda f: f.__name__.lstrip("_"))
    def test_dict_build_keeps_instants_and_zone(
        self, build: Callable[[dict[str, list[Any]]], Any], zone_source: str
    ) -> None:
        if zone_source == "dateutil":
            tz: Any = pytest.importorskip("dateutil.tz").gettz("America/New_York")
        else:
            tz = NY
        values = [NOON_NY.replace(tzinfo=tz), LATER_NY.replace(tzinfo=tz)]
        table = _arrow_of(build({"t": values}))
        field_type = table.schema.field("t").type
        assert pa.types.is_timestamp(field_type)
        assert field_type.tz == "America/New_York"
        assert _utc(table.column("t").to_pylist()) == _utc([NOON_NY, LATER_NY])

    @pytest.mark.parametrize("hours", [5, -3], ids=["plus_05", "minus_03"])
    @pytest.mark.parametrize("build", _ENTRY_POINTS, ids=lambda f: f.__name__.lstrip("_"))
    def test_dict_build_keeps_whole_hour_offset(self, build: Callable[[dict[str, list[Any]]], Any], hours: int) -> None:
        offset = timezone(timedelta(hours=hours))
        values = [NOON_NY.astimezone(offset), LATER_NY.astimezone(offset)]
        table = _arrow_of(build({"t": values}))
        assert pa.types.is_timestamp(table.schema.field("t").type)
        got = table.column("t").to_pylist()
        assert _utc(got) == _utc(values)
        assert [v.utcoffset() for v in got] == [v.utcoffset() for v in values]

    @pytest.mark.parametrize(
        "values",
        [[NOON_NY, LATER_NY], [NOON_NY.astimezone(KOLKATA), LATER_NY], [NOON_NY.astimezone(KOLKATA), None, LATER_NY]],
        ids=["same_zone", "kolkata_and_ny", "kolkata_none_ny"],
    )
    def test_value_set_keeps_instants_and_zone(self, values: list[Any]) -> None:
        column = pa.array([NOON_NY.astimezone(UTC)], type=pa.timestamp("us", tz="America/New_York"))
        result = value_set(column, values)
        assert result.type == pa.timestamp("us", tz="America/New_York")
        assert _utc(result.to_pylist()) == _utc(values)

    def test_value_set_on_dictionary_column_with_aware_values_keeps_instants_and_zone(self) -> None:
        values = [NOON_NY.astimezone(KOLKATA), LATER_NY.astimezone(KOLKATA)]
        column = pa.array(_utc([NOON_NY, LATER_NY]), type=pa.timestamp("us", tz="America/New_York")).dictionary_encode()
        result = value_set(column, values)
        assert result.type == pa.timestamp("us", tz="America/New_York")
        assert _utc(result.to_pylist()) == _utc(values)

    def test_value_set_naive_values_on_aware_column_stay_naive(self) -> None:
        column = pa.array([NOON_NY.astimezone(UTC)], type=pa.timestamp("us", tz="America/New_York"))
        assert value_set(column, [_NAIVE]).type == pa.timestamp("us")


class TestEntryPointsRejectUnnameableZones:
    @pytest.mark.parametrize("kind", ["mixed", "keyless", "custom", "aware_naive", "naive_aware"])
    @pytest.mark.parametrize("build", _ENTRY_POINTS, ids=lambda f: f.__name__.lstrip("_"))
    def test_dict_build_raises_naming_column_and_zone(
        self, build: Callable[[dict[str, list[Any]]], Any], kind: str
    ) -> None:
        values, texts = bad_zone_values(kind)
        with pytest.raises(ValueError) as error:
            build({"zoned": values})
        for text in ["zoned", *texts]:
            assert text in str(error.value)

    @pytest.mark.parametrize("build", _POLARS_ENTRY_POINTS, ids=lambda f: f.__name__.lstrip("_"))
    def test_polars_rejects_sub_hour_offset_naming_column_and_zone(
        self, build: Callable[[dict[str, list[Any]]], Any]
    ) -> None:
        values = [NOON_NY.astimezone(KOLKATA), LATER_NY.astimezone(KOLKATA)]
        with pytest.raises(ValueError) as error:
            build({"zoned": values})
        for text in ["zoned", "+05:30"]:
            assert text in str(error.value)


_NY_TYPE = pa.timestamp("us", tz="America/New_York")


def _ny_column(values: list[datetime | None]) -> pa.Array:
    return pa.array([None if v is None else v.astimezone(UTC) for v in values], type=_NY_TYPE)


def _time_window_max(tmp_path: Path) -> tuple[pa.Array, list[datetime | None], str]:
    values = [NOON_NY, None, LATER_NY]
    table = pa.table({"ref": _ny_column([NOON_NY, NOON_NY + timedelta(hours=1), LATER_NY]), "v": _ny_column(values)})
    span = TimeWindowFeatureGroup._window_span(1, "second")
    column = PyArrowTimeWindow._perform_window_operation(table, "max", span, ["v"], "ref")
    return column, values, "America/New_York"


def _missing_value_ffill(tmp_path: Path) -> tuple[pa.Array, list[datetime | None], str]:
    table = pa.table({"v": _ny_column([None, NOON_NY, None, LATER_NY])})
    column = PyArrowMissingValueImputation._perform_imputation(table, "ffill", ["v"])
    return column, [None, NOON_NY, NOON_NY, LATER_NY], "America/New_York"


def _missing_value_grouped_ffill(tmp_path: Path) -> tuple[pa.Array, list[datetime | None], str]:
    table = pa.table({"v": _ny_column([NOON_NY, None, LATER_NY, None]), "g": ["a", "a", "b", "b"]})
    column = PyArrowMissingValueImputation._perform_grouped_imputation(table, "ffill", "v", None, ["g"])
    return column, [NOON_NY, NOON_NY, LATER_NY, LATER_NY], "America/New_York"


def _sqlite_produce(tmp_path: Path, feature: Feature, zones: tuple[Any, ...] = (NY,)) -> pa.Array:
    values = [NOON_NY, None, LATER_NY]
    path = tmp_path / "zoned.db"
    write_sqlite(path, {"t": {"c": [None if v is None else v.isoformat() for v in values]}})
    feature_set = FeatureSet()
    feature_set.add(feature)
    with contextlib.closing(SqliteFG.connect({"sqlite": str(path)})) as connection:
        cycle = itertools.cycle(zones)

        def to_zone(_cursor: Any, row: tuple[Any, ...]) -> tuple[Any, ...]:
            return tuple(datetime.fromisoformat(v).astimezone(next(cycle)) if isinstance(v, str) else v for v in row)

        connection.row_factory = to_zone
        table = SqliteFG.produce_rows(connection, "t", feature_set)
    return table.column("c")


def _sqlite_inferred(tmp_path: Path) -> tuple[pa.Array, list[datetime | None], str | None]:
    return _sqlite_produce(tmp_path, Feature.not_typed("c")), [NOON_NY, None, LATER_NY], "America/New_York"


def _sqlite_declared_naive(tmp_path: Path) -> tuple[pa.Array, list[datetime | None], str | None]:
    column = _sqlite_produce(tmp_path, Feature("c", data_type=DataType.TIMESTAMP_MICROS))
    return (
        column,
        [None if v is None else v.astimezone(UTC).replace(tzinfo=None) for v in (NOON_NY, None, LATER_NY)],
        None,
    )


def _aggregation_python_value(tmp_path: Path) -> tuple[pa.Array, list[datetime | None], str]:
    table = PyArrowAggregation._add_result_to_data(pa.table({"n": [1, 2]}), "out", NOON_NY)
    return table.column("out"), [NOON_NY, NOON_NY], "America/New_York"


_FEATURE_GROUP_BUILDS: list[Callable[[Path], tuple[pa.Array, list[datetime | None], str | None]]] = [
    _time_window_max,
    _missing_value_ffill,
    _missing_value_grouped_ffill,
    _sqlite_inferred,
    _sqlite_declared_naive,
    _aggregation_python_value,
]


@ENV_VALUES
@pytest.mark.usefixtures("ignore_timezone_env")
class TestFeatureGroupBuildsKeepNonUtcInstants:
    @pytest.mark.parametrize("build", _FEATURE_GROUP_BUILDS, ids=lambda f: f.__name__.lstrip("_"))
    def test_build_keeps_instants_and_zone(
        self, build: Callable[[Path], tuple[pa.Array, list[datetime | None], str | None]], tmp_path: Path
    ) -> None:
        column, expected_values, expected_zone = build(tmp_path)
        assert pa.types.is_timestamp(column.type)
        assert column.type.tz == expected_zone
        got = column.to_pylist()
        assert got == expected_values if expected_zone is None else _utc(got) == _utc(expected_values)


def test_sqlite_produce_rows_rejects_mixed_zones(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as error:
        _sqlite_produce(tmp_path, Feature.not_typed("c"), zones=(NY, KOLKATA))
    for text in ["c", "America/New_York", "+05:30"]:
        assert text in str(error.value)
