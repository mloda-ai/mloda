"""FrameworkAdapter: one test adapter per compute framework, plus the Arrow table builder."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType

_NAMES = ["PythonDict", "PyArrow", "Pandas", "Polars", "PolarsLazy", "DuckDB", "Sqlite", "Iceberg"]
_TYPED = {
    "PythonDict": False,
    "PyArrow": True,
    "Pandas": True,
    "Polars": True,
    "PolarsLazy": True,
    "DuckDB": True,
    "Sqlite": True,
    "Iceberg": True,
}
_CONNECTION = ["DuckDB", "Sqlite", "Iceberg"]


def _make(name: str) -> Any:
    import mloda._testing.framework_adapter as module

    return getattr(module, f"{name}Adapter")()


def _data() -> TypedTestData:
    return TypedTestData(
        {
            "i": (DataType.INT64, [1, None, 3]),
            "d": (DataType.DOUBLE, [1.5, 2.5, None]),
            "s": (DataType.STRING, ["a", None, "c"]),
            "b": (DataType.BOOLEAN, [True, False, None]),
        }
    )


def _zero_rows() -> TypedTestData:
    return TypedTestData({"i": (DataType.INT64, []), "d": (DataType.DOUBLE, [])})


@pytest.fixture(params=_NAMES)
def name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def adapter(name: str) -> Any:
    instance = _make(name)
    if not instance.is_available():
        pytest.skip(f"{name} backend unavailable")
    return instance


def test_capabilities_typed_columns(name: str) -> None:
    assert _make(name).capabilities.typed_columns is _TYPED[name]


def test_round_trip(adapter: Any) -> None:
    with adapter:
        result = adapter.from_typed(_data())

        assert adapter.column_names(result) == ["i", "d", "s", "b"]
        assert adapter.row_count(result) == 3
        assert adapter.column_values(result, "i") == [1, None, 3]
        assert adapter.column_values(result, "d") == [1.5, 2.5, None]
        assert adapter.column_values(result, "s") == ["a", None, "c"]
        assert adapter.column_values(result, "b") == [True, False, None]


def test_rows_reads_the_result_as_a_list_of_dicts(adapter: Any) -> None:
    with adapter:
        result = adapter.from_typed(_data())

        assert adapter.rows(result) == [
            {"i": 1, "d": 1.5, "s": "a", "b": True},
            {"i": None, "d": 2.5, "s": None, "b": False},
            {"i": 3, "d": None, "s": "c", "b": None},
        ]


def test_rows_zero_row_result_is_empty_list(adapter: Any) -> None:
    with adapter:
        assert adapter.rows(adapter.from_typed(_zero_rows())) == []


def test_null_timestamp_reads_back_as_none(adapter: Any) -> None:
    values = [datetime(2024, 1, 1), None]
    with adapter:
        result = adapter.from_typed(TypedTestData({"t": (DataType.TIMESTAMP_MICROS, values)}))

        read = adapter.column_values(result, "t")

    assert read[0] == datetime(2024, 1, 1)
    assert read[1] is None


def test_list_cells_read_back_as_lists(adapter: Any) -> None:
    if type(adapter).__name__ == "SqliteAdapter":
        pytest.skip("SQLite cannot hold LIST")
    values = [[1, 2], None, [3]]
    with adapter:
        result = adapter.from_typed(TypedTestData({"l": (DataType.LIST, values)}))

        assert adapter.column_values(result, "l") == values


def test_pandas_string_column_keeps_string_dtype() -> None:
    adapter = _make("Pandas")
    if not adapter.is_available():
        pytest.skip("Pandas backend unavailable")

    assert adapter.lost_type(adapter.from_typed(_data()), "s") is False


@pytest.mark.parametrize("name", _CONNECTION)
def test_open_twice_raises(name: str) -> None:
    adapter = _make(name)
    if not adapter.is_available():
        pytest.skip(f"{name} backend unavailable")
    with adapter:
        with pytest.raises(RuntimeError):
            adapter.open()


def test_to_arrow_table_only_empty_lists_raises_naming_column() -> None:
    pytest.importorskip("pyarrow")
    from mloda._testing.framework_adapter import to_arrow_table

    with pytest.raises(ValueError, match="items"):
        to_arrow_table(TypedTestData({"items": (DataType.LIST, [[], []])}))


def test_zero_row_keeps_names(adapter: Any) -> None:
    with adapter:
        result = adapter.from_typed(_zero_rows())

        assert adapter.row_count(result) == 0
        assert adapter.column_names(result) == ["i", "d"]


def test_lost_type_false_for_numeric_columns(adapter: Any) -> None:
    with adapter:
        result = adapter.from_typed(_data())
        zero = adapter.from_typed(_zero_rows())

        assert adapter.lost_type(result, "i") is False
        assert adapter.lost_type(result, "d") is False
        assert adapter.lost_type(zero, "i") is False
        assert adapter.lost_type(zero, "d") is False


@pytest.mark.parametrize("name", ["PyArrow", "Pandas", "Polars", "PolarsLazy"])
def test_nullable_int_and_bool_keep_exact_type(adapter: Any) -> None:
    with adapter:
        result = adapter.from_typed(_data())

        assert adapter.has_type(result, "i", DataType.INT64) is True
        assert adapter.has_type(result, "b", DataType.BOOLEAN) is True
        assert adapter.lost_type(result, "i") is False
        assert adapter.lost_type(result, "b") is False


@pytest.mark.parametrize("name", ["PyArrow", "Pandas", "Polars", "PolarsLazy"])
def test_lost_type_true_for_null_typed_column(name: str) -> None:
    pa = pytest.importorskip("pyarrow")
    adapter = _make(name)
    if not adapter.is_available():
        pytest.skip(f"{name} backend unavailable")

    result = adapter.from_arrow(pa.table({"n": pa.array([None, None], type=pa.null())}))

    assert adapter.lost_type(result, "n") is True


_INT32_CHECKS: dict[str, Callable[[Any], bool]] = {
    "PyArrow": lambda t: str(t) == "int32",
    "Pandas": lambda t: str(t) == "Int32",
    "Polars": lambda t: str(t) == "Int32",
    "PolarsLazy": lambda t: str(t) == "Int32",
    "DuckDB": lambda t: str(t) == "int32",
    "Iceberg": lambda t: str(t) == "int32",
}


@pytest.mark.parametrize("name", list(_INT32_CHECKS))
def test_int32_keeps_native_width(name: str) -> None:
    adapter = _make(name)
    if not adapter.is_available():
        pytest.skip(f"{name} backend unavailable")
    with adapter:
        result = adapter.from_typed(TypedTestData({"c": (DataType.INT32, [1, 2])}))

        assert _INT32_CHECKS[name](adapter.column_type(result, "c"))


_EXACT = ["PyArrow", "Pandas", "Polars", "PolarsLazy"]
_NEIGHBORS = [
    (DataType.INT32, DataType.INT64, [1, 2]),
    (DataType.FLOAT, DataType.DOUBLE, [1.5, 2.5]),
    (DataType.TIMESTAMP_MILLIS, DataType.TIMESTAMP_MICROS, [datetime(2024, 1, 1), datetime(2024, 1, 2)]),
]
_UTC = datetime(2024, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize("name", _EXACT)
@pytest.mark.parametrize(("own", "neighbor", "values"), _NEIGHBORS)
def test_has_type_is_exact(adapter: Any, own: DataType, neighbor: DataType, values: list[Any]) -> None:
    with adapter:
        result = adapter.from_typed(TypedTestData({"c": (own, values)}))

        assert adapter.has_type(result, "c", own) is True
        assert adapter.has_type(result, "c", neighbor) is False


@pytest.mark.parametrize("name", ["PythonDict", "DuckDB", "Sqlite", "Iceberg"])
def test_has_type_is_none_without_exact_native_type(name: str, adapter: Any) -> None:
    with adapter:
        result = adapter.from_typed(TypedTestData({"c": (DataType.INT64, [1, 2])}))

        assert adapter.has_type(result, "c", DataType.INT64) is None


@pytest.mark.parametrize("name", _EXACT)
def test_has_type_timezone_aware_vs_naive(name: str, adapter: Any) -> None:
    with adapter:
        aware = adapter.from_typed(TypedTestData({"t": (DataType.TIMESTAMP_MICROS_TZ, [_UTC])}))
        naive = adapter.from_typed(TypedTestData({"t": (DataType.TIMESTAMP_MICROS, [datetime(2024, 1, 1)])}))

        assert adapter.has_type(aware, "t", DataType.TIMESTAMP_MICROS_TZ) is True
        assert adapter.has_type(naive, "t", DataType.TIMESTAMP_MICROS_TZ) is False


@pytest.mark.parametrize("name", ["PyArrow", "Polars", "PolarsLazy"])
def test_has_type_list(name: str, adapter: Any) -> None:
    with adapter:
        result = adapter.from_typed(TypedTestData({"l": (DataType.LIST, [[1, 2], [3]])}))

        assert adapter.has_type(result, "l", DataType.LIST) is True
        assert adapter.has_type(result, "l", DataType.INT64) is False


def test_pandas_object_date_has_no_exact_type() -> None:
    adapter = _make("Pandas")
    if not adapter.is_available():
        pytest.skip("Pandas backend unavailable")
    result = adapter.from_typed(TypedTestData({"d": (DataType.DATE, [datetime(2024, 1, 1).date()])}))

    assert adapter.has_type(result, "d", DataType.DATE) is None


def test_to_arrow_table_fixed_offset_timezone() -> None:
    pa = pytest.importorskip("pyarrow")
    from mloda._testing.framework_adapter import to_arrow_table

    tz = timezone(timedelta(hours=2))
    values = [datetime(2024, 1, 1, 12, 0, tzinfo=tz), None]
    table = to_arrow_table(TypedTestData({"t": (DataType.TIMESTAMP_MICROS_TZ, values)}))

    field_type = table.schema.field("t").type
    assert field_type.unit == "us"
    assert field_type.tz == "+02:00"
    assert pa.types.is_timestamp(field_type)


def test_to_arrow_table_zero_row_timezone_is_utc() -> None:
    pa = pytest.importorskip("pyarrow")
    from mloda._testing.framework_adapter import to_arrow_table

    table = to_arrow_table(TypedTestData({"t": (DataType.TIMESTAMP_MICROS_TZ, [])}))

    assert table.schema.field("t").type == pa.timestamp("us", tz="UTC")


def test_to_arrow_table_empty_list_raises() -> None:
    pytest.importorskip("pyarrow")
    from mloda._testing.framework_adapter import to_arrow_table

    with pytest.raises(ValueError, match="items"):
        to_arrow_table(TypedTestData({"items": (DataType.LIST, [])}))


def test_to_arrow_table_override_applies() -> None:
    pa = pytest.importorskip("pyarrow")
    from mloda._testing.framework_adapter import to_arrow_table

    table = to_arrow_table(TypedTestData({"c": (DataType.INT32, [1, 2])}), arrow_types={"c": pa.int16()})

    assert table.schema.field("c").type == pa.int16()


@pytest.mark.parametrize("name", _CONNECTION)
def test_from_typed_outside_session_raises(name: str) -> None:
    adapter = _make(name)
    if not adapter.is_available():
        pytest.skip(f"{name} backend unavailable")

    with pytest.raises(RuntimeError):
        adapter.from_typed(_data())


@pytest.mark.parametrize("name", _CONNECTION)
def test_connection_lifecycle(name: str) -> None:
    adapter = _make(name)
    if not adapter.is_available():
        pytest.skip(f"{name} backend unavailable")

    with adapter:
        assert adapter.connection is not None
    assert adapter.connection is None


def _iceberg() -> Any:
    adapter = _make("Iceberg")
    if not adapter.is_available():
        pytest.skip("Iceberg backend unavailable")
    return adapter


def test_iceberg_from_typed_returns_real_table_and_reads_plain_arrow() -> None:
    import pyarrow as pa

    adapter = _iceberg()
    from pyiceberg.table import Table

    with adapter:
        result = adapter.from_typed(_data())
        plain = pa.table({"x": [1, 2], "y": ["a", "b"]})

        assert isinstance(result, Table)
        assert adapter.namespace == "mloda_test"
        assert result.name()[:-1] == (adapter.namespace,)
        assert adapter.column_names(plain) == ["x", "y"]
        assert adapter.column_values(plain, "y") == ["a", "b"]


def test_iceberg_close_removes_warehouse_directory() -> None:
    """The warehouse temp dir exists while open and is removed and reset to None on close."""
    from pathlib import Path

    adapter = _iceberg()
    with adapter:
        warehouse = Path(adapter.warehouse)
        assert warehouse.is_dir()

    assert not warehouse.exists()
    assert adapter.warehouse is None


def test_iceberg_null_typed_column_raises_value_error() -> None:
    pa = pytest.importorskip("pyarrow")
    adapter = _iceberg()
    with adapter:
        with pytest.raises(ValueError):
            adapter.from_arrow(pa.table({"n": pa.array([None, None], type=pa.null())}))


def test_iceberg_fixed_offset_timestamp_raises_unsupported() -> None:
    pa = pytest.importorskip("pyarrow")
    adapter = _iceberg()
    from pyiceberg.io.pyarrow import UnsupportedPyArrowTypeException

    table = pa.table({"t": pa.array([], type=pa.timestamp("us", tz="+02:00"))})
    with adapter:
        with pytest.raises(UnsupportedPyArrowTypeException):
            adapter.from_arrow(table)


def _pandas() -> Any:
    pytest.importorskip("pandas")
    adapter = _make("Pandas")
    if not adapter.is_available():
        pytest.skip("Pandas backend unavailable")
    return adapter


def test_pandas_object_string_column_fits_by_value() -> None:
    import pandas as pd

    frame = pd.DataFrame({"s": pd.Series(["a", "b"], dtype=object)})

    assert _pandas().has_type(frame, "s", DataType.STRING) is None


@pytest.mark.parametrize(
    ("unit", "data_type", "expected"),
    [
        ("s", DataType.TIMESTAMP_MICROS, True),
        ("ms", DataType.TIMESTAMP_MICROS, False),
        ("ns", DataType.TIMESTAMP_MICROS, True),
        ("ns", DataType.TIMESTAMP_MILLIS, False),
        ("s", DataType.TIMESTAMP_MILLIS, False),
        ("ms", DataType.TIMESTAMP_MILLIS, True),
    ],
)
def test_pandas_timestamp_micros_accepts_s_us_ns(unit: str, data_type: DataType, expected: bool) -> None:
    adapter = _pandas()
    frame = adapter.from_typed(TypedTestData({"t": (DataType.TIMESTAMP_MICROS, [datetime(2024, 1, 1)])}))
    frame["t"] = frame["t"].astype(f"datetime64[{unit}]")

    assert adapter.has_type(frame, "t", data_type) is expected


def test_pandas_sparse_column_has_type_does_not_raise() -> None:
    import pandas as pd

    frame = pd.DataFrame({"c": pd.arrays.SparseArray([1, 0, 2], dtype="int64")})

    assert _pandas().has_type(frame, "c", DataType.INT64) in (True, False, None)


def _column_types(name: str, frame: Any) -> list[str]:
    if name == "Pandas":
        return [str(frame.dtypes.iloc[i]) for i in range(2)]
    if name == "DuckDB":
        return [str(t) for t in frame.types]
    return [str(t) for t in frame.schema.types]


_DUP_TYPES = {
    "PyArrow": {DataType.INT64: "int64", DataType.STRING: "string"},
    "Pandas": {DataType.INT64: "Int64", DataType.STRING: "string"},
    "DuckDB": {DataType.INT64: "BIGINT", DataType.STRING: "VARCHAR"},
    "Iceberg": {DataType.INT64: "int64", DataType.STRING: "string"},
}


@pytest.mark.parametrize("name", ["PyArrow", "Pandas", "DuckDB", "Iceberg"])
@pytest.mark.parametrize("pair", [(DataType.INT64, DataType.STRING), (DataType.STRING, DataType.INT64)])
def test_duplicate_column_frame_keeps_names_and_types(name: str, adapter: Any, pair: tuple[DataType, DataType]) -> None:
    with adapter:
        frame = adapter.duplicate_column_frame(*pair)

        assert adapter.column_names(frame) == ["a", "a"]
        assert _column_types(name, frame) == [_DUP_TYPES[name][pair[0]], _DUP_TYPES[name][pair[1]]]


def test_iceberg_duplicate_column_frame_is_plain_arrow() -> None:
    import pyarrow as pa

    adapter = _iceberg()
    with adapter:
        assert isinstance(adapter.duplicate_column_frame(DataType.INT64, DataType.STRING), pa.Table)


@pytest.mark.parametrize("name", ["PythonDict", "Polars", "PolarsLazy", "Sqlite"])
def test_duplicate_column_frame_cannot_hold(adapter: Any) -> None:
    with adapter:
        with pytest.raises(NotImplementedError, match="cannot hold"):
            adapter.duplicate_column_frame(DataType.INT64, DataType.STRING)


@pytest.mark.parametrize("name", ["PythonDict", "PyArrow", "Pandas", "Polars", "PolarsLazy", "Iceberg"])
def test_zero_column_frame_has_no_columns(adapter: Any) -> None:
    with adapter:
        assert adapter.column_names(adapter.zero_column_frame()) == []


@pytest.mark.parametrize("name", ["DuckDB", "Sqlite"])
def test_zero_column_frame_cannot_hold(adapter: Any) -> None:
    with adapter:
        with pytest.raises(NotImplementedError, match="cannot hold"):
            adapter.zero_column_frame()
