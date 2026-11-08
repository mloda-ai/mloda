"""One test adapter per compute framework, plus the Arrow table builder for typed test data."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, Self

from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.optional_dependency import require
from mloda.user.python_dict import row_count

if TYPE_CHECKING:
    import pyarrow as pa

    from mloda.core.abstract_plugins.compute_framework import ComputeFramework


@dataclass(frozen=True)
class AdapterCapabilities:
    """typed_columns: native type per column (False for PythonDict); sql_relation: serves the SQL relation suites."""

    typed_columns: bool = True
    sql_relation: bool = False


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def to_arrow_table(data: TypedTestData, arrow_types: Mapping[str, pa.DataType] | None = None) -> pa.Table:
    """Build an Arrow table from typed data; arrow_types overrides a column's physical type."""

    pa = require("pyarrow", "building Arrow test tables")
    overrides = arrow_types or {}
    columns = data.to_python_dict()
    arrays = []
    for name, data_type in data.types.items():
        values = columns[name]
        if name in overrides:
            arrays.append(pa.array(values, type=overrides[name]))
        elif data_type == DataType.TIMESTAMP_MICROS_TZ:
            if all(v is None for v in values):
                arrays.append(pa.array(values, type=pa.timestamp("us", tz="UTC")))
            else:
                arrays.append(pa.array(values))
        elif data_type == DataType.LIST:
            array = pa.array(values)
            if pa.types.is_null(array.type) or (
                pa.types.is_list(array.type) and pa.types.is_null(array.type.value_type)
            ):
                raise ValueError(f"column '{name}': a LIST column without element values needs an arrow_types override")
            arrays.append(array)
        else:
            arrays.append(pa.array(values, type=DataType.to_arrow_type(data_type)))
    return pa.table(arrays, names=list(data.types))


_SAMPLE_VALUES: dict[DataType, Any] = {
    DataType.INT64: 1,
    DataType.STRING: "x",
    DataType.DOUBLE: 1.5,
    DataType.BOOLEAN: True,
}


def _duplicate_column_table(first_type: DataType, second_type: DataType) -> pa.Table:
    """A one-row Arrow table with columns "a" (first_type) and "b" (second_type); only _SAMPLE_VALUES types."""
    pa = require("pyarrow", "building Arrow test tables")
    return pa.table(
        [
            pa.array([_SAMPLE_VALUES[first_type]], type=DataType.to_arrow_type(first_type)),
            pa.array([_SAMPLE_VALUES[second_type]], type=DataType.to_arrow_type(second_type)),
        ],
        names=["a", "b"],
    )


def _arrow_has_type(arrow_type: Any, data_type: DataType) -> bool:
    if data_type in (DataType.TIMESTAMP_MICROS_TZ, DataType.LIST):
        return bool(DataType.from_arrow_type_safe(arrow_type) == data_type)
    return bool(arrow_type == DataType.to_arrow_type(data_type))


_SYNC_ONLY_REASON = "the framework serves SYNC only, so its steps stay in the parent process"
_ICEBERG_NO_NATIVE_TYPE = (DataType.INT8, DataType.UINT8, DataType.UINT64, DataType.TIMESTAMP_MILLIS)


def _plain_iceberg_type(arrow_type: Any) -> Any:
    # Iceberg string/binary/list have no width; large variants are a zero-row scan artefact of stored tables.
    pa = require("pyarrow", "reading Iceberg test tables")

    if pa.types.is_large_string(arrow_type):
        return pa.string()
    if pa.types.is_large_binary(arrow_type):
        return pa.binary()
    if pa.types.is_large_list(arrow_type):
        field = arrow_type.value_field
        return pa.list_(field.with_type(_plain_iceberg_type(field.type)))
    return arrow_type


_PANDAS_NUMERIC = {
    DataType.INT8: ("i", 1),
    DataType.INT32: ("i", 4),
    DataType.INT64: ("i", 8),
    DataType.UINT8: ("u", 1),
    DataType.UINT64: ("u", 8),
    DataType.FLOAT: ("f", 4),
    DataType.DOUBLE: ("f", 8),
    DataType.BOOLEAN: ("b", 1),
}
_PANDAS_OBJECT_VALUE_FIT = (DataType.LIST, DataType.BINARY, DataType.DATE, DataType.DECIMAL, DataType.STRING)


def _pandas_has_type(dtype: Any, data_type: DataType) -> bool | None:
    pd = require("pandas", "reading pandas test frames")

    if isinstance(dtype, pd.ArrowDtype):
        return _arrow_has_type(dtype.pyarrow_dtype, data_type)
    if isinstance(dtype, pd.DatetimeTZDtype):
        return data_type == DataType.TIMESTAMP_MICROS_TZ
    if isinstance(dtype, pd.StringDtype):
        return data_type == DataType.STRING
    numpy_dtype = getattr(dtype, "numpy_dtype", dtype)
    if numpy_dtype.kind == "O":
        return None if data_type in _PANDAS_OBJECT_VALUE_FIT else False
    if numpy_dtype.kind == "M":
        unit = str(numpy_dtype).removeprefix("datetime64[").removesuffix("]")
        if data_type == DataType.TIMESTAMP_MICROS:
            return unit in ("s", "us", "ns")
        return unit == "ms" and data_type == DataType.TIMESTAMP_MILLIS
    return _PANDAS_NUMERIC.get(data_type) == (numpy_dtype.kind, getattr(numpy_dtype, "itemsize", None))


def _polars_has_type(dtype: Any, data_type: DataType) -> bool:
    pl = require("polars", "reading polars test frames")

    if data_type == DataType.TIMESTAMP_MICROS_TZ:
        return isinstance(dtype, pl.Datetime) and dtype.time_zone is not None
    if data_type == DataType.LIST:
        return isinstance(dtype, (pl.List, pl.Array))
    exact = {
        DataType.INT8: pl.Int8(),
        DataType.INT32: pl.Int32(),
        DataType.INT64: pl.Int64(),
        DataType.UINT8: pl.UInt8(),
        DataType.UINT64: pl.UInt64(),
        DataType.FLOAT: pl.Float32(),
        DataType.DOUBLE: pl.Float64(),
        DataType.BOOLEAN: pl.Boolean(),
        DataType.STRING: pl.String(),
        DataType.BINARY: pl.Binary(),
        DataType.DATE: pl.Date(),
        DataType.TIMESTAMP_MILLIS: pl.Datetime("ms"),
        DataType.TIMESTAMP_MICROS: pl.Datetime("us"),
        DataType.DECIMAL: pl.Decimal(38, 18),
    }
    return bool(dtype == exact[data_type])


class FrameworkAdapter(ABC):
    """Builds frames for one compute framework from typed data and reads them back."""

    capabilities: ClassVar[AdapterCapabilities]
    # Set when the framework serves no THREADING, so run-all suites run it in SYNC only; the value is the reason.
    sync_only_reason: ClassVar[str | None] = None

    @abstractmethod
    def compute_framework(self) -> type[ComputeFramework]: ...

    def is_available(self) -> bool:
        return bool(self.compute_framework().is_available())

    @abstractmethod
    def from_arrow(self, table: pa.Table) -> Any: ...

    def from_typed(self, data: TypedTestData, arrow_types: Mapping[str, pa.DataType] | None = None) -> Any:
        return self.from_arrow(to_arrow_table(data, arrow_types))

    @abstractmethod
    def column_names(self, result: Any) -> list[str]: ...

    @abstractmethod
    def row_count(self, result: Any) -> int:
        """The row count; also reads the frame run_all returns."""

    @abstractmethod
    def column_values(self, result: Any, column: str) -> list[Any]:
        """The column as Python values with nulls as None."""

    def _cannot_hold(self, shape: str) -> NotImplementedError:
        return NotImplementedError(f"{self.compute_framework().__name__} cannot hold {shape}")

    def duplicate_column_frame(self, first_type: DataType, second_type: DataType) -> Any:
        """Two columns both named "a", typed in order.

        A backend that cannot hold it raises NotImplementedError or returns its post-transform shape.
        """
        return self.from_arrow(_duplicate_column_table(first_type, second_type).rename_columns(["a", "a"]))

    def zero_column_frame(self) -> Any:
        """A frame with no columns."""
        pa = require("pyarrow", "building Arrow test tables")
        return self.from_arrow(pa.table({}))

    def rows(self, result: Any) -> list[dict[str, Any]]:
        """Row dicts, nulls as None, also for run_all frames; reads per column, so lazy adapters override."""
        names = self.column_names(result)
        columns = [self.column_values(result, name) for name in names]
        return [dict(zip(names, values)) for values in zip(*columns, strict=True)]

    @abstractmethod
    def column_type(self, result: Any, column: str) -> Any:
        """The framework's native type object for the column."""

    @abstractmethod
    def lost_type(self, result: Any, column: str) -> bool:
        """True when the column carries no usable type (null, object)."""

    def has_type(self, result: Any, column: str, data_type: DataType) -> bool | None:
        """Whether the column's native type is exactly data_type; None when the adapter has no exact type for it."""
        return None

    @property
    def connection(self) -> Any:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class PythonDictAdapter(FrameworkAdapter):
    capabilities = AdapterCapabilities(typed_columns=False)

    def compute_framework(self) -> type[ComputeFramework]:
        from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
            PythonDictFramework,
        )

        return PythonDictFramework

    def from_arrow(self, table: pa.Table) -> Any:
        return table.to_pydict()

    def duplicate_column_frame(self, first_type: DataType, second_type: DataType) -> Any:
        raise self._cannot_hold("a frame with duplicate column names")

    def from_typed(self, data: TypedTestData, arrow_types: Mapping[str, pa.DataType] | None = None) -> Any:
        """arrow_types does not apply to Python dicts."""
        return data.to_python_dict()

    def column_names(self, result: Any) -> list[str]:
        return list(result)

    def row_count(self, result: Any) -> int:
        return row_count(result)

    def column_values(self, result: Any, column: str) -> list[Any]:
        return list(result[column])

    def column_type(self, result: Any, column: str) -> Any:
        return None

    def lost_type(self, result: Any, column: str) -> bool:
        return False


class _ArrowReadAdapter(FrameworkAdapter):
    """Reads results through an Arrow table, so pyarrow is required."""

    def is_available(self) -> bool:
        return super().is_available() and _module_available("pyarrow")

    @abstractmethod
    def to_table(self, result: Any) -> pa.Table: ...

    def _table(self, result: Any) -> pa.Table:
        pa = require("pyarrow", "reading Arrow test tables")
        if isinstance(result, pa.Table):
            return result
        return self.to_table(result)

    def column_names(self, result: Any) -> list[str]:
        return list(self._table(result).column_names)

    def row_count(self, result: Any) -> int:
        return int(self._table(result).num_rows)

    def column_values(self, result: Any, column: str) -> list[Any]:
        return list(self._table(result).column(column).to_pylist())

    def rows(self, result: Any) -> list[dict[str, Any]]:
        return list(self._table(result).to_pylist())

    def column_type(self, result: Any, column: str) -> Any:
        return self._table(result).schema.field(column).type

    def lost_type(self, result: Any, column: str) -> bool:
        pa = require("pyarrow", "reading Arrow test tables")

        return bool(pa.types.is_null(self.column_type(result, column)))


class PyArrowAdapter(_ArrowReadAdapter):
    capabilities = AdapterCapabilities(typed_columns=True)

    def compute_framework(self) -> type[ComputeFramework]:
        from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

        return PyArrowTable

    def from_arrow(self, table: pa.Table) -> Any:
        return table

    def to_table(self, result: Any) -> pa.Table:
        return result

    def has_type(self, result: Any, column: str, data_type: DataType) -> bool | None:
        return _arrow_has_type(self.column_type(result, column), data_type)


class PandasAdapter(FrameworkAdapter):
    capabilities = AdapterCapabilities(typed_columns=True)

    def compute_framework(self) -> type[ComputeFramework]:
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        return PandasDataFrame

    def from_arrow(self, table: pa.Table) -> Any:
        pd = require("pandas", "building pandas test frames")
        pa = require("pyarrow", "building pandas test frames")

        mapping = {
            pa.string(): pd.StringDtype(),
            pa.bool_(): pd.BooleanDtype(),
            pa.int8(): pd.Int8Dtype(),
            pa.int16(): pd.Int16Dtype(),
            pa.int32(): pd.Int32Dtype(),
            pa.int64(): pd.Int64Dtype(),
            pa.uint8(): pd.UInt8Dtype(),
            pa.uint16(): pd.UInt16Dtype(),
            pa.uint32(): pd.UInt32Dtype(),
            pa.uint64(): pd.UInt64Dtype(),
        }
        return table.to_pandas(
            types_mapper=mapping.get,
            coerce_temporal_nanoseconds=False,
        )

    def duplicate_column_frame(self, first_type: DataType, second_type: DataType) -> Any:
        frame = self.from_arrow(_duplicate_column_table(first_type, second_type))
        frame.columns = ["a", "a"]
        return frame

    def column_names(self, result: Any) -> list[str]:
        return [str(c) for c in result.columns]

    def row_count(self, result: Any) -> int:
        return int(len(result))

    def column_values(self, result: Any, column: str) -> list[Any]:
        """NaT, NaN and NA read as None and array cells as lists. Not a NaN-versus-null oracle."""
        pd = require("pandas", "reading pandas test frames")

        def plain(v: Any) -> Any:
            if v is pd.NA or v is pd.NaT or (isinstance(v, float) and v != v):
                return None
            return v.tolist() if hasattr(v, "tolist") else v

        return [plain(v) for v in result[column].tolist()]

    def column_type(self, result: Any, column: str) -> Any:
        return result[column].dtype

    def lost_type(self, result: Any, column: str) -> bool:
        return str(self.column_type(result, column)) == "object"

    def has_type(self, result: Any, column: str, data_type: DataType) -> bool | None:
        return _pandas_has_type(self.column_type(result, column), data_type)


class PolarsAdapter(FrameworkAdapter):
    capabilities = AdapterCapabilities(typed_columns=True)

    def compute_framework(self) -> type[ComputeFramework]:
        from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

        return PolarsDataFrame

    def from_arrow(self, table: pa.Table) -> Any:
        pl = require("polars", "building polars test frames")
        require("pyarrow", "building polars test frames")

        return pl.from_arrow(table)

    def duplicate_column_frame(self, first_type: DataType, second_type: DataType) -> Any:
        raise self._cannot_hold("a frame with duplicate column names")

    def _eager(self, result: Any) -> Any:
        return result

    def column_names(self, result: Any) -> list[str]:
        return list(self._eager(result).columns)

    def row_count(self, result: Any) -> int:
        return int(self._eager(result).height)

    def column_values(self, result: Any, column: str) -> list[Any]:
        return list(self._eager(result)[column].to_list())

    def rows(self, result: Any) -> list[dict[str, Any]]:
        return list(self._eager(result).to_dicts())

    def column_type(self, result: Any, column: str) -> Any:
        return self._eager(result).schema[column]

    def lost_type(self, result: Any, column: str) -> bool:
        pl = require("polars", "reading polars test frames")

        return bool(self.column_type(result, column) in (pl.Null, pl.Object))

    def has_type(self, result: Any, column: str, data_type: DataType) -> bool | None:
        return _polars_has_type(self.column_type(result, column), data_type)


class PolarsLazyAdapter(PolarsAdapter):
    def compute_framework(self) -> type[ComputeFramework]:
        from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

        return PolarsLazyDataFrame

    def from_arrow(self, table: pa.Table) -> Any:
        return super().from_arrow(table).lazy()

    def _eager(self, result: Any) -> Any:
        pl = require("polars", "reading polars test frames")
        return result.collect() if isinstance(result, pl.LazyFrame) else result

    def column_names(self, result: Any) -> list[str]:
        return list(result.collect_schema().names())

    def column_type(self, result: Any, column: str) -> Any:
        return result.collect_schema()[column]


class DuckDBAdapter(_ArrowReadAdapter):
    capabilities = AdapterCapabilities(typed_columns=True, sql_relation=True)
    sync_only_reason: ClassVar[str | None] = _SYNC_ONLY_REASON

    def __init__(self) -> None:
        self._connection: Any = None

    @property
    def connection(self) -> Any:
        return self._connection

    def compute_framework(self) -> type[ComputeFramework]:
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework

        return DuckDBFramework

    def open(self) -> None:
        if self._connection is not None:
            raise RuntimeError("DuckDBAdapter is already open")
        duckdb = require("duckdb", "opening a DuckDB test session")

        from mloda.user import ParallelizationMode
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework

        self._connection = duckdb.connect()
        # Building the framework object is what pins the session time zone to UTC.
        framework = DuckDBFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        framework.set_framework_connection_object(self._connection)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._connection = None

    def from_arrow(self, table: pa.Table) -> Any:
        if self._connection is None:
            raise RuntimeError("DuckDBAdapter needs an open session: use it as a context manager")
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        return DuckdbRelation.from_arrow(self._connection, table)

    def duplicate_column_frame(self, first_type: DataType, second_type: DataType) -> Any:
        return self.from_arrow(_duplicate_column_table(first_type, second_type)).project('"a", "b" AS "a"')

    def zero_column_frame(self) -> Any:
        raise self._cannot_hold("a frame with no columns")

    def to_table(self, result: Any) -> pa.Table:
        return result.to_arrow_table()

    def has_type(self, result: Any, column: str, data_type: DataType) -> bool | None:
        return _arrow_has_type(self.column_type(result, column), data_type)


class SqliteAdapter(_ArrowReadAdapter):
    # has_type stays None: readback types are replayed from the source Arrow schema or inferred from values.
    capabilities = AdapterCapabilities(typed_columns=True, sql_relation=True)
    sync_only_reason: ClassVar[str | None] = _SYNC_ONLY_REASON

    def __init__(self) -> None:
        self._connection: Any = None

    @property
    def connection(self) -> Any:
        return self._connection

    def compute_framework(self) -> type[ComputeFramework]:
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

        return SqliteFramework

    def open(self) -> None:
        import sqlite3

        if self._connection is not None:
            raise RuntimeError("SqliteAdapter is already open")

        from mloda.user import ParallelizationMode
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

        self._connection = sqlite3.connect(":memory:")
        # Building the framework object is what registers REGEXP and DQS handling.
        framework = SqliteFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        framework.set_framework_connection_object(self._connection)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._connection = None

    def from_arrow(self, table: pa.Table) -> Any:
        if self._connection is None:
            raise RuntimeError("SqliteAdapter needs an open session: use it as a context manager")
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

        return SqliteRelation.from_arrow(self._connection, table)

    def duplicate_column_frame(self, first_type: DataType, second_type: DataType) -> Any:
        raise self._cannot_hold("a frame with duplicate column names")

    def zero_column_frame(self) -> Any:
        raise self._cannot_hold("a frame with no columns")

    def to_table(self, result: Any) -> pa.Table:
        return result.to_arrow_table()


class IcebergAdapter(_ArrowReadAdapter):
    """Iceberg over a temp SQLite-backed catalog; Iceberg rejects null-typed columns and non-UTC time zones."""

    capabilities = AdapterCapabilities(typed_columns=True)
    namespace = "mloda_test"

    def __init__(self) -> None:
        self._catalog: Any = None
        self._tmp: Any = None
        self._warehouse: str | None = None

    @property
    def connection(self) -> Any:
        return self._catalog

    @property
    def warehouse(self) -> str | None:
        return self._warehouse

    def compute_framework(self) -> type[ComputeFramework]:
        from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import IcebergFramework

        return IcebergFramework

    def is_available(self) -> bool:
        return super().is_available() and _module_available("sqlalchemy")

    def open(self) -> None:
        if self._catalog is not None:
            raise RuntimeError("IcebergAdapter is already open")

        import tempfile

        SqlCatalog = require("pyiceberg.catalog.sql", "opening an Iceberg test catalog").SqlCatalog

        # The TemporaryDirectory finalizer removes the dir if catalog creation fails below.
        tmp = tempfile.TemporaryDirectory()
        catalog = SqlCatalog(
            "mloda_test",
            uri=f"sqlite:///{tmp.name}/catalog.db",
            warehouse=f"file://{tmp.name}",
        )
        catalog.create_namespace(self.namespace)
        self._tmp = tmp
        self._catalog = catalog
        self._warehouse = tmp.name

    def close(self) -> None:
        if self._catalog is not None:
            self._catalog.engine.dispose()
        if self._tmp is not None:
            self._tmp.cleanup()
        self._catalog = None
        self._tmp = None
        self._warehouse = None

    def from_arrow(self, table: pa.Table) -> Any:
        if self._catalog is None:
            raise RuntimeError("IcebergAdapter needs an open session: use it as a context manager")
        iceberg_table = self._catalog.create_table(f"{self.namespace}.t{uuid.uuid4().hex}", schema=table.schema)
        iceberg_table.append(table)
        return iceberg_table

    def duplicate_column_frame(self, first_type: DataType, second_type: DataType) -> Any:
        """Native Iceberg schemas cannot carry duplicate names, so this is the post-transform PyArrow shape."""
        return _duplicate_column_table(first_type, second_type).rename_columns(["a", "a"])

    def to_table(self, result: Any) -> pa.Table:
        pa = require("pyarrow", "reading Iceberg test tables")
        if isinstance(result, pa.Table):
            return result
        return result.scan().to_arrow()

    def column_type(self, result: Any, column: str) -> Any:
        raw = super().column_type(result, column)
        if isinstance(result, require("pyarrow", "reading Iceberg test tables").Table):
            return raw
        return _plain_iceberg_type(raw)

    def has_type(self, result: Any, column: str, data_type: DataType) -> bool | None:
        if data_type in _ICEBERG_NO_NATIVE_TYPE and not isinstance(
            result, require("pyarrow", "reading Iceberg test tables").Table
        ):
            return None
        return _arrow_has_type(self.column_type(result, column), data_type)


class SparkAdapter(_ArrowReadAdapter):
    """Borrows the caller's active SparkSession, never starts or stops one; Spark rejects unsigned integer types."""

    # has_type stays None: a Spark-type mapping is not implemented (no unsigned or millisecond types in Spark).
    capabilities = AdapterCapabilities(typed_columns=True)

    def __init__(self) -> None:
        self._session: Any = None

    @property
    def connection(self) -> Any:
        return self._session

    def compute_framework(self) -> type[ComputeFramework]:
        from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import SparkFramework

        return SparkFramework

    def open(self) -> None:
        if self._session is not None:
            raise RuntimeError("SparkAdapter is already open")
        session = require("pyspark.sql", "opening a Spark test session").SparkSession.getActiveSession()
        if session is None:
            raise RuntimeError("SparkAdapter needs an active SparkSession")
        self._session = session

    def close(self) -> None:
        self._session = None

    def from_arrow(self, table: pa.Table) -> Any:
        if self._session is None:
            raise RuntimeError("SparkAdapter needs an open session: use it as a context manager")
        from_arrow_schema = require("pyspark.sql.pandas.types", "building Spark test frames").from_arrow_schema

        # The NTZ schema keeps naive timestamps from being shifted by the session time zone.
        return self._session.createDataFrame(table, schema=from_arrow_schema(table.schema, prefer_timestamp_ntz=True))

    def to_table(self, result: Any) -> pa.Table:
        return result.toArrow()

    def column_names(self, result: Any) -> list[str]:
        return list(result.columns)

    def row_count(self, result: Any) -> int:
        return int(result.count())

    def column_type(self, result: Any, column: str) -> Any:
        return result.schema[column].dataType

    def lost_type(self, result: Any, column: str) -> bool:
        null_type = require("pyspark.sql.types", "reading Spark test frames").NullType
        return isinstance(self.column_type(result, column), null_type)
