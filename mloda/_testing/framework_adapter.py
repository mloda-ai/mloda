"""One test adapter per compute framework, plus the Arrow table builder for typed test data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar

from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.optional_dependency import require

if TYPE_CHECKING:
    import pyarrow as pa

    from mloda.core.abstract_plugins.compute_framework import ComputeFramework


@dataclass(frozen=True)
class AdapterCapabilities:
    """typed_columns: the frame carries a native type per column (False for PythonDict)."""

    typed_columns: bool


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
            if all(v is None for v in values):
                raise ValueError(f"column '{name}': a LIST column without values needs an arrow_types override")
            arrays.append(pa.array(values))
        else:
            arrays.append(pa.array(values, type=DataType.to_arrow_type(data_type)))
    return pa.table(arrays, names=list(data.types))


class FrameworkAdapter(ABC):
    """Builds frames for one compute framework from typed data and reads them back."""

    capabilities: ClassVar[AdapterCapabilities]

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
    def row_count(self, result: Any) -> int: ...

    @abstractmethod
    def column_values(self, result: Any, column: str) -> list[Any]:
        """The column as Python values with nulls as None."""

    @abstractmethod
    def column_type(self, result: Any, column: str) -> Any:
        """The framework's native type object for the column."""

    @abstractmethod
    def lost_type(self, result: Any, column: str) -> bool:
        """True when the column carries no usable type (null, object)."""

    @property
    def connection(self) -> Any:
        return None

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> FrameworkAdapter:
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

    def from_typed(self, data: TypedTestData, arrow_types: Mapping[str, pa.DataType] | None = None) -> Any:
        return data.to_python_dict()

    def column_names(self, result: Any) -> list[str]:
        return list(result)

    def row_count(self, result: Any) -> int:
        return len(next(iter(result.values()), []))

    def column_values(self, result: Any, column: str) -> list[Any]:
        return list(result[column])

    def column_type(self, result: Any, column: str) -> Any:
        return None

    def lost_type(self, result: Any, column: str) -> bool:
        return False


class _ArrowBuiltAdapter(FrameworkAdapter):
    """Frames are built from Arrow tables, so pyarrow is required."""

    def is_available(self) -> bool:
        return super().is_available() and _module_available("pyarrow")


class _ArrowReadAdapter(_ArrowBuiltAdapter):
    """Reads results through an Arrow table."""

    @abstractmethod
    def to_table(self, result: Any) -> pa.Table: ...

    def column_names(self, result: Any) -> list[str]:
        return list(self.to_table(result).column_names)

    def row_count(self, result: Any) -> int:
        return int(self.to_table(result).num_rows)

    def column_values(self, result: Any, column: str) -> list[Any]:
        return list(self.to_table(result).column(column).to_pylist())

    def column_type(self, result: Any, column: str) -> Any:
        return self.to_table(result).schema.field(column).type

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


class PandasAdapter(_ArrowBuiltAdapter):
    capabilities = AdapterCapabilities(typed_columns=True)

    def compute_framework(self) -> type[ComputeFramework]:
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

        return PandasDataFrame

    def from_arrow(self, table: pa.Table) -> Any:
        return table.to_pandas()

    def column_names(self, result: Any) -> list[str]:
        return [str(c) for c in result.columns]

    def row_count(self, result: Any) -> int:
        return int(len(result))

    def column_values(self, result: Any, column: str) -> list[Any]:
        """Scalar NaN and NA read as None; list cells are untouched. Not a NaN-versus-null oracle."""
        pd = require("pandas", "reading pandas test frames")

        return [None if v is pd.NA or (isinstance(v, float) and v != v) else v for v in result[column].tolist()]

    def column_type(self, result: Any, column: str) -> Any:
        return result[column].dtype

    def lost_type(self, result: Any, column: str) -> bool:
        return str(self.column_type(result, column)) == "object"


class PolarsAdapter(_ArrowBuiltAdapter):
    capabilities = AdapterCapabilities(typed_columns=True)

    def compute_framework(self) -> type[ComputeFramework]:
        from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame

        return PolarsDataFrame

    def from_arrow(self, table: pa.Table) -> Any:
        pl = require("polars", "reading polars test frames")

        return pl.from_arrow(table)

    def _eager(self, result: Any) -> Any:
        return result

    def column_names(self, result: Any) -> list[str]:
        return list(self._eager(result).columns)

    def row_count(self, result: Any) -> int:
        return int(self._eager(result).height)

    def column_values(self, result: Any, column: str) -> list[Any]:
        return list(self._eager(result)[column].to_list())

    def column_type(self, result: Any, column: str) -> Any:
        return self._eager(result).schema[column]

    def lost_type(self, result: Any, column: str) -> bool:
        pl = require("polars", "reading polars test frames")

        return bool(self.column_type(result, column) in (pl.Null, pl.Object))


class PolarsLazyAdapter(PolarsAdapter):
    def compute_framework(self) -> type[ComputeFramework]:
        from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

        return PolarsLazyDataFrame

    def from_arrow(self, table: pa.Table) -> Any:
        return super().from_arrow(table).lazy()

    def _eager(self, result: Any) -> Any:
        return result.collect()


class DuckDBAdapter(_ArrowReadAdapter):
    capabilities = AdapterCapabilities(typed_columns=True)

    def __init__(self) -> None:
        self._connection: Any = None

    @property
    def connection(self) -> Any:
        return self._connection

    def compute_framework(self) -> type[ComputeFramework]:
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework

        return DuckDBFramework

    def open(self) -> None:
        duckdb = require("duckdb", "opening a DuckDB test session")

        from mloda.user import ParallelizationMode
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework

        self._connection = duckdb.connect()
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

    def to_table(self, result: Any) -> pa.Table:
        return result.to_arrow_table()


class SqliteAdapter(_ArrowReadAdapter):
    capabilities = AdapterCapabilities(typed_columns=True)

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

        from mloda.user import ParallelizationMode
        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework

        self._connection = sqlite3.connect(":memory:")
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

    def to_table(self, result: Any) -> pa.Table:
        return result.to_arrow_table()
