from typing import Any

from mloda_plugins.compute_framework.base_implementations.sql.sql_base_pyarrow_transformer import (
    SqlBasePyArrowTransformer,
)

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]


class DuckDBPyArrowTransformer(SqlBasePyArrowTransformer):
    @classmethod
    def framework(cls) -> Any:
        if duckdb is None:
            raise ImportError("DuckDB is not installed. Install it with: pip install duckdb")
        return duckdb.DuckDBPyRelation

    @classmethod
    def import_fw(cls) -> None:
        import duckdb

    @classmethod
    def _convert_to_arrow(cls, data: Any) -> Any:
        return data.to_arrow_table()

    @classmethod
    def _convert_to_native(cls, data: Any, connection: Any) -> Any:
        return connection.from_arrow(data)

    @classmethod
    def _validate_connection(cls, connection: Any) -> None:
        if duckdb is None:
            raise ImportError("DuckDB is not installed. To be able to use this framework, please install duckdb.")
        if not isinstance(connection, duckdb.DuckDBPyConnection):
            raise ValueError(f"Expected a DuckDB connection object, got {type(connection)}")
