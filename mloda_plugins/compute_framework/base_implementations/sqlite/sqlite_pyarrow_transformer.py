import sqlite3
from typing import Any

from mloda_plugins.compute_framework.base_implementations.sql.sql_base_pyarrow_transformer import (
    SqlBasePyArrowTransformer,
)
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation


class SqlitePyArrowTransformer(SqlBasePyArrowTransformer):
    @classmethod
    def framework(cls) -> Any:
        return SqliteRelation

    @classmethod
    def import_fw(cls) -> None:
        pass

    @classmethod
    def _convert_to_arrow(cls, data: Any) -> Any:
        return data.to_arrow_table()

    @classmethod
    def _convert_to_native(cls, data: Any, connection: Any) -> Any:
        return SqliteRelation.from_arrow(connection, data)

    @classmethod
    def _validate_connection(cls, connection: Any) -> None:
        if not isinstance(connection, sqlite3.Connection):
            raise ValueError(f"Expected a sqlite3.Connection object, got {type(connection)}")
