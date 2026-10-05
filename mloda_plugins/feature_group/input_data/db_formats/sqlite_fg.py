"""SQLite database format group: read-only connections, quoted queries, a pyarrow neutral and a stdlib dict loader."""

import os
import sqlite3
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, ClassVar, cast

from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.abstract_plugins.components.input_data.read_db_fg import DBTable
from mloda.core.optional_dependency import require
from mloda.provider import ReadDBFG
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_affinity import sqlite_affinity_class


class SqliteFG(ReadDBFG):
    """Reads SQLite files from ``Credential(sqlite=<path>)``; ``table_name`` restricts the lookup to one table.

    The value is a plain file path, opened read-only; ``file:`` URIs and ``:memory:`` are not supported.
    """

    CREDENTIAL_KEY: ClassVar[str] = "sqlite"
    CATALOG_ERRORS: ClassVar[tuple[type[BaseException], ...]] = (*ReadDBFG.CATALOG_ERRORS, sqlite3.Error)

    @classmethod
    def is_valid_credentials(cls, credentials: Any) -> bool:
        return isinstance(credentials, Mapping) and isinstance(credentials.get(cls.CREDENTIAL_KEY), str)

    @classmethod
    def database_identity(cls, credentials: Any) -> str:
        return str(os.path.abspath(credentials[cls.CREDENTIAL_KEY]))

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        uri = Path(credentials[cls.CREDENTIAL_KEY]).absolute().as_uri() + "?mode=ro"
        return sqlite3.connect(uri, uri=True)

    @classmethod
    def list_tables(cls, connection: Any) -> Collection[str]:
        return [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table';")]

    @classmethod
    def table_columns(cls, connection: Any, table: str) -> Collection[str]:
        return [row[1] for row in connection.execute(f"PRAGMA table_info({quote_ident(table)});")]

    @classmethod
    def _select(cls, connection: Any, table: str, features: Any) -> tuple[list[str], list[Any]]:
        names = ", ".join(quote_ident(name) for name in sorted(features.get_all_names()))
        query = f"select {names} from {quote_ident(table)};"  # nosec B608 - every identifier is quoted
        cursor = connection.execute(query)
        return [description[0] for description in cursor.description], cursor.fetchall()

    @classmethod
    def produce_rows(cls, connection: Any, table: str, features: Any) -> Any:
        pa = require("pyarrow", "reading SQLite into a pyarrow table")
        column_names, rows = cls._select(connection, table, features)
        data_types = {feature.name: feature.data_type for feature in features.features}
        fields = []
        for index, name in enumerate(column_names):
            data_type = data_types.get(name)
            if data_type:
                arrow_type = DataType.to_arrow_type(data_type)
            else:
                arrow_type = DataType.infer_arrow_type(rows[0][index]) if rows else pa.null()
            fields.append((name, arrow_type))
        data = [dict(zip(column_names, row)) for row in rows]
        return pa.Table.from_pylist(data, schema=pa.schema(fields))

    @classmethod
    def describe_columns(cls, match: SourceMatch) -> dict[str, DataType | None]:
        """Column name to DataType by declared-type affinity; NUMERIC or undeclared maps to None."""
        access = cast(DBTable, match.access)
        connection = cls.get_connection(access.credentials)
        try:
            rows = list(connection.execute(f"PRAGMA table_info({quote_ident(str(access.table))});"))
        finally:
            cls.close_connection(connection)
        if not rows:
            raise ValueError(f"{cls.__name__}.describe_columns: no such table '{access.table}'.")
        return {row[1]: cls._affinity_to_datatype(str(row[2])) for row in rows}

    @staticmethod
    def _affinity_to_datatype(declared_type: str) -> DataType | None:
        label = sqlite_affinity_class(declared_type)
        if label == "INTEGER":
            return DataType.INT64
        if label == "TEXT":
            return DataType.STRING
        if label == "BLOB":
            return DataType.BINARY
        if label == "REAL":
            return DataType.DOUBLE
        return None


def _load_dict(match: SourceMatch, features: Any) -> dict[str, list[Any]]:
    access = cast(DBTable, match.access)
    connection = SqliteFG.get_connection(access.credentials)
    try:
        column_names, rows = SqliteFG._select(connection, cast(str, access.table), features)
    finally:
        SqliteFG.close_connection(connection)
    return {name: [row[index] for row in rows] for index, name in enumerate(column_names)}


SqliteFG.register_loader(PythonDictFramework, _load_dict)
