"""SQLite database format group: read-only connections, quoted queries, a pyarrow neutral and a stdlib dict implementation."""

import math
import os
import sqlite3
from collections.abc import Collection, Hashable, Mapping
from pathlib import Path
from typing import Any, ClassVar, cast

from mloda.core.abstract_plugins.components.mask.null_or_nan import split_null_or_nan
from mloda.core.filter.single_filter import SingleFilter
from mloda.core.optional_dependency import require
from mloda.provider import DBTable, Implementation, ReadDBFG, SourceMatch
from mloda.user import DataType
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_affinity import sqlite_affinity_class

_PUSHABLE_AFFINITIES = ("INTEGER", "REAL", "TEXT")
_AFFINITY_ARROW = {"INTEGER": "int64", "REAL": "float64", "TEXT": "string"}
_MAX_PARAMS = 900  # stays under SQLite's historical 999 bound-parameter limit
_EXACT_INT = 2**53  # beyond this a float64 comparison could differ from SQLite's exact integer comparison


def _numeric_bound_ok(value: Any, affinity: str) -> bool:
    if type(value) is int:
        return abs(value) <= _EXACT_INT
    return affinity == "REAL" and type(value) is float and not math.isnan(value)


def _condition(single_filter: SingleFilter, affinity: str | None) -> tuple[str, list[Any]] | None:
    """The WHERE condition and its parameters for a filter shape that matches the framework exactly, else None."""
    if affinity not in _PUSHABLE_AFFINITIES:
        return None
    column = quote_ident(single_filter.name)
    parameter = single_filter.parameter
    kind = single_filter.filter_type
    text = affinity == "TEXT"
    if text:
        column += " COLLATE BINARY"  # a NOCASE column would match more rows than the framework's exact comparison
    if kind == "categorical_inclusion":
        present, has_null = split_null_or_nan(parameter.values or [])
        if has_null or not present or not all(_bound_ok(v, affinity) for v in present):
            return None
        return f"{column} in ({', '.join('?' * len(present))})", present
    if kind == "equal":
        bounds = [parameter.value]
        return (f"{column} = ?", bounds) if _bound_ok(bounds[0], affinity) else None
    if text:
        return None
    if kind == "min" and parameter.value is not None:
        return (f"{column} >= ?", [parameter.value]) if _bound_ok(parameter.value, affinity) else None
    if kind == "max":
        bound = parameter.max_value if parameter.max_value is not None else parameter.value
        operator = "<" if parameter.max_value is not None and parameter.max_exclusive else "<="
        return (f"{column} {operator} ?", [bound]) if _bound_ok(bound, affinity) else None
    if kind == "range" and parameter.min_value is not None and parameter.max_value is not None:
        bounds = [parameter.min_value, parameter.max_value]
        if all(_bound_ok(b, affinity) for b in bounds):
            return f"{column} >= ? and {column} {'<' if parameter.max_exclusive else '<='} ?", bounds
    return None


def _bound_ok(value: Any, affinity: str) -> bool:
    return isinstance(value, str) if affinity == "TEXT" else _numeric_bound_ok(value, affinity)


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
    def login_key(cls, credentials: Any) -> Hashable:
        return ()

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        uri = Path(credentials[cls.CREDENTIAL_KEY]).absolute().as_uri() + "?mode=ro"
        return sqlite3.connect(uri, uri=True)

    @classmethod
    def _raw_reason(cls, match: SourceMatch) -> str | None:
        if isinstance(match.access, DBTable) and not os.path.exists(match.access.credentials[cls.CREDENTIAL_KEY]):
            return "it does not exist"
        return super()._raw_reason(match)

    @classmethod
    def list_tables(cls, connection: Any) -> Collection[str]:
        return [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table';")]

    @classmethod
    def table_columns(cls, connection: Any, table: str) -> Collection[str]:
        return [row[1] for row in connection.execute(f"PRAGMA table_info({quote_ident(table)});")]

    @classmethod
    def select_rows(cls, connection: Any, table: str, features: Any) -> tuple[list[str], list[Any]]:
        names = ", ".join(quote_ident(name) for name in sorted(features.get_all_names()))
        where, params, _ = cls._pushdown(connection, table, features)
        query = f"select {names} from {quote_ident(table)}{where};"  # nosec B608 - identifiers quoted, values bound
        cursor = connection.execute(query, params)
        return [description[0] for description in cursor.description], cursor.fetchall()

    @classmethod
    def _pushdown(cls, connection: Any, table: str, features: Any) -> tuple[str, list[Any], dict[str, str]]:
        """The WHERE text, its parameters and the column affinities; empty unless a filter is pushed."""
        filters = cls.rechecked_filters(features)
        if not filters:
            return "", [], {}
        rows = connection.execute(f"PRAGMA table_info({quote_ident(table)});").fetchall()
        affinities = {row[1]: sqlite_affinity_class(str(row[2])) for row in rows}
        untyped = [f.name for f in features.features if not f.data_type]
        if any(affinities.get(name) not in _PUSHABLE_AFFINITIES for name in untyped):
            return "", [], {}
        conditions: list[str] = []
        params: list[Any] = []
        for single_filter in filters:
            condition = _condition(single_filter, affinities.get(single_filter.name))
            if condition is None or len(params) + len(condition[1]) > _MAX_PARAMS:
                continue
            conditions.append(condition[0])
            params.extend(condition[1])
        if not conditions:
            return "", [], {}
        return " where " + " and ".join(conditions), params, affinities

    @classmethod
    def produce_rows(cls, connection: Any, table: str, features: Any) -> Any:
        pa = require("pyarrow", "reading SQLite into a pyarrow table")
        column_names, rows = cls.select_rows(connection, table, features)
        affinities = cls._pushdown(connection, table, features)[2]
        data_types = {feature.name: feature.data_type for feature in features.features}
        fields = []
        for index, name in enumerate(column_names):
            data_type = data_types.get(name)
            if data_type:
                arrow_type = DataType.to_arrow_type(data_type)
            else:
                arrow_type = pa.array([row[index] for row in rows]).type if rows else pa.null()
                if pa.types.is_null(arrow_type) and affinities.get(name) in _AFFINITY_ARROW:
                    arrow_type = getattr(pa, _AFFINITY_ARROW[affinities[name]])()
            fields.append((name, arrow_type))
        data = [dict(zip(column_names, row)) for row in rows]
        return pa.Table.from_pylist(data, schema=pa.schema(fields))

    @classmethod
    def describe_columns(cls, match: SourceMatch) -> dict[str, DataType | None]:
        """Column name to DataType by declared-type affinity; NUMERIC or undeclared maps to None."""
        if not isinstance(match.access, DBTable):
            return super().describe_columns(match)
        access = match.access
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


def _load_dict(group: type[SqliteFG], match: SourceMatch, features: Any) -> dict[str, list[Any]]:
    access = cast(DBTable, match.access)
    connection = group.get_connection(access.credentials)
    try:
        column_names, rows = group.select_rows(connection, cast(str, access.table), features)
    finally:
        group.close_connection(connection)
    return {name: [row[index] for row in rows] for index, name in enumerate(column_names)}


class SqlitePythonDict(Implementation):
    feature_group = SqliteFG
    compute_framework = PythonDictFramework

    @classmethod
    def calculate(cls, group: Any, match: SourceMatch, features: Any, framework: Any) -> Any:
        return _load_dict(group, match, features)
