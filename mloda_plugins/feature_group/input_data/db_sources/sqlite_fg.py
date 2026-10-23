"""SQLite database source group: read-only connections, quoted queries, a pyarrow neutral and a stdlib dict implementation."""

import math
import os
import sqlite3
from collections.abc import Collection, Hashable, Mapping
from pathlib import Path
from typing import Any, ClassVar, cast

from mloda.core.abstract_plugins.components.mask.null_or_nan import split_null_or_nan
from mloda.core.filter.single_filter import SingleFilter
from mloda.core.optional_dependency import require
from mloda.provider import ComputeFramework, DBTable, FeatureSet, Implementation, ReadDBFG, SourceMatch
from mloda.user import DataType
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_affinity import sqlite_affinity_class

_PUSHABLE_AFFINITIES = ("INTEGER", "REAL", "TEXT")
_CLASS_ARROW = {"integer": "int64", "real": "float64", "text": "string", "blob": "binary", "null": "null"}
_CLASS_AFFINITY = {"integer": "INTEGER", "real": "REAL", "text": "TEXT"}
_AFFINITY_DATA_TYPE = {"INTEGER": DataType.INT64, "REAL": DataType.DOUBLE, "TEXT": DataType.STRING}
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


def _filter_affinity(
    name: str, data_types: dict[str, Any], classes: dict[str, set[str]], affinities: dict[str, str]
) -> str | None:
    """The comparison rules a filter column may use: its one scanned class if untyped, else a matching declared one."""
    if name not in data_types:
        return None
    if not data_types[name]:
        found = classes.get(name, set())
        return _CLASS_AFFINITY.get(next(iter(found))) if len(found) == 1 else None
    affinity = affinities.get(name)
    return affinity if affinity in _AFFINITY_DATA_TYPE and data_types[name] == _AFFINITY_DATA_TYPE[affinity] else None


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
        names, rows, _ = cls._fetch(connection, table, features)
        return names, rows

    @classmethod
    def _fetch(cls, connection: Any, table: str, features: Any) -> tuple[list[str], list[Any], dict[str, str]]:
        """Column names, rows and, for a pushed filter, the full-table arrow type of each untyped column."""
        where, params, arrow_types = cls._pushdown(connection, table, features)
        names = ", ".join(quote_ident(name) for name in sorted(features.get_all_names()))
        query = f"select {names} from {quote_ident(table)}{where};"  # nosec B608 - identifiers quoted, values bound
        cursor = connection.execute(query, params)
        return [description[0] for description in cursor.description], cursor.fetchall(), arrow_types

    @classmethod
    def _pushdown(cls, connection: Any, table: str, features: Any) -> tuple[str, list[Any], dict[str, str]]:
        """The WHERE text, its parameters and the untyped columns' arrow types; empty unless a filter is pushed."""
        filters = cls.rechecked_filters(features)
        if not filters:
            return "", [], {}
        data_types = {f.name: f.data_type for f in features.features}
        filter_names = {f.name for f in filters}
        typed_filters = [name for name in filter_names if data_types.get(name)]
        affinities: dict[str, str] = {}
        if typed_filters:
            rows = connection.execute(f"PRAGMA table_info({quote_ident(table)});").fetchall()
            affinities = {row[1]: sqlite_affinity_class(str(row[2])) for row in rows}
        untyped = sorted(name for name, data_type in data_types.items() if not data_type)
        classes: dict[str, set[str]] = {}
        if untyped:
            scan = ", ".join(f"group_concat(distinct typeof({quote_ident(name)}))" for name in untyped)
            row = connection.execute(f"select {scan} from {quote_ident(table)};").fetchone()  # nosec B608
            classes = {name: set((value or "").split(",")) - {"null", ""} for name, value in zip(untyped, row)}
            if any(len(found) > 1 for found in classes.values()):
                return "", [], {}
        conditions: list[str] = []
        params: list[Any] = []
        for single_filter in filters:
            affinity = _filter_affinity(single_filter.name, data_types, classes, affinities)
            condition = _condition(single_filter, affinity)
            if condition is None or len(params) + len(condition[1]) > _MAX_PARAMS:
                continue
            conditions.append(condition[0])
            params.extend(condition[1])
        if not conditions:
            return "", [], {}
        arrow_types = {name: _CLASS_ARROW[next(iter(found), "null")] for name, found in classes.items()}
        return " where " + " and ".join(conditions), params, arrow_types

    @classmethod
    def produce_rows(cls, connection: Any, table: str, features: Any) -> Any:
        pa = require("pyarrow", "reading SQLite into a pyarrow table")
        column_names, rows, arrow_types = cls._fetch(connection, table, features)
        data_types = {feature.name: feature.data_type for feature in features.features}
        fields = []
        for index, name in enumerate(column_names):
            data_type = data_types.get(name)
            if data_type:
                arrow_type = DataType.to_arrow_type(data_type)
            elif name in arrow_types:
                arrow_type = getattr(pa, arrow_types[name])()
            else:
                arrow_type = pa.array([row[index] for row in rows]).type if rows else pa.null()
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


class SqlitePythonDict(Implementation[SqliteFG]):
    feature_group = SqliteFG
    compute_framework = PythonDictFramework

    @classmethod
    def calculate(
        cls, group: type[SqliteFG], data: SourceMatch, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return _load_dict(group, data, features)
