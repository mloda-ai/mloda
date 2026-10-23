"""DuckDB connection format group: opt-in by connection_handle, lazy on DuckDB, Arrow elsewhere, never closes your connection."""

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy, SourceMatch
from mloda.core.abstract_plugins.components.input_data.match_cache import run_cached
from mloda.core.abstract_plugins.components.match_rejection import INPUT_DATA_STAGE, record_match_rejection
from mloda.core.optional_dependency import require
from mloda.provider import FormatFeatureGroup, Implementation, PropertySpec
from mloda.user import DataAccessCollection, Options
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

HANDLE_KEY = "connection_handle"
_CATALOG_SQL = (
    "SELECT table_name, column_name FROM information_schema.columns "
    "WHERE table_schema = 'main' AND table_catalog = current_database() ORDER BY table_name, ordinal_position"
)
_FILE_SQL = "SELECT path, current_database() FROM duckdb_databases() WHERE database_name = current_database()"


@dataclass(frozen=True, eq=False)
class DuckdbTable:
    """The access of a DuckDB match: the user's connection (never shown) and one table or view."""

    connection: Any = field(repr=False)
    table: str

    def __deepcopy__(self, memo: dict[int, Any]) -> "DuckdbTable":
        return self  # a plan copy shares the live connection

    def __reduce__(self) -> Any:
        raise TypeError("A DuckdbFG match holds a live DuckDB connection and cannot be sent to another process.")


def _catalog(connection: Any) -> tuple[str, dict[str, list[str]], str]:
    """The database identity, its main-schema tables with their columns and its name, read once per run."""

    def read() -> tuple[str, dict[str, list[str]], str]:
        cursor = connection.cursor()
        try:
            found: dict[str, list[str]] = {}
            for table, column in cursor.execute(_CATALOG_SQL).fetchall():
                found.setdefault(table, []).append(column)
            path, database = cursor.execute(_FILE_SQL).fetchone()
        finally:
            cursor.close()
        return path or "memory", found, database

    return run_cached((DuckdbFG, "catalog", connection), read)


def _select(access: DuckdbTable, features: Any) -> str:
    names = ", ".join(quote_ident(name) for name in sorted(features.get_all_names()))
    database = quote_ident(_catalog(access.connection)[2])
    return f'select {names} from {database}."main".{quote_ident(access.table)}'  # nosec B608 - identifiers quoted


class DuckdbFG(FormatFeatureGroup):
    """Reads tables of a DuckDB connection in the DataAccessCollection, selected with ``connection_handle``."""

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (
        ClaimRoute("connection", NamePolicy.CHECKED, True, (HANDLE_KEY,)),
    )
    PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {
        HANDLE_KEY: PropertySpec("Name of the DuckDB connection in the DataAccessCollection to read.", default=None)
    }

    @classmethod
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        name = cls.get_class_name()
        if set(cls.pointer_keys()) & set(options.keys()):
            reason = f"{name} has no pointer; select a connection with {HANDLE_KEY}"
            record_match_rejection(name, reason, stage=INPUT_DATA_STAGE)
            return []
        handle = cls.read_option("connection_handle", options)
        connections = {} if data_access_collection is None else data_access_collection.connections
        if not isinstance(handle, str) or handle not in connections:
            reason = f"{HANDLE_KEY} {handle!r} is not a connection handle; known: {sorted(connections)}"
            record_match_rejection(name, reason, stage=INPUT_DATA_STAGE)
            return []
        connection = connections[handle]
        # only a DuckDB connection object can need the import
        duckdb = (
            require("duckdb", "reading DuckDB connections") if type(connection).__module__.endswith("duckdb") else None
        )
        if duckdb is None or not isinstance(connection, duckdb.DuckDBPyConnection):
            reason = f"{HANDLE_KEY} '{handle}' is not a duckdb connection"
            record_match_rejection(name, reason, stage=INPUT_DATA_STAGE)
            return []
        identity, tables, _ = _catalog(connection)
        return [SourceMatch(f"{identity}::{table}", DuckdbTable(connection, table)) for table in tables]

    @classmethod
    def columns(cls, match: SourceMatch) -> Collection[str] | None:
        access = cast(DuckdbTable, match.access)
        return _catalog(access.connection)[1].get(access.table)

    @classmethod
    def match_connection(cls, match: SourceMatch) -> Any | None:
        return cast(DuckdbTable, match.access).connection

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        access = cast(DuckdbTable, match.access)
        cursor = access.connection.cursor()
        try:
            return cursor.sql(_select(access, features)).to_arrow_table()
        finally:
            cursor.close()


class DuckdbFGDuckDB(Implementation):
    feature_group = DuckdbFG
    compute_framework = DuckDBFramework

    @classmethod
    def calculate(cls, group: Any, match: SourceMatch, features: Any, framework: Any) -> Any:
        access = cast(DuckdbTable, match.access)
        return DuckdbRelation(access.connection, access.connection.sql(_select(access, features)))
