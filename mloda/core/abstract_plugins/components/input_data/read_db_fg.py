"""Abstract database format group: finds credentials, lists table catalogs per run, loads one table.

A subclass writes ``is_valid_credentials``, ``database_identity``, ``connect``, ``list_tables``, ``table_columns``
and ``produce_rows``.
"""

from abc import abstractmethod
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

from mloda.core.abstract_plugins.components.credential_scrub import scrub_credentials
from mloda.core.abstract_plugins.components.credential import Credential, RegisteredCredential
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy, SourceMatch
from mloda.core.abstract_plugins.components.input_data.format_feature_group import (
    HANDLE_OPTION,
    HANDLE_SPEC,
    FormatFeatureGroup,
)
from mloda.core.abstract_plugins.components.input_data.match_cache import run_cached
from mloda.core.abstract_plugins.components.match_rejection import INPUT_DATA_STAGE, record_match_rejection
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.utils import is_match_abort

_MAX_DESCRIBED = 5


@dataclass(frozen=True, eq=False)
class DBTable:
    """The access of a database match: the credentials (never shown) and the table, None when unlistable."""

    credentials: Any = field(repr=False)
    table: str | None


def _capped(names: list[str]) -> str:
    text = ", ".join(repr(name) for name in names[:_MAX_DESCRIBED])
    return text if len(names) <= _MAX_DESCRIBED else f"{text}, and {len(names) - _MAX_DESCRIBED} more"


class ReadDBFG(FormatFeatureGroup):
    """Abstract database format group: claims features found as columns of tables in matching credentials.

    The base serves table catalogs only. Third-party groups should reuse the contract mixin
    database_format_feature_group_test_mixin from tests/mixins/reader_feature_groups in mloda's repository.
    """

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("credentials", NamePolicy.CHECKED, True),)
    PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {HANDLE_OPTION: HANDLE_SPEC}
    TABLE_KEY: ClassVar[str] = "table_name"
    CATALOG_ERRORS: ClassVar[tuple[type[BaseException], ...]] = (OSError, ValueError, ImportError)
    CREDENTIAL_KEY: ClassVar[str] = "<credential key>"

    @classmethod
    @abstractmethod
    def is_valid_credentials(cls, credentials: Any) -> bool:
        """True when the credentials hold this database's own key; never raises."""

    @classmethod
    @abstractmethod
    def database_identity(cls, credentials: Any) -> str:
        """A credential-free name of the database, used as source prefix and cache key."""

    @classmethod
    @abstractmethod
    def connect(cls, credentials: Any) -> Any:
        """Open a connection."""

    @classmethod
    @abstractmethod
    def list_tables(cls, connection: Any) -> Collection[str]:
        """Names of the tables."""

    @classmethod
    @abstractmethod
    def table_columns(cls, connection: Any, table: str) -> Collection[str]:
        """Column names of one table."""

    @classmethod
    @abstractmethod
    def produce_rows(cls, connection: Any, table: str, features: Any) -> Any:
        """The requested columns of the table in a neutral form, fully materialized (the connection closes after)."""

    @classmethod
    def close_connection(cls, connection: Any) -> None:
        connection.close()

    @classmethod
    def wrap_feature_scoped_access(cls, value: Any) -> Any:
        """A redacting RegisteredCredential copy of a mapping or Credential, None for anything else."""
        if isinstance(value, RegisteredCredential):
            return value
        if isinstance(value, Mapping):
            return RegisteredCredential(dict(value))
        if isinstance(value, Credential):
            return RegisteredCredential(value.data)
        return None

    @classmethod
    def get_connection(cls, credentials: Any) -> Any:
        connection = cls.connect(credentials)
        if connection is None:
            raise ValueError(
                f"Connection to database failed for {cls.__name__}.\n"
                f"Credentials type: {type(credentials).__name__}.\n"
                "The connect() method returned None. Verify that:\n"
                "  - The database server is reachable.\n"
                "  - The credentials (host, port, user, password, database name) are correct.\n"
                "  - The required database driver is installed."
            )
        return connection

    @classmethod
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        name = cls.get_class_name()
        key = next((key for key in cls.pointer_keys() if key in options), None)
        if key is not None:
            value = options.get(key)
            wrapped = cls.wrap_feature_scoped_access(value)
            if wrapped is None:
                reason = f"the pointer value must be a credential mapping, got {type(value).__name__}"
                record_match_rejection(name, reason, stage=INPUT_DATA_STAGE)
                return []
            if not cls.is_valid_credentials(wrapped):
                record_match_rejection(
                    name, "the pointed credentials are not valid for this group", stage=INPUT_DATA_STAGE
                )
                return []
            candidates = [wrapped]
        elif data_access_collection is None:
            return []
        else:
            candidates = cls._handle_credentials(options, data_access_collection)
        sources: dict[str, SourceMatch] = {}
        for credentials in candidates:
            if cls.is_valid_credentials(credentials):
                for match in cls._tables(cls.database_identity(credentials), credentials):
                    sources.setdefault(match.source, match)
        return list(sources.values())

    @classmethod
    def _handle_credentials(cls, options: Options, dac: DataAccessCollection) -> list[Any]:
        handle = options.get(HANDLE_OPTION)
        if not isinstance(handle, str):
            return list(dac.credentials.values())
        if handle in dac.credentials:
            return [dac.credentials[handle]]
        if handle not in dac.handles():
            record_match_rejection(
                cls.get_class_name(),
                f"data_access_handle '{handle}' is unknown; credential handles: {sorted(dac.credentials)}",
                stage=INPUT_DATA_STAGE,
            )
        return []

    @classmethod
    def _tables(cls, identity: str, credentials: Any) -> list[SourceMatch]:
        catalog, _ = cls._catalog(credentials)
        if catalog is None:
            return [SourceMatch(identity, DBTable(credentials, None))]
        preset = credentials.get(cls.TABLE_KEY) if isinstance(credentials, Mapping) else None
        tables = [preset] if isinstance(preset, str) and preset else list(catalog)
        return [SourceMatch(f"{identity}::{table}", DBTable(credentials, table)) for table in tables]

    @classmethod
    def _catalog(cls, credentials: Any) -> tuple[dict[str, tuple[str, ...]] | None, str | None]:
        identity = cls.database_identity(credentials)
        return run_cached((cls, "catalog", identity), lambda: cls._read_catalog(credentials))

    @classmethod
    def _read_catalog(cls, credentials: Any) -> tuple[dict[str, tuple[str, ...]] | None, str | None]:
        try:
            connection = cls.get_connection(credentials)
            try:
                catalog = {table: tuple(cls.table_columns(connection, table)) for table in cls.list_tables(connection)}
            finally:
                cls.close_connection(connection)
        except cls.CATALOG_ERRORS as exc:
            if is_match_abort(exc):
                raise
            text = str(exc)
            if isinstance(credentials, Mapping):
                for secret in sorted(
                    (v for v in credentials.values() if isinstance(v, str) and v), key=len, reverse=True
                ):
                    text = text.replace(secret, "***")
            return None, f"could not read its tables: {scrub_credentials(text)}"
        return catalog, None

    @classmethod
    def columns(cls, match: SourceMatch) -> Collection[str] | None:
        access = cast(DBTable, match.access)
        if access.table is None:
            return None
        catalog, _ = cls._catalog(access.credentials)
        return None if catalog is None else catalog.get(access.table)

    @classmethod
    def unknown_columns_reason(cls, match: SourceMatch) -> str | None:
        access = cast(DBTable, match.access)
        catalog, reason = cls._catalog(access.credentials)
        if catalog is not None and access.table is not None and access.table not in catalog:
            return f"table '{access.table}' does not exist"
        return reason

    @classmethod
    def ambiguity_fix(
        cls, feature_name: str, matches: list[SourceMatch], data_access_collection: DataAccessCollection | None
    ) -> str:
        name = cls.get_class_name()
        identities = {cls.database_identity(cast(DBTable, match.access).credentials) for match in matches}
        if len(identities) > 1:
            handles = sorted(
                handle
                for handle, credentials in (
                    data_access_collection.credentials.items() if data_access_collection else ()
                )
                if cls.is_valid_credentials(credentials) and cls.database_identity(credentials) in identities
            )
            hint = f" (credential handles: {_capped(handles)})" if handles else ""
            return f"select one database with data_access_handle{hint}, or point {name} at one credential."
        table = cast(DBTable, matches[0].access).table
        return (
            f"point {name} at one table with options={{{name!r}: "
            f"Credential({cls.CREDENTIAL_KEY}=<value>, {cls.TABLE_KEY}={table!r})}}, "
            f"or set {cls.TABLE_KEY} on the credential."
        )

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        access = cast(DBTable, match.access)
        connection = cls.get_connection(access.credentials)
        try:
            return cls.produce_rows(connection, cast(str, access.table), features)
        finally:
            cls.close_connection(connection)
