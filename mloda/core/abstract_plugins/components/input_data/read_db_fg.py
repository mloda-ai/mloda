"""Abstract database format group: finds credentials, lists table catalogs per run, loads one table or query.

A subclass writes ``is_valid_credentials``, ``database_identity``, ``connect``, ``list_tables``, ``table_columns``
and ``produce_rows``; the opt-in ``QUERY_ROUTE`` adds ``produce_query_rows``.
"""

from abc import abstractmethod
from collections.abc import Collection, Hashable, Iterable, Mapping
from dataclasses import dataclass, field
import secrets
from hashlib import blake2b, sha256
from typing import TYPE_CHECKING, Any, ClassVar, cast

from mloda.core.abstract_plugins.components.credential_scrub import scrub_credentials
from mloda.core.abstract_plugins.components.credential import Credential, RegisteredCredential
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy, SourceMatch
from mloda.core.abstract_plugins.components.input_data.format_feature_group import (
    HANDLE_OPTION,
    HANDLE_SPEC,
    QUERY_OPTION as QUERY_OPTION,
    REASON_DETAIL,
    _MAX_DESCRIBED,
    FormatFeatureGroup,
)
from mloda.core.abstract_plugins.components.input_data.match_cache import run_cached
from mloda.core.abstract_plugins.components.match_rejection import INPUT_DATA_STAGE, record_match_rejection
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.utils import is_match_abort
from mloda.core.abstract_plugins.compute_framework import ComputeFramework

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.implementation import Implementation

QUERY_SPEC = PropertySpec("SQL text this group runs through QUERY_ROUTE.", default=None, context=False)

_LOGIN_KEY_SECRET = secrets.token_bytes(16)  # login keys are compared within one process only
_PLAIN_LEAVES = (str, bytes, int, float, bool, type(None))


def _login_text(value: Any, others: list[Any], seen: frozenset[int] = frozenset()) -> str:
    """Canonical text of a credential value; a value that is not a plain scalar goes to others, compared by ==."""
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        if id(value) in seen:
            return "cycle"
        seen = seen | {id(value)}
        if isinstance(value, Mapping):
            items = sorted(value.items(), key=lambda item: repr(item[0]))
            return (
                "{" + ",".join(f"{_login_text(k, others, seen)}:{_login_text(v, others, seen)}" for k, v in items) + "}"
            )
        parts = [_login_text(item, others, seen) for item in value]
        if isinstance(value, (set, frozenset)):
            return "<" + ",".join(sorted(parts)) + ">"
        return ("[" if isinstance(value, list) else "(") + ",".join(parts) + ("]" if isinstance(value, list) else ")")
    if type(value) in _PLAIN_LEAVES:
        return f"{type(value).__name__}:{value!r}"
    others.append(value)
    return "?"


@dataclass(frozen=True, eq=False)
class _LoginKey:
    """Hashes by the digest of the plain values; equal only when the other values compare equal too."""

    digest: str
    others: tuple[Any, ...] = field(default=(), repr=False)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _LoginKey) and self.digest == other.digest and self.others == other.others

    def __hash__(self) -> int:
        return hash(self.digest)


def _capped(names: list[str]) -> str:
    text = ", ".join(repr(name) for name in names[:_MAX_DESCRIBED])
    return text if len(names) <= _MAX_DESCRIBED else f"{text}, and {len(names) - _MAX_DESCRIBED} more"


@dataclass(frozen=True, eq=False)
class DBTable:
    """The access of a database match: the credentials (never shown) and the table, None when unlistable."""

    credentials: Any = field(repr=False)
    table: str | None


@dataclass(frozen=True, eq=False)
class DBQuery:
    """The access of a query match: the credentials (never shown) and the query text."""

    credentials: Any = field(repr=False)
    query_text: str


class ReadDBFG(FormatFeatureGroup):
    """Abstract database format group: claims features found as columns of tables in matching credentials.

    The base serves table catalogs; QUERY_ROUTE opts in to query text. Third-party groups should reuse the contract
    mixin database_format_feature_group_test_mixin from tests/mixins/reader_feature_groups in mloda's repository; it
    needs the rejection_window fixture from mloda's tests/conftest.py.
    """

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = (ClaimRoute("credentials", NamePolicy.CHECKED, True),)
    QUERY_ROUTE: ClassVar[ClaimRoute] = ClaimRoute("credentials", NamePolicy.OPEN, False, (QUERY_OPTION,))
    PROPERTY_MAPPING: ClassVar[dict[str, PropertySpec]] = {HANDLE_OPTION: HANDLE_SPEC}
    TABLE_KEY: ClassVar[str] = "table_name"
    CATALOG_ERRORS: ClassVar[tuple[type[BaseException], ...]] = (OSError, ValueError, ImportError)
    CREDENTIAL_KEY: ClassVar[str] = "<credential key>"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        cls._merge_property_mapping()
        if cls.serves_query_text() and QUERY_OPTION not in cls.PROPERTY_MAPPING:
            cls.PROPERTY_MAPPING = {**cls.PROPERTY_MAPPING, QUERY_OPTION: QUERY_SPEC}
        super().__init_subclass__(**kwargs)

    @classmethod
    @abstractmethod
    def is_valid_credentials(cls, credentials: Any) -> bool:
        """True when the credentials hold this database's own key; never raises."""

    @classmethod
    @abstractmethod
    def database_identity(cls, credentials: Any) -> str:
        """A credential-free name of the database, used as source prefix; catalogs are cached per credential."""

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
    def produce_query_rows(cls, connection: Any, query_text: str, features: Any) -> Any:
        """The requested columns of the query result in a neutral form."""
        raise NotImplementedError(f"{cls.get_class_name()} lists QUERY_ROUTE but does not implement produce_query_rows")

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
    def _matches_by_default_rules(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> bool:
        """On a match, replace a plain mapping pointer with a redacting RegisteredCredential."""
        if not super()._matches_by_default_rules(feature_name, options, data_access_collection):
            return False
        for key in cls.pointer_keys():
            if key in options:
                value = options.get(key)
                if isinstance(value, Mapping) and not isinstance(value, RegisteredCredential):
                    options.set(key, cls.wrap_feature_scoped_access(value))
        return True

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
        query = QUERY_OPTION in route.required_options
        candidates = [c for c in cls._candidates(options, data_access_collection) if cls.is_valid_credentials(c)]
        if query:
            query_text = options.get(QUERY_OPTION)
            if not isinstance(query_text, str) or not query_text:
                record_match_rejection(
                    cls.get_class_name(), f"{QUERY_OPTION} must be a non-empty string", stage=INPUT_DATA_STAGE
                )
                return []
        elif cls.serves_query_text() and QUERY_OPTION in options:
            if candidates:
                record_match_rejection(
                    cls.get_class_name(),
                    f"{QUERY_OPTION} is set, so the table route is not used",
                    stage=INPUT_DATA_STAGE,
                )
            return []
        digest = sha256(query_text.encode()).hexdigest()[:16] if query else ""
        by_source: dict[str, dict[Any, tuple[Any, SourceMatch]]] = {}
        for credentials in candidates:
            identity = cls.database_identity(credentials)
            if query:
                found = [SourceMatch(f"{identity}::query:{digest}", DBQuery(credentials, query_text))]
            else:
                found = cls._tables(identity, credentials)
            login = cls.login_key(credentials)
            for match in found:
                by_source.setdefault(match.source, {}).setdefault(login, (credentials, match))
        matches: list[SourceMatch] = []
        for source, logins in by_source.items():
            if len(logins) == 1:
                matches.append(next(iter(logins.values()))[1])
                continue
            for index, (credentials, match) in enumerate(logins.values(), 1):
                handle = cls._handle_of(credentials, data_access_collection, index)
                matches.append(SourceMatch(f"{source}@{handle}", match.access))
        return matches

    @classmethod
    def _handle_of(cls, credentials: Any, data_access_collection: DataAccessCollection | None, index: int) -> str:
        items = data_access_collection.credentials.items() if data_access_collection else ()
        return next((handle for handle, value in items if value is credentials), f"credential {index}")

    @classmethod
    def login_key(cls, credentials: Any) -> Hashable:
        """A credential-free key; equal keys mean the same login. Default: a keyed digest of all but the table."""
        # A custom _candidates may hand over a Credential.
        data = credentials.data if isinstance(credentials, Credential) else credentials
        if isinstance(data, Mapping):
            data = {k: v for k, v in data.items() if k != cls.TABLE_KEY}
        others: list[Any] = []
        text = _login_text(data, others)
        return _LoginKey(blake2b(text.encode(), key=_LOGIN_KEY_SECRET).hexdigest(), tuple(others))

    @classmethod
    def data_access_identity(cls, match: SourceMatch) -> str:
        access = match.access
        identity = cls.database_identity(access.credentials)
        if isinstance(access, DBQuery):
            return f"{identity}::query:{sha256(access.query_text.encode()).hexdigest()[:16]}"
        return identity if access.table is None else f"{identity}::{access.table}"

    @classmethod
    def _candidates(cls, options: Options, data_access_collection: DataAccessCollection | None) -> list[Any]:
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
            return [wrapped]
        if data_access_collection is None:
            return []
        return cls._handle_credentials(options, data_access_collection)

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
        preset = cls._preset_table(credentials)
        tables = [preset] if preset else list(catalog)
        return [SourceMatch(f"{identity}::{table}", DBTable(credentials, table)) for table in tables]

    @classmethod
    def _preset_table(cls, credentials: Any) -> str | None:
        preset = credentials.get(cls.TABLE_KEY) if isinstance(credentials, Mapping) else None
        return preset if isinstance(preset, str) and preset else None

    @classmethod
    def _catalog(cls, credentials: Any) -> tuple[dict[str, tuple[str, ...]] | None, str | None]:
        identity = cls.database_identity(credentials)
        key = (cls, "catalog", identity, cls.login_key(credentials), cls._preset_table(credentials))
        return run_cached(key, lambda: cls._read_catalog(credentials))

    @classmethod
    def _read_catalog(cls, credentials: Any) -> tuple[dict[str, tuple[str, ...]] | None, str | None]:
        try:
            connection = cls.get_connection(credentials)
            try:
                tables = list(cls.list_tables(connection))
                preset = cls._preset_table(credentials)
                if preset:
                    tables = [preset] if preset in tables else []
                catalog = {table: tuple(cls.table_columns(connection, table)) for table in tables}
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
            detail = ": " if isinstance(exc, ImportError) else REASON_DETAIL
            return None, f"could not read its tables{detail}{scrub_credentials(text)}"
        return catalog, None

    @classmethod
    def columns(cls, match: SourceMatch) -> Collection[str] | None:
        access = match.access
        if isinstance(access, DBQuery) or access.table is None:
            return None
        catalog, _ = cls._catalog(access.credentials)
        return None if catalog is None else catalog.get(access.table)

    @classmethod
    def unknown_columns_reason(cls, match: SourceMatch) -> str | None:
        raw = cls._raw_reason(match)
        return None if raw is None else raw.replace(REASON_DETAIL, ": ")

    @classmethod
    def _hidden_reason(cls, match: SourceMatch) -> str | None:
        raw = cls._raw_reason(match)
        return None if raw is None else raw.split(REASON_DETAIL)[0]

    @classmethod
    def _raw_reason(cls, match: SourceMatch) -> str | None:
        access = match.access
        if isinstance(access, DBQuery):
            return None
        catalog, reason = cls._catalog(access.credentials)
        if catalog is not None and access.table is not None and access.table not in catalog:
            return "its preset table does not exist"
        return reason

    @classmethod
    def _hidden_source(cls, match: SourceMatch) -> str:
        return cls.database_identity(match.access.credentials)

    @classmethod
    def ambiguity_fix(
        cls, feature_name: str, matches: list[SourceMatch], data_access_collection: DataAccessCollection | None
    ) -> str:
        name = cls.get_class_name()
        identities = {cls.database_identity(match.access.credentials) for match in matches}
        logins = {cls.login_key(match.access.credentials) for match in matches}
        if len(identities) > 1 or len(logins) > 1:
            handles = sorted(
                handle
                for handle, credentials in (
                    data_access_collection.credentials.items() if data_access_collection else ()
                )
                if cls.is_valid_credentials(credentials) and cls.database_identity(credentials) in identities
            )
            hint = f" (credential handles: {_capped(handles)})" if handles else ""
            what = "one database" if len(identities) > 1 else "one credential handle"
            return f"select {what} with data_access_handle{hint}, or point {name} at one credential."
        table = cast(DBTable, matches[0].access).table
        return (
            f"point {name} at one table with options={{{name!r}: "
            f"Credential({cls.CREDENTIAL_KEY}=<value>, {cls.TABLE_KEY}={table!r})}}, "
            f"or set {cls.TABLE_KEY} on the credential."
        )

    @classmethod
    def select_implementation(
        cls,
        compute_framework: type[ComputeFramework],
        match: SourceMatch,
        implementations: Iterable[type["Implementation"]],
    ) -> "type[Implementation] | None":
        """No implementation for a query match: it loads through the neutral path."""
        if isinstance(match.access, DBQuery):
            return None
        return super().select_implementation(compute_framework, match, implementations)

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        access = match.access
        connection = cls.get_connection(access.credentials)
        try:
            if isinstance(access, DBQuery):
                return cls.produce_query_rows(connection, access.query_text, features)
            return cls.produce_rows(connection, cast(str, access.table), features)
        finally:
            cls.close_connection(connection)
