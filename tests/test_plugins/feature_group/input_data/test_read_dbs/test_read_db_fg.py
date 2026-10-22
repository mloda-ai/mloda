"""ReadDBFG base behaviour through small test-local database groups, no real driver.

The groups below read the credential key ``toyfmt_basedb`` only, so they stay inert for every other test's inputs.
"""

from __future__ import annotations

import hashlib
import inspect
from collections import OrderedDict
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import pytest

import mloda.provider as provider
from mloda.core.abstract_plugins.components.credential import RegisteredCredential
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy, SourceMatch
from mloda.core.abstract_plugins.components.input_data.match_cache import _RUN_MATCH_CACHE, run_match_cache
from mloda.core.abstract_plugins.components.input_data.read_db_fg import DBQuery, DBTable
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import (
    FeatureResolutionError,
    IdentifyFeatureGroupClass,
    resolve_or_raise,
)
from mloda.core.abstract_plugins.components.match_rejection import INPUT_DATA_STAGE, MatchRejection
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import FormatFeatureGroup, ReadDBFG
from mloda.user import Credential, DataAccessCollection, Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG

KEY = "toyfmt_basedb"


class FakeConnection:
    def __init__(self) -> None:
        self.close_count = 0
        self.fail_rows = False

    def close(self) -> None:
        self.close_count += 1


class ToyBaseDB(ReadDBFG):
    """Complete test-local group: records every hook call in class-level lists."""

    created: ClassVar[list[FakeConnection]] = []
    received_credentials: ClassVar[list[Any]] = []
    row_calls: ClassVar[list[tuple[Any, str, Any]]] = []
    list_calls: ClassVar[list[Any]] = []
    connect_mode: ClassVar[str] = "ok"

    @classmethod
    def is_valid_credentials(cls, credentials: Any) -> bool:
        return isinstance(credentials, Mapping) and isinstance(credentials.get(KEY), str)

    @classmethod
    def database_identity(cls, credentials: Any) -> str:
        return f"toyfmt-db:{credentials[KEY]}"

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        cls.received_credentials.append(credentials)
        if cls.connect_mode == "none":
            return None
        if cls.connect_mode == "raise":
            raise RuntimeError("toyfmt no connect")
        connection = FakeConnection()
        cls.created.append(connection)
        return connection

    @classmethod
    def list_tables(cls, connection: Any) -> Collection[str]:
        cls.list_calls.append(connection)
        return ["toy_table"]

    @classmethod
    def table_columns(cls, connection: Any, table: str) -> Collection[str]:
        return ("toy_a", "toy_b")

    @classmethod
    def produce_rows(cls, connection: Any, table: str, features: Any) -> Any:
        cls.row_calls.append((connection, table, features))
        if connection.fail_rows:
            raise RuntimeError("toyfmt boom")
        return {"toy_a": [1, 2], "toy_b": [3, 4]}


class ToyCustomCloseDB(ToyBaseDB):
    """Overrides close_connection for a driver without close()."""

    closed: ClassVar[list[Any]] = []

    @classmethod
    def close_connection(cls, connection: Any) -> None:
        cls.closed.append(connection)


class ToyStaticRowsDB(ToyBaseDB):
    """Defines produce_rows as a staticmethod."""

    @staticmethod
    def produce_rows(connection: Any, table: str, features: Any) -> Any:
        return {"toy_a": [table]}


class ToyPartialDB(ReadDBFG):
    """Leaves produce_rows abstract."""

    @classmethod
    def is_valid_credentials(cls, credentials: Any) -> bool:
        return False

    @classmethod
    def database_identity(cls, credentials: Any) -> str:
        return "toyfmt-partial"

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        return FakeConnection()

    @classmethod
    def list_tables(cls, connection: Any) -> Collection[str]:
        return []

    @classmethod
    def table_columns(cls, connection: Any, table: str) -> Collection[str]:
        return ()


@pytest.fixture(autouse=True)
def _reset_toy() -> None:
    ToyBaseDB.created.clear()
    ToyBaseDB.received_credentials.clear()
    ToyBaseDB.row_calls.clear()
    ToyBaseDB.list_calls.clear()
    ToyBaseDB.connect_mode = "ok"
    ToyCustomCloseDB.closed.clear()


def _match(credentials: Any, table: str | None = "toy_table") -> SourceMatch:
    return SourceMatch(source=f"toyfmt-db:{credentials[KEY]}::{table}", access=DBTable(credentials, table))


class TestReadDBFGBase:
    def test_it_is_an_abstract_format_group_exported_from_the_provider_with_its_access_carriers(self) -> None:
        assert provider.ReadDBFG is ReadDBFG
        assert "ReadDBFG" in provider.__all__
        assert issubclass(ReadDBFG, FormatFeatureGroup)
        assert inspect.isabstract(ReadDBFG)
        assert provider.DBTable is DBTable
        assert provider.DBQuery is DBQuery
        assert "DBTable" in provider.__all__
        assert "DBQuery" in provider.__all__

    def test_the_provider_hooks_are_abstract_and_the_inherited_ones_are_not(self) -> None:
        required = {
            "is_valid_credentials",
            "database_identity",
            "connect",
            "list_tables",
            "table_columns",
            "produce_rows",
        }
        assert required <= set(ReadDBFG.__abstractmethods__)
        inherited = {"find_sources", "load_neutral", "get_connection", "close_connection", "columns"}
        assert not inherited & set(ReadDBFG.__abstractmethods__)
        assert not inspect.isabstract(ToyBaseDB)

    def test_a_group_missing_one_provider_hook_stays_abstract_and_never_claims(self) -> None:
        assert inspect.isabstract(ToyPartialDB)
        assert ToyPartialDB.__abstractmethods__ == frozenset({"produce_rows"})
        dac = DataAccessCollection(credentials={"h": {KEY: "x"}})
        result = IdentifyFeatureGroupClass.evaluate(
            Feature("toy_a", feature_group=ToyPartialDB), {ToyPartialDB: {PythonDictFramework}}, None, dac
        )
        assert ToyPartialDB not in result.identified

    def test_the_claim_route_is_one_checked_searched_credentials_route(self) -> None:
        assert ReadDBFG.CLAIM_ROUTES == (ClaimRoute("credentials", NamePolicy.CHECKED, True),)

    def test_the_property_mapping_declares_only_the_handle(self) -> None:
        assert set(ReadDBFG.PROPERTY_MAPPING) == {"data_access_handle"}
        handle = ReadDBFG.PROPERTY_MAPPING["data_access_handle"]
        assert handle.default is None
        assert handle.strict_validation is True
        assert handle.context is True
        assert handle.match_guard is not None
        assert handle.match_guard("a_handle")
        assert not handle.match_guard(["a_handle"])
        assert set(ToyBaseDB.PROPERTY_MAPPING) == {"data_access_handle"}
        assert "query_text" in ToyQueryDB.PROPERTY_MAPPING

    def test_the_table_name_key_and_catalog_errors_are_declared(self) -> None:
        assert ReadDBFG.TABLE_KEY == "table_name"
        assert issubclass(OSError, ReadDBFG.CATALOG_ERRORS)
        assert issubclass(ValueError, ReadDBFG.CATALOG_ERRORS)
        assert not issubclass(TypeError, ReadDBFG.CATALOG_ERRORS)

    def test_the_docstring_points_third_party_groups_at_the_contract_mixins_and_states_the_scope(self) -> None:
        doc = ReadDBFG.__doc__ or ""
        assert "tests/mixins/reader_feature_groups" in doc
        assert "database_format_feature_group_test_mixin" in doc
        assert "catalog" in doc.lower()

    def test_the_catalog_is_not_versioned_by_file_stat(self) -> None:
        assert not hasattr(ReadDBFG, "catalog_version")


class TestGetConnection:
    def test_it_returns_what_connect_returns(self) -> None:
        connection = ToyBaseDB.get_connection({KEY: "a"})
        assert isinstance(connection, FakeConnection)
        assert ToyBaseDB.created == [connection]

    def test_a_none_connection_raises_naming_the_class_the_credential_type_and_the_checklist(self) -> None:
        ToyBaseDB.connect_mode = "none"
        with pytest.raises(ValueError) as excinfo:
            ToyBaseDB.get_connection({KEY: "a"})
        message = str(excinfo.value)
        assert "ToyBaseDB" in message
        assert "Credentials type: dict" in message
        assert "database server is reachable" in message

    def test_a_str_credential_shows_its_type(self) -> None:
        ToyBaseDB.connect_mode = "none"
        with pytest.raises(ValueError, match="Credentials type: str"):
            ToyBaseDB.get_connection("dsn")

    def test_the_message_never_echoes_credential_values(self) -> None:
        ToyBaseDB.connect_mode = "none"
        with pytest.raises(ValueError) as excinfo:
            ToyBaseDB.get_connection({KEY: "toyfmt-secret-value"})  # nosec B105
        assert "toyfmt-secret-value" not in str(excinfo.value)


class TestCloseSemantics:
    def test_the_default_close_calls_close_once(self) -> None:
        connection = FakeConnection()
        ToyBaseDB.close_connection(connection)
        assert connection.close_count == 1

    def test_load_neutral_connects_with_the_matched_credentials_reads_the_table_and_closes_once(self) -> None:
        credentials = {KEY: "a"}
        features = object()

        result = ToyBaseDB.load_neutral(_match(credentials), features)

        assert result == {"toy_a": [1, 2], "toy_b": [3, 4]}
        assert ToyBaseDB.received_credentials == [credentials]
        assert ToyBaseDB.received_credentials[0] is credentials
        connection, table, passed = ToyBaseDB.row_calls[0]
        assert table == "toy_table"
        assert passed is features
        assert connection is ToyBaseDB.created[0]
        assert connection.close_count == 1

    def test_a_staticmethod_produce_rows_runs_end_to_end_and_the_connection_closes(self) -> None:
        result = ToyStaticRowsDB.load_neutral(_match({KEY: "a"}), object())

        assert result == {"toy_a": ["toy_table"]}
        assert ToyBaseDB.created[0].close_count == 1

    def test_the_connection_is_closed_when_produce_rows_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = ToyBaseDB.connect

        def failing_rows(credentials: Any) -> Any:
            connection = original(credentials)
            connection.fail_rows = True
            return connection

        monkeypatch.setattr(ToyBaseDB, "connect", failing_rows)
        with pytest.raises(RuntimeError, match="toyfmt boom"):
            ToyBaseDB.load_neutral(_match({KEY: "a"}), object())
        assert ToyBaseDB.created[-1].close_count == 1

    def test_a_connect_that_raises_skips_close_and_produce_rows(self) -> None:
        ToyBaseDB.connect_mode = "raise"
        with pytest.raises(RuntimeError, match="toyfmt no connect"):
            ToyBaseDB.load_neutral(_match({KEY: "a"}), object())
        assert ToyBaseDB.row_calls == []
        assert ToyBaseDB.created == []

    def test_a_connect_returning_none_raises_before_produce_rows(self) -> None:
        ToyBaseDB.connect_mode = "none"
        with pytest.raises(ValueError, match="ToyBaseDB"):
            ToyBaseDB.load_neutral(_match({KEY: "a"}), object())
        assert ToyBaseDB.row_calls == []

    def test_an_overridden_close_connection_is_the_one_load_neutral_uses(self) -> None:
        ToyCustomCloseDB.load_neutral(_match({KEY: "a"}), object())
        assert len(ToyCustomCloseDB.closed) == 1
        assert ToyCustomCloseDB.closed[0].close_count == 0


class TestWrapFeatureScopedAccess:
    def test_a_dict_is_wrapped_in_a_registered_credential_copy(self) -> None:
        original = {KEY: "a"}
        wrapped = ToyBaseDB.wrap_feature_scoped_access(original)
        assert type(wrapped) is RegisteredCredential
        assert wrapped == original
        assert wrapped is not original

    def test_a_dict_subclass_is_wrapped(self) -> None:
        wrapped = ToyBaseDB.wrap_feature_scoped_access(OrderedDict({KEY: "a"}))
        assert type(wrapped) is RegisteredCredential
        assert wrapped == {KEY: "a"}

    def test_a_credential_becomes_a_registered_credential_of_its_data(self) -> None:
        credential = Credential({KEY: "a"})
        wrapped = ToyBaseDB.wrap_feature_scoped_access(credential)
        assert type(wrapped) is RegisteredCredential
        assert wrapped == {KEY: "a"}
        assert credential.data == {KEY: "a"}

    def test_a_registered_credential_is_returned_as_is(self) -> None:
        credential = RegisteredCredential({KEY: "a"})
        assert ToyBaseDB.wrap_feature_scoped_access(credential) is credential

    def test_any_mapping_is_wrapped_in_a_registered_credential_copy(self) -> None:
        wrapped = ToyBaseDB.wrap_feature_scoped_access(MappingProxyType({KEY: "a"}))
        assert type(wrapped) is RegisteredCredential
        assert wrapped == {KEY: "a"}

    def test_a_pointer_given_as_a_non_dict_mapping_claims(self) -> None:
        feature = Feature("toy_a", options={"ToyBaseDB": MappingProxyType({KEY: "a"})})
        result = IdentifyFeatureGroupClass.evaluate(feature, {ToyBaseDB: {PythonDictFramework}}, None, None)
        assert ToyBaseDB in result.identified

    @pytest.mark.parametrize("value", ["a path", 5, None, ["a"]])
    def test_anything_else_is_none(self, value: Any) -> None:
        assert ToyBaseDB.wrap_feature_scoped_access(value) is None


class TestInheritedDefaults:
    def test_columns_of_a_match_come_from_the_catalog_and_an_unreadable_one_has_none(self) -> None:
        assert set(ToyBaseDB.columns(_match({KEY: "a"})) or ()) == {"toy_a", "toy_b"}
        assert ToyBaseDB.columns(_match({KEY: "a"}, None)) is None

    def test_describe_columns_names_every_column_with_no_type(self) -> None:
        assert ToyBaseDB.describe_columns(_match({KEY: "a"})) == {"toy_a": None, "toy_b": None}

    def test_count_rows_is_none_and_the_identity_is_the_credential_free_source(self) -> None:
        match = _match({KEY: "a", "password": "toyfmt-secret-value"})  # nosec B105
        assert ToyBaseDB.count_rows(match, PythonDictFramework) is None
        assert ToyBaseDB.data_access_identity(match) == match.source
        assert "toyfmt-secret-value" not in match.source
        assert "toyfmt-secret-value" not in repr(match)


class RoleConnection(FakeConnection):
    def __init__(self, role: str) -> None:
        super().__init__()
        self.role = role


class ToyRoleDB(ToyBaseDB):
    """The catalog depends on the credential's role: only the admin reads the extra column."""

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        return RoleConnection(str(credentials.get("role")))

    @classmethod
    def table_columns(cls, connection: Any, table: str) -> Collection[str]:
        if connection.role == "admin":
            return ("toyfmt_shared", "toyfmt_admin_only")
        return ("toyfmt_shared",)


class ToySecretIdentityDB(ToyBaseDB):
    """Puts the credential into the database identity, as a driver that embeds it in a URL would."""

    @classmethod
    def database_identity(cls, credentials: Any) -> str:
        return f"kv://shop?pw={credentials[KEY]}"


@dataclass(frozen=True)
class _OpaqueSecret:
    """Equality separates two values whose repr is identical."""

    value: str = field(repr=False)


@dataclass
class _MutableOpaqueSecret:
    """Unhashable, equality by value, repr hides the value."""

    value: str = field(repr=False)


class _SecretText(str):
    pass


_SHARED_OPAQUE = _OpaqueSecret("one")


class TestCatalogCache:
    def test_a_run_reads_a_database_catalog_once_and_a_second_database_separately(self) -> None:
        first, second = _match({KEY: "a"}), _match({KEY: "b"})
        with run_match_cache():
            for _ in range(3):
                ToyBaseDB.columns(first)
            ToyBaseDB.columns(second)
        assert len(ToyBaseDB.list_calls) == 2

    def test_outside_a_run_every_read_lists_again(self) -> None:
        match = _match({KEY: "a"})
        ToyBaseDB.columns(match)
        ToyBaseDB.columns(match)
        assert len(ToyBaseDB.list_calls) == 2

    def test_credentials_differing_only_in_a_secret_read_two_catalogs_in_one_run(self) -> None:
        first = _match({KEY: "a", "password": "toyfmt-secret-one"})  # nosec B105
        second = _match({KEY: "a", "password": "toyfmt-secret-two"})  # nosec B105
        with run_match_cache():
            ToyBaseDB.columns(first)
            ToyBaseDB.columns(second)
        assert len(ToyBaseDB.list_calls) == 2

    @pytest.mark.parametrize("secret_type", [_OpaqueSecret, _MutableOpaqueSecret])
    def test_secrets_with_an_identical_redacted_repr_still_read_two_catalogs_in_one_run(
        self, secret_type: type
    ) -> None:
        first = _match({KEY: "a", "password": secret_type("one")})  # nosec B105
        second = _match({KEY: "a", "password": secret_type("two")})  # nosec B105
        assert repr(first.access.credentials["password"]) == repr(second.access.credentials["password"])
        with run_match_cache():
            ToyBaseDB.columns(first)
            ToyBaseDB.columns(second)
        assert len(ToyBaseDB.list_calls) == 2

    def test_credentials_differing_only_in_the_preset_table_read_two_catalogs_in_one_run(self) -> None:
        first = _match({KEY: "a", ToyBaseDB.TABLE_KEY: "t1"})
        second = _match({KEY: "a", ToyBaseDB.TABLE_KEY: "t2"})
        with run_match_cache():
            ToyBaseDB.columns(first)
            ToyBaseDB.columns(second)
        assert len(ToyBaseDB.list_calls) == 2

    def test_the_run_cache_keys_carry_no_credential_value(self) -> None:
        with run_match_cache():
            ToyBaseDB.columns(_match({KEY: "a", "password": _ADMIN_SECRET}))
            cache = _RUN_MATCH_CACHE.get()
            assert cache
            for key in cache:
                assert _ADMIN_SECRET not in str(key)
                assert _ADMIN_SECRET not in repr(key)

    def test_equal_but_distinct_credential_objects_read_one_catalog_in_a_run(self) -> None:
        first = _match({KEY: "a", "password": "toyfmt-secret-one", "options": {"x": [1]}})  # nosec B105
        second = _match({KEY: "a", "password": "toyfmt-secret-one", "options": {"x": [1]}})  # nosec B105
        assert first.access is not second.access
        with run_match_cache():
            ToyBaseDB.columns(first)
            ToyBaseDB.columns(second)
        assert len(ToyBaseDB.list_calls) == 1

    def test_an_analyst_never_sees_or_claims_a_column_only_the_admin_credential_reads(self) -> None:
        admin = {KEY: "shop", "role": "admin"}
        analyst = {KEY: "shop", "role": "analyst"}
        plugins: FeatureGroupEnvironmentMapping = {ToyRoleDB: {PythonDictFramework}}
        with run_match_cache():
            assert "toyfmt_admin_only" in (ToyRoleDB.columns(_match(admin)) or ())

            dac = DataAccessCollection(credentials={"analyst_handle": analyst})
            claim = Feature("toyfmt_admin_only")
            assert ToyRoleDB not in IdentifyFeatureGroupClass.evaluate(claim, plugins, None, dac).identified

            typo = Feature("toyfmt_missing_col", options={"ToyRoleDB": analyst})
            with pytest.raises(ValueError) as excinfo:
                resolve_or_raise(typo, plugins, None, None)
        message = str(excinfo.value)
        assert "toyfmt_shared" in message
        assert "toyfmt_admin_only" not in message


class ToyRoleQueryDB(ToyRoleDB):
    """ToyRoleDB that also opts into the query route."""

    CLAIM_ROUTES = (*ReadDBFG.CLAIM_ROUTES, ReadDBFG.QUERY_ROUTE)


class ToyRoleOrdersDB(ToyRoleDB):
    """ToyRoleDB over a distinctively named table."""

    @classmethod
    def list_tables(cls, connection: Any) -> Collection[str]:
        return ["zq_orders_tbl"]


_ADMIN_SECRET = "toyfmt-admin-secret"  # nosec B105
_ADMIN = {KEY: "shop", "role": "admin", "password": _ADMIN_SECRET}
_ANALYST = {KEY: "shop", "role": "analyst", "password": "toyfmt-analyst-secret"}  # nosec B105


def _two_logins() -> DataAccessCollection:
    return DataAccessCollection(credentials={"analyst_h": _ANALYST, "admin_h": _ADMIN})


class TestTwoLoginsOnOneSource:
    def test_a_column_only_the_second_login_sees_is_claimed_with_its_match_and_loaded_with_its_credentials(
        self,
    ) -> None:
        feature = Feature("toyfmt_admin_only")
        plugins: FeatureGroupEnvironmentMapping = {ToyRoleDB: {PythonDictFramework}}
        result = IdentifyFeatureGroupClass.evaluate(feature, plugins, None, _two_logins())
        assert ToyRoleDB in result.identified
        assert feature.input_data_match is not None
        match = feature.input_data_match[1]
        assert match.source == "toyfmt-db:shop::toy_table@admin_h"
        ToyRoleDB.load_neutral(match, object())
        assert ToyBaseDB.row_calls[-1][0].role == "admin"

    @pytest.mark.parametrize(
        ("feature", "group"),
        [
            (Feature("toyfmt_shared"), ToyRoleDB),
            (Feature("anything", options={"query_text": "SELECT 1"}), ToyRoleQueryDB),
        ],
        ids=["table_route", "query_route"],
    )
    def test_two_logins_on_one_identity_abort_naming_both_handles_without_a_secret(
        self, feature: Feature, group: type[ToyRoleDB]
    ) -> None:
        plugins: FeatureGroupEnvironmentMapping = {group: {PythonDictFramework}}
        with pytest.raises(ValueError) as excinfo:
            resolve_or_raise(feature, plugins, None, _two_logins())
        message = str(excinfo.value)
        assert "credential handles: 'admin_h', 'analyst_h'" in message
        assert "@admin_h" not in message
        assert "@analyst_h" not in message
        assert "(2 sources)" in message
        if group is ToyRoleDB:
            assert "toy_table" not in message
        assert "data_access_handle" in message
        assert "select one credential handle" in message
        assert "select one database" not in message
        assert _ADMIN_SECRET not in message

    def test_the_data_access_identity_of_a_suffixed_match_is_the_unsuffixed_source(self) -> None:
        matches = ToyRoleDB.find_sources(ToyRoleDB.CLAIM_ROUTES[0], "toyfmt_shared", Options({}), _two_logins())
        assert len(matches) == 2
        assert {ToyRoleDB.data_access_identity(match) for match in matches} == {"toyfmt-db:shop::toy_table"}

    def test_unpointed_missing_column_text_shows_neither_a_handle_suffix_nor_the_table_name(self) -> None:
        plugins: FeatureGroupEnvironmentMapping = {ToyRoleOrdersDB: {PythonDictFramework}}
        with pytest.raises(ValueError) as excinfo:
            resolve_or_raise(Feature("toyfmt_missing_col"), plugins, None, _two_logins())
        message = str(excinfo.value)
        assert "toyfmt-db:shop" in message
        assert "@" not in message
        assert "zq_orders_tbl" not in message


class ToyFlakyRoleDB(ToyRoleDB):
    """ToyRoleDB whose broken role cannot read the catalog."""

    @classmethod
    def list_tables(cls, connection: Any) -> Collection[str]:
        if connection.role == "broken":
            raise OSError("toyfmt catalog down")
        return super().list_tables(connection)


class ToyOutsideCollectionDB(ToyRoleDB):
    """Supplies two logins from outside the data access collection."""

    @classmethod
    def _candidates(cls, options: Options, data_access_collection: DataAccessCollection | None) -> list[Any]:
        return [dict(_ANALYST), dict(_ADMIN)]


class TestTwoLoginsEdges:
    def test_unpointed_missing_column_does_not_blame_a_failed_read_when_another_login_read_the_catalog(
        self,
    ) -> None:
        broken = {KEY: "shop", "role": "broken"}
        dac = DataAccessCollection(credentials={"a_broken_h": broken, "b_analyst_h": _ANALYST})
        plugins: FeatureGroupEnvironmentMapping = {ToyFlakyRoleDB: {PythonDictFramework}}
        with pytest.raises(ValueError) as excinfo:
            resolve_or_raise(Feature("toyfmt_missing_col"), plugins, None, dac)
        message = str(excinfo.value)
        assert "toyfmt-db:shop" in message
        assert "could not read" not in message

    def test_logins_found_outside_the_collection_still_get_distinct_sources(self) -> None:
        matches = ToyOutsideCollectionDB.find_sources(
            ToyOutsideCollectionDB.CLAIM_ROUTES[0], "toyfmt_shared", Options({}), None
        )
        assert len(matches) == 2
        assert len({match.source for match in matches}) == 2


class TestLoginKey:
    @pytest.mark.parametrize(
        ("first", "second", "equal"),
        [
            (
                {KEY: "a", "password": "p", ToyBaseDB.TABLE_KEY: "t1"},  # nosec B105
                {KEY: "a", "password": "p", ToyBaseDB.TABLE_KEY: "t2"},  # nosec B105
                True,
            ),
            ({KEY: "a", "role": "admin"}, {KEY: "a", "role": "analyst"}, False),
            (
                Credential({KEY: "a", "password": "toyfmt-pw-one"}),  # nosec B105
                {KEY: "a", "password": "toyfmt-pw-one"},  # nosec B105
                True,
            ),
            (
                {KEY: "a", "password": _OpaqueSecret("one")},  # nosec B105
                {KEY: "a", "password": _OpaqueSecret("two")},  # nosec B105
                False,
            ),
            (
                Credential({KEY: "a", "password": "toyfmt-pw-one", ToyBaseDB.TABLE_KEY: "t1"}),  # nosec B105
                Credential({KEY: "a", "password": "toyfmt-pw-one", ToyBaseDB.TABLE_KEY: "t2"}),  # nosec B105
                True,
            ),
            (
                {KEY: "a", "password": _SHARED_OPAQUE},  # nosec B105
                {KEY: "a", "password": _SHARED_OPAQUE},  # nosec B105
                True,
            ),
            (
                {KEY: "a", "password": _MutableOpaqueSecret("one")},  # nosec B105
                {KEY: "a", "password": _MutableOpaqueSecret("two")},  # nosec B105
                False,
            ),
            (
                {KEY: "a", "cert": Path("toyfmt-cert.pem")},
                {KEY: "a", "cert": Path("toyfmt-cert.pem")},
                True,
            ),
            (
                {KEY: "a", "hosts": ["h1", "h2"]},
                {KEY: "a", "hosts": ("h1", "h2")},
                False,
            ),
        ],
        ids=[
            "differ_only_in_table_name",
            "differ_in_another_field",
            "credential_object_equals_its_mapping",
            "opaque_secrets_with_one_redacted_repr_differ",
            "credential_objects_differ_only_in_table_name",
            "same_opaque_secret_instance_is_one_login",
            "unhashable_opaque_secrets_with_one_redacted_repr_differ",
            "equal_but_distinct_value_objects_are_one_login",
            "a_list_and_a_tuple_value_differ",
        ],
    )
    def test_credentials_are_one_login_only_up_to_the_table_name(self, first: Any, second: Any, equal: bool) -> None:
        one = ToyBaseDB.login_key(first)
        two = ToyBaseDB.login_key(second)
        assert (one == two) is equal

    def test_a_non_mapping_credential_is_a_login_by_value(self) -> None:
        assert ToyBaseDB.login_key("dsn-one") == ToyBaseDB.login_key("dsn-one")
        assert ToyBaseDB.login_key("dsn-one") != ToyBaseDB.login_key("dsn-two")

    def test_the_default_key_carries_no_credential_value(self) -> None:
        key = ToyBaseDB.login_key({KEY: "a", "password": _ADMIN_SECRET})
        assert _ADMIN_SECRET not in str(key)
        assert _ADMIN_SECRET not in repr(key)
        key = ToyBaseDB.login_key({KEY: "a", "password": _SecretText(_ADMIN_SECRET)})
        assert _ADMIN_SECRET not in str(key)
        assert _ADMIN_SECRET not in repr(key)


class TestMultipleMessageSourceIsScrubbed:
    def test_the_source_part_of_the_multiple_message_lacks_a_secret_in_the_identity(self, tmp_path: Path) -> None:
        secret = "toyfmt-secret"  # nosec B105
        csv_path = tmp_path / "twin.csv"
        csv_path.write_text("toy_a\n1\n2\n", encoding="utf-8")
        dac = DataAccessCollection(credentials={"toy_handle": {KEY: secret}}, files={"csv_handle": str(csv_path)})
        plugins: FeatureGroupEnvironmentMapping = {
            ToySecretIdentityDB: {PythonDictFramework},
            CsvFG: {PythonDictFramework},
        }

        with pytest.raises(ValueError, match="Multiple feature groups found") as excinfo:
            resolve_or_raise(Feature("toy_a"), plugins, None, dac)

        message = str(excinfo.value)
        assert "[source: " in message
        assert "kv://shop" in message
        assert secret not in message


class TestCatalogFailures:
    def test_a_missing_driver_declines_and_a_sibling_group_still_resolves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def missing_driver(credentials: Any) -> Any:
            raise ImportError("toyfmt driver is not installed")

        monkeypatch.setattr(ToyBaseDB, "connect", missing_driver)
        csv_path = tmp_path / "sibling.csv"
        csv_path.write_text("toyfmt_csv_col\n1\n2\n", encoding="utf-8")
        dac = DataAccessCollection(credentials={"toy_handle": {KEY: "a"}}, files={"csv_handle": str(csv_path)})

        declined = IdentifyFeatureGroupClass.evaluate(Feature("toy_a"), {ToyBaseDB: {PythonDictFramework}}, None, dac)
        assert ToyBaseDB not in declined.identified
        assert "could not read its tables" in declined.eliminations[ToyBaseDB].reason

        result = mloda.run_all(
            ["toyfmt_csv_col"],
            compute_frameworks=[PythonDictFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({ToyBaseDB, CsvFG}),
            data_access_collection=dac,
        )
        assert result[0] == {"toyfmt_csv_col": [1, 2]}

    def test_a_credential_value_in_the_driver_error_is_masked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        secret = "toyfmt-dsn-secret"  # nosec B105

        def leaking(credentials: Any) -> Any:
            raise OSError(f"cannot connect with password {secret}")

        monkeypatch.setattr(ToyBaseDB, "connect", leaking)
        dac = DataAccessCollection(credentials={"toy_handle": {KEY: "a", "password": secret}})
        result = IdentifyFeatureGroupClass.evaluate(Feature("toy_a"), {ToyBaseDB: {PythonDictFramework}}, None, dac)
        reason = result.eliminations[ToyBaseDB].reason
        assert "could not read its tables" in reason
        assert secret not in reason

        pointed = Feature("toy_a", options={"ToyBaseDB": {KEY: "a", "password": secret}})
        with pytest.raises(ValueError, match="could not read its tables") as excinfo:
            resolve_or_raise(pointed, {ToyBaseDB: {PythonDictFramework}}, None, None)
        assert secret not in str(excinfo.value)


class ToyQueryDB(ToyBaseDB):
    """Opts into the query route."""

    CLAIM_ROUTES = (*ReadDBFG.CLAIM_ROUTES, ReadDBFG.QUERY_ROUTE)


class ToyOwnQueryRouteDB(ToyBaseDB):
    """Lists its own query route, which differs from ReadDBFG.QUERY_ROUTE."""

    CLAIM_ROUTES = (*ReadDBFG.CLAIM_ROUTES, ClaimRoute("own_credentials", NamePolicy.OPEN, False, ("query_text",)))


class TestQueryRoute:
    def test_a_group_with_its_own_query_text_route_declares_query_text_and_claims_a_query_source(self) -> None:
        assert ReadDBFG.QUERY_ROUTE not in ToyOwnQueryRouteDB.CLAIM_ROUTES
        assert "query_text" in ToyOwnQueryRouteDB.PROPERTY_MAPPING
        feature = Feature("anything", options={"ToyOwnQueryRouteDB": {KEY: "a"}, "query_text": "SELECT 1"})
        plugins: FeatureGroupEnvironmentMapping = {ToyOwnQueryRouteDB: {PythonDictFramework}}
        result = IdentifyFeatureGroupClass.evaluate(feature, plugins, None, None)
        assert ToyOwnQueryRouteDB in result.identified
        assert feature.input_data_match is not None
        assert (
            feature.input_data_match[1].source == f"toyfmt-db:a::query:{hashlib.sha256(b'SELECT 1').hexdigest()[:16]}"
        )

    def test_the_query_route_is_an_open_pointed_only_route_unlocked_by_query_text_and_not_a_default(self) -> None:
        assert ReadDBFG.QUERY_ROUTE == ClaimRoute("credentials", NamePolicy.OPEN, False, ("query_text",))
        assert ReadDBFG.QUERY_ROUTE not in ReadDBFG.CLAIM_ROUTES
        assert ReadDBFG.QUERY_ROUTE not in ToyBaseDB.CLAIM_ROUTES

    def test_a_group_without_the_route_aborts_pointed_on_query_text(self) -> None:
        feature = Feature("toy_a", options={"ToyBaseDB": {KEY: "a"}, "query_text": "SELECT 1"})
        with pytest.raises(FeatureResolutionError, match="query_text is set, but ToyBaseDB lists no query route"):
            resolve_or_raise(feature, {ToyBaseDB: {PythonDictFramework}}, None, None)
        assert ToyBaseDB.created == []

    def test_the_default_produce_query_rows_raises_naming_the_group(self) -> None:
        with pytest.raises(NotImplementedError, match="ToyBaseDB"):
            ToyBaseDB.produce_query_rows(FakeConnection(), "SELECT 1", object())

    def test_the_query_source_hashes_the_query_and_holds_no_credential_value(self) -> None:
        secret = "toyfmt-query-secret"  # nosec B105
        feature = Feature("anything", options={"ToyQueryDB": {KEY: "a", "password": secret}, "query_text": "SELECT 1"})
        result = IdentifyFeatureGroupClass.evaluate(feature, {ToyQueryDB: {PythonDictFramework}}, None, None)
        assert ToyQueryDB in result.identified
        assert feature.input_data_match is not None
        source = feature.input_data_match[1].source
        assert source == f"toyfmt-db:a::query:{hashlib.sha256(b'SELECT 1').hexdigest()[:16]}"
        assert secret not in repr(feature.input_data_match[1])
        assert ToyBaseDB.list_calls == []
        assert ToyBaseDB.created == []

    def test_query_text_suppresses_the_table_route_with_an_attributable_rejection(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        options = Options({"query_text": "SELECT 1"})
        dac = DataAccessCollection(credentials={"toy_handle": {KEY: "a"}})
        assert ToyQueryDB.find_sources(ToyQueryDB.CLAIM_ROUTES[0], "toy_a", options, dac) == []
        rejection = rejection_window[ToyQueryDB.get_class_name()]
        assert rejection.stage == INPUT_DATA_STAGE
        assert "query_text" in rejection.reason

    @pytest.mark.parametrize("query", [5, "", None, ["SELECT 1"]], ids=["int", "empty", "none", "list"])
    def test_a_non_str_or_empty_query_text_declines_with_a_rejection(self, query: Any) -> None:
        feature = Feature("toy_a", options={"ToyQueryDB": {KEY: "a"}, "query_text": query})
        result = IdentifyFeatureGroupClass.evaluate(feature, {ToyQueryDB: {PythonDictFramework}}, None, None)
        assert ToyQueryDB not in result.identified
        assert "query_text" in result.eliminations[ToyQueryDB].reason

    def test_load_neutral_runs_produce_query_rows_for_a_query_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[Any, str, Any]] = []

        def produce(connection: Any, query_text: str, features: Any) -> Any:
            calls.append((connection, query_text, features))
            return {"toy_a": [7]}

        monkeypatch.setattr(ToyQueryDB, "produce_query_rows", produce)
        credentials = {KEY: "a"}
        match = SourceMatch("toyfmt-db:a::query:x", DBQuery(credentials, "SELECT 1"))
        features = object()
        assert ToyQueryDB.load_neutral(match, features) == {"toy_a": [7]}
        assert calls == [(ToyBaseDB.created[0], "SELECT 1", features)]
        assert ToyBaseDB.created[0].close_count == 1
        assert ToyBaseDB.row_calls == []


class TestEndToEndOnAThirdPartyGroup:
    def test_a_complete_local_group_claims_and_loads_through_the_base(self) -> None:
        dac = DataAccessCollection(credentials={"toy_handle": {KEY: "a"}})

        result = mloda.run_all(
            ["toy_a"],
            compute_frameworks=[PythonDictFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({ToyBaseDB}),
            data_access_collection=dac,
        )

        assert result[0] == {"toy_a": [1, 2]}
        assert all(connection.close_count == 1 for connection in ToyBaseDB.created)

    def test_the_abstract_base_never_claims_a_feature_named_after_it(self) -> None:
        dac = DataAccessCollection(credentials={"toy_handle": {KEY: "a"}})
        result = IdentifyFeatureGroupClass.evaluate(Feature("ReadDBFG"), {ReadDBFG: {PythonDictFramework}}, None, dac)
        assert ReadDBFG not in result.identified
