"""ReadDBFG base behaviour through small test-local database groups, no real driver.

The groups below read the credential key ``toyfmt_basedb`` only, so they stay inert for every other test's inputs.
"""

from __future__ import annotations

import hashlib
import inspect
from collections import OrderedDict
from collections.abc import Collection, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import pytest

import mloda.provider as provider
from mloda.core.abstract_plugins.components.credential import RegisteredCredential
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy, SourceMatch
from mloda.core.abstract_plugins.components.input_data.match_cache import run_match_cache
from mloda.core.abstract_plugins.components.input_data.read_db_fg import DBQuery, DBTable
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass, resolve_or_raise
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
    def test_it_is_an_abstract_format_group_exported_from_the_provider_without_its_table_carrier(self) -> None:
        assert provider.ReadDBFG is ReadDBFG
        assert "ReadDBFG" in provider.__all__
        assert issubclass(ReadDBFG, FormatFeatureGroup)
        assert inspect.isabstract(ReadDBFG)
        assert not hasattr(provider, "DBTable")

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


class TestQueryRoute:
    def test_the_query_route_is_an_open_pointed_only_route_unlocked_by_query_text_and_not_a_default(self) -> None:
        assert ReadDBFG.QUERY_ROUTE == ClaimRoute("credentials", NamePolicy.OPEN, False, ("query_text",))
        assert ReadDBFG.QUERY_ROUTE not in ReadDBFG.CLAIM_ROUTES
        assert ReadDBFG.QUERY_ROUTE not in ToyBaseDB.CLAIM_ROUTES

    def test_a_group_without_the_route_ignores_query_text_and_claims_by_table(self) -> None:
        feature = Feature("toy_a", options={"ToyBaseDB": {KEY: "a"}, "query_text": "SELECT 1"})
        result = IdentifyFeatureGroupClass.evaluate(feature, {ToyBaseDB: {PythonDictFramework}}, None, None)
        assert ToyBaseDB in result.identified
        assert feature.input_data_match is not None
        assert feature.input_data_match[1].source == "toyfmt-db:a::toy_table"

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
        assert source == f"toyfmt-db:a::query:{hashlib.sha256(b'SELECT 1').hexdigest()[:12]}"
        assert secret not in repr(feature.input_data_match[1])
        assert ToyBaseDB.list_calls == []
        assert ToyBaseDB.created == []

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
