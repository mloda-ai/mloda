"""Shared contract tests for ReadDBFG implementations (one database kind per group).

A concrete test class sets the FormatFeatureGroup attributes, ``credential_key`` and ``write_database``. Not
collected on its own. Every test-local subclass of the group is gated by an explicit plugin mapping.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from mloda.core.abstract_plugins.components.credential import RegisteredCredential
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.abstract_plugins.components.input_data.match_cache import run_match_cache
from mloda.core.abstract_plugins.components.input_data.read_db_fg import DBTable
from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
from mloda.core.abstract_plugins.components.utils import escalate_match_abort
from mloda.provider import CHAIN_SEPARATOR
from mloda.user import Credential, Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow import PyArrowAggregatedFeatureGroup
from tests.mixins.reader_feature_groups.format_feature_group_test_mixin import (
    FormatFeatureGroupTestMixin,
    _LoadCapture,
)

if TYPE_CHECKING:
    from mloda.provider import ReadDBFG

HANDLE_OPTION = "data_access_handle"
SECRET = "toyfmt-secret-value"  # nosec B105
OWN_TABLE = "own_table"


class _Spy:
    """Calls the group's connection hooks received since the spy was installed."""

    def __init__(self) -> None:
        self.opened: list[Any] = []
        self.closed: list[Any] = []
        self.list_tables = 0
        self.table_columns: list[str] = []

    def all_closed(self) -> bool:
        return all(any(conn is closed for closed in self.closed) for conn in self.opened)


class DatabaseFormatFeatureGroupTestMixin(FormatFeatureGroupTestMixin):
    """Contract of every ReadDBFG; adds credential, catalog, handle, cache, lifecycle and failure checks."""

    feature_group_class: type[ReadDBFG]
    credential_key: str
    tmp_path: Path

    def write_database(self, path: Path, tables: dict[str, dict[str, list[Any]]]) -> None:
        """Create a readable database holding one table per entry with exactly these columns."""
        raise NotImplementedError

    def write_corrupt_database(self, path: Path) -> None:
        """Write a file at path that the group's driver rejects (default: invalid bytes)."""
        path.write_bytes(b"\xff\xfe\x00\x81" * 64)

    def make_credential(self, path: Path, **extra: Any) -> dict[str, Any]:
        return {self.credential_key: str(path), **extra}

    def identity_of(self, path: Path) -> str:
        """The credential-free identity of the database at path."""
        return os.path.abspath(str(path))

    @pytest.fixture(autouse=True)
    def _database_format_setup(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.own_path = self._make("own_db", {OWN_TABLE: {self.present_column: [1, 2]}})
        self.expected_source = f"{self.identity_of(self.own_path)}::{OWN_TABLE}"

    def own_pointer(self) -> Any:
        return self.make_credential(self.own_path)

    def other_pointer(self) -> tuple[Any, str]:
        other = self._make("other_db", {"other_table": {self.present_column: [9]}})
        return self.make_credential(other), self._source_of(other, "other_table")

    def foreign_handle_dac(self) -> tuple[DataAccessCollection, str]:
        file_path = self.tmp_path / "plain_file.txt"
        file_path.write_text("x")
        dac = DataAccessCollection(
            credentials={"own_handle": self.make_credential(self.own_path)}, files={"file_h": str(file_path)}
        )
        return dac, "file_h"

    def known_handles_dac(self) -> tuple[DataAccessCollection, list[str]]:
        dac = DataAccessCollection(credentials={"known_credential_handle": self.make_credential(self.own_path)})
        return dac, ["known_credential_handle"]

    def own_dac(self) -> DataAccessCollection:
        return DataAccessCollection(credentials={"own_handle": self.make_credential(self.own_path)})

    def foreign_dac(self) -> DataAccessCollection:
        return DataAccessCollection(credentials={"foreign_handle": {"toyfmt_foreign_key": "toyfmt_value"}})

    # helpers

    def _make(self, name: str, tables: dict[str, dict[str, list[Any]]]) -> Path:
        path = self.tmp_path / f"{name}.db"
        self.write_database(path, tables)
        return path

    def _source_of(self, path: Path, table: str) -> str:
        return f"{self.identity_of(path)}::{table}"

    def _dac_of(self, *paths: Path, **extra: Any) -> DataAccessCollection:
        return DataAccessCollection(
            credentials={f"handle_{n}": self.make_credential(path, **extra) for n, path in enumerate(paths)}
        )

    def _spy(self, monkeypatch: pytest.MonkeyPatch) -> _Spy:
        """Record connections opened and closed and catalog calls from now on; the real hooks still run."""
        cls = self.feature_group_class
        spy = _Spy()
        original_connect = cls.connect
        original_close = cls.close_connection
        original_list = cls.list_tables
        original_columns = cls.table_columns

        def connect(klass: Any, credentials: Any) -> Any:
            connection = original_connect(credentials)
            spy.opened.append(connection)
            return connection

        def close_connection(klass: Any, connection: Any) -> None:
            spy.closed.append(connection)
            original_close(connection)

        def list_tables(klass: Any, connection: Any) -> Any:
            spy.list_tables += 1
            return original_list(connection)

        def table_columns(klass: Any, connection: Any, table: str) -> Any:
            spy.table_columns.append(table)
            return original_columns(connection, table)

        monkeypatch.setattr(cls, "connect", classmethod(connect))
        monkeypatch.setattr(cls, "close_connection", classmethod(close_connection))
        monkeypatch.setattr(cls, "list_tables", classmethod(list_tables))
        monkeypatch.setattr(cls, "table_columns", classmethod(table_columns))
        return spy

    def _pointed(self, name: str, value: Any, context: dict[str, Any] | None = None) -> Feature:
        return Feature(name, Options({self._group_name(): value}, context=context))

    # claiming

    def test_db_claims_via_a_credentials_handle_with_a_credential_free_source(self) -> None:
        feature = Feature(self.present_column)
        assert self._claims(feature, self.own_dac())
        match = self._claimed_match(feature)
        assert match.source == self._source_of(self.own_path, OWN_TABLE)
        assert isinstance(match.access, DBTable)
        assert match.access.table == OWN_TABLE

    @pytest.mark.parametrize(
        "wrap",
        [
            lambda mapping: dict(mapping),
            lambda mapping: Credential(mapping),
            lambda mapping: RegisteredCredential(mapping),
        ],
        ids=["dict", "credential", "registered_credential"],
    )
    def test_db_claims_via_a_pointer_in_each_credential_form_and_leaves_it_unchanged(self, wrap: Any) -> None:
        mapping = self.make_credential(self.own_path)
        value = wrap(mapping)
        before = copy.deepcopy(value.data if isinstance(value, Credential) else dict(value))
        feature = self._pointed(self.present_column, value)
        assert self._claims(feature, None)
        assert self._claimed_match(feature).source == self.expected_source
        assert (value.data if isinstance(value, Credential) else dict(value)) == before
        assert mapping == self.make_credential(self.own_path)

    def test_db_an_invalid_pointer_credential_declines_with_a_rejection_and_never_searches_the_collection(self) -> None:
        result = self._evaluate(self._pointed(self.present_column, {"toyfmt_foreign_key": 1}), self.own_dac())
        assert self.feature_group_class not in result.identified
        assert result.eliminations[self.feature_group_class].reason

    def test_db_foreign_credentials_stay_silent(self, rejection_window: dict[str, MatchRejection]) -> None:
        options = Options()
        assert not self.feature_group_class.match_feature_group_criteria(
            self.present_column, options, self.foreign_dac()
        )
        assert self._group_name() not in rejection_window

    # missing columns, tables and ambiguity

    def test_db_two_tables_with_the_column_abort_naming_both_and_the_fix_without_any_secret(self) -> None:
        path = self._make("two_tables", {"table_a": {self.present_column: [1]}, "table_b": {self.present_column: [2]}})
        dac = self._dac_of(path, password=SECRET)
        with pytest.raises(ValueError) as exc_info:
            self._resolve(Feature(self.present_column), dac)
        message = str(exc_info.value)
        assert self._source_of(path, "table_a") in message
        assert self._source_of(path, "table_b") in message
        assert self._group_name() in message
        assert "table_name" in message
        assert self.credential_key in message
        assert "narrow with a data_access_handle or column_to_file." not in message
        assert SECRET not in message

    def test_db_two_databases_with_the_column_abort_naming_both_and_the_handle_fix_without_any_secret(self) -> None:
        other = self._make("second_db", {"second_table": {self.present_column: [1]}})
        dac = DataAccessCollection(
            credentials={
                "db_a": self.make_credential(self.own_path, password=SECRET),
                "db_b": self.make_credential(other, password=SECRET),
            }
        )
        with pytest.raises(ValueError) as exc_info:
            self._resolve(Feature(self.present_column), dac)
        message = str(exc_info.value)
        assert self.expected_source in message
        assert self._source_of(other, "second_table") in message
        assert "data_access_handle" in message
        assert "db_a" in message
        assert "db_b" in message
        assert SECRET not in message

    def test_db_a_preset_table_name_restricts_the_lookup_to_that_table(self) -> None:
        path = self._make("preset", {"table_a": {self.present_column: [1]}, "table_b": {self.present_column: [2]}})
        dac = DataAccessCollection(credentials={"preset_handle": self.make_credential(path, table_name="table_b")})
        feature = Feature(self.present_column)
        assert self._claims(feature, dac)
        assert self._claimed_match(feature).source == self._source_of(path, "table_b")

    def test_db_a_preset_table_name_without_the_column_declines_and_aborts_when_pointed(self) -> None:
        path = self._make("preset_lacks", {"table_a": {"toyfmt_other_a": [1]}, "table_b": {self.present_column: [2]}})
        credential = self.make_credential(path, table_name="table_a")
        dac = DataAccessCollection(credentials={"preset_handle": credential})
        result = self._evaluate(Feature(self.present_column), dac)
        assert self.feature_group_class not in result.identified
        assert self._source_of(path, "table_a") in result.eliminations[self.feature_group_class].reason
        with pytest.raises(ValueError) as exc_info:
            self._resolve(self._pointed(self.present_column, credential), None)
        assert self._source_of(path, "table_a") in str(exc_info.value)

    def test_db_a_preset_table_name_missing_from_the_catalog_declines_and_aborts_when_pointed(self) -> None:
        credential = self.make_credential(self.own_path, table_name="toyfmt_no_such_table")
        dac = DataAccessCollection(credentials={"preset_handle": credential})
        result = self._evaluate(Feature(self.present_column), dac)
        assert self.feature_group_class not in result.identified
        assert "toyfmt_no_such_table" in result.eliminations[self.feature_group_class].reason
        with pytest.raises(ValueError, match="toyfmt_no_such_table"):
            self._resolve(self._pointed(self.present_column, credential), None)

    def test_db_the_same_database_under_two_handles_is_one_source(self) -> None:
        dac = DataAccessCollection(
            credentials={
                "first_handle": self.make_credential(self.own_path),
                "second_handle": self.make_credential(self.own_path),
            }
        )
        feature = Feature(self.present_column)
        assert self._claims(feature, dac)
        assert self._claimed_match(feature).source == self.expected_source

    def test_db_only_tables_with_the_column_are_considered(self) -> None:
        path = self._make("mixed", {"fits": {self.present_column: [1]}, "other": {"toyfmt_unrelated_column": [1]}})
        feature = Feature(self.present_column)
        assert self._claims(feature, self._dac_of(path))
        assert self._claimed_match(feature).source == self._source_of(path, "fits")

    # data_access_handle

    def test_db_handle_names_one_credential(self) -> None:
        other = self._make("handle_other", {"handle_table": {self.present_column: [2]}})
        dac = DataAccessCollection(
            credentials={"hand_a": self.make_credential(self.own_path), "hand_b": self.make_credential(other)}
        )
        feature = Feature(self.present_column, Options(context={HANDLE_OPTION: "hand_b"}))
        assert self._claims(feature, dac)
        assert self._claimed_match(feature).source == self._source_of(other, "handle_table")

    def test_db_a_foreign_credential_next_to_an_own_one_is_not_ambiguous(self) -> None:
        dac = DataAccessCollection(
            credentials={
                "own_handle": self.make_credential(self.own_path),
                "foreign_handle": {"toyfmt_foreign_key": "toyfmt_value"},
            }
        )
        feature = Feature(self.present_column)
        assert self._claims(feature, dac)
        assert self._claimed_match(feature).source == self.expected_source

    def test_db_a_handle_naming_a_foreign_credential_declines_instead_of_rescanning(self) -> None:
        dac = DataAccessCollection(
            credentials={
                "own_handle": self.make_credential(self.own_path),
                "foreign_handle": {"toyfmt_foreign_key": "toyfmt_value"},
            }
        )
        feature = Feature(self.present_column, Options(context={HANDLE_OPTION: "foreign_handle"}))
        assert not self._claims(feature, dac)

    def test_db_claims_a_typed_credential_registered_in_the_collection(self) -> None:
        dac = DataAccessCollection(credentials=Credential(self.make_credential(self.own_path)))
        feature = Feature(self.present_column)
        assert self._claims(feature, dac)
        assert self._claimed_match(feature).source == self.expected_source

    # per-run cache

    def test_db_one_run_reads_each_catalog_once_for_many_features_and_tables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = [f"{self.present_column}_cache_{n}" for n in ("a", "b", "c")]
        second = [f"{self.present_column}_cache_{n}" for n in ("d", "e")]
        path = self._make(
            "cache_db",
            {"table_one": {name: [1, 2] for name in first}, "table_two": {name: [3, 4] for name in second}},
        )
        dac = self._dac_of(path)
        spy = self._spy(monkeypatch)

        with run_match_cache():
            for column in (*first, *second, *first):
                assert self._claims(Feature(column), dac)

        assert len(spy.opened) == 1
        assert spy.list_tables == 1
        assert sorted(spy.table_columns) == ["table_one", "table_two"]
        assert spy.all_closed()

    def test_db_a_run_lists_the_tables_of_a_database_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        columns = [f"{self.present_column}_run_{n}" for n in ("a", "b", "c")]
        path = self._make("run_db", {"run_table": {name: [1, 2, 3] for name in columns}})
        spy = self._spy(monkeypatch)

        mloda.run_all(
            [*columns, f"{columns[0]}{CHAIN_SEPARATOR}sum_aggr"],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {self.feature_group_class, PyArrowAggregatedFeatureGroup}
            ),
            data_access_collection=self._dac_of(path),
        )

        assert spy.list_tables == 1
        assert spy.all_closed()

    # listing failures

    def _own_match(self) -> SourceMatch:
        credential = self.make_credential(self.own_path)
        return SourceMatch(source=self.expected_source, access=DBTable(credential, OWN_TABLE))

    def test_db_columns_of_a_match_are_the_columns_of_its_table(self) -> None:
        assert set(self.feature_group_class.columns(self._own_match()) or ()) == {self.present_column}

    def test_db_describe_columns_names_exactly_the_table_columns(self) -> None:
        described = self.feature_group_class.describe_columns(self._own_match())
        assert set(described) == {self.present_column}

    @pytest.mark.parametrize("error", [OSError("toyfmt os failure"), ValueError("toyfmt value failure")])
    def test_db_listing_failures_are_none_with_a_reason(
        self, error: Exception, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing(klass: Any, connection: Any) -> Any:
            raise error

        monkeypatch.setattr(self.feature_group_class, "list_tables", classmethod(failing))
        match = self._own_match()
        assert self.feature_group_class.columns(match) is None
        assert str(error) in (self.feature_group_class.unknown_columns_reason(match) or "")

    def test_db_a_type_error_from_list_tables_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def defect(klass: Any, connection: Any) -> Any:
            raise TypeError("toyfmt code defect")

        monkeypatch.setattr(self.feature_group_class, "list_tables", classmethod(defect))
        with pytest.raises(TypeError, match="toyfmt code defect"):
            self.feature_group_class.columns(self._own_match())

    def test_db_a_marked_match_abort_from_list_tables_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def abort(klass: Any, connection: Any) -> Any:
            raise escalate_match_abort(ValueError("toyfmt marked abort"))

        monkeypatch.setattr(self.feature_group_class, "list_tables", classmethod(abort))
        with pytest.raises(ValueError, match="toyfmt marked abort"):
            self.feature_group_class.columns(self._own_match())

    def test_db_a_failing_listing_still_closes_the_connection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        spy = self._spy(monkeypatch)
        original = self.feature_group_class.list_tables

        def failing(klass: Any, connection: Any) -> Any:
            original(connection)
            raise OSError("toyfmt os failure")

        monkeypatch.setattr(self.feature_group_class, "list_tables", classmethod(failing))
        assert self.feature_group_class.columns(self._own_match()) is None
        assert spy.opened
        assert spy.all_closed()

    def test_db_corrupt_database_declines_unpointed_and_aborts_pointed(self) -> None:
        path = self.tmp_path / "corrupt.db"
        self.write_corrupt_database(path)
        identity = self.identity_of(path)

        result = self._evaluate(Feature(self.present_column), self._dac_of(path))
        assert self.feature_group_class not in result.identified
        reason = result.eliminations[self.feature_group_class].reason
        assert identity in reason
        assert "could not read" in reason

        with pytest.raises(ValueError) as exc_info:
            self._resolve(self._pointed(self.present_column, self.make_credential(path)), None)
        assert identity in str(exc_info.value)
        assert "could not read" in str(exc_info.value)

    def test_db_a_missing_database_declines_unpointed_aborts_pointed_and_is_never_created(self) -> None:
        path = self.tmp_path / "never_created.db"
        identity = self.identity_of(path)

        result = self._evaluate(Feature(self.present_column), self._dac_of(path))
        assert self.feature_group_class not in result.identified
        assert identity in result.eliminations[self.feature_group_class].reason

        with pytest.raises(ValueError) as exc_info:
            self._resolve(self._pointed(self.present_column, self.make_credential(path)), None)
        assert identity in str(exc_info.value)
        assert not path.exists()

    def test_db_a_corrupt_database_next_to_a_good_one_does_not_block_the_good_one(self) -> None:
        corrupt = self.tmp_path / "corrupt_sibling.db"
        self.write_corrupt_database(corrupt)
        dac = self._dac_of(corrupt, self.own_path)
        feature = Feature(self.present_column)
        assert self._claims(feature, dac)
        assert self._claimed_match(feature).source == self.expected_source

    # secrets

    def test_db_match_and_rejections_hold_no_credential_values(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        credential = self.make_credential(self.own_path, password=SECRET)
        feature = self._pointed(self.present_column, credential)
        assert self._claims(feature, None)
        assert SECRET not in repr(self._claimed_match(feature))

        missing = Feature(self.missing_column, Options(context={HANDLE_OPTION: "toyfmt_unknown_handle"}))
        dac = DataAccessCollection(credentials={"secret_handle": credential})
        assert not self.feature_group_class.match_feature_group_criteria(self.missing_column, missing.options, dac)
        assert SECRET not in "".join(entry.reason for entry in rejection_window.values())

    # names

    def test_db_a_chain_shaped_name_resolves_to_the_aggregation_only(self) -> None:
        column = f"{self.present_column}_agg"
        path = self._make("agg_db", {"agg_table": {column: [1, 2, 3]}})
        result = mloda.run_all(
            [f"{column}{CHAIN_SEPARATOR}sum_aggr"],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {self.feature_group_class, PyArrowAggregatedFeatureGroup}
            ),
            data_access_collection=self._dac_of(path),
        )
        assert result[0].to_pydict()[f"{column}{CHAIN_SEPARATOR}sum_aggr"] == [6, 6, 6]

    # loading

    def test_db_load_returns_exactly_the_requested_columns_and_closes_every_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = self._make("load_db", {"load_table": {"toyfmt_a": [1, 2], "toyfmt_b": [3, 4], "toyfmt_c": [5, 6]}})
        spy = self._spy(monkeypatch)

        result = mloda.run_all(
            ["toyfmt_a", "toyfmt_c"],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({self.feature_group_class}),
            data_access_collection=self._dac_of(path),
        )

        assert len(result) == 1
        assert result[0].to_pydict() == {"toyfmt_a": [1, 2], "toyfmt_c": [5, 6]}
        assert len(spy.opened) >= 2
        assert spy.all_closed()

    @pytest.mark.parametrize("batches", [[["toyfmt_left", "toyfmt_right"]], [["toyfmt_left"], ["toyfmt_right"]]])
    def test_db_tables_per_match_leave_the_shared_credential_untouched(self, batches: list[list[str]]) -> None:
        path = self._make("shared_db", {"left_table": {"toyfmt_left": [1, 2]}, "right_table": {"toyfmt_right": [7]}})
        dac = self._dac_of(path)
        before = copy.deepcopy(dac.credentials)
        seen: dict[str, list[Any]] = {}
        for batch in batches:
            for table in mloda.run_all(
                list[Feature | str](batch),
                compute_frameworks=[PyArrowTable],
                plugin_collector=PluginCollector.enabled_feature_groups({self.feature_group_class}),
                data_access_collection=dac,
            ):
                seen.update(table.to_pydict())
        assert seen == {"toyfmt_left": [1, 2], "toyfmt_right": [7]}
        assert dac.credentials == before

    def test_db_input_data_load_identity_ignores_credential_values(self) -> None:
        path = self._make("identity_db", {"identity_table": {"toyfmt_identity": [1]}})
        capture = _LoadCapture()

        mloda.run_all(
            ["toyfmt_identity"],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({self.feature_group_class}),
            data_access_collection=self._dac_of(path, password=SECRET),
            function_extender={capture},
        )

        assert len(capture.contexts) == 1
        identity = capture.contexts[0].data_access_identity
        assert identity == self._source_of(path, "identity_table")
        assert SECRET not in str(identity)

    # subclass takeover
