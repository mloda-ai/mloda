"""SqliteFG: the shared database contract, framework loads, and the SQLite-specific connection and query behaviour."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.abstract_plugins.components.input_data.read_db_fg import DBTable
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import FeatureSet, HashableDict
from mloda.user import (
    Credential,
    DataAccessCollection,
    DataType,
    Feature,
    Options,
    ParallelizationMode,
    PluginCollector,
    PluginLoader,
    mloda,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.input_data.db_formats.sqlite_fg import SqliteFG
from tests.mixins.compute_frameworks.framework_adapter_mixins import (
    DatabaseLoadsIntoFrameworkMixin,
    PandasDataFrameAdapter,
    PolarsDataFrameAdapter,
    PyArrowTableAdapter,
    PythonDictAdapter,
)
from tests.mixins.reader_feature_groups.database_format_feature_group_test_mixin import (
    DatabaseFormatFeatureGroupTestMixin,
)
from tests.mixins.reader_feature_groups.format_file_writers import write_sqlite
from tests.test_core.test_integration.test_core.test_runner_one_compute_framework import SumFeature

KEY = "sqlite"


class TestSqliteFGContract(DatabaseFormatFeatureGroupTestMixin):
    feature_group_class = SqliteFG
    credential_key = KEY
    present_column = "impl_sqlite_present"
    missing_column = "impl_sqlite_missing"

    def write_database(self, path: Path, tables: dict[str, dict[str, list[Any]]]) -> None:
        write_sqlite(path, tables)

    def test_sqlite_the_same_database_by_relative_and_absolute_path_is_one_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(self.tmp_path)
        dac = DataAccessCollection(
            credentials={
                "relative_handle": {KEY: os.path.relpath(self.own_path, self.tmp_path)},
                "absolute_handle": {KEY: str(self.own_path)},
            }
        )
        feature = Feature(self.present_column)
        assert self._claims(feature, dac)
        assert self._matched_source(feature).source == self.expected_source


class TestSqliteLoadsIntoPyArrowTable(PyArrowTableAdapter, DatabaseLoadsIntoFrameworkMixin):
    db_group = SqliteFG
    credential_key = KEY
    expected_loader = "neutral"

    def write_database(self, path: Path, tables: dict[str, dict[str, list[Any]]]) -> None:
        write_sqlite(path, tables)

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result.column(column).to_pylist())


class TestSqliteLoadsIntoPythonDict(PythonDictAdapter, DatabaseLoadsIntoFrameworkMixin):
    db_group = SqliteFG
    credential_key = KEY
    expected_loader = "PythonDictFramework"

    def write_database(self, path: Path, tables: dict[str, dict[str, list[Any]]]) -> None:
        write_sqlite(path, tables)

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result[column])


class TestSqliteLoadsIntoPandas(PandasDataFrameAdapter, DatabaseLoadsIntoFrameworkMixin):
    db_group = SqliteFG
    credential_key = KEY
    expected_loader = "neutral"

    def write_database(self, path: Path, tables: dict[str, dict[str, list[Any]]]) -> None:
        write_sqlite(path, tables)

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result[column].tolist())


class TestSqliteLoadsIntoPolars(PolarsDataFrameAdapter, DatabaseLoadsIntoFrameworkMixin):
    db_group = SqliteFG
    credential_key = KEY
    expected_loader = "neutral"

    def write_database(self, path: Path, tables: dict[str, dict[str, list[Any]]]) -> None:
        write_sqlite(path, tables)

    def values_of(self, result: Any, column: str) -> list[Any]:
        return list(result[column].to_list())


@pytest.fixture()
def people_db(tmp_path: Path) -> str:
    path = tmp_path / "people.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    connection.executemany(
        "INSERT INTO test_table (name, age) VALUES (?, ?)", [("Alice", 30), ("Bob", 25), ("Charlie", 35)]
    )
    connection.execute("CREATE TABLE orders (order_amount INTEGER)")
    connection.execute("INSERT INTO orders (order_amount) VALUES (5)")
    connection.commit()
    connection.close()
    return str(path)


def _match(path: str, table: str) -> SourceMatch:
    return SourceMatch(source=f"{os.path.abspath(path)}::{table}", access=DBTable({KEY: path}, table))


def _features(*features: Feature) -> FeatureSet:
    feature_set = FeatureSet()
    for feature in features:
        feature_set.add(feature)
    return feature_set


class TestSqliteCredentials:
    @pytest.mark.parametrize(
        "credentials, expected",
        [
            ({KEY: "some.db"}, True),
            ({KEY: "some.db", "user": "alice"}, True),
            ({KEY: Path("some.db")}, False),
            ({KEY: 5}, False),
            ({"user": "alice"}, False),
            ({}, False),
            (None, False),
            ("some.db", False),
        ],
    )
    def test_is_valid_credentials_checks_only_the_own_key_and_never_raises(
        self, credentials: Any, expected: bool
    ) -> None:
        assert SqliteFG.is_valid_credentials(credentials) is expected

    def test_a_missing_file_is_not_an_invalid_credential(self, tmp_path: Path) -> None:
        assert SqliteFG.is_valid_credentials({KEY: str(tmp_path / "missing.db")}) is True

    def test_database_identity_is_the_absolute_path_and_carries_no_other_credential_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        identity = SqliteFG.database_identity({KEY: "rel.db", "password": "toyfmt-secret-value"})  # nosec B105
        assert identity == os.path.abspath("rel.db")

    def test_a_path_credential_declines_and_creates_no_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.db"
        feature = Feature("t_col", Options({SqliteFG.get_class_name(): {KEY: missing}}))
        result = IdentifyFeatureGroupClass.evaluate(feature, {SqliteFG: {PyArrowTable}}, None, None)
        assert SqliteFG not in result.identified
        assert not missing.exists()


class TestSqliteConnection:
    def test_connect_returns_a_read_only_connection(self, people_db: str) -> None:
        connection = SqliteFG.connect({KEY: people_db})
        try:
            assert isinstance(connection, sqlite3.Connection)
            with pytest.raises(sqlite3.OperationalError):
                connection.execute("CREATE TABLE written (a)")
        finally:
            connection.close()

    def test_connect_never_creates_a_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.db"
        with pytest.raises(sqlite3.Error):
            SqliteFG.connect({KEY: str(missing)})
        assert not missing.exists()

    def test_a_path_with_a_space_and_a_hash_connects_and_loads(self, tmp_path: Path) -> None:
        folder = tmp_path / "dir with space #1"
        folder.mkdir()
        path = folder / "odd name.db"
        write_sqlite(path, {"odd_table": {"odd_column": [4, 5]}})

        result = mloda.run_all(
            ["odd_column"],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({SqliteFG}),
            data_access_collection=DataAccessCollection(credentials={"odd_handle": {KEY: str(path)}}),
        )

        assert result[0].to_pydict() == {"odd_column": [4, 5]}

    def test_list_tables_and_table_columns_read_the_catalog(self, people_db: str) -> None:
        connection = SqliteFG.connect({KEY: people_db})
        try:
            assert set(SqliteFG.list_tables(connection)) == {"test_table", "orders"}
            assert list(SqliteFG.table_columns(connection, "test_table")) == ["id", "name", "age"]
        finally:
            connection.close()

    def test_a_table_name_with_a_quote_and_a_space_is_listed_described_and_loaded(self, tmp_path: Path) -> None:
        path = tmp_path / "quoted.db"
        write_sqlite(path, {'we"ird table': {"quoted_col": [1, 2]}})
        dac = DataAccessCollection(credentials={"quoted_handle": {KEY: str(path)}})

        result = mloda.run_all(
            ["quoted_col"],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({SqliteFG}),
            data_access_collection=dac,
        )

        assert result[0].to_pydict() == {"quoted_col": [1, 2]}


class TestSqliteProduceRows:
    def _produce(self, path: str, table: str, *features: Feature) -> Any:
        connection = SqliteFG.connect({KEY: path})
        try:
            return SqliteFG.produce_rows(connection, table, _features(*features))
        finally:
            connection.close()

    def test_columns_come_back_sorted_with_their_rows(self, people_db: str) -> None:
        names = ["name", "id", "age"]
        table = self._produce(people_db, "test_table", *(Feature(name) for name in names))
        assert isinstance(table, pa.Table)
        assert table.column_names == sorted(names)
        assert table.to_pydict() == {"age": [30, 25, 35], "id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}

    def test_identifiers_are_quoted_so_a_crafted_name_cannot_inject_sql(self, people_db: str) -> None:
        connection = sqlite3.connect(people_db)
        connection.execute("CREATE TABLE secrets (api_key TEXT)")
        connection.execute("INSERT INTO secrets VALUES ('SUPER_SECRET_KEY')")
        connection.commit()
        connection.close()

        malicious = "name FROM test_table UNION SELECT api_key FROM secrets --"
        table = self._produce(people_db, "test_table", Feature(malicious))

        leaked = {cell for column in table.to_pydict().values() for cell in column}
        assert "SUPER_SECRET_KEY" not in leaked

    def test_a_crafted_table_name_selects_nothing_and_drops_nothing(self, people_db: str) -> None:
        with pytest.raises(sqlite3.Error):
            self._produce(people_db, "test_table; DROP TABLE test_table; --", Feature("id"))
        assert self._produce(people_db, "test_table", Feature("id")).to_pydict() == {"id": [1, 2, 3]}

    def test_declared_types_win_over_inference(self, people_db: str) -> None:
        table = self._produce(people_db, "test_table", Feature.int32_of("id"), Feature.int64_of("age"))
        assert pa.types.is_int32(table.schema.field("id").type)
        assert pa.types.is_int64(table.schema.field("age").type)

    def test_untyped_features_are_inferred_from_the_data(self, people_db: str) -> None:
        table = self._produce(people_db, "test_table", Feature.not_typed("id"), Feature.not_typed("name"))
        assert pa.types.is_integer(table.schema.field("id").type)
        assert pa.types.is_string(table.schema.field("name").type)

    def test_declared_and_inferred_types_mix(self, people_db: str) -> None:
        table = self._produce(
            people_db, "test_table", Feature.int64_of("id"), Feature.not_typed("age"), Feature.str_of("name")
        )
        assert pa.types.is_int64(table.schema.field("id").type)
        assert pa.types.is_integer(table.schema.field("age").type)
        assert pa.types.is_string(table.schema.field("name").type)


class TestSqliteDescribeColumns:
    @pytest.fixture()
    def affinity_db(self, tmp_path: Path) -> str:
        path = tmp_path / "affinity.db"
        connection = sqlite3.connect(path)
        connection.execute(
            """
            CREATE TABLE affinity_table (
                price REAL, weight FLOAT, distance DOUBLE, payload BLOB, label VARCHAR(255), notes CLOB,
                untyped_col, amount NUMERIC, precise DECIMAL(10,5), text_then_blob TEXT BLOB,
                blob_sub_type_text BLOB SUB_TYPE TEXT, char_then_double CHAR DOUBLE, double_then_blob DOUBLE BLOB
            )
            """
        )
        connection.commit()
        connection.close()
        return str(path)

    def test_declared_types_map_to_data_types(self, people_db: str) -> None:
        result = SqliteFG.describe_columns(_match(people_db, "test_table"))
        assert result == {"id": DataType.INT64, "name": DataType.STRING, "age": DataType.INT64}

    def test_affinity_edge_cases_resolve_by_sqlite_precedence(self, affinity_db: str) -> None:
        result = SqliteFG.describe_columns(_match(affinity_db, "affinity_table"))

        assert result["price"] == DataType.DOUBLE
        assert result["weight"] == DataType.DOUBLE
        assert result["distance"] == DataType.DOUBLE
        assert result["payload"] == DataType.BINARY
        assert result["label"] == DataType.STRING
        assert result["notes"] == DataType.STRING
        assert result["untyped_col"] is None
        assert result["amount"] is None
        assert result["precise"] is None
        assert result["text_then_blob"] == DataType.STRING
        assert result["blob_sub_type_text"] == DataType.STRING
        assert result["char_then_double"] == DataType.STRING
        assert result["double_then_blob"] == DataType.BINARY

    def test_an_unknown_table_raises_naming_it(self, people_db: str) -> None:
        with pytest.raises(ValueError, match="does_not_exist"):
            SqliteFG.describe_columns(_match(people_db, "does_not_exist"))

    def test_a_missing_database_is_not_created(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.db"
        with pytest.raises((OSError, ValueError, sqlite3.Error)):
            SqliteFG.describe_columns(_match(str(missing), "t"))
        assert not missing.exists()

    def test_a_crafted_preset_table_name_declines_and_leaves_the_table_intact(self, people_db: str) -> None:
        credential = {KEY: people_db, "table_name": "test_table); DROP TABLE test_table; --"}
        result = IdentifyFeatureGroupClass.evaluate(
            Feature("id"),
            {SqliteFG: {PyArrowTable}},
            None,
            DataAccessCollection(credentials={"crafted_handle": credential}),
        )
        assert SqliteFG not in result.identified
        assert SqliteFG.describe_columns(_match(people_db, "test_table")) == {
            "id": DataType.INT64,
            "name": DataType.STRING,
            "age": DataType.INT64,
        }


class TestSqliteFGRunAll:
    @pytest.mark.parametrize(
        "wrap",
        [lambda mapping: mapping, lambda mapping: Credential(mapping)],
        ids=["plain_dict", "credential"],
    )
    def test_a_pointer_with_a_table_name_loads_that_table(self, people_db: str, wrap: Any) -> None:
        feature = Feature(
            name="id", options={SqliteFG.get_class_name(): wrap({KEY: people_db, "table_name": "test_table"})}
        )

        result = mloda.run_all(
            [feature],
            compute_frameworks=["PyArrowTable"],
            plugin_collector=PluginCollector.enabled_feature_groups({SqliteFG}),
        )

        assert result[0].to_pydict() == {"id": [1, 2, 3]}

    def test_a_credentials_list_in_the_collection_loads_its_columns(self, people_db: str) -> None:
        result = mloda.run_all(
            ["name", "id"],
            compute_frameworks=["PyArrowTable"],
            data_access_collection=DataAccessCollection(credentials=[{KEY: people_db}]),
            plugin_collector=PluginCollector.enabled_feature_groups({SqliteFG}),
        )
        assert result[0].to_pydict() == {"name": ["Alice", "Bob", "Charlie"], "id": [1, 2, 3]}

    def test_a_column_found_in_two_tables_aborts_instead_of_taking_the_first(self, people_db: str) -> None:
        connection = sqlite3.connect(people_db)
        connection.execute("CREATE TABLE test_table_2 (id INTEGER PRIMARY KEY, name TEXT)")
        connection.commit()
        connection.close()

        with pytest.raises(ValueError, match="test_table_2"):
            mloda.run_all(
                ["name"],
                compute_frameworks=["PyArrowTable"],
                data_access_collection=DataAccessCollection(credentials=[{KEY: people_db}]),
                plugin_collector=PluginCollector.enabled_feature_groups({SqliteFG}),
            )

    def test_an_aggregation_over_a_collection_credential(self, people_db: str) -> None:
        feature = Feature(name="sum_of_", options={"sum": ("id", "id")})

        result = mloda.run_all(
            [feature],
            compute_frameworks=["PyArrowTable"],
            data_access_collection=DataAccessCollection(credentials=[{KEY: people_db}]),
            plugin_collector=PluginCollector.enabled_feature_groups({SqliteFG, SumFeature}),
        )

        assert "SumFeature_idid" in result[0].to_pydict()

    def test_run_all_closes_every_sqlite_connection(self, people_db: str, monkeypatch: pytest.MonkeyPatch) -> None:
        opened: list[sqlite3.Connection] = []
        original_connect = SqliteFG.connect

        def tracking_connect(credentials: Any) -> Any:
            connection = original_connect(credentials)
            opened.append(connection)
            return connection

        monkeypatch.setattr(SqliteFG, "connect", tracking_connect)

        mloda.run_all(
            ["name", "id"],
            compute_frameworks=["PyArrowTable"],
            data_access_collection=DataAccessCollection(credentials=[{KEY: people_db}]),
            plugin_collector=PluginCollector.enabled_feature_groups({SqliteFG}),
        )

        assert opened, "expected SqliteFG.connect to be invoked at least once"
        for connection in opened:
            with pytest.raises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")


class TestSqliteFGMultiprocessing:
    PluginLoader().all()

    def test_hashable_dict_credentials_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError) as excinfo:
            DataAccessCollection(credentials=[HashableDict({KEY: str(tmp_path / "mp_hashable.sqlite")})])
        assert "credential" in str(excinfo.value).lower()

    def test_plain_dict_credentials_under_multiprocessing(self, people_db: str, flight_server: Any) -> None:
        result = mloda.run_all(
            ["name", "id"],
            compute_frameworks=["PyArrowTable"],
            parallelization_modes={ParallelizationMode.MULTIPROCESSING},
            flight_server=flight_server,
            data_access_collection=DataAccessCollection(credentials=[{KEY: people_db}]),
            plugin_collector=PluginCollector.enabled_feature_groups({SqliteFG}),
        )

        assert result[0].to_pydict() == {"name": ["Alice", "Bob", "Charlie"], "id": [1, 2, 3]}
