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
from mloda.core.filter.filter_engine import BaseFilterEngine
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import FeatureSet, HashableDict, ReadDBFG
from mloda.user import (
    Credential,
    DataAccessCollection,
    DataType,
    Feature,
    FilterType,
    Options,
    ParallelizationMode,
    PluginCollector,
    PluginLoader,
    SingleFilter,
    mloda,
)
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_filter_engine import PyArrowFilterEngine
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.feature_group.input_data.db_sources.sqlite_fg import SqliteFG
from tests.mixins.compute_frameworks.framework_adapter_mixins import (
    DatabaseLoadsIntoFrameworkMixin,
    PandasDataFrameAdapter,
    PolarsDataFrameAdapter,
    PyArrowTableAdapter,
    PythonDictAdapter,
)
from tests.mixins.reader_feature_groups.read_db_fg_test_mixin import (
    ReadDBFGTestMixin,
)
from tests.mixins.reader_feature_groups.format_file_writers import write_sqlite
from tests.test_core.test_integration.test_core.test_runner_one_compute_framework import SumFeature

KEY = "sqlite"


class TestSqliteFGContract(ReadDBFGTestMixin):
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
        assert self._claimed_match(feature).source == self.expected_source


QUERY_KEY = "toyfmt_sqlite_query"


class SqliteQueryToyfmtFG(SqliteFG):
    """Test-local SqliteFG that opts into the query route; its own credential key keeps it inert elsewhere and makes the database contract cover loaders a subclass inherits."""

    CREDENTIAL_KEY = QUERY_KEY
    CLAIM_ROUTES = (*ReadDBFG.CLAIM_ROUTES, ReadDBFG.QUERY_ROUTE)

    @classmethod
    def produce_query_rows(cls, connection: Any, query_text: str, features: Any) -> Any:
        cursor = connection.execute(query_text)
        names = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        wanted = sorted(features.get_all_names())
        return pa.table({name: [row[names.index(name)] for row in rows] for name in wanted})


class TestSqliteQueryRouteContract(ReadDBFGTestMixin):
    feature_group_class = SqliteQueryToyfmtFG
    credential_key = QUERY_KEY
    present_column = "impl_sqlite_query_present"
    missing_column = "impl_sqlite_query_missing"

    def write_database(self, path: Path, tables: dict[str, dict[str, list[Any]]]) -> None:
        write_sqlite(path, tables)

    def query_for(self, table: str, columns: list[str]) -> str:
        return f"SELECT {', '.join(quote_ident(c) for c in columns)} FROM {quote_ident(table)}"  # nosec B608  # quoted names


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
    expected_loader = "mloda_plugins.feature_group.input_data.db_sources.sqlite_fg:SqlitePythonDict"

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


class TestSqliteProduceRowsInference:
    def _produce(self, tmp_path: Path, values: list[Any], feature: Feature) -> pa.Table:
        path = tmp_path / "infer.db"
        write_sqlite(path, {"infer_table": {"infer_col": values}})
        connection = SqliteFG.connect({KEY: str(path)})
        try:
            table = SqliteFG.produce_rows(connection, "infer_table", _features(feature))
        finally:
            connection.close()
        assert isinstance(table, pa.Table)
        return table

    @pytest.mark.parametrize(
        "values",
        [[None, 5, 7], [None, "a", "b"], [None, 1.5, 2.5], [1, None, 3]],
        ids=["int", "str", "float", "null_in_middle"],
    )
    def test_a_null_in_the_first_row_does_not_break_inference(self, tmp_path: Path, values: list[Any]) -> None:
        table = self._produce(tmp_path, values, Feature.not_typed("infer_col"))
        assert table.column("infer_col").to_pylist() == values
        assert not pa.types.is_null(table.schema.field("infer_col").type)

    @pytest.mark.parametrize(
        "values, check",
        [
            ([1, 2.5], pa.types.is_floating),
            ([1, 2, 3_000_000_000], pa.types.is_int64),
        ],
        ids=["int_then_float", "int_then_large_int"],
    )
    def test_a_later_value_widens_the_inferred_type(self, tmp_path: Path, values: list[Any], check: Any) -> None:
        table = self._produce(tmp_path, values, Feature.not_typed("infer_col"))
        assert table.column("infer_col").to_pylist() == values
        assert check(table.schema.field("infer_col").type)

    def test_an_all_null_column_loads_as_null_type(self, tmp_path: Path) -> None:
        table = self._produce(tmp_path, [None, None], Feature.not_typed("infer_col"))
        assert table.column("infer_col").to_pylist() == [None, None]
        assert pa.types.is_null(table.schema.field("infer_col").type)

    def test_a_declared_type_still_wins_over_a_leading_null(self, tmp_path: Path) -> None:
        table = self._produce(tmp_path, [None, 5], Feature.int32_of("infer_col"))
        assert pa.types.is_int32(table.schema.field("infer_col").type)
        assert table.column("infer_col").to_pylist() == [None, 5]

    def test_a_leading_null_table_loads_through_run_all(self, tmp_path: Path) -> None:
        path = tmp_path / "leading_null.db"
        write_sqlite(path, {"leading_null_table": {"leading_null_col": [None, 1, 2.5]}})

        result = mloda.run_all(
            ["leading_null_col"],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({SqliteFG}),
            data_access_collection=DataAccessCollection(credentials={"null_handle": {KEY: str(path)}}),
        )

        assert result[0].to_pydict() == {"leading_null_col": [None, 1.0, 2.5]}

    def test_a_leading_null_table_loads_through_run_all_into_pandas(self, tmp_path: Path) -> None:
        path = tmp_path / "leading_null_pd.db"
        write_sqlite(path, {"leading_null_pd_table": {"leading_null_pd_col": [None, 3, 4]}})

        result = mloda.run_all(
            ["leading_null_pd_col"],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups({SqliteFG}),
            data_access_collection=DataAccessCollection(credentials={"null_pd_handle": {KEY: str(path)}}),
        )

        values = result[0]["leading_null_pd_col"].tolist()
        assert values[0] != values[0] or values[0] is None
        assert values[1:] == [3, 4]


class NoRecheckSqliteFG(SqliteFG):
    """Test-local SqliteFG that opts out of the post-read filter pass; its own credential key keeps it inert."""

    CREDENTIAL_KEY = "norecheck_sqlite"

    @classmethod
    def final_filters(cls) -> bool | None:
        return False


def _filtered_features(features: list[Feature], filters: list[SingleFilter], engine: Any = PyArrowFilterEngine) -> Any:
    feature_set = _features(*features)
    feature_set.add_filters(set(filters))
    feature_set.filter_engine = engine
    return feature_set


def _flt(column: str, filter_type: FilterType, parameter: dict[str, Any]) -> SingleFilter:
    return SingleFilter(Feature(column), filter_type, parameter)


T = FilterType
TYPED_COLUMNS = [
    Feature.int64_of("id"),
    Feature.int64_of("i"),
    Feature.double_of("r"),
    Feature.str_of("t"),
    Feature.int64_of("n"),
    Feature.str_of("nc"),
]
ALL_IDS = [1, 2, 3, 4, 5]
BIG = 2**53 + 1

PUSHED = [
    ("int_equal", "i", T.EQUAL, {"value": 3}, [3]),
    ("int_min", "i", T.MIN, {"value": 4}, [4, 5]),
    ("int_max_inclusive", "i", T.MAX, {"value": 2}, [1, 2]),
    ("int_max_exclusive", "i", T.MAX, {"max": 2, "max_exclusive": True}, [1]),
    ("int_range", "i", T.RANGE, {"min": 2, "max": 4}, [2, 3, 4]),
    ("int_range_exclusive", "i", T.RANGE, {"min": 2, "max": 4, "max_exclusive": True}, [2, 3]),
    ("int_inclusion", "i", T.CATEGORICAL_INCLUSION, {"values": [1, 5]}, [1, 5]),
    ("real_equal_float", "r", T.EQUAL, {"value": 2.5}, [2]),
    ("real_min_float", "r", T.MIN, {"value": 4.5}, [4, 5]),
    ("real_range_int_bounds", "r", T.RANGE, {"min": 2, "max": 4}, [2, 3]),
    ("text_equal", "t", T.EQUAL, {"value": "c"}, [3]),
    ("text_inclusion", "t", T.CATEGORICAL_INCLUSION, {"values": ["a", "e"]}, [1, 5]),
    ("nocase_equal_is_binary", "nc", T.EQUAL, {"value": "abc"}, [2, 5]),
]

NOT_PUSHED = [
    ("regex", "t", T.REGEX, {"value": "^a"}),
    ("text_range", "t", T.RANGE, {"min": "b", "max": "d"}),
    ("text_min", "t", T.MIN, {"value": "c"}),
    ("bool_bound", "i", T.EQUAL, {"value": True}),
    ("big_int_bound", "i", T.MIN, {"value": BIG}),
    ("big_negative_int_bound", "i", T.MAX, {"value": -BIG}),
    ("float_bound_on_integer_column", "i", T.MIN, {"value": 2.5}),
    ("numeric_column", "n", T.EQUAL, {"value": 3}),
    ("inclusion_with_none", "i", T.CATEGORICAL_INCLUSION, {"values": [1, None]}),
    ("inclusion_with_nan", "r", T.CATEGORICAL_INCLUSION, {"values": [1.5, float("nan")]}),
    ("inclusion_empty", "i", T.CATEGORICAL_INCLUSION, {"values": []}),
    ("inclusion_over_900", "i", T.CATEGORICAL_INCLUSION, {"values": list(range(10, 1010))}),
    ("inclusion_text_on_integer_column", "i", T.CATEGORICAL_INCLUSION, {"values": ["1"]}),
]


@pytest.fixture()
def filter_db(tmp_path: Path) -> str:
    path = tmp_path / "filters.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE ft (id INTEGER, i INTEGER, r REAL, t TEXT, n NUMERIC, nc TEXT COLLATE NOCASE)")
    rows = [
        (1, 1, 1.5, "a", 1, "Abc"),
        (2, 2, 2.5, "b", 2, "abc"),
        (3, 3, 3.5, "c", 3, "ABC"),
        (4, 4, 4.5, "d", 4, "xyz"),
        (5, 5, 5.5, "e", 5, "abc"),
    ]
    connection.executemany("INSERT INTO ft VALUES (?, ?, ?, ?, ?, ?)", rows)
    connection.execute("CREATE TABLE mixed (id INTEGER, v INTEGER)")
    connection.executemany("INSERT INTO mixed VALUES (?, ?)", [(1, 5), (2, "abc"), (3, 10)])
    connection.execute("CREATE TABLE loose (id INTEGER, u, num NUMERIC, i INTEGER)")
    connection.executemany("INSERT INTO loose VALUES (?, ?, ?, ?)", [(1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3)])
    connection.execute("CREATE TABLE wild (id INTEGER, mi INTEGER, mt TEXT)")
    connection.executemany("INSERT INTO wild VALUES (?, ?, ?)", [(1, 1, "a"), (2, 2.5, b"\x01\x02"), (3, 3, "c")])
    connection.execute("CREATE TABLE kinds (id, a, b, c, d, e)")
    connection.executemany(
        "INSERT INTO kinds VALUES (?, ?, ?, ?, ?, ?)",
        [(1, 1, 1.5, "x", b"\x01", None), (2, 2, 2.5, "y", b"\x02", None)],
    )
    connection.commit()
    connection.close()
    return str(path)


def _ids(db: str, table: str, features: Any, group: type[SqliteFG] = SqliteFG) -> list[int]:
    connection = SqliteFG.connect({KEY: db})
    try:
        names, rows = group.select_rows(connection, table, features)
    finally:
        connection.close()
    return [row[names.index("id")] for row in rows]


class TestSqliteFilterPushdown:
    @pytest.mark.parametrize("case", PUSHED, ids=[case[0] for case in PUSHED])
    def test_a_pushed_filter_fetches_only_the_matching_rows(self, filter_db: str, case: Any) -> None:
        _, column, filter_type, parameter, expected = case
        features = _filtered_features(TYPED_COLUMNS, [_flt(column, filter_type, parameter)])
        assert _ids(filter_db, "ft", features) == expected

    @pytest.mark.parametrize("case", NOT_PUSHED, ids=[case[0] for case in NOT_PUSHED])
    def test_a_filter_that_is_not_pushed_leaves_every_row_to_the_post_read_pass(
        self, filter_db: str, case: Any
    ) -> None:
        _, column, filter_type, parameter = case
        features = _filtered_features(TYPED_COLUMNS, [_flt(column, filter_type, parameter)])
        assert _ids(filter_db, "ft", features) == ALL_IDS

    def test_a_filter_on_a_column_the_step_does_not_request_is_irrelevant(self, filter_db: str) -> None:
        features = _filtered_features([Feature.int64_of("id")], [_flt("i", T.EQUAL, {"value": 3})])
        assert _ids(filter_db, "ft", features) == ALL_IDS

    def test_no_filter_fetches_every_row(self, filter_db: str) -> None:
        assert _ids(filter_db, "ft", _features(*TYPED_COLUMNS)) == ALL_IDS

    def test_a_text_value_in_an_integer_column_does_not_break_a_pushed_max(self, filter_db: str) -> None:
        features = _filtered_features([Feature.int64_of("id"), Feature.int64_of("v")], [_flt("v", T.MAX, {"value": 7})])
        assert _ids(filter_db, "mixed", features) == [1]

    def test_an_untyped_numeric_column_holding_one_storage_class_does_not_block_the_pushdown(
        self, filter_db: str
    ) -> None:
        features = _filtered_features(
            [Feature.not_typed("id"), Feature.not_typed("i"), Feature.not_typed("num")],
            [_flt("i", T.MIN, {"value": 3})],
        )
        assert _ids(filter_db, "loose", features) == [3]

    def test_an_undeclared_untyped_column_holding_one_storage_class_does_not_block_the_pushdown(
        self, filter_db: str
    ) -> None:
        features = _filtered_features(
            [Feature.not_typed("id"), Feature.not_typed("i"), Feature.not_typed("u")],
            [_flt("i", T.MIN, {"value": 3})],
        )
        assert _ids(filter_db, "loose", features) == [3]

    def test_an_untyped_column_with_mixed_storage_classes_blocks_the_pushdown(self, filter_db: str) -> None:
        features = _filtered_features(
            [Feature.not_typed("id"), Feature.not_typed("mi"), Feature.not_typed("mt")],
            [_flt("id", T.EQUAL, {"value": 1})],
        )
        assert _ids(filter_db, "wild", features) == [1, 2, 3]

    @pytest.mark.parametrize(
        ("column", "declared", "filter_value", "pushed"),
        [
            ("r", Feature.float_of, 2.5, False),
            ("r", Feature.double_of, 2.5, True),
            ("i", Feature.int64_of, 3, True),
            ("i", Feature.int32_of, 3, False),
            ("i", Feature.double_of, 3, False),
            ("t", Feature.str_of, "c", True),
        ],
        ids=["real_float", "real_double", "int_int64", "int_int32", "int_double", "text_string"],
    )
    def test_a_typed_filter_column_is_pushed_only_when_its_type_matches_the_affinity(
        self, filter_db: str, column: str, declared: Any, filter_value: Any, pushed: bool
    ) -> None:
        features = _filtered_features(
            [Feature.int64_of("id"), declared(column)], [_flt(column, T.EQUAL, {"value": filter_value})]
        )
        ids = _ids(filter_db, "ft", features)
        assert (len(ids) < len(ALL_IDS)) is pushed

    def test_a_typed_numeric_column_does_not_block_other_pushed_filters(self, filter_db: str) -> None:
        features = _filtered_features(
            [Feature.not_typed("id"), Feature.not_typed("i"), Feature.int64_of("num")],
            [_flt("i", T.MIN, {"value": 3})],
        )
        assert _ids(filter_db, "loose", features) == [3]

    def test_nothing_is_pushed_when_the_group_skips_the_post_read_pass(self, filter_db: str) -> None:
        features = _filtered_features(TYPED_COLUMNS, [_flt("i", T.EQUAL, {"value": 3})])
        assert _ids(filter_db, "ft", features, NoRecheckSqliteFG) == ALL_IDS

    def test_nothing_is_pushed_when_the_filter_engine_skips_the_post_read_pass(self, filter_db: str) -> None:
        features = _filtered_features(TYPED_COLUMNS, [_flt("i", T.EQUAL, {"value": 3})], BaseFilterEngine)
        assert _ids(filter_db, "ft", features) == ALL_IDS


class TestSqliteFilteredInference:
    def _table(self, db: str, table: str, features: Any) -> pa.Table:
        connection = SqliteFG.connect({KEY: db})
        try:
            result = SqliteFG.produce_rows(connection, table, features)
        finally:
            connection.close()
        assert isinstance(result, pa.Table)
        return result

    def test_a_pushed_filter_matching_no_rows_gives_the_full_table_storage_class_types(self, filter_db: str) -> None:
        names = ["a", "b", "c", "d", "e"]
        features = _filtered_features(
            [Feature.not_typed("id"), *[Feature.not_typed(n) for n in names]],
            [_flt("id", T.EQUAL, {"value": 99})],
        )
        table = self._table(filter_db, "kinds", features)
        assert table.num_rows == 0
        assert pa.types.is_int64(table.schema.field("a").type)
        assert pa.types.is_float64(table.schema.field("b").type)
        assert pa.types.is_string(table.schema.field("c").type)
        assert pa.types.is_binary(table.schema.field("d").type)
        assert pa.types.is_null(table.schema.field("e").type)

    def test_a_filter_with_mixed_untyped_columns_returns_the_same_types_as_without_it(self, filter_db: str) -> None:
        columns = [Feature.not_typed("id"), Feature.not_typed("mi"), Feature.not_typed("mt")]
        plain = self._table(filter_db, "wild", _features(*columns))
        filtered = self._table(filter_db, "wild", _filtered_features(columns, [_flt("id", T.EQUAL, {"value": 1})]))
        assert pa.types.is_float64(filtered.schema.field("mi").type)
        assert pa.types.is_binary(filtered.schema.field("mt").type)
        assert filtered.schema.field("mi").type == plain.schema.field("mi").type
        assert filtered.schema.field("mt").type == plain.schema.field("mt").type

    def test_a_pushed_filter_matching_only_null_rows_keeps_the_full_table_type(self, tmp_path: Path) -> None:
        path = tmp_path / "nulls.db"
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE nt (a INTEGER, b TEXT, c REAL)")
        connection.executemany("INSERT INTO nt VALUES (?, ?, ?)", [(1, None, None), (2, "x", 2.5)])
        connection.commit()
        connection.close()
        features = _filtered_features(
            [Feature.not_typed("a"), Feature.not_typed("b"), Feature.not_typed("c")],
            [_flt("a", T.EQUAL, {"value": 1})],
        )
        table = self._table(str(path), "nt", features)
        assert table.to_pydict() == {"a": [1], "b": [None], "c": [None]}
        assert pa.types.is_string(table.schema.field("b").type)
        assert pa.types.is_float64(table.schema.field("c").type)

    def test_without_a_filter_an_empty_table_still_loads_as_null_type(self, filter_db: str) -> None:
        connection = sqlite3.connect(filter_db)
        connection.execute("CREATE TABLE empty_t (a INTEGER)")
        connection.commit()
        connection.close()
        table = self._table(filter_db, "empty_t", _features(Feature.not_typed("a")))
        assert pa.types.is_null(table.schema.field("a").type)


class TestSqliteRecheckedFilters:
    FILTER = _flt("i", T.EQUAL, {"value": 3})

    def test_returns_the_applicable_filters_when_the_post_read_pass_runs(self) -> None:
        elsewhere = _flt("not_requested", T.EQUAL, {"value": 1})
        features = _filtered_features(TYPED_COLUMNS, [self.FILTER, elsewhere])
        assert SqliteFG.rechecked_filters(features) == [self.FILTER]

    def test_returns_nothing_when_the_group_skips_the_pass(self) -> None:
        features = _filtered_features(TYPED_COLUMNS, [self.FILTER])
        assert NoRecheckSqliteFG.rechecked_filters(features) == []

    def test_returns_nothing_when_the_filter_engine_skips_the_pass(self) -> None:
        features = _filtered_features(TYPED_COLUMNS, [self.FILTER], BaseFilterEngine)
        assert SqliteFG.rechecked_filters(features) == []

    def test_returns_nothing_without_filters(self) -> None:
        assert SqliteFG.rechecked_filters(_features(*TYPED_COLUMNS)) == []


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

    def test_a_missing_database_is_not_created_and_reports_that_it_does_not_exist(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.db"
        with pytest.raises((OSError, ValueError, sqlite3.Error)):
            SqliteFG.describe_columns(_match(str(missing), "t"))
        assert not missing.exists()
        assert SqliteFG.unknown_columns_reason(_match(str(missing), "t")) == "it does not exist"

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

    def test_an_aggregation_over_a_collection_credential(self, people_db: str) -> None:
        feature = Feature(name="sum_of_", options={"sum": ("id", "id")})

        result = mloda.run_all(
            [feature],
            compute_frameworks=["PyArrowTable"],
            data_access_collection=DataAccessCollection(credentials=[{KEY: people_db}]),
            plugin_collector=PluginCollector.enabled_feature_groups({SqliteFG, SumFeature}),
        )

        assert "SumFeature_idid" in result[0].to_pydict()


class TestSqliteFGMultiprocessing:
    PluginLoader().all()

    def test_hashable_dict_credentials_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError) as excinfo:
            DataAccessCollection(credentials=[HashableDict({KEY: str(tmp_path / "mp_hashable.sqlite")})])
        assert "credential" in str(excinfo.value).lower()

    @pytest.mark.subprocess
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
