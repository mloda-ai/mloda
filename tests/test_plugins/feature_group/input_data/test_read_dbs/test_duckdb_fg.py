"""DuckdbFG: the shared source contract plus its connection-sourced, opt-in, in-place DuckDB behaviour."""

from __future__ import annotations

import pickle  # nosec B403
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.input_data.base_input_data import RESERVED_READER_OPTION_KEY
from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import ComputeFramework
from mloda.user import DataAccessCollection, Feature, FilterType, GlobalFilter, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow import PyArrowAggregation  # noqa: F401
from mloda_plugins.feature_group.input_data.db_sources import duckdb_fg as duckdb_fg_module
from mloda_plugins.feature_group.input_data.db_sources.duckdb_fg import DuckdbFG, DuckdbTable
from tests.mixins.reader_feature_groups.source_feature_group_test_mixin import SourceFeatureGroupTestMixin

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed.")

TABLE = "duck_table"
HANDLE = "own_handle"


def _database(opened: list[Any], rows: dict[str, list[int]], path: str = ":memory:", table: str = TABLE) -> Any:
    connection = duckdb.connect(path)
    opened.append(connection)
    columns = ", ".join(f"{name} INTEGER" for name in rows)
    connection.execute(f"CREATE TABLE {table} ({columns})")
    marks = ", ".join("?" for _ in rows)
    connection.executemany(f"INSERT INTO {table} VALUES ({marks})", list(zip(*rows.values())))  # nosec B608
    return connection


def _run(
    dac: DataAccessCollection,
    features: list[Feature],
    framework: type[ComputeFramework] = PyArrowTable,
    output: type[ComputeFramework] | None = None,
    global_filter: GlobalFilter | None = None,
) -> Any:
    return mloda.run_all(
        features,
        compute_frameworks=[framework],
        plugin_collector=PluginCollector.enabled_feature_groups({DuckdbFG}),
        data_access_collection=dac,
        output_framework=output,
        global_filter=global_filter,
    )


def _evaluate(feature: Feature, dac: DataAccessCollection | None) -> Any:
    return IdentifyFeatureGroupClass.evaluate(feature, {DuckdbFG: {DuckDBFramework}}, None, dac)


def _handle_feature(name: str, handle: str = HANDLE) -> Feature:
    return Feature(name, Options({"connection_handle": handle}))


@pytest.fixture()
def opened() -> Iterator[list[Any]]:
    connections: list[Any] = []
    yield connections
    for connection in connections:
        connection.close()


class TestDuckdbFGContract(SourceFeatureGroupTestMixin):
    feature_group_class = DuckdbFG
    present_column = "impl_duckdb_present"
    missing_column = "impl_duckdb_missing"
    expected_source = f"memory::{TABLE}"
    context_options = {"connection_handle": HANDLE}

    @pytest.fixture(autouse=True)
    def _duckdb_setup(self, opened: list[Any]) -> None:
        self.opened = opened

    def own_dac(self) -> DataAccessCollection:
        connection = _database(self.opened, {self.present_column: [1, 2], "impl_duckdb_other": [3, 4]})
        return DataAccessCollection(connections={HANDLE: connection})

    def foreign_dac(self) -> DataAccessCollection:
        foreign = sqlite3.connect(":memory:")
        self.opened.append(foreign)
        return DataAccessCollection(connections={"foreign_handle": foreign})

    def own_pointer(self) -> Any:
        return "unused"

    def test_leaves_collection_and_options_unchanged(self) -> None:
        dac = self.own_dac()
        connections = dict(dac.connections)
        feature = Feature(self.present_column, Options({"unrelated_option": 1}))
        group = dict(feature.options.group)
        context = dict(feature.options.context)

        assert self._claims(feature, dac)

        assert dict(dac.connections) == connections
        assert feature.options.group == group
        assert feature.options.context == context
        claimed = _handle_feature(self.present_column)
        _evaluate(claimed, dac)
        assert set(claimed.options.group) <= {"connection_handle", RESERVED_READER_OPTION_KEY}
        assert claimed.options.group["connection_handle"] == HANDLE

    def test_pointer_value_is_used_alone_and_ignores_the_collection(self) -> None:
        pytest.skip("the group has no pointer; a value under its key declines (see the DuckdbFG option key test)")

    def test_a_pointer_of_the_wrong_type_declines_naming_its_type_and_never_searches_the_collection(self) -> None:
        pytest.skip("the group has no pointer; a value under its key declines (see the DuckdbFG option key test)")

    def test_a_handle_of_another_kind_yields_no_sources(self) -> None:
        pytest.skip("the group selects by connection_handle, not data_access_handle (see the non-DuckDB handle test)")

    def test_an_unknown_handle_records_a_rejection_listing_the_known_handles(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        pytest.skip("the group selects by connection_handle, not data_access_handle (see the unknown handle test)")

    def test_a_non_str_handle_is_no_narrowing_and_is_rejected_by_the_option_check(self) -> None:
        pytest.skip("data_access_handle is not an option of this group; connection_handle is always required")

    def test_pointed_with_a_missing_column_aborts_naming_source_and_columns(self) -> None:
        pytest.skip("the group has no pointer to abort on; pointing at it declines")

    def test_pointed_with_query_text_aborts_naming_the_query_route_when_the_group_lists_none(self) -> None:
        pytest.skip("the group has no pointer to abort on; pointing at it declines")

    def test_missing_column_text_hides_the_columns_unless_a_handle_narrows_the_search(self) -> None:
        pytest.skip("narrowing is by connection_handle, which every match requires; data_access_handle is unused")

    def test_a_typo_of_a_known_column_suggests_it_only_where_the_columns_are_shown(self) -> None:
        pytest.skip("the typo suggestion needs a pointed group or data_access_handle; neither applies")

    def test_a_subclass_takes_over_a_pointer_keyed_on_its_parent(self) -> None:
        pytest.skip("the group has no pointer for a subclass to take over")


class TestDuckdbFGInPlace:
    def test_the_reason_names_the_framework_that_lost_the_connection(self, opened: list[Any]) -> None:
        connection = _database(opened, {"reason_col": [1, 2]})

        # DuckDB last, so the reason is "reads in place" rather than the list order
        steps = mloda.explain(
            [Feature("reason_col", connection=connection)],
            compute_frameworks=[PyArrowTable, SqliteFramework, DuckDBFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({DuckdbFG}),
        )

        (step,) = [s for s in steps if s.step_kind == "compute"]
        assert step.compute_framework_reason == (
            "reads in place; SqliteFramework: Connection handle '<unnamed>' holds a connection "
            "this framework does not accept."
        )

    def test_under_duckdb_the_data_is_a_lazy_relation_on_the_users_connection_and_stays_open(
        self, opened: list[Any]
    ) -> None:
        connection = _database(opened, {"inplace_a": [1, 2, 3], "inplace_b": [4, 5, 6]})
        dac = DataAccessCollection(connections={HANDLE: connection})

        result = _run(dac, [_handle_feature("inplace_a")], DuckDBFramework, output=DuckDBFramework)

        assert isinstance(result[0], DuckdbRelation)
        assert result[0].connection is connection
        assert result[0].columns == ["inplace_a"]
        assert connection.execute("SELECT 1").fetchone() == (1,)
        assert result[0].to_arrow_table().to_pydict() == {"inplace_a": [1, 2, 3]}

    def test_the_neutral_load_returns_the_requested_columns_as_an_arrow_table(self, opened: list[Any]) -> None:
        connection = _database(opened, {"neutral_a": [1, 2], "neutral_b": [3, 4], "neutral_c": [5, 6]})
        dac = DataAccessCollection(connections={HANDLE: connection})

        result = _run(dac, [_handle_feature("neutral_a"), _handle_feature("neutral_c")])

        assert isinstance(result[0], pa.Table)
        assert result[0].to_pydict() == {"neutral_a": [1, 2], "neutral_c": [5, 6]}
        assert connection.execute("SELECT 1").fetchone() == (1,)

    @pytest.mark.parametrize("handle, expected", [("first", [1]), ("second", [2])])
    def test_connection_handle_selects_one_of_two_duckdb_connections(
        self, opened: list[Any], handle: str, expected: list[int]
    ) -> None:
        first = _database(opened, {"pick_col": [1]})
        second = _database(opened, {"pick_col": [2]})
        dac = DataAccessCollection(connections={"first": first, "second": second})

        result = _run(dac, [_handle_feature("pick_col", handle)])

        assert result[0].to_pydict() == {"pick_col": expected}

    def test_a_filter_returns_only_the_filtered_rows(self, opened: list[Any]) -> None:
        connection = _database(opened, {"filter_id": [1, 2, 3], "filter_val": [10, 20, 30]})
        dac = DataAccessCollection(connections={HANDLE: connection})
        global_filter = GlobalFilter()
        global_filter.add_filter("filter_val", FilterType.EQUAL, {"value": 20})

        result = _run(
            dac,
            [_handle_feature("filter_id"), _handle_feature("filter_val")],
            DuckDBFramework,
            output=DuckDBFramework,
            global_filter=global_filter,
        )

        assert result[0].to_arrow_table().to_pydict() == {"filter_id": [2], "filter_val": [20]}

    @pytest.mark.parametrize("framework", [DuckDBFramework, PyArrowTable])
    def test_a_temp_table_shadowing_the_source_table_does_not_change_the_rows_read(
        self, opened: list[Any], framework: type[ComputeFramework]
    ) -> None:
        connection = _database(opened, {"shadow_col": [1, 2]})
        connection.execute(f"CREATE TEMP TABLE {TABLE} (shadow_col INTEGER)")
        connection.execute(f"INSERT INTO {TABLE} VALUES (9)")  # nosec B608
        dac = DataAccessCollection(connections={HANDLE: connection})

        result = _run(dac, [_handle_feature("shadow_col")], framework, output=framework)

        rows = result[0].to_arrow_table() if framework is DuckDBFramework else result[0]
        assert rows.to_pydict() == {"shadow_col": [1, 2]}

    def test_a_table_read_on_duckdb_feeds_a_sum_aggregation_on_pyarrow(self, opened: list[Any]) -> None:
        connection = _database(opened, {"hop_amount": [1, 2, 3]})

        result = mloda.run_all(
            [Feature("hop_amount__sum_aggr", connection=connection)],
            compute_frameworks=[DuckDBFramework, PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({DuckdbFG, AggregatedFeatureGroup}),
        )

        compute = [step for step in result.plan if step.step_kind == "compute"]
        assert [(step.feature_group, step.compute_framework) for step in compute] == [
            (DuckdbFG, DuckDBFramework),
            (AggregatedFeatureGroup, PyArrowTable),
        ]
        assert result[0].to_pydict() == {"hop_amount__sum_aggr": [6, 6, 6]}


class TestDuckdbTablePickling:
    def test_pickling_a_match_names_the_live_connection(self, opened: list[Any]) -> None:
        table = DuckdbTable(_database(opened, {"pickle_col": [1]}), TABLE)

        with pytest.raises(TypeError, match="live DuckDB connection"):
            pickle.dumps(table)


class TestDuckdbFGOptIn:
    def test_a_non_duckdb_handle_needs_no_duckdb_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def no_duckdb(*args: Any, **kwargs: Any) -> Any:
            raise ImportError("duckdb is not installed")

        monkeypatch.setattr(duckdb_fg_module, "require", no_duckdb)
        foreign = sqlite3.connect(":memory:")
        try:
            dac = DataAccessCollection(connections={"lite_handle": foreign})

            result = _evaluate(_handle_feature("lite_col", "lite_handle"), dac)
        finally:
            foreign.close()

        assert DuckdbFG not in result.identified
        assert "is not a duckdb connection" in result.eliminations[DuckdbFG].reason

    def test_a_duckdb_connection_without_connection_handle_is_not_read(self, opened: list[Any]) -> None:
        connection = _database(opened, {"optin_col": [1]})
        dac = DataAccessCollection(connections={HANDLE: connection})

        result = _evaluate(Feature("optin_col"), dac)

        assert DuckdbFG not in result.identified

    @pytest.mark.parametrize("handle", ["no_such_handle", 5], ids=["unknown", "non_str"])
    def test_an_unknown_handle_declines_with_a_rejection(self, opened: list[Any], handle: Any) -> None:
        connection = _database(opened, {"unknown_col": [1]})
        dac = DataAccessCollection(connections={HANDLE: connection, "other_handle": connection})

        result = _evaluate(Feature("unknown_col", Options({"connection_handle": handle})), dac)

        assert DuckdbFG not in result.identified
        reason = result.eliminations[DuckdbFG].reason
        assert repr(handle) in reason
        assert str(sorted(["other_handle", HANDLE])) in reason

    def test_a_column_the_selected_table_lacks_declines_naming_the_column_and_source_only(
        self, opened: list[Any]
    ) -> None:
        connection = _database(opened, {"have_col": [1]})
        dac = DataAccessCollection(connections={HANDLE: connection})

        result = _evaluate(_handle_feature("lack_col"), dac)

        assert DuckdbFG not in result.identified
        reason = result.eliminations[DuckdbFG].reason
        assert "'lack_col'" in reason
        assert f"memory::{TABLE}" in reason
        assert "have_col" not in reason

    def test_a_non_duckdb_handle_declines_with_a_rejection(self, opened: list[Any]) -> None:
        foreign = sqlite3.connect(":memory:")
        opened.append(foreign)
        dac = DataAccessCollection(connections={"lite_handle": foreign})

        result = _evaluate(_handle_feature("lite_col", "lite_handle"), dac)

        assert DuckdbFG not in result.identified
        reason = result.eliminations[DuckdbFG].reason
        assert "lite_handle" in reason
        assert "duckdb" in reason.lower()

    @pytest.mark.parametrize("value", ["pointer", {"duckdb": "pointer"}], ids=["str", "mapping"])
    def test_any_value_under_the_group_key_declines_pointing_at_connection_handle(
        self, opened: list[Any], value: Any
    ) -> None:
        connection = _database(opened, {"pointer_col": [1]})
        dac = DataAccessCollection(connections={HANDLE: connection})
        feature = Feature("pointer_col", Options({"connection_handle": HANDLE, DuckdbFG.get_class_name(): value}))

        result = _evaluate(feature, dac)

        assert DuckdbFG not in result.identified
        assert "connection_handle" in result.eliminations[DuckdbFG].reason


class TestDuckdbFGIdentity:
    @staticmethod
    def _identity(connection: Any, column: str) -> str:
        feature = _handle_feature(column)
        _evaluate(feature, DataAccessCollection(connections={HANDLE: connection}))
        pair = feature.input_data_match
        assert pair is not None
        assert pair[0] is DuckdbFG
        return DuckdbFG.data_access_identity(cast(SourceMatch, pair[1]))

    def test_a_file_database_identity_is_its_path(self, tmp_path: Path, opened: list[Any]) -> None:
        path = tmp_path / "identity.duckdb"
        connection = _database(opened, {"file_identity_col": [1]}, str(path))

        identity = self._identity(connection, "file_identity_col")

        assert identity.endswith(f"::{TABLE}")
        assert identity.startswith(str(path))
        assert "DuckDBPyConnection" not in identity
        assert "0x" not in identity

    def test_an_in_memory_database_identity_is_memory_and_the_table(self, opened: list[Any]) -> None:
        connection = _database(opened, {"memory_identity_col": [1]})

        identity = self._identity(connection, "memory_identity_col")

        assert identity == f"memory::{TABLE}"


class TestMatchConnectionHook:
    def test_duckdb_fg_returns_the_connection_of_its_match(self, opened: list[Any]) -> None:
        connection = _database(opened, {"hook_col": [1]})
        feature = _handle_feature("hook_col")
        _evaluate(feature, DataAccessCollection(connections={HANDLE: connection}))
        pair = feature.input_data_match
        assert pair is not None

        assert DuckdbFG.match_connection(cast(SourceMatch, pair[1])) is connection
