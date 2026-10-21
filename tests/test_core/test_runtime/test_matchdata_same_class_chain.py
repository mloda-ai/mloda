"""Regression: a same-class MatchData chain (a feature's input_features resolved by the SAME
MatchData class) must keep receiving the framework connection on the upstream step. Blocking
inherit_from from ever copying the connection key breaks this (the upstream step's
run_calculation reads the key off its own Options), even though the connection must still
never survive pickling into a multiprocessing worker."""

import sqlite3
from typing import Any

import pytest

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda, DataAccessCollection
from mloda.core.abstract_plugins.components.connection_requirement import ConnectionRequirement
from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet, MatchData, DataCreator
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework


class ChainSameClassDuckDBFG(FeatureGroup, MatchData):
    """chain_b's input_features resolves chain_a via the SAME class, feature-scope entry point."""

    SEEN_CONNECTIONS: list[Any] = []

    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
        framework_connection_object: Any | None = None,
    ) -> Any:
        if feature_name not in {"chain_a", "chain_b"}:
            return None
        if isinstance(framework_connection_object, duckdb.DuckDBPyConnection):
            return framework_connection_object
        if data_access_collection is not None and options.get("connection_handle") is not None:
            return data_access_collection.resolve("connection", hint=options.get("connection_handle"))
        if data_access_collection is not None:
            for conn in data_access_collection.connections.values():
                if isinstance(conn, duckdb.DuckDBPyConnection):
                    return conn
        return None

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        if str(feature_name) == "chain_b":
            return {Feature("chain_a")}
        return None

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {DuckDBFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        names = {str(f.name) for f in features.features}
        if names == {"chain_a"}:
            return {"chain_a": [1, 2, 3]}
        cls.SEEN_CONNECTIONS.append(data.connection)
        return data.project("*, chain_a + 1 AS chain_b")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"chain_a", "chain_b"}


class ChainSameClassRootDuckDBFG(ChainSameClassDuckDBFG):
    """Same shape, but chain_a is a DataCreator root, matched globally through a DataAccessCollection."""

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({"chain_a"})


@pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed.")
class TestMatchDataSameClassChainKeepsConnection:
    """Both entry points must resolve, not raise 'connection object required'."""

    def test_feature_scope_same_class_chain_resolves(self) -> None:
        """chain_b selects the DAC connection via connection_handle; chain_a is the SAME class's own input."""
        conn = duckdb.connect()
        plugin_collector = PluginCollector.enabled_feature_groups({ChainSameClassDuckDBFG})
        ChainSameClassDuckDBFG.SEEN_CONNECTIONS.clear()

        result = mloda.run_all(
            [Feature("chain_b", Options({"connection_handle": "main"}))],
            data_access_collection=DataAccessCollection(connections={"other": duckdb.connect(), "main": conn}),
            compute_frameworks=[DuckDBFramework],
            plugin_collector=plugin_collector,
        )

        assert ChainSameClassDuckDBFG.SEEN_CONNECTIONS == [conn]
        assert result[0].to_pydict() == {"chain_b": [2, 3, 4]}

    def test_a_connection_object_under_the_class_name_key_raises_the_move_error(self) -> None:
        conn = duckdb.connect()
        plugin_collector = PluginCollector.enabled_feature_groups({ChainSameClassDuckDBFG})

        with pytest.raises(ValueError, match="ChainSameClassDuckDBFG") as raised:
            mloda.run_all(
                [Feature("chain_b", Options({"ChainSameClassDuckDBFG": conn}))],
                compute_frameworks=[DuckDBFramework],
                plugin_collector=plugin_collector,
            )

        assert "DuckDBFramework" in str(raised.value)
        assert "DataAccessCollection" in str(raised.value)
        assert "connection_handle" in str(raised.value)

    def test_global_scope_dac_same_class_chain_resolves(self) -> None:
        """chain_a is a DataCreator root matched via DataAccessCollection; chain_b needs the same connection."""
        conn = duckdb.connect()
        plugin_collector = PluginCollector.enabled_feature_groups({ChainSameClassRootDuckDBFG})

        result = mloda.run_all(
            [Feature("chain_b")],
            data_access_collection=DataAccessCollection(connections={conn}),
            compute_frameworks=[DuckDBFramework],
            plugin_collector=plugin_collector,
        )

        final = result[0]
        rows = final.fetchall() if hasattr(final, "fetchall") else final
        assert rows is not None


class TwoClassSqliteRootFG(FeatureGroup, MatchData):
    """Root whose own matcher selects the SQLite connection of the DataAccessCollection."""

    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
        framework_connection_object: Any | None = None,
    ) -> Any:
        if feature_name != "two_class_y" or data_access_collection is None:
            return None
        return next((c for c in data_access_collection.connections.values() if isinstance(c, sqlite3.Connection)), None)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"two_class_y"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {SqliteFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"two_class_y": [1, 2]}


class TwoClassDuckDBConsumerFG(FeatureGroup, MatchData):
    """Consumer whose matcher selects the DuckDB connection; its input belongs to another MatchData class."""

    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
        framework_connection_object: Any | None = None,
    ) -> Any:
        if feature_name != "two_class_x" or data_access_collection is None:
            return None
        return next(
            (c for c in data_access_collection.connections.values() if isinstance(c, duckdb.DuckDBPyConnection)), None
        )

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("two_class_y")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"two_class_x"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {DuckDBFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.project("two_class_y * 2 AS two_class_x")


@pytest.mark.skipif(duckdb is None, reason="DuckDB is not installed.")
class TestMatchDataTwoClassChain:
    """Each MatchData class resolves its own connection handle; a handle only inherited from the consumer yields."""

    def test_two_matchdata_classes_on_two_connections_in_one_chain_run(self) -> None:
        access = DataAccessCollection(connections={"d": duckdb.connect(), "s": sqlite3.connect(":memory:")})

        result = mloda.run_all(
            [Feature("two_class_x")],
            data_access_collection=access,
            compute_frameworks=[DuckDBFramework, SqliteFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({TwoClassSqliteRootFG, TwoClassDuckDBConsumerFG}),
        )

        assert result[0].to_pydict() == {"two_class_x": [2, 4]}

    def test_a_handle_the_feature_set_itself_still_conflicts_with_its_matcher(self) -> None:
        access = DataAccessCollection(connections={"d": duckdb.connect(), "s": sqlite3.connect(":memory:")})

        with pytest.raises(ValueError, match="already set with different values"):
            mloda.run_all(
                [Feature("two_class_y", Options({"connection_handle": "d"}))],
                data_access_collection=access,
                compute_frameworks=[DuckDBFramework, SqliteFramework],
                plugin_collector=PluginCollector.enabled_feature_groups({TwoClassSqliteRootFG}),
            )


class _SessionLike:
    """Stands in for a SparkSession or an Iceberg catalog."""


class SelfManagedFramework(PyArrowTable):
    """A framework that manages its own connection (like Spark and Iceberg)."""

    @classmethod
    def connection_requirement(cls) -> ConnectionRequirement:
        return ConnectionRequirement.SELF_MANAGED

    @classmethod
    def _connection_matches(cls, conn: Any) -> bool:
        return isinstance(conn, _SessionLike)


class SelfManagedRootFG(FeatureGroup):
    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({"self_managed_col"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {SelfManagedFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"self_managed_col": [1]}


def test_a_self_managed_connection_under_a_feature_option_raises_the_move_error() -> None:
    with pytest.raises(ValueError, match="some_key") as raised:
        mloda.run_all(
            [Feature("self_managed_col", Options({"some_key": _SessionLike()}))],
            compute_frameworks=[SelfManagedFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({SelfManagedRootFG}),
        )

    assert "DataAccessCollection" in str(raised.value)
