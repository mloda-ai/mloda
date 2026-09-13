"""Regression: a same-class MatchData chain (a feature's input_features resolved by the SAME
MatchData class) must keep receiving the framework connection on the upstream step. Blocking
inherit_from from ever copying the connection key breaks this (the upstream step's
run_calculation reads the key off its own Options), even though the connection must still
never survive pickling into a multiprocessing worker."""

from typing import Any

import pytest

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

from mloda.user import Feature, FeatureName, Options, PluginCollector, mloda, DataAccessCollection
from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet, MatchData, DataCreator
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework


class ChainSameClassDuckDBFG(FeatureGroup, MatchData):
    """chain_b's input_features resolves chain_a via the SAME class, feature-scope entry point."""

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
        """chain_b declares the connection via Options; chain_a inherits it as the SAME class's own input."""
        conn = duckdb.connect()
        plugin_collector = PluginCollector.enabled_feature_groups({ChainSameClassDuckDBFG})

        result = mloda.run_all(
            [Feature("chain_b", Options({"ChainSameClassDuckDBFG": conn}))],
            compute_frameworks={DuckDBFramework},
            plugin_collector=plugin_collector,
        )

        final = result[0]
        rows = final.fetchall() if hasattr(final, "fetchall") else final
        assert rows is not None

    def test_global_scope_dac_same_class_chain_resolves(self) -> None:
        """chain_a is a DataCreator root matched via DataAccessCollection; chain_b needs the same connection."""
        conn = duckdb.connect()
        plugin_collector = PluginCollector.enabled_feature_groups({ChainSameClassRootDuckDBFG})

        result = mloda.run_all(
            [Feature("chain_b")],
            data_access_collection=DataAccessCollection(connections={conn}),
            compute_frameworks={DuckDBFramework},
            plugin_collector=plugin_collector,
        )

        final = result[0]
        rows = final.fetchall() if hasattr(final, "fetchall") else final
        assert rows is not None
