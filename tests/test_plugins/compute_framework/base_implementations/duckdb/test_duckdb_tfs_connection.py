"""Reproducer for issue #440: PyArrow -> DuckDB TFS connection propagation."""

from typing import Any, Optional

import pytest

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment]

from mloda.user import mloda, Feature, DataAccessCollection, ParallelizationMode, PluginCollector
from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet, MatchData
from mloda.user import FeatureName, Options
from mloda.core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda.provider import BaseInputData
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework


class TfsIssue440DataCreator(FeatureGroup):
    """Provides raw data via PyArrowTable (simulates the source framework in a TFS)."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"raw_val"})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"raw_val": [1, 2, 3]}


class TfsIssue440DuckDBFG(FeatureGroup, MatchData):
    """Consumes data in DuckDB, triggering a PyArrow -> DuckDB TransformFrameworkStep."""

    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
        framework_connection_object: Optional[Any] = None,
    ) -> Any:
        if duckdb is None or not DuckDBFramework.is_available():
            return None
        if feature_name not in cls.feature_names_supported():
            return None
        if isinstance(framework_connection_object, duckdb.DuckDBPyConnection):
            return framework_connection_object
        if data_access_collection is None:
            return None
        for conn in data_access_collection.initialized_connection_objects:
            if isinstance(conn, duckdb.DuckDBPyConnection):
                return conn
        return None

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature("raw_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {DuckDBFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Project via DuckDB SQL -- requires framework_connection_object to be set
        return data.project("*, raw_val * 2 AS tfs_doubled")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"tfs_doubled"}


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow not installed.")
class TestPyArrowToDuckDBTfsConnection:
    def test_tfs_connection_reaches_duckdb_framework(self) -> None:
        """DataAccessCollection connection must reach the DuckDB cfw created by TFS.

        Reproducer for issue #440: when a TransformFrameworkStep converts data from
        PyArrowTable to DuckDBFramework, the destination DuckDBFramework instance must
        have its framework_connection_object set from DataAccessCollection so that
        the transformer can use it. Without the fix, this raises ValueError because
        framework_connection_object is None inside the transformer.
        """
        conn = duckdb.connect()
        plugin_collector = PluginCollector.enabled_feature_groups({TfsIssue440DataCreator, TfsIssue440DuckDBFG})
        data_access_collection = DataAccessCollection(initialized_connection_objects={conn})
        result = mloda.run_all(
            [Feature("tfs_doubled")],
            compute_frameworks={PyArrowTable, DuckDBFramework},
            plugin_collector=plugin_collector,
            data_access_collection=data_access_collection,
            parallelization_modes={ParallelizationMode.SYNC},
        )
        conn.close()
        assert result is not None
        assert len(result) == 1
        final = result[0]
        assert isinstance(final, pa.Table)
        assert "tfs_doubled" in final.column_names
        assert final.column("tfs_doubled").to_pylist() == [2, 4, 6]
