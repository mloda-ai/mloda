"""End-to-end TFS connection-propagation test for DuckDB.

Reproducer for issue #440. PyArrow -> DuckDB TFS path; without the fix the
destination DuckDBFramework has no connection and `data.project(...)` raises."""

from typing import Any, Optional

import pytest

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]

from mloda.user import Feature, DataAccessCollection
from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet, MatchData
from mloda.user import FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from tests.test_plugins.compute_framework.base_implementations.tfs_connection_e2e_mixin import (
    TfsConnectionEndToEndMixin,
)


class TfsDoubledDuckDBFG(FeatureGroup, MatchData):
    """Doubles raw_val using a DuckDB SQL projection that REQUIRES the connection."""

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
        for conn in data_access_collection.connections.values():
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
        return data.project("*, raw_val * 2 AS tfs_doubled")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"tfs_doubled"}


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow not installed.")
class TestDuckDBTfsConnectionE2E(TfsConnectionEndToEndMixin):
    destination_framework_class = DuckDBFramework
    destination_fg_class = TfsDoubledDuckDBFG

    @pytest.fixture
    def live_connection(self) -> Any:
        conn = duckdb.connect()
        yield conn
        conn.close()

    def extract_doubled(self, final: Any) -> list[int]:
        assert isinstance(final, pa.Table)
        assert "tfs_doubled" in final.column_names
        return list(final.column("tfs_doubled").to_pylist())
