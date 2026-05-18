"""End-to-end TFS connection-propagation test for SQLite."""

import sqlite3
from typing import Any, Optional

import pytest

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment]

from mloda.user import Feature, DataAccessCollection
from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet, MatchData
from mloda.user import FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from tests.test_plugins.compute_framework.base_implementations.tfs_connection_e2e_mixin import (
    TfsConnectionEndToEndMixin,
)


class TfsDoubledSqliteFG(FeatureGroup, MatchData):
    """Doubles raw_val via a SQLite SELECT that REQUIRES the connection."""

    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
        framework_connection_object: Optional[Any] = None,
    ) -> Any:
        if feature_name not in cls.feature_names_supported():
            return None
        if isinstance(framework_connection_object, sqlite3.Connection):
            return framework_connection_object
        if data_access_collection is None:
            return None
        for conn in data_access_collection.initialized_connection_objects:
            if isinstance(conn, sqlite3.Connection):
                return conn
        return None

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature("raw_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {SqliteFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.select(_raw_sql="*, raw_val * 2 AS tfs_doubled")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"tfs_doubled"}


@pytest.mark.skipif(pa is None, reason="PyArrow not installed (needed for the source FG).")
class TestSqliteTfsConnectionE2E(TfsConnectionEndToEndMixin):
    destination_framework_class = SqliteFramework
    destination_fg_class = TfsDoubledSqliteFG

    @pytest.fixture
    def live_connection(self) -> Any:
        conn = sqlite3.connect(":memory:")
        yield conn
        conn.close()

    def extract_doubled(self, final: Any) -> list[int]:
        arrow_table = final.to_arrow_table()
        assert "tfs_doubled" in arrow_table.column_names
        return list(arrow_table.column("tfs_doubled").to_pylist())
