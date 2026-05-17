"""End-to-end TFS connection-propagation test for Spark."""

from typing import Any, Optional

import pytest

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment]

from mloda.user import Feature, DataAccessCollection
from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet, MatchData
from mloda.user import FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import SparkFramework
from tests.test_plugins.compute_framework.base_implementations.spark.conftest import (
    PYSPARK_AVAILABLE,
    SKIP_REASON,
)
from tests.test_plugins.compute_framework.base_implementations.tfs_connection_e2e_mixin import (
    TfsConnectionEndToEndMixin,
)

if PYSPARK_AVAILABLE:
    from pyspark.sql import SparkSession
else:
    SparkSession = None


class TfsDoubledSparkFG(FeatureGroup, MatchData):
    """Doubles raw_val via Spark selectExpr; requires the SparkSession to be set."""

    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
        framework_connection_object: Optional[Any] = None,
    ) -> Any:
        if not PYSPARK_AVAILABLE:
            return None
        if feature_name not in cls.feature_names_supported():
            return None
        if isinstance(framework_connection_object, SparkSession):
            return framework_connection_object
        if data_access_collection is None:
            return None
        for conn in data_access_collection.initialized_connection_objects:
            if isinstance(conn, SparkSession):
                return conn
        return None

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {Feature("raw_val")}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {SparkFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.selectExpr("*", "raw_val * 2 AS tfs_doubled")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"tfs_doubled"}


@pytest.mark.skipif(not PYSPARK_AVAILABLE or pa is None, reason=SKIP_REASON or "PySpark/PyArrow not available")
class TestSparkTfsConnectionE2E(TfsConnectionEndToEndMixin):
    destination_framework_class = SparkFramework
    destination_fg_class = TfsDoubledSparkFG

    @pytest.fixture
    def live_connection(self, spark_session: Any) -> Any:
        return spark_session

    def extract_doubled(self, final: Any) -> list[int]:
        rows = final.select("tfs_doubled").collect()
        return [row["tfs_doubled"] for row in rows]
