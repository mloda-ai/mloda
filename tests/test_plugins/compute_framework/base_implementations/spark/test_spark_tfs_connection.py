"""End-to-end TFS connection-propagation test for Spark."""

from typing import Any

import pytest

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]

from mloda.user import Feature
from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet
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


class TfsDoubledSparkFG(FeatureGroup):
    """Doubles raw_val via Spark selectExpr; requires the SparkSession to be set."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
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

    def test_a_root_runs_on_the_one_dac_connection_without_a_class_name_option(self, live_connection: Any) -> None:
        pytest.skip("Spark data does not expose .connection and the result is not a pa.Table")

    def test_a_connection_on_the_feature_needs_no_data_access_collection(self, live_connection: Any) -> None:
        pytest.skip("Spark data does not expose .connection and the result is not a pa.Table")

    @pytest.fixture
    def live_connection(self, spark_session: Any) -> Any:
        return spark_session

    def extract_doubled(self, final: Any) -> list[int]:
        rows = final.select("tfs_doubled").collect()
        return [row["tfs_doubled"] for row in rows]
