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
)
from mloda._testing.compute_framework.connection_suite import ConnectionEndToEndSuite
from mloda._testing.declarations import Cause
from tests.test_plugins.compute_framework.base_implementations.spark.spark_session_adapter import SparkSessionAdapter

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


class TestSparkTfsConnectionE2E(SparkSessionAdapter, ConnectionEndToEndSuite):
    destination_fg_class = TfsDoubledSparkFG
    switched_off = {
        "test_a_root_runs_on_the_one_dac_connection_without_a_class_name_option": Cause.BACKEND,
        "test_a_connection_on_the_feature_needs_no_data_access_collection": Cause.BACKEND,
    }

    @pytest.mark.skip(reason="Spark data does not expose .connection and the result is not a pa.Table")
    def test_a_root_runs_on_the_one_dac_connection_without_a_class_name_option(
        self, *args: Any, **kwargs: Any
    ) -> None: ...

    @pytest.mark.skip(reason="Spark data does not expose .connection and the result is not a pa.Table")
    def test_a_connection_on_the_feature_needs_no_data_access_collection(self, *args: Any, **kwargs: Any) -> None: ...

    def extract_doubled(self, final: Any) -> list[int]:
        rows = final.select("tfs_doubled").collect()
        return [row["tfs_doubled"] for row in rows]
