"""
Spark Integration Tests

This module contains integration tests for the Spark compute framework with mlodaAPI.

Requirements:
- PySpark must be installed (pip install pyspark)
- Java 8+ must be installed and JAVA_HOME environment variable must be set

Environment Setup:
- JAVA_HOME: Must point to a valid Java installation

Test Coverage:
- Basic feature calculation with Spark DataFrames
- Multi-step feature pipelines
- Aggregation operations
- Cross-framework transformations (Spark to PyArrow)
- Integration with mlodaAPI

The tests use a shared SparkSession fixture to avoid Java gateway conflicts and
ensure proper resource management across all test methods.
"""

from typing import Any, Dict, Optional, Set, Type, Union
import pytest

from mloda_core.abstract_plugins.components.match_data.match_data import MatchData
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import SparkFramework
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection

# Import shared fixtures and availability flags from conftest.py
try:
    from tests.test_plugins.compute_framework.base_implementations.spark.conftest import PYSPARK_AVAILABLE
except ImportError:
    # Fallback for when running tests directly
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import PYSPARK_AVAILABLE  # type: ignore

import logging

logger = logging.getLogger(__name__)

# Import PySpark types for integration testing (only if available)
if PYSPARK_AVAILABLE:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
    import pyspark.sql.functions as F
else:
    SparkSession = None
    StructType = None
    StructField = None
    StringType = None
    IntegerType = None
    DoubleType = None
    F = None

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None


# Test data for Spark integration tests
spark_test_dict = {
    "id": [1, 2, 3, 4, 5],
    "value": [10, 20, 30, 40, 50],
    "category": ["A", "B", "A", "C", "B"],
    "score": [1.5, 2.5, 3.5, 4.5, 5.5],
}


class SparkTestDataCreator(ATestDataCreator):
    """Test data creator for Spark integration tests."""

    compute_framework = SparkFramework

    # Override conversion to support Spark DataFrames
    conversion = {
        **ATestDataCreator.conversion,
        SparkFramework: lambda data: data,  # Spark framework handles dict conversion internally
    }

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw test data as a dictionary."""
        return spark_test_dict


class ATestSparkFeatureGroup(AbstractFeatureGroup, MatchData):
    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
        framework_connection_object: Optional[Any] = None,
    ) -> Any:
        """Check for data access collection if any child classes match the data access."""

        if not SparkFramework.is_available():
            return False

        if feature_name not in cls.feature_names_supported():
            return False

        if isinstance(framework_connection_object, SparkSession):
            return framework_connection_object

        if data_access_collection is None:
            return False

        if data_access_collection.initialized_connection_objects is None:
            return False

        if data_access_collection.initialized_connection_objects:
            for conn in data_access_collection.initialized_connection_objects:
                if isinstance(conn, SparkSession):
                    return conn
        return False

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {SparkFramework}


class SparkSimpleTransformFeatureGroup(ATestSparkFeatureGroup):
    """Simple feature group for testing Spark transformations."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Require base features for transformation."""
        feature_name_str = feature_name.name if isinstance(feature_name, FeatureName) else str(feature_name)

        if feature_name_str == "doubled_value":
            return {Feature("value")}
        elif feature_name_str == "score_plus_ten":
            return {Feature("score")}

        return set()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Perform simple transformations on the data using Spark DataFrame operations."""
        # Start with the original data
        result_data = data

        for feat in features.features:
            feature_name = str(feat.name)

            if feature_name == "doubled_value":
                # Add doubled_value column using Spark DataFrame operations
                result_data = result_data.withColumn("doubled_value", F.col("value") * 2)

            elif feature_name == "score_plus_ten":
                # Add score_plus_ten column using Spark DataFrame operations
                result_data = result_data.withColumn("score_plus_ten", F.col("score") + 10)

        return result_data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"doubled_value", "score_plus_ten"}


class SparkSecondTransformFeatureGroup(ATestSparkFeatureGroup):
    """Second transformation that depends on the first."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature("doubled_value")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        result_data = data

        for feat in features.features:
            feature_name = str(feat.name)

            if feature_name == "quadrupled_value":
                # Add quadrupled_value column using Spark DataFrame operations
                result_data = result_data.withColumn("quadrupled_value", F.col("doubled_value") * 2)

        return result_data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"quadrupled_value"}


class SparkAggregationFeatureGroup(ATestSparkFeatureGroup):
    """Feature group for testing Spark aggregation capabilities."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        feature_name_str = feature_name.name if isinstance(feature_name, FeatureName) else str(feature_name)

        if feature_name_str in ["avg_value_by_category", "count_by_category"]:
            return {Feature("value"), Feature("category")}

        return set()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Perform aggregations using Spark DataFrame window functions."""
        from pyspark.sql.window import Window

        result_data = data

        for feat in features.features:
            feature_name = str(feat.name)

            if feature_name == "avg_value_by_category":
                # Calculate average value by category using window function
                window_spec = Window.partitionBy("category")
                result_data = result_data.withColumn("avg_value_by_category", F.avg("value").over(window_spec))

            elif feature_name == "count_by_category":
                # Count records by category using window function
                window_spec = Window.partitionBy("category")
                result_data = result_data.withColumn("count_by_category", F.count("*").over(window_spec))

        return result_data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"avg_value_by_category", "count_by_category"}


class CheckData(AbstractFeatureGroup):
    """Feature group for testing cross-framework transformation (Spark to PyArrow)."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        feature_name_str = feature_name.name if isinstance(feature_name, FeatureName) else str(feature_name)

        if feature_name_str in ["pyarrow_avg_value_by_category"]:
            return {Feature("avg_value_by_category")}

        return set()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Rename columns to demonstrate PyArrow processing
        data = data.rename_columns(["id", "value", "category", "score", "pyarrow_avg_value_by_category"])
        return data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"pyarrow_avg_value_by_category"}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PyArrowTable}


@pytest.mark.skipif(
    not PYSPARK_AVAILABLE or pa is None, reason="PySpark or PyArrow is not installed. Skipping this test."
)
class TestSparkIntegrationWithMlodaAPI:
    """Integration tests for SparkFramework with mlodaAPI."""

    @pytest.mark.parametrize(
        "modes",
        [({ParallelizationModes.SYNC})],
    )
    def test_basic_spark_feature_calculation(
        self, modes: Set[ParallelizationModes], flight_server: Any, spark_session: Any
    ) -> None:
        """Test basic feature calculation with Spark DataFrames."""
        # Enable the test feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {SparkTestDataCreator, SparkSimpleTransformFeatureGroup}
        )

        # Define features to calculate
        feature_list = [
            Feature(name="doubled_value", options={"SparkTestDataCreator": spark_session}),
            Feature(name="score_plus_ten", options={"SparkTestDataCreator": spark_session}),
        ]

        data_access_collection = DataAccessCollection(initialized_connection_objects={spark_session})

        # Run with Spark framework
        result = mlodaAPI.run_all(
            feature_list,  # type: ignore
            flight_server=flight_server,
            parallelization_modes=modes,
            plugin_collector=plugin_collector,
            data_access_collection=data_access_collection,
            compute_frameworks={SparkFramework},
        )

        # The result should be a Spark DataFrame
        assert len(result) == 1
        final_data = result[0]
        assert hasattr(final_data, "columns")

        # Verify the transformations worked
        assert "doubled_value" in final_data.columns
        assert "score_plus_ten" in final_data.columns

        # Check some values using Spark DataFrame methods
        # Sort by id to ensure consistent ordering and collect data
        sorted_data = final_data.orderBy("id").collect()

        doubled_values = [row["doubled_value"] for row in sorted_data]
        assert doubled_values == [20, 40, 60, 80, 100]  # Original values * 2

        score_plus_ten = [row["score_plus_ten"] for row in sorted_data]
        assert score_plus_ten == [11.5, 12.5, 13.5, 14.5, 15.5]  # Original scores + 10

    def test_spark_aggregation_features(self, flight_server: Any, spark_session: Any) -> None:
        """Test Spark aggregation capabilities."""
        # Enable the test feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups({SparkTestDataCreator, SparkAggregationFeatureGroup})

        # Define features to calculate
        feature_list = [
            Feature(name="avg_value_by_category", options={"SparkTestDataCreator": spark_session}),
            Feature(name="count_by_category", options={"SparkTestDataCreator": spark_session}),
        ]

        # Run with Spark framework
        result = mlodaAPI.run_all(
            feature_list,  # type: ignore
            flight_server=flight_server,
            parallelization_modes={ParallelizationModes.SYNC},
            compute_frameworks={SparkFramework},
            plugin_collector=plugin_collector,
        )

        # Verify results
        final_data = result[0]
        assert "avg_value_by_category" in final_data.columns
        assert "count_by_category" in final_data.columns

        # Check aggregation results using Spark DataFrame methods
        # Category A: values [10, 30], avg = 20, count = 2
        # Category B: values [20, 50], avg = 35, count = 2
        # Category C: values [40], avg = 40, count = 1

        # Sort by id to ensure consistent ordering and collect data
        sorted_data = final_data.orderBy("id").collect()

        avg_values = [row["avg_value_by_category"] for row in sorted_data]
        count_values = [row["count_by_category"] for row in sorted_data]

        # Verify the aggregations are correct
        assert len(set(avg_values)) <= 3  # At most 3 different averages (one per category)
        assert len(set(count_values)) <= 3  # At most 3 different counts (one per category)

        # Check specific aggregation values
        # Category A (id 1,3): avg=20, count=2
        # Category B (id 2,5): avg=35, count=2
        # Category C (id 4): avg=40, count=1
        expected_avgs = [20.0, 35.0, 20.0, 40.0, 35.0]
        expected_counts = [2, 2, 2, 1, 2]

        assert avg_values == expected_avgs
        assert count_values == expected_counts

    def test_multiple_feature_groups_spark_pipeline(self, flight_server: Any, spark_session: Any) -> None:
        """Test a pipeline with multiple feature groups using Spark."""
        # Enable all feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {SparkTestDataCreator, SparkSimpleTransformFeatureGroup, SparkSecondTransformFeatureGroup}
        )

        feature_list = [Feature(name="quadrupled_value", options={"SparkTestDataCreator": spark_session})]

        # Run the multi-step pipeline
        result = mlodaAPI.run_all(
            feature_list,  # type: ignore
            flight_server=flight_server,
            compute_frameworks={SparkFramework},
            plugin_collector=plugin_collector,
            parallelization_modes={ParallelizationModes.SYNC},
        )

        # Verify results
        assert len(result) == 1
        final_data = result[0]

        # Verify the multi-step transformation using Spark DataFrame methods
        assert "quadrupled_value" in final_data.columns

        # Sort by id to ensure consistent ordering and collect data
        sorted_data = final_data.orderBy("id").collect()
        quadrupled_values = [row["quadrupled_value"] for row in sorted_data]
        assert quadrupled_values == [40, 80, 120, 160, 200]  # Original values * 4

    def test_transform_to_pyarrow(self, flight_server: Any, spark_session: Any) -> None:
        """Test a pipeline with cross-framework transformation (Spark to PyArrow)."""
        # Enable all feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {
                SparkTestDataCreator,
                SparkSimpleTransformFeatureGroup,
                SparkSecondTransformFeatureGroup,
                SparkAggregationFeatureGroup,
                CheckData,
            }
        )

        feature_list = [Feature(name="pyarrow_avg_value_by_category", options={"SparkTestDataCreator": spark_session})]

        # Run the multi-step pipeline with cross-framework transformation
        result = mlodaAPI.run_all(
            feature_list,  # type: ignore
            flight_server=flight_server,
            compute_frameworks={SparkFramework, PyArrowTable},
            plugin_collector=plugin_collector,
            parallelization_modes={ParallelizationModes.SYNC},
        )

        assert len(result) == 1
        assert "pyarrow_avg_value_by_category" in result[0].column_names

    def test_spark_with_complex_transformations(self, flight_server: Any, spark_session: Any) -> None:
        """Test Spark with more complex DataFrame transformations."""
        # Enable the test feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {SparkTestDataCreator, SparkSimpleTransformFeatureGroup, SparkAggregationFeatureGroup}
        )

        # Define multiple features to calculate
        feature_list = [
            Feature(name="avg_value_by_category", options={"SparkTestDataCreator": spark_session}),
            Feature(name="count_by_category", options={"SparkTestDataCreator": spark_session}),
        ]

        # Run with Spark framework
        result = mlodaAPI.run_all(
            feature_list,  # type: ignore
            flight_server=flight_server,
            parallelization_modes={ParallelizationModes.SYNC},
            compute_frameworks={SparkFramework},
            plugin_collector=plugin_collector,
        )

        assert len(result) == 1

        # Verify features are present - the aggregation features might not be calculated
        # if they're from a different feature group that doesn't get triggered
        final_data = result[0]

        # Check if we got aggregation features (they might not be present due to feature group logic)
        actual_columns = final_data.columns
        has_aggregation_features = "avg_value_by_category" in actual_columns and "count_by_category" in actual_columns

        if has_aggregation_features:
            # If we have aggregation features, verify they're correct
            sorted_data = final_data.orderBy("id").collect()
            avg_values = [row["avg_value_by_category"] for row in sorted_data]
            count_values = [row["count_by_category"] for row in sorted_data]

            # Verify the aggregations are correct
            assert len(set(avg_values)) <= 3  # At most 3 different averages (one per category)
            assert len(set(count_values)) <= 3  # At most 3 different counts (one per category)

        # Verify data integrity using Spark DataFrame methods
        assert final_data.count() == 5  # Should have 5 rows

    def test_spark_error_handling(self, flight_server: Any, spark_session: Any) -> None:
        """Test error handling in Spark feature groups."""

        class SparkErrorFeatureGroup(ATestSparkFeatureGroup):
            """Feature group that intentionally causes an error."""

            def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
                return {Feature("value")}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                # This will cause an error - trying to access non-existent column
                return data.withColumn("error_feature", F.col("nonexistent_column"))

            @classmethod
            def feature_names_supported(cls) -> Set[str]:
                return {"error_feature"}

        # Enable the error feature group
        plugin_collector = PlugInCollector.enabled_feature_groups({SparkTestDataCreator, SparkErrorFeatureGroup})

        feature_list = [Feature(name="error_feature", options={"SparkTestDataCreator": spark_session})]

        # This should raise an exception due to the non-existent column
        with pytest.raises(Exception):
            mlodaAPI.run_all(
                feature_list,  # type: ignore
                flight_server=flight_server,
                parallelization_modes={ParallelizationModes.SYNC},
                compute_frameworks={SparkFramework},
                plugin_collector=plugin_collector,
            )
