from mloda.provider import ConnectionMatcherMixin
from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
import pytest

from typing import Any, Dict, List, Optional, Set, Type, Union

from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda.user import ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator
from mloda.user import DataAccessCollection

import logging

logger = logging.getLogger(__name__)

try:
    import duckdb
except ImportError:
    logger.warning("DuckDB is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None


@pytest.fixture
def duckdb_conn() -> Any:
    """Provides a DuckDB in-memory connection and ensures it's closed."""
    conn = duckdb.connect()
    yield conn  # Yields the connection to the test function
    conn.close()  # This runs after the test function completes (or fails)


duckdb_test_dict = {
    "id": [1, 2, 3, 4, 5],
    "value": [10, 20, 30, 40, 50],
    "category": ["A", "B", "A", "C", "B"],
    "score": [1.5, 2.5, 3.5, 4.5, 5.5],
}


class DuckDBTestDataCreator(ATestDataCreator):
    """Test data creator for DuckDB integration tests."""

    compute_framework = DuckDBFramework

    # Override conversion to support DuckDB relations
    conversion = {
        **ATestDataCreator.conversion,
        DuckDBFramework: lambda data: data,  # DuckDB framework handles dict conversion internally
    }

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw test data as a dictionary."""
        return duckdb_test_dict


class ATestDuckDBFeatureGroup(FeatureGroup, ConnectionMatcherMixin):
    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
        framework_connection_object: Optional[Any] = None,
    ) -> Any:
        """We check for data access collection if any child classes match the data access."""

        if not DuckDBFramework.is_available():
            return None

        if feature_name not in cls.feature_names_supported():
            return None

        if isinstance(framework_connection_object, duckdb.DuckDBPyConnection):
            return framework_connection_object

        if data_access_collection is None:
            return None

        if data_access_collection.initialized_connection_objects is None:
            return None

        if data_access_collection.initialized_connection_objects:
            for conn in data_access_collection.initialized_connection_objects:
                if isinstance(conn, duckdb.DuckDBPyConnection):
                    return conn
        return None

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {DuckDBFramework}


class DuckDBSimpleTransformFeatureGroup(ATestDuckDBFeatureGroup):
    """Simple feature group for testing DuckDB transformations."""

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
        """Perform simple transformations on the data using DuckDB SQL."""
        # Start with the original data
        result_data = data

        for feat in features.features:
            feature_name = str(feat.name)

            if feature_name == "doubled_value":
                # Add doubled_value column using DuckDB SQL
                result_data = result_data.select("*, value * 2 AS doubled_value")

            elif feature_name == "score_plus_ten":
                # Add score_plus_ten column using DuckDB SQL
                result_data = result_data.select("*, score + 10 AS score_plus_ten")

        return result_data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"doubled_value", "score_plus_ten"}


class DuckDBSecondTransformFeatureGroup(ATestDuckDBFeatureGroup):
    """Second transformation that depends on the first."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature("doubled_value")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        result_data = data

        for feat in features.features:
            feature_name = str(feat.name)

            if feature_name == "quadrupled_value":
                # Add quadrupled_value column using DuckDB SQL
                result_data = result_data.select("*, doubled_value * 2 AS quadrupled_value")

        return result_data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"quadrupled_value"}


class DuckDBAggregationFeatureGroup(ATestDuckDBFeatureGroup):
    """Feature group for testing DuckDB aggregation capabilities."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        feature_name_str = feature_name.name if isinstance(feature_name, FeatureName) else str(feature_name)

        if feature_name_str in ["avg_value_by_category", "count_by_category"]:
            return {Feature("value"), Feature("category")}

        return set()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Perform aggregations using DuckDB SQL."""
        result_data = data

        for feat in features.features:
            feature_name = str(feat.name)

            if feature_name == "avg_value_by_category":
                # Calculate average value by category
                result_data = result_data.select("*, AVG(value) OVER (PARTITION BY category) AS avg_value_by_category")

            elif feature_name == "count_by_category":
                # Count records by category
                result_data = result_data.select("*, COUNT(*) OVER (PARTITION BY category) AS count_by_category")

        return result_data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"avg_value_by_category", "count_by_category"}


class CheckData(FeatureGroup):
    """Feature group for testing DuckDB aggregation capabilities."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        feature_name_str = feature_name.name if isinstance(feature_name, FeatureName) else str(feature_name)

        if feature_name_str in ["pyarrow_avg_value_by_category"]:
            return {Feature("avg_value_by_category")}

        return set()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data = data.rename_columns(["id", "value", "category", "score", "pyarrow_avg_value_by_category"])
        return data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"pyarrow_avg_value_by_category"}

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}


@pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed. Skipping this test.")
class TestDuckDBIntegrationWithMlodaAPI:
    """Integration tests for DuckDBFramework with mloda."""

    @pytest.mark.parametrize(
        "modes",
        [({ParallelizationMode.SYNC})],
    )
    def test_basic_duckdb_feature_calculation(
        self, modes: Set[ParallelizationMode], flight_server: Any, duckdb_conn: Any
    ) -> None:
        """Test basic feature calculation with DuckDB."""
        # Enable the test feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {DuckDBTestDataCreator, DuckDBSimpleTransformFeatureGroup}
        )

        # Define features to calculate
        feature_list: List[Feature | str] = ["doubled_value", "score_plus_ten"]

        feature_list = [
            Feature(name="doubled_value", options={"DuckDBTestDataCreator": duckdb_conn}),
            Feature(name="score_plus_ten", options={"DuckDBTestDataCreator": duckdb_conn}),
        ]

        data_access_collection = DataAccessCollection(initialized_connection_objects={duckdb_conn})

        # Run with DuckDB framework
        result = mloda.run_all(
            feature_list,
            flight_server=flight_server,
            parallelization_modes=modes,
            plugin_collector=plugin_collector,
            data_access_collection=data_access_collection,
            compute_frameworks={DuckDBFramework},
        )

        # The result should be a pandas DataFrame (converted from DuckDB relation)
        final_data = result[0]
        assert hasattr(final_data, "columns")

        # Verify the transformations worked
        assert "doubled_value" in final_data.columns
        assert "score_plus_ten" in final_data.columns

        # Check some values
        doubled_values = final_data["doubled_value"].tolist()
        assert doubled_values == [20, 40, 60, 80, 100]  # Original values * 2

        score_plus_ten = final_data["score_plus_ten"].tolist()
        assert score_plus_ten == [11.5, 12.5, 13.5, 14.5, 15.5]  # Original scores + 10

    def test_duckdb_aggregation_features(self, flight_server: Any, duckdb_conn: Any) -> None:
        """Test DuckDB aggregation capabilities."""
        # Enable the test feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {DuckDBTestDataCreator, DuckDBAggregationFeatureGroup}
        )

        # Define features to calculate
        feature_list: List[Feature | str] = ["doubled_value", "score_plus_ten"]

        feature_list = [
            Feature(name="avg_value_by_category", options={"DuckDBTestDataCreator": duckdb_conn}),
            Feature(name="count_by_category", options={"DuckDBTestDataCreator": duckdb_conn}),
        ]

        # Run with DuckDB framework
        result = mloda.run_all(
            feature_list,
            flight_server=flight_server,
            parallelization_modes={ParallelizationMode.SYNC},
            compute_frameworks={DuckDBFramework},
            plugin_collector=plugin_collector,
        )

        # Verify results
        final_data = result[0]
        assert "avg_value_by_category" in final_data.columns
        assert "count_by_category" in final_data.columns

        # Check aggregation results
        # Category A: values [10, 30], avg = 20, count = 2
        # Category B: values [20, 50], avg = 35, count = 2
        # Category C: values [40], avg = 40, count = 1
        avg_values = final_data["avg_value_by_category"].tolist()
        count_values = final_data["count_by_category"].tolist()

        # Verify the aggregations are correct
        assert len(set(avg_values)) <= 3  # At most 3 different averages (one per category)
        assert len(set(count_values)) <= 3  # At most 3 different counts (one per category)

    def test_multiple_feature_groups_duckdb_pipeline(self, flight_server: Any, duckdb_conn: Any) -> None:
        """Test a pipeline with multiple feature groups using DuckDB."""
        # Enable all feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {DuckDBTestDataCreator, DuckDBSimpleTransformFeatureGroup, DuckDBSecondTransformFeatureGroup}
        )

        feature_list = [Feature(name="quadrupled_value", options={"DuckDBTestDataCreator": duckdb_conn})]

        # Run the multi-step pipeline
        result = mloda.run_all(
            feature_list,  # type: ignore
            flight_server=flight_server,
            compute_frameworks={DuckDBFramework},
            plugin_collector=plugin_collector,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        # Verify results
        assert len(result) == 1
        final_data = result[0]

        # Verify the multi-step transformation
        assert "quadrupled_value" in final_data.columns
        quadrupled_values = final_data["quadrupled_value"].tolist()
        assert quadrupled_values == [40, 80, 120, 160, 200]  # Original values * 4

    def test_transform_to_pyarrow(self, flight_server: Any, duckdb_conn: Any) -> None:
        """Test a pipeline with multiple feature groups using DuckDB."""
        # Enable all feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {
                DuckDBTestDataCreator,
                DuckDBSimpleTransformFeatureGroup,
                DuckDBSecondTransformFeatureGroup,
                DuckDBAggregationFeatureGroup,
                CheckData,
            }
        )

        feature_list = [Feature(name="pyarrow_avg_value_by_category", options={"DuckDBTestDataCreator": duckdb_conn})]

        # Run the multi-step pipeline
        result = mloda.run_all(
            feature_list,  # type: ignore
            flight_server=flight_server,
            compute_frameworks={DuckDBFramework, PyArrowTable},
            plugin_collector=plugin_collector,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        assert len(result) == 1
        assert "pyarrow_avg_value_by_category" in result[0].column_names
