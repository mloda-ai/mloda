import pytest
from typing import Any, Dict, List, Optional, Set

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore


polars_test_dict = {
    "id": [1, 2, 3, 4, 5],
    "value": [10, 20, 30, 40, 50],
    "category": ["A", "B", "A", "C", "B"],
    "score": [1.5, 2.5, 3.5, 4.5, 5.5],
}


class PolarsLazyTestDataCreator(ATestDataCreator):
    """Test data creator for Polars lazy integration tests."""

    compute_framework = PolarsLazyDataFrame

    # Override conversion to support Polars LazyFrame
    conversion = {
        **ATestDataCreator.conversion,
        PolarsLazyDataFrame: lambda data: pl.LazyFrame(data),
    }

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw test data as a dictionary."""
        return polars_test_dict


class PolarsEagerTestDataCreator(ATestDataCreator):
    """Test data creator for Polars eager integration tests."""

    compute_framework = PolarsDataFrame

    # Override conversion to support Polars DataFrame
    conversion = {
        **ATestDataCreator.conversion,
        PolarsDataFrame: lambda data: pl.DataFrame(data),
    }

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw test data as a dictionary."""
        return polars_test_dict


class PolarsLazySimpleTransformFeatureGroup(AbstractFeatureGroup):
    """Simple feature group for testing lazy transformations."""

    @classmethod
    def compute_framework_rule(cls) -> Set[Any]:
        """Support both lazy and eager Polars frameworks."""
        return {PolarsDataFrame, PolarsLazyDataFrame}

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
        """Perform simple transformations on the data."""
        # Start with the original data
        result_data = data

        for feat in features.features:
            feature_name = str(feat.name)

            if feature_name == "doubled_value":
                # Add doubled_value column to the data
                result_data = result_data.with_columns((pl.col("value") * 2).alias("doubled_value"))

            elif feature_name == "score_plus_ten":
                # Add score_plus_ten column to the data
                result_data = result_data.with_columns((pl.col("score") + 10).alias("score_plus_ten"))
        return result_data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return set(["doubled_value", "score_plus_ten"])


class SecondTransformFeatureGroup(AbstractFeatureGroup):
    """Second transformation that depends on the first."""

    @classmethod
    def compute_framework_rule(cls) -> Set[Any]:
        return {PolarsLazyDataFrame}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature("doubled_value")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        result_data = data

        for feat in features.features:
            feature_name = str(feat.name)

            if feature_name == "quadrupled_value":
                # Add doubled_value column to the data
                result_data = result_data.with_columns((pl.col("doubled_value") * 2).alias("quadrupled_value"))
        return result_data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return set(["quadrupled_value"])


@pytest.mark.skipif(pl is None, reason="Polars is not installed. Skipping this test.")
class TestPolarsLazyIntegrationWithMlodaAPI:
    """Integration tests for PolarsLazyDataFrame with mlodaAPI."""

    @pytest.mark.parametrize(
        "modes",
        [({ParallelizationModes.SYNC})],
    )
    def test_basic_lazy_feature_calculation(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        """Test basic feature calculation with lazy evaluation."""
        # Enable the test feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {PolarsLazyTestDataCreator, PolarsLazySimpleTransformFeatureGroup}
        )

        # Define features to calculate
        feature_list: List[Feature | str] = ["doubled_value", "score_plus_ten"]

        # Run with lazy framework
        result = mlodaAPI.run_all(
            feature_list,
            flight_server=flight_server,
            parallelization_modes=modes,
            compute_frameworks={PolarsLazyDataFrame},
            plugin_collector=plugin_collector,
        )

        # The result should be collected at this point
        final_data = result[0]
        assert hasattr(final_data, "columns")

        # Verify the transformations worked
        assert "doubled_value" in final_data.columns
        assert "score_plus_ten" in final_data.columns

        # Check some values
        doubled_values = final_data["doubled_value"].to_list()
        assert doubled_values == [20, 40, 60, 80, 100]  # Original values * 2

        score_plus_ten = final_data["score_plus_ten"].to_list()
        assert score_plus_ten == [11.5, 12.5, 13.5, 14.5, 15.5]  # Original scores + 10

    def test_lazy_vs_eager_equivalence(self) -> None:
        """Test that lazy and eager frameworks produce equivalent results."""
        # Define features to calculate
        feature_list: List[Feature | str] = ["id", "value", "score", "doubled_value", "score_plus_ten"]

        # Run with eager framework
        eager_plugin_collector = PlugInCollector.enabled_feature_groups(
            {PolarsEagerTestDataCreator, PolarsLazySimpleTransformFeatureGroup}
        )
        eager_result = mlodaAPI.run_all(
            feature_list,
            compute_frameworks={PolarsDataFrame},
            plugin_collector=eager_plugin_collector,
        )

        # Run with lazy framework
        lazy_plugin_collector = PlugInCollector.enabled_feature_groups(
            {PolarsLazyTestDataCreator, PolarsLazySimpleTransformFeatureGroup}
        )
        lazy_result = mlodaAPI.run_all(
            feature_list,
            compute_frameworks={PolarsLazyDataFrame},
            plugin_collector=lazy_plugin_collector,
        )

        # Both should produce results
        assert len(eager_result) > 0
        assert len(lazy_result) > 0

        # Get the final data
        eager_data = eager_result[0]
        lazy_data = lazy_result[0]

        # If lazy result is still lazy, collect it
        if hasattr(lazy_data, "collect"):
            lazy_data = lazy_data.collect()

        # Results should be equivalent
        assert eager_data.equals(lazy_data)

    def test_multiple_feature_groups_lazy_pipeline(self, flight_server: Any) -> None:
        """Test a pipeline with multiple feature groups using lazy evaluation."""

        # Enable all feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {PolarsLazyTestDataCreator, PolarsLazySimpleTransformFeatureGroup, SecondTransformFeatureGroup}
        )

        # Define a pipeline: base data -> doubled -> quadrupled
        feature_list: List[Feature | str] = ["quadrupled_value"]

        # Run the multi-step pipeline
        result = mlodaAPI.run_all(
            feature_list,
            flight_server=flight_server,
            compute_frameworks={PolarsLazyDataFrame},
            plugin_collector=plugin_collector,
            parallelization_modes={ParallelizationModes.SYNC},
        )

        # Verify results
        assert len(result) == 1
        final_data = result[0]

        # Verify the multi-step transformation
        assert "quadrupled_value" in final_data.columns
        quadrupled_values = final_data["quadrupled_value"].to_list()
        assert quadrupled_values == [40, 80, 120, 160, 200]  # Original values * 4
