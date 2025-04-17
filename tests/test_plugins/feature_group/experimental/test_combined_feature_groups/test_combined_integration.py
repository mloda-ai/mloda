"""
Integration tests for combined feature groups.
"""

from typing import List

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable

from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow import PyArrowAggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.data_quality.missing_value.pandas import PandasMissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.data_quality.missing_value.pyarrow import PyArrowMissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindowFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pyarrow import PyArrowTimeWindowFeatureGroup

from tests.test_plugins.feature_group.experimental.test_combined_feature_groups.test_combined_utils import (
    PandasCombinedFeatureTestDataCreator,
    PyArrowCombinedFeatureTestDataCreator,
    validate_combined_features,
)


class TestCombinedFeatureGroupsPandas:
    """Integration tests for combining multiple feature groups using Pandas."""

    def test_max_aggr_sum_7_day_window_mean_imputed_price(self) -> None:
        """
        Test a feature that combines missing value imputation, time window, and aggregation.

        This test demonstrates the composability of feature groups in the mloda framework
        by creating a feature that:
        1. First imputes missing values in a price feature using mean imputation
        2. Then applies a 7-day time window sum operation on the imputed price
        3. Finally applies a max aggregation on the time-windowed feature
        """

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {
                PandasCombinedFeatureTestDataCreator,
                PandasMissingValueFeatureGroup,
                PandasTimeWindowFeatureGroup,
                PandasAggregatedFeatureGroup,
            }
        )

        # Define the feature chain
        features: List[Feature | str] = [
            "price",  # Source data with missing values
            "mean_imputed__price",  # Step 1: Mean imputation
            "sum_7_day_window__mean_imputed__price",  # Step 2: 7-day window sum
            "max_aggr__sum_7_day_window__mean_imputed__price",  # Step 3: Max aggregation
        ]

        # Run the API with the feature chain
        result = mlodaAPI.run_all(
            features,
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Validate the results
        validate_combined_features(result)

        result2 = mlodaAPI.run_all(
            ["max_aggr__sum_7_day_window__mean_imputed__price"],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        for res in result:
            if "max_aggr__sum_7_day_window__mean_imputed__price" in res.columns:
                res_check = res["max_aggr__sum_7_day_window__mean_imputed__price"]

        for res in result2:
            if "max_aggr__sum_7_day_window__mean_imputed__price" in res.columns:
                res2_check = res["max_aggr__sum_7_day_window__mean_imputed__price"]

        assert res_check.equals(res2_check)

    def test_max_aggr_sum_7_day_window_mean_imputed_price_with_configuration(self) -> None:
        """
        Test a feature that combines missing value imputation, time window, and aggregation
        using the configuration-based approach with FeatureChainParserConfiguration.

        This test demonstrates the composability of feature groups in the mloda framework
        using the configuration-based approach to create a feature that:
        1. First imputes missing values in a price feature using mean imputation
        2. Then applies a 7-day time window sum operation on the imputed price
        3. Finally applies a max aggregation on the time-windowed feature
        """

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {
                PandasCombinedFeatureTestDataCreator,
                PandasMissingValueFeatureGroup,
                PandasTimeWindowFeatureGroup,
                PandasAggregatedFeatureGroup,
            }
        )

        # Step 1: Create the mean imputation feature using configuration
        missing_value_parser = MissingValueFeatureGroup.configurable_feature_chain_parser()
        if missing_value_parser is None:
            raise ValueError("MissingValueFeatureGroup parser is not available.")

        mean_imputed_feature = Feature(
            "placeholder",
            Options(
                {
                    MissingValueFeatureGroup.IMPUTATION_METHOD: "mean",
                    DefaultOptionKeys.mloda_source_feature: "price",
                }
            ),
        )

        # Step 2: Create the time window feature using configuration
        time_window_parser = TimeWindowFeatureGroup.configurable_feature_chain_parser()
        if time_window_parser is None:
            raise ValueError("TimeWindowFeatureGroup parser is not available.")

        time_window_feature = Feature(
            "placeholder",
            Options(
                {
                    TimeWindowFeatureGroup.WINDOW_FUNCTION: "sum",
                    TimeWindowFeatureGroup.WINDOW_SIZE: 7,
                    TimeWindowFeatureGroup.TIME_UNIT: "day",
                    DefaultOptionKeys.mloda_source_feature: "mean_imputed__price",
                }
            ),
        )

        # Step 3: Create the aggregation feature using configuration
        agg_parser = AggregatedFeatureGroup.configurable_feature_chain_parser()
        if agg_parser is None:
            raise ValueError("AggregatedFeatureGroup parser is not available.")

        agg_feature = Feature(
            "placeholder",
            Options(
                {
                    AggregatedFeatureGroup.AGGREGATION_TYPE: "max",
                    DefaultOptionKeys.mloda_source_feature: "sum_7_day_window__mean_imputed__price",
                }
            ),
        )

        # Run the API with the feature chain
        result = mlodaAPI.run_all(
            [mean_imputed_feature, time_window_feature, agg_feature],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Validate the results
        validate_combined_features(result)

        # Test with just the final feature to verify the engine can resolve the entire chain
        result2 = mlodaAPI.run_all(
            [agg_feature],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Verify the results are the same
        for res in result:
            if "max_aggr__sum_7_day_window__mean_imputed__price" in res.columns:
                res_check = res["max_aggr__sum_7_day_window__mean_imputed__price"]

        for res in result2:
            if "max_aggr__sum_7_day_window__mean_imputed__price" in res.columns:
                res2_check = res["max_aggr__sum_7_day_window__mean_imputed__price"]

        assert res_check.equals(res2_check)


class TestCombinedFeatureGroupsPyArrow:
    """Integration tests for combining multiple feature groups using PyArrow."""

    def test_max_aggr_sum_7_day_window_mean_imputed_price(self) -> None:
        """
        Test a feature that combines missing value imputation, time window, and aggregation.

        This test demonstrates the composability of feature groups in the mloda framework
        by creating a feature that:
        1. First imputes missing values in a price feature using mean imputation
        2. Then applies a 7-day time window sum operation on the imputed price
        3. Finally applies a max aggregation on the time-windowed feature
        """

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {
                PyArrowCombinedFeatureTestDataCreator,
                PyArrowMissingValueFeatureGroup,
                PyArrowTimeWindowFeatureGroup,
                PyArrowAggregatedFeatureGroup,
            }
        )

        # Define the feature chain
        features: List[Feature | str] = [
            "price",  # Source data with missing values
            "mean_imputed__price",  # Step 1: Mean imputation
            "sum_7_day_window__mean_imputed__price",  # Step 2: 7-day window sum
            "max_aggr__sum_7_day_window__mean_imputed__price",  # Step 3: Max aggregation
        ]

        # Run the API with the feature chain
        result = mlodaAPI.run_all(
            features,
            compute_frameworks={PyarrowTable},
            plugin_collector=plugin_collector,
        )

        # Validate the results
        validate_combined_features(result)
