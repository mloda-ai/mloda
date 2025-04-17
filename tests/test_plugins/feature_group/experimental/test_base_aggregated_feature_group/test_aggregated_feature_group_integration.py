"""
Integration tests for the AggregatedFeatureGroup with mlodaAPI.
"""

from typing import Any, Dict

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class AggregatedParserTestDataCreator(ATestDataCreator):
    """Test data creator for aggregation parser tests."""

    compute_framework = PandasDataframe

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "Sales": [100, 200, 300, 400, 500],
            "Revenue": [1000, 2000, 3000, 4000, 5000],
        }


class TestAggregatedFeatureGroupIntegration:
    def test_integration_with_feature_parser(self) -> None:
        """Test integration with mlodaAPI using the parser."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {AggregatedParserTestDataCreator, PandasAggregatedFeatureGroup}
        )

        parser = AggregatedFeatureGroup.configurable_feature_chain_parser()
        if parser is None:
            raise ValueError("Feature chain parser is not available.")

        f1 = Feature(
            "x",
            Options(
                {
                    AggregatedFeatureGroup.AGGREGATION_TYPE: "sum",
                    DefaultOptionKeys.mloda_source_feature: "Sales",
                }
            ),
        )

        f2 = Feature(
            "x",
            Options(
                {AggregatedFeatureGroup.AGGREGATION_TYPE: "avg", DefaultOptionKeys.mloda_source_feature: "Revenue"}
            ),
        )

        feature1 = parser.create_feature_without_options(f1)
        feature2 = parser.create_feature_without_options(f2)

        if feature1 is None or feature2 is None:
            raise ValueError("Failed to create features using the parser.")

        # test with pre parsing the features
        results = mlodaAPI.run_all(
            [feature1, feature2],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        assert len(results) == 1

        # Find the DataFrame with the aggregated feature
        agg_df = None
        for df in results:
            if "sum_aggr__Sales" in df.columns and "avg_aggr__Revenue" in df.columns:
                agg_df = df
                break

        assert agg_df is not None, "DataFrame with aggregated feature not found"
        assert agg_df["sum_aggr__Sales"].iloc[0] == 1500  # Sum of [100, 200, 300, 400, 500]

        # test with mloda parsing the features
        results2 = mlodaAPI.run_all(
            [f1, f2],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        assert len(results2) == 1
        assert results[0].sort_index(axis=1).equals(results2[0].sort_index(axis=1))
