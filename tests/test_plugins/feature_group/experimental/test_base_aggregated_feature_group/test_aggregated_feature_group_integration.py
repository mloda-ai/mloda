"""
Integration tests for the AggregatedFeatureGroup with API.
"""

from typing import Any, Dict

import mloda
from mloda import Feature
from mloda import Options
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class AggregatedParserTestDataCreator(ATestDataCreator):
    """Test data creator for aggregation parser tests."""

    compute_framework = PandasDataFrame

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "Sales": [100, 200, 300, 400, 500],
            "Revenue": [1000, 2000, 3000, 4000, 5000],
        }


class TestAggregatedFeatureGroupIntegration:
    def test_integration_with_feature_parser(self) -> None:
        """Test integration with API using the parser."""
        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {AggregatedParserTestDataCreator, PandasAggregatedFeatureGroup}
        )

        f1 = Feature(
            "sum_sales",
            Options(
                context={
                    AggregatedFeatureGroup.AGGREGATION_TYPE: "sum",
                    DefaultOptionKeys.in_features: "Sales",
                }
            ),
        )

        f2 = Feature(
            "avg_revenue",
            Options(
                context={
                    AggregatedFeatureGroup.AGGREGATION_TYPE: "avg",
                    DefaultOptionKeys.in_features: "Revenue",
                }
            ),
        )

        # test with pre parsing the features
        results = mloda.run_all(
            [f1, f2],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        assert len(results) == 1

        # Find the DataFrame with the aggregated feature
        agg_df = None
        for df in results:
            if "avg_revenue" in df.columns and "sum_sales" in df.columns:
                agg_df = df
                break

        assert agg_df is not None, "DataFrame with aggregated feature not found"
        assert agg_df["sum_sales"].iloc[0] == 1500  # Sum of [100, 200, 300, 400, 500]

        # test with mloda parsing the features
        results2 = mloda.run_all(
            [f1, f2],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        assert len(results2) == 1
        assert results[0].sort_index(axis=1).equals(results2[0].sort_index(axis=1))
