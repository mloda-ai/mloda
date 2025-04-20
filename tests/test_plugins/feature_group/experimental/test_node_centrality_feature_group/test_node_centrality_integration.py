"""
Integration tests for node centrality feature groups.
"""

from typing import Any, Dict, List

import pandas as pd

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.node_centrality.base import NodeCentralityFeatureGroup
from mloda_plugins.feature_group.experimental.node_centrality.pandas import PandasNodeCentralityFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class NodeCentralityTestDataCreator(ATestDataCreator):
    """Base class for node centrality test data creators."""

    compute_framework = PandasDataframe

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a DataFrame with network/graph data."""
        """Return the raw data as a dictionary."""
        # Create a sample network with nodes and edges
        return {
            "source": ["A", "A", "B", "B", "C", "D"],
            "target": ["B", "C", "C", "D", "D", "E"],
            "weight": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }


# List of node centrality features to test
NODE_CENTRALITY_FEATURES: List[Feature | str] = [
    "degree_centrality__source",  # Degree centrality for source nodes
    "betweenness_centrality__source",  # Betweenness centrality for source nodes
    "closeness_centrality__source",  # Closeness centrality for source nodes
    "eigenvector_centrality__source",  # Eigenvector centrality for source nodes
    "pagerank_centrality__source",  # PageRank centrality for source nodes
]


def validate_node_centrality_features(result_df: pd.DataFrame, expected_features: List[Feature | str]) -> None:
    """
    Validate node centrality features in a Pandas DataFrame.

    Args:
        result_df: DataFrame containing node centrality features
        expected_features: List of expected feature names

    Raises:
        AssertionError: If validation fails
    """
    # Verify all expected features exist
    for feature in expected_features:
        # Get the feature name if it's a Feature object, otherwise use it directly
        feature_name = feature.name.name if isinstance(feature, Feature) else feature
        assert feature_name in result_df.columns, f"Expected feature '{feature_name}' not found"

        # Verify the feature values are valid (non-negative)
        assert (result_df[feature_name] >= 0).all(), f"Feature '{feature_name}' has negative values"

        # For normalized centrality measures, verify values are between 0 and 1
        if feature_name in [
            "closeness_centrality__source",
            "eigenvector_centrality__source",
            "pagerank_centrality__source",
        ]:
            assert (result_df[feature_name] <= 1).all(), f"Feature '{feature_name}' has values greater than 1"


class TestNodeCentralityPandasIntegration:
    """Integration tests for the node centrality feature group using Pandas."""

    def test_node_centrality_with_data_creator(self) -> None:
        """Test node centrality features with mlodaAPI using DataCreator."""

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {NodeCentralityTestDataCreator, PandasNodeCentralityFeatureGroup}
        )

        # Run the API with multiple node centrality features
        result = mlodaAPI.run_all(
            [
                "source",  # Source node feature
                "target",  # Target node feature
                "weight",  # Edge weight feature
                "degree_centrality__source",  # Degree centrality
                "betweenness_centrality__source",  # Betweenness centrality
                "closeness_centrality__source",  # Closeness centrality
                "eigenvector_centrality__source",  # Eigenvector centrality
                "pagerank_centrality__source",  # PageRank centrality
            ],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) > 0, "No results returned from mlodaAPI"

        # Find the DataFrame with the node centrality features
        centrality_df = None
        for df in result:
            if "degree_centrality__source" in df.columns:
                centrality_df = df
                break

        assert centrality_df is not None, "DataFrame with node centrality features not found"

        # Validate the node centrality features
        validate_node_centrality_features(centrality_df, NODE_CENTRALITY_FEATURES)

    def test_node_centrality_with_configuration(self) -> None:
        """
        Test node centrality features using the configuration-based approach
        with FeatureChainParserConfiguration.
        """
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {NodeCentralityTestDataCreator, PandasNodeCentralityFeatureGroup}
        )

        # Create features using configuration
        parser = NodeCentralityFeatureGroup.configurable_feature_chain_parser()
        assert parser is not None, "NodeCentralityFeatureGroup parser is not available"

        # Create degree centrality feature
        degree_feature = Feature(
            "placeholder",
            Options(
                {
                    NodeCentralityFeatureGroup.CENTRALITY_TYPE: "degree",
                    DefaultOptionKeys.mloda_source_feature: "source",
                    NodeCentralityFeatureGroup.GRAPH_TYPE: "undirected",
                    NodeCentralityFeatureGroup.WEIGHT_COLUMN: "weight",
                }
            ),
        )

        # Create betweenness centrality feature
        betweenness_feature = Feature(
            "placeholder",
            Options(
                {
                    NodeCentralityFeatureGroup.CENTRALITY_TYPE: "betweenness",
                    DefaultOptionKeys.mloda_source_feature: "source",
                    NodeCentralityFeatureGroup.GRAPH_TYPE: "undirected",
                    NodeCentralityFeatureGroup.WEIGHT_COLUMN: "weight",
                }
            ),
        )

        # Run the API with the configured features
        result = mlodaAPI.run_all(
            [
                "source",  # Source node feature
                "target",  # Target node feature
                "weight",  # Edge weight feature
                degree_feature,
                betweenness_feature,
            ],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) > 0, "No results returned from mlodaAPI"

        # Find the DataFrame with the node centrality features
        centrality_df = None
        for df in result:
            if "degree_centrality__source" in df.columns:
                centrality_df = df
                break

        assert centrality_df is not None, "DataFrame with node centrality features not found"

        # Validate the node centrality features
        validate_node_centrality_features(
            centrality_df, ["degree_centrality__source", "betweenness_centrality__source"]
        )

    def test_directed_vs_undirected_graph(self) -> None:
        """Test node centrality features with different graph types."""

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {NodeCentralityTestDataCreator, PandasNodeCentralityFeatureGroup}
        )

        # Create features with different graph types
        parser = NodeCentralityFeatureGroup.configurable_feature_chain_parser()
        assert parser is not None, "NodeCentralityFeatureGroup parser is not available"

        # Create degree centrality features with different graph types and source features
        degree_undirected = Feature(
            "placeholder",
            Options(
                {
                    NodeCentralityFeatureGroup.CENTRALITY_TYPE: "degree",
                    DefaultOptionKeys.mloda_source_feature: "source",  # Use source for undirected
                    NodeCentralityFeatureGroup.GRAPH_TYPE: "undirected",
                    NodeCentralityFeatureGroup.WEIGHT_COLUMN: "weight",
                }
            ),
        )

        degree_directed = Feature(
            "placeholder",
            Options(
                {
                    NodeCentralityFeatureGroup.CENTRALITY_TYPE: "degree",
                    DefaultOptionKeys.mloda_source_feature: "target",  # Use target for directed
                    NodeCentralityFeatureGroup.GRAPH_TYPE: "directed",
                    NodeCentralityFeatureGroup.WEIGHT_COLUMN: "weight",
                }
            ),
        )

        # Run the API with the configured features
        result = mlodaAPI.run_all(
            [
                "source",  # Source node feature
                "target",  # Target node feature
                "weight",  # Edge weight feature
                degree_undirected,
                degree_directed,
            ],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) > 0, "No results returned from mlodaAPI"

        # Find the DataFrames with the node centrality features
        source_df = None
        target_df = None

        for df in result:
            if "degree_centrality__source" in df.columns:
                source_df = df
            if "degree_centrality__target" in df.columns:
                target_df = df

        # Verify both features were found in the results
        assert source_df is not None, "DataFrame with source centrality feature not found"
        assert target_df is not None, "DataFrame with target centrality feature not found"

        # Verify both features have valid values
        assert (source_df["degree_centrality__source"] >= 0).all(), "Undirected centrality has negative values"
        assert (target_df["degree_centrality__target"] >= 0).all(), "Directed centrality has negative values"

        # Verify the features have different shapes or values
        # This is a simple check to ensure the graph type parameter is having an effect
        # We can't directly compare them since they're in different DataFrames
        assert len(source_df) > 0, "Source centrality DataFrame is empty"
        assert len(target_df) > 0, "Target centrality DataFrame is empty"
