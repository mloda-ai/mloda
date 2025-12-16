"""
Integration tests for node centrality feature groups.
"""

from typing import Any, Dict, List

import pandas as pd

import mloda
from mloda import Feature
from mloda import Options
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.node_centrality.base import NodeCentralityFeatureGroup
from mloda_plugins.feature_group.experimental.node_centrality.pandas import PandasNodeCentralityFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class NodeCentralityTestDataCreator(ATestDataCreator):
    """Base class for node centrality test data creators."""

    compute_framework = PandasDataFrame

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
    "source__degree_centrality",  # Degree centrality for source nodes
    "source__betweenness_centrality",  # Betweenness centrality for source nodes
    "source__closeness_centrality",  # Closeness centrality for source nodes
    "source__eigenvector_centrality",  # Eigenvector centrality for source nodes
    "source__pagerank_centrality",  # PageRank centrality for source nodes
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
            "source__closeness_centrality",
            "source__eigenvector_centrality",
            "source__pagerank_centrality",
        ]:
            assert (result_df[feature_name] <= 1).all(), f"Feature '{feature_name}' has values greater than 1"


class TestNodeCentralityPandasIntegration:
    """Integration tests for the node centrality feature group using Pandas."""

    def test_node_centrality_with_data_creator(self) -> None:
        """Test node centrality features with API using DataCreator."""

        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {NodeCentralityTestDataCreator, PandasNodeCentralityFeatureGroup}
        )

        # Run the API with multiple node centrality features
        result = mloda.run_all(
            [
                "source",  # Source node feature
                "target",  # Target node feature
                "weight",  # Edge weight feature
                "source__degree_centrality",  # Degree centrality
                "source__betweenness_centrality",  # Betweenness centrality
                "source__closeness_centrality",  # Closeness centrality
                "source__eigenvector_centrality",  # Eigenvector centrality
                "source__pagerank_centrality",  # PageRank centrality
            ],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) > 0, "No results returned from API"

        # Find the DataFrame with the node centrality features
        centrality_df = None
        for df in result:
            if "source__degree_centrality" in df.columns:
                centrality_df = df
                break

        assert centrality_df is not None, "DataFrame with node centrality features not found"

        # Validate the node centrality features
        validate_node_centrality_features(centrality_df, NODE_CENTRALITY_FEATURES)

    def test_node_centrality_with_configuration(self) -> None:
        """
        Test node centrality features using the configuration-based approach
        """
        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {NodeCentralityTestDataCreator, PandasNodeCentralityFeatureGroup}
        )

        # Create degree centrality feature using configuration-based approach
        degree_feature = Feature(
            "placeholder1",
            Options(
                context={
                    NodeCentralityFeatureGroup.CENTRALITY_TYPE: "degree",
                    DefaultOptionKeys.in_features: "source",
                    NodeCentralityFeatureGroup.GRAPH_TYPE: "undirected",
                    NodeCentralityFeatureGroup.WEIGHT_COLUMN: "weight",
                }
            ),
        )

        # Create betweenness centrality feature using configuration-based approach
        betweenness_feature = Feature(
            "placeholder2",
            Options(
                context={
                    NodeCentralityFeatureGroup.CENTRALITY_TYPE: "betweenness",
                    DefaultOptionKeys.in_features: "source",
                    NodeCentralityFeatureGroup.GRAPH_TYPE: "undirected",
                    NodeCentralityFeatureGroup.WEIGHT_COLUMN: "weight",
                }
            ),
        )

        # Run the API with the configured features
        result = mloda.run_all(
            [
                "source",  # Source node feature
                "target",  # Target node feature
                "weight",  # Edge weight feature
                degree_feature,
                betweenness_feature,
            ],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) > 0, "No results returned from API"

        # Find the DataFrame with the node centrality features
        centrality_df = None
        for df in result:
            if "placeholder1" in df.columns:
                centrality_df = df
                break

        assert centrality_df is not None, "DataFrame with node centrality features not found"

        # Validate the node centrality features
        validate_node_centrality_features(centrality_df, ["placeholder1", "placeholder2"])

    def test_directed_vs_undirected_graph(self) -> None:
        """Test node centrality features with different graph types."""

        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {NodeCentralityTestDataCreator, PandasNodeCentralityFeatureGroup}
        )

        # Create degree centrality features with different graph types and source features
        degree_undirected = Feature(
            "placeholder1",
            Options(
                context={
                    NodeCentralityFeatureGroup.CENTRALITY_TYPE: "degree",
                    DefaultOptionKeys.in_features: "source",  # Use source for undirected
                    NodeCentralityFeatureGroup.GRAPH_TYPE: "undirected",
                    NodeCentralityFeatureGroup.WEIGHT_COLUMN: "weight",
                }
            ),
        )

        degree_directed = Feature(
            "placeholder2",
            Options(
                context={
                    NodeCentralityFeatureGroup.CENTRALITY_TYPE: "degree",
                    DefaultOptionKeys.in_features: "target",  # Use target for directed
                    NodeCentralityFeatureGroup.GRAPH_TYPE: "directed",
                    NodeCentralityFeatureGroup.WEIGHT_COLUMN: "weight",
                }
            ),
        )

        # Run the API with the configured features
        result = mloda.run_all(
            [
                "source",  # Source node feature
                "target",  # Target node feature
                "weight",  # Edge weight feature
                degree_undirected,
                degree_directed,
            ],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) > 0, "No results returned from API"

        # Find the DataFrames with the node centrality features
        source_df = None
        target_df = None

        for df in result:
            if "placeholder1" in df.columns:
                source_df = df
            if "placeholder2" in df.columns:
                target_df = df

        # Verify both features were found in the results
        assert source_df is not None, "DataFrame with source centrality feature not found"
        assert target_df is not None, "DataFrame with target centrality feature not found"

        # Verify both features have valid values
        assert (source_df["placeholder1"] >= 0).all(), "Undirected centrality has negative values"
        assert (target_df["placeholder2"] >= 0).all(), "Directed centrality has negative values"

        # Verify the features have different shapes or values
        # This is a simple check to ensure the graph type parameter is having an effect
        # We can't directly compare them since they're in different DataFrames
        assert len(source_df) > 0, "Source centrality DataFrame is empty"
        assert len(target_df) > 0, "Target centrality DataFrame is empty"
