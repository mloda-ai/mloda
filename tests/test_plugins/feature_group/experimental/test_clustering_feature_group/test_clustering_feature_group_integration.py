"""
Integration tests for the ClusteringFeatureGroup.
"""

from typing import Any, Dict, List

import numpy as np

from mloda.user import mloda
from mloda.user import Feature
from mloda.user import Options
from mloda.user import PluginCollector

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.clustering.base import ClusteringFeatureGroup
from mloda_plugins.feature_group.experimental.clustering.pandas import PandasClusteringFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class ClusteringFeatureTestDataCreator(ATestDataCreator):
    """Base class for clustering feature test data creators."""

    compute_framework = PandasDataFrame

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary with features for clustering."""
        # Create two clusters of points
        np.random.seed(42)

        # Create two clusters of points
        cluster1_x = np.random.normal(0, 1, 50)
        cluster1_y = np.random.normal(0, 1, 50)

        cluster2_x = np.random.normal(5, 1, 50)
        cluster2_y = np.random.normal(5, 1, 50)

        # Combine the clusters
        x = np.concatenate([cluster1_x, cluster2_x])
        y = np.concatenate([cluster1_y, cluster2_y])

        # Create a dictionary
        return {
            "feature1": x,
            "feature2": y,
            "category": ["A"] * 50 + ["B"] * 50,
        }


def validate_clustering_results(result: List) -> None:  # type: ignore
    """
    Validate the results of the clustering feature test.

    Args:
        result: List of DataFrames from the mloda.run_all call

    Raises:
        AssertionError: If validation fails
    """
    # Verify we have at least one result
    assert len(result) >= 1, "Expected at least one result"

    # Convert all results to pandas DataFrames for consistent validation
    dfs = []
    for res in result:
        if hasattr(res, "to_pandas"):
            dfs.append(res.to_pandas())
        else:
            dfs.append(res)

    # Find the DataFrame with the clustering features
    kmeans_feature = "feature1&feature2__cluster_kmeans_2"
    dbscan_feature = "feature1&feature2__cluster_dbscan_auto"
    hierarchical_feature = "feature1&feature2__cluster_hierarchical_2"

    # Check that all features exist in the results
    kmeans_df = None
    dbscan_df = None
    hierarchical_df = None

    for df in dfs:
        if kmeans_feature in df.columns:
            kmeans_df = df
        if dbscan_feature in df.columns:
            dbscan_df = df
        if hierarchical_feature in df.columns:
            hierarchical_df = df

    # Verify that all features were found
    assert kmeans_df is not None, f"DataFrame with {kmeans_feature} not found"
    assert dbscan_df is not None, f"DataFrame with {dbscan_feature} not found"
    assert hierarchical_df is not None, f"DataFrame with {hierarchical_feature} not found"

    # Verify the clustering results

    # K-means should have exactly 2 clusters
    assert len(np.unique(kmeans_df[kmeans_feature])) == 2, "K-means should have exactly 2 clusters"

    # DBSCAN may have noise points (labeled as -1)
    # So we check that there is at least 1 unique cluster (excluding noise)
    unique_clusters = np.unique(dbscan_df[dbscan_feature])
    assert len(unique_clusters[unique_clusters >= 0]) >= 1, "DBSCAN should have at least 1 cluster"

    # Hierarchical should have exactly 2 clusters
    assert len(np.unique(hierarchical_df[hierarchical_feature])) == 2, "Hierarchical should have exactly 2 clusters"


class TestClusteringFeatureGroupIntegration:
    """Integration tests for the ClusteringFeatureGroup."""

    def test_integration_with_feature_names(self) -> None:
        """Test integration with mloda using explicit feature names."""
        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {
                ClusteringFeatureTestDataCreator,
                PandasClusteringFeatureGroup,
            }
        )

        # Define the features
        features: List[Feature | str] = [
            "feature1",
            "feature2",
            "feature1&feature2__cluster_kmeans_2",
            "feature1&feature2__cluster_dbscan_auto",
            "feature1&feature2__cluster_hierarchical_2",
        ]

        # Run the mloda
        result = mloda.run_all(
            features,
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        # Validate the results
        validate_clustering_results(result)

    def test_integration_with_feature_parser(self) -> None:
        """Test integration with mloda using the parser."""
        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {
                ClusteringFeatureTestDataCreator,
                PandasClusteringFeatureGroup,
            }
        )

        # Create features using the parser configuration
        kmeans_feature = Feature(
            "feature1&feature2__cluster_kmeans_2",
            Options(
                context={
                    ClusteringFeatureGroup.ALGORITHM: "kmeans",
                    ClusteringFeatureGroup.K_VALUE: 2,
                    DefaultOptionKeys.in_features: "feature1,feature2",
                }
            ),
        )

        dbscan_feature = Feature(
            "feature1&feature2__cluster_dbscan_auto",
            Options(
                context={
                    ClusteringFeatureGroup.ALGORITHM: "dbscan",
                    ClusteringFeatureGroup.K_VALUE: "auto",
                    DefaultOptionKeys.in_features: "feature1,feature2",
                }
            ),
        )

        hierarchical_feature = Feature(
            "feature1&feature2__cluster_hierarchical_2",
            Options(
                context={
                    ClusteringFeatureGroup.ALGORITHM: "hierarchical",
                    ClusteringFeatureGroup.K_VALUE: 2,
                    DefaultOptionKeys.in_features: "feature1,feature2",
                }
            ),
        )

        # Define the features
        features: List[str | Feature] = [
            "feature1",
            "feature2",
            kmeans_feature,
            dbscan_feature,
            hierarchical_feature,
        ]

        # Run the mloda
        result = mloda.run_all(
            features,
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        validate_clustering_results(result)

    def test_integration_with_different_algorithms(self) -> None:
        """Test integration with mloda using different clustering algorithms."""
        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {
                ClusteringFeatureTestDataCreator,
                PandasClusteringFeatureGroup,
            }
        )

        # Define the features
        features: List[str | Feature] = [
            "feature1",
            "feature2",
            "feature1&feature2__cluster_kmeans_2",
            "feature1&feature2__cluster_spectral_2",
            "feature1&feature2__cluster_agglomerative_2",
        ]

        # Run the mloda
        result = mloda.run_all(
            features,
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        # Verify we have at least one result
        assert len(result) >= 1, "Expected at least one result"

        # Convert all results to pandas DataFrames for consistent validation
        dfs = []
        for res in result:
            if hasattr(res, "to_pandas"):
                dfs.append(res.to_pandas())
            else:
                dfs.append(res)

        # Find the DataFrame with the clustering features
        kmeans_feature = "feature1&feature2__cluster_kmeans_2"
        spectral_feature = "feature1&feature2__cluster_spectral_2"
        agglomerative_feature = "feature1&feature2__cluster_agglomerative_2"

        # Check that all features exist in the results
        kmeans_df = None
        spectral_df = None
        agglomerative_df = None

        for df in dfs:
            if kmeans_feature in df.columns:
                kmeans_df = df
            if spectral_feature in df.columns:
                spectral_df = df
            if agglomerative_feature in df.columns:
                agglomerative_df = df

        # Verify that all features were found
        assert kmeans_df is not None, f"DataFrame with {kmeans_feature} not found"
        assert spectral_df is not None, f"DataFrame with {spectral_feature} not found"
        assert agglomerative_df is not None, f"DataFrame with {agglomerative_feature} not found"

        # Verify the clustering results

        # All algorithms should have exactly 2 clusters
        assert len(np.unique(kmeans_df[kmeans_feature])) == 2, "K-means should have exactly 2 clusters"
        assert len(np.unique(spectral_df[spectral_feature])) == 2, "Spectral should have exactly 2 clusters"
        assert len(np.unique(agglomerative_df[agglomerative_feature])) == 2, (
            "Agglomerative should have exactly 2 clusters"
        )
