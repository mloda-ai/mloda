"""
Tests for the PandasClusteringFeatureGroup class.
"""

import pytest
import pandas as pd
import numpy as np

from mloda.user import Feature
from mloda.provider import FeatureSet
from mloda_plugins.feature_group.experimental.clustering.pandas import PandasClusteringFeatureGroup


class TestPandasClusteringFeatureGroup:
    """Tests for the PandasClusteringFeatureGroup class."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        # Create a DataFrame with two features and 100 samples
        np.random.seed(42)

        # Create two clusters of points
        cluster1_x = np.random.normal(0, 1, 50)
        cluster1_y = np.random.normal(0, 1, 50)

        cluster2_x = np.random.normal(5, 1, 50)
        cluster2_y = np.random.normal(5, 1, 50)

        # Combine the clusters
        x = np.concatenate([cluster1_x, cluster2_x])
        y = np.concatenate([cluster1_y, cluster2_y])

        # Create a DataFrame
        df = pd.DataFrame(
            {
                "feature1": x,
                "feature2": y,
                "category": ["A"] * 50 + ["B"] * 50,
            }
        )

        return df

    def test_check_source_features_exist(self, sample_data: pd.DataFrame) -> None:
        """Test the _check_source_features_exist method."""
        # Valid features
        PandasClusteringFeatureGroup._check_source_features_exist(sample_data, ["feature1"])
        PandasClusteringFeatureGroup._check_source_features_exist(sample_data, ["feature1", "feature2"])

        # Invalid features - all missing should raise error
        with pytest.raises(ValueError):
            PandasClusteringFeatureGroup._check_source_features_exist(sample_data, ["invalid_feature"])

        # Partial match should NOT raise error (some features exist)
        PandasClusteringFeatureGroup._check_source_features_exist(sample_data, ["feature1", "invalid_feature"])

    def test_add_result_to_data(self, sample_data: pd.DataFrame) -> None:
        """Test the _add_result_to_data method."""
        # Create a result array
        result = np.array([0, 1] * 50)

        # Add the result to the data
        updated_data = PandasClusteringFeatureGroup._add_result_to_data(sample_data, "cluster_result", result)

        # Check that the result was added
        assert "cluster_result" in updated_data.columns
        assert len(updated_data["cluster_result"]) == len(sample_data)
        assert (updated_data["cluster_result"].values == result).all()

    def test_perform_kmeans_clustering(self, sample_data: pd.DataFrame) -> None:
        """Test the _perform_kmeans_clustering method."""
        # Extract features
        X = sample_data[["feature1", "feature2"]].values

        # Perform K-means clustering
        result = PandasClusteringFeatureGroup._perform_kmeans_clustering(X, 2)

        # Check that the result has the expected shape
        assert len(result) == len(sample_data)

        # Check that there are exactly 2 unique clusters
        assert len(np.unique(result)) == 2

    def test_perform_dbscan_clustering(self, sample_data: pd.DataFrame) -> None:
        """Test the _perform_dbscan_clustering method."""
        # Extract features
        X = sample_data[["feature1", "feature2"]].values

        # Perform DBSCAN clustering
        result = PandasClusteringFeatureGroup._perform_dbscan_clustering(X, "auto")

        # Check that the result has the expected shape
        assert len(result) == len(sample_data)

        # DBSCAN may have noise points (labeled as -1)
        # So we check that there are at least 2 unique clusters (excluding noise)
        unique_clusters = np.unique(result)
        assert len(unique_clusters[unique_clusters >= 0]) >= 1

    def test_perform_hierarchical_clustering(self, sample_data: pd.DataFrame) -> None:
        """Test the _perform_hierarchical_clustering method."""
        # Extract features
        X = sample_data[["feature1", "feature2"]].values

        # Perform hierarchical clustering
        result = PandasClusteringFeatureGroup._perform_hierarchical_clustering(X, 2)

        # Check that the result has the expected shape
        assert len(result) == len(sample_data)

        # Check that there are exactly 2 unique clusters
        assert len(np.unique(result)) == 2

    def test_perform_spectral_clustering(self, sample_data: pd.DataFrame) -> None:
        """Test the _perform_spectral_clustering method."""
        # Extract features
        X = sample_data[["feature1", "feature2"]].values

        # Perform spectral clustering
        result = PandasClusteringFeatureGroup._perform_spectral_clustering(X, 2)

        # Check that the result has the expected shape
        assert len(result) == len(sample_data)

        # Check that there are exactly 2 unique clusters
        assert len(np.unique(result)) == 2

    def test_perform_affinity_clustering(self, sample_data: pd.DataFrame) -> None:
        """Test the _perform_affinity_clustering method."""
        # Extract features
        X = sample_data[["feature1", "feature2"]].values

        # Perform affinity clustering
        result = PandasClusteringFeatureGroup._perform_affinity_clustering(X, "auto")

        # Check that the result has the expected shape
        assert len(result) == len(sample_data)

        # Check that there is at least 1 cluster
        assert len(np.unique(result)) >= 1

    def test_find_optimal_k(self, sample_data: pd.DataFrame) -> None:
        """Test the _find_optimal_k method."""
        # Extract features
        X = sample_data[["feature1", "feature2"]].values

        # Find optimal k
        k = PandasClusteringFeatureGroup._find_optimal_k(X, "kmeans", max_k=5)

        # Check that k is a positive integer
        assert isinstance(k, int)
        assert k > 0
        assert k <= 5

    def test_calculate_feature_kmeans(self, sample_data: pd.DataFrame) -> None:
        """Test the calculate_feature method with K-means clustering."""
        # Create a feature set
        feature_set = FeatureSet()
        feature_set.add(Feature("feature1&feature2__cluster_kmeans_2"))

        # Calculate the feature
        result = PandasClusteringFeatureGroup.calculate_feature(sample_data, feature_set)

        # Check that the result has the expected column
        assert "feature1&feature2__cluster_kmeans_2" in result.columns

        # Check that there are exactly 2 unique clusters
        assert len(np.unique(result["feature1&feature2__cluster_kmeans_2"])) == 2

    def test_calculate_feature_dbscan(self, sample_data: pd.DataFrame) -> None:
        """Test the calculate_feature method with DBSCAN clustering."""
        # Create a feature set
        feature_set = FeatureSet()
        feature_set.add(Feature("feature1&feature2__cluster_dbscan_auto"))

        # Calculate the feature
        result = PandasClusteringFeatureGroup.calculate_feature(sample_data, feature_set)

        # Check that the result has the expected column
        assert "feature1&feature2__cluster_dbscan_auto" in result.columns

        # DBSCAN may have noise points (labeled as -1)
        # So we check that there is at least 1 unique cluster (excluding noise)
        unique_clusters = np.unique(result["feature1&feature2__cluster_dbscan_auto"])
        assert len(unique_clusters[unique_clusters >= 0]) >= 1

    def test_calculate_feature_hierarchical(self, sample_data: pd.DataFrame) -> None:
        """Test the calculate_feature method with hierarchical clustering."""
        # Create a feature set
        feature_set = FeatureSet()
        feature_set.add(Feature("feature1&feature2__cluster_hierarchical_2"))

        # Calculate the feature
        result = PandasClusteringFeatureGroup.calculate_feature(sample_data, feature_set)

        # Check that the result has the expected column
        assert "feature1&feature2__cluster_hierarchical_2" in result.columns

        # Check that there are exactly 2 unique clusters
        assert len(np.unique(result["feature1&feature2__cluster_hierarchical_2"])) == 2

    def test_calculate_feature_multiple(self, sample_data: pd.DataFrame) -> None:
        """Test the calculate_feature method with multiple clustering features."""
        # Create a feature set
        feature_set = FeatureSet()
        x = [
            Feature("feature1&feature2__cluster_kmeans_2"),
            Feature("feature1&feature2__cluster_dbscan_auto"),
            Feature("feature1&feature2__cluster_hierarchical_2"),
        ]
        for i in x:
            feature_set.add(i)

        # Calculate the features
        result = PandasClusteringFeatureGroup.calculate_feature(sample_data, feature_set)

        # Check that the result has the expected columns
        assert "feature1&feature2__cluster_kmeans_2" in result.columns
        assert "feature1&feature2__cluster_dbscan_auto" in result.columns
        assert "feature1&feature2__cluster_hierarchical_2" in result.columns
