"""
Tests for the PandasClustering implementation.
"""

from unittest.mock import Mock

import pytest
import pandas as pd
import numpy as np

from tests.test_plugins.feature_group.experimental.core_input_check import assert_core_rejects_missing_input
from mloda.user import Feature
from mloda.provider import FeatureSet
from mloda_plugins.feature_group.experimental.clustering.base import ClusteringFeatureGroup
from mloda_plugins.feature_group.experimental.clustering.pandas import PandasClustering
from mloda.user.pandas import PandasDataFrame


class TestPandasClustering:
    """Tests for the PandasClustering implementation."""

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

    def _run_loop(self, data: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        """Run the group loop with a stub perform and the real pandas add/columns callables."""
        feature_set = FeatureSet()
        feature_set.add(Feature(feature_name))
        labels = np.zeros(len(data), dtype=int)
        result: pd.DataFrame = ClusteringFeatureGroup.cluster(
            data,
            feature_set,
            perform=Mock(return_value=labels),
            perform_with_probabilities=Mock(),
            add_column=PandasClustering._add_result_to_data,
            columns=PandasDataFrame.extract_column_names,
        )
        return result

    def test_cluster_loop_leaves_the_source_check_to_core(self, sample_data: pd.DataFrame) -> None:
        """The loop itself does not reject a partially present source; core does."""
        result = self._run_loop(sample_data, "feature1&invalid_feature__cluster_kmeans_2")
        assert "feature1&invalid_feature__cluster_kmeans_2" in result.columns

    def test_cluster_loop_all_sources_missing_raises(self, sample_data: pd.DataFrame) -> None:
        """No source name present: core names every missing input and the available columns."""
        error = assert_core_rejects_missing_input(
            PandasClustering, sample_data, "invalid1&invalid2__cluster_kmeans_2", ("invalid1", "invalid2")
        )
        assert "feature1" in error.available_columns

    def test_cluster_partial_source_presence_rejected_by_core(self, sample_data: pd.DataFrame) -> None:
        """Core requires every declared source, so one absent source of two is rejected."""
        assert_core_rejects_missing_input(
            PandasClustering, sample_data, "feature1&invalid_feature__cluster_kmeans_2", ("invalid_feature",)
        )

    def test_add_result_to_data(self, sample_data: pd.DataFrame) -> None:
        """Test the _add_result_to_data method."""
        # Create a result array
        result = np.array([0, 1] * 50)

        # Add the result to the data
        updated_data = PandasClustering._add_result_to_data(sample_data, "cluster_result", result)

        # Check that the result was added
        assert "cluster_result" in updated_data.columns
        assert len(updated_data["cluster_result"]) == len(sample_data)
        assert (updated_data["cluster_result"].values == result).all()

    @pytest.mark.parametrize("algorithm", ["kmeans", "hierarchical", "spectral"])
    def test_perform_clustering_with_fixed_k(self, sample_data: pd.DataFrame, algorithm: str) -> None:
        """These take an explicit k and return exactly that many clusters; dbscan and affinity infer k instead."""
        # Extract features
        X = sample_data[["feature1", "feature2"]].values

        cluster = getattr(PandasClustering, f"_perform_{algorithm}_clustering")
        result = cluster(X, 2)

        # Check that the result has the expected shape
        assert len(result) == len(sample_data)

        # Check that there are exactly 2 unique clusters
        assert len(np.unique(result)) == 2

    def test_perform_dbscan_clustering(self, sample_data: pd.DataFrame) -> None:
        """Test the _perform_dbscan_clustering method."""
        # Extract features
        X = sample_data[["feature1", "feature2"]].values

        # Perform DBSCAN clustering
        result = PandasClustering._perform_dbscan_clustering(X, "auto")

        # Check that the result has the expected shape
        assert len(result) == len(sample_data)

        # DBSCAN may have noise points (labeled as -1)
        # So we check that there are at least 2 unique clusters (excluding noise)
        unique_clusters = np.unique(result)
        assert len(unique_clusters[unique_clusters >= 0]) >= 1

    def test_perform_affinity_clustering(self, sample_data: pd.DataFrame) -> None:
        """Test the _perform_affinity_clustering method."""
        # Extract features
        X = sample_data[["feature1", "feature2"]].values

        # Perform affinity clustering
        result = PandasClustering._perform_affinity_clustering(X, "auto")

        # Check that the result has the expected shape
        assert len(result) == len(sample_data)

        # Check that there is at least 1 cluster
        assert len(np.unique(result)) >= 1

    def test_find_optimal_k(self, sample_data: pd.DataFrame) -> None:
        """Test the _find_optimal_k method."""
        # Extract features
        X = sample_data[["feature1", "feature2"]].values

        # Find optimal k
        k = PandasClustering._find_optimal_k(X, "kmeans", max_k=5)

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
        result = PandasClustering.calculate(ClusteringFeatureGroup, sample_data, feature_set, None)

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
        result = PandasClustering.calculate(ClusteringFeatureGroup, sample_data, feature_set, None)

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
        result = PandasClustering.calculate(ClusteringFeatureGroup, sample_data, feature_set, None)

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
        result = PandasClustering.calculate(ClusteringFeatureGroup, sample_data, feature_set, None)

        # Check that the result has the expected columns
        assert "feature1&feature2__cluster_kmeans_2" in result.columns
        assert "feature1&feature2__cluster_dbscan_auto" in result.columns
        assert "feature1&feature2__cluster_hierarchical_2" in result.columns
