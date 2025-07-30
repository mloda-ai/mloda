"""
Tests for the base ClusteringFeatureGroup class.
"""

import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.clustering.base import ClusteringFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class TestClusteringFeatureGroup:
    """Tests for the ClusteringFeatureGroup class."""

    def test_match_feature_group_criteria(self) -> None:
        """Test the match_feature_group_criteria method."""
        # Valid feature names
        assert ClusteringFeatureGroup.match_feature_group_criteria("cluster_kmeans_5__customer_behavior", Options())
        assert ClusteringFeatureGroup.match_feature_group_criteria("cluster_dbscan_auto__sensor_readings", Options())
        assert ClusteringFeatureGroup.match_feature_group_criteria(
            "cluster_hierarchical_3__transaction_patterns", Options()
        )

        # Invalid feature names
        assert not ClusteringFeatureGroup.match_feature_group_criteria("kmeans_cluster_5__customer_behavior", Options())
        assert not ClusteringFeatureGroup.match_feature_group_criteria(
            "cluster_invalid_5__customer_behavior", Options()
        )
        assert not ClusteringFeatureGroup.match_feature_group_criteria(
            "cluster_kmeans_invalid__customer_behavior", Options()
        )
        assert not ClusteringFeatureGroup.match_feature_group_criteria("cluster_kmeans_5_customer_behavior", Options())

    def test_parse_clustering_prefix(self) -> None:
        """Test the parse_clustering_prefix method."""
        # Valid feature names
        algorithm, k_value = ClusteringFeatureGroup.parse_clustering_prefix("cluster_kmeans_5__customer_behavior")
        assert algorithm == "kmeans"
        assert k_value == "5"

        algorithm, k_value = ClusteringFeatureGroup.parse_clustering_prefix("cluster_dbscan_auto__sensor_readings")
        assert algorithm == "dbscan"
        assert k_value == "auto"

        # Invalid feature names
        with pytest.raises(ValueError):
            ClusteringFeatureGroup.parse_clustering_prefix("kmeans_cluster_5__customer_behavior")

        with pytest.raises(ValueError):
            ClusteringFeatureGroup.parse_clustering_prefix("cluster_invalid_5__customer_behavior")

        with pytest.raises(ValueError):
            ClusteringFeatureGroup.parse_clustering_prefix("cluster_kmeans_invalid__customer_behavior")

        with pytest.raises(ValueError):
            ClusteringFeatureGroup.parse_clustering_prefix("cluster_kmeans_5_customer_behavior")

    def test_get_algorithm(self) -> None:
        """Test extracting algorithm from feature names using parse_clustering_prefix."""
        algorithm, _ = ClusteringFeatureGroup.parse_clustering_prefix("cluster_kmeans_5__customer_behavior")
        assert algorithm == "kmeans"

        algorithm, _ = ClusteringFeatureGroup.parse_clustering_prefix("cluster_dbscan_auto__sensor_readings")
        assert algorithm == "dbscan"

        algorithm, _ = ClusteringFeatureGroup.parse_clustering_prefix("cluster_hierarchical_3__transaction_patterns")
        assert algorithm == "hierarchical"

    def test_get_k_value(self) -> None:
        """Test the get_k_value method."""
        assert ClusteringFeatureGroup.get_k_value("cluster_kmeans_5__customer_behavior") == 5
        assert ClusteringFeatureGroup.get_k_value("cluster_dbscan_auto__sensor_readings") == "auto"
        assert ClusteringFeatureGroup.get_k_value("cluster_hierarchical_3__transaction_patterns") == 3

    def test_input_features(self) -> None:
        """Test the input_features method."""
        feature_group = ClusteringFeatureGroup()

        # Single source feature
        input_features = feature_group.input_features(Options(), FeatureName("cluster_kmeans_5__customer_behavior"))
        assert input_features is not None
        assert len(input_features) == 1
        assert Feature("customer_behavior") in input_features

        # Multiple source features (comma-separated)
        input_features = feature_group.input_features(
            Options(), FeatureName("cluster_kmeans_5__feature1,feature2,feature3")
        )
        assert input_features is not None
        assert len(input_features) == 3
        assert Feature("feature1") in input_features
        assert Feature("feature2") in input_features
        assert Feature("feature3") in input_features
