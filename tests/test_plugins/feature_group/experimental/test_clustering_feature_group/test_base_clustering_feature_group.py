"""
Tests for the base ClusteringFeatureGroup class.
"""

import pytest

from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda_plugins.feature_group.experimental.clustering.base import ClusteringFeatureGroup
from mloda_plugins.feature_group.experimental.clustering.pandas import PandasClusteringFeatureGroup


class TestClusteringFeatureGroup:
    """Tests for the ClusteringFeatureGroup class."""

    def test_match_feature_group_criteria(self) -> None:
        """Test the match_feature_group_criteria method."""
        # Valid feature names
        assert ClusteringFeatureGroup.match_feature_group_criteria("customer_behavior__cluster_kmeans_5", Options())
        assert ClusteringFeatureGroup.match_feature_group_criteria("sensor_readings__cluster_dbscan_auto", Options())
        assert ClusteringFeatureGroup.match_feature_group_criteria(
            "transaction_patterns__cluster_hierarchical_3", Options()
        )

        # Invalid feature names
        assert not ClusteringFeatureGroup.match_feature_group_criteria("customer_behavior__kmeans_cluster_5", Options())
        assert not ClusteringFeatureGroup.match_feature_group_criteria(
            "customer_behavior__cluster_invalid_5", Options()
        )
        assert not ClusteringFeatureGroup.match_feature_group_criteria(
            "customer_behavior__cluster_kmeans_invalid", Options()
        )
        assert not ClusteringFeatureGroup.match_feature_group_criteria("customer_behavior_cluster_kmeans_5", Options())

    def test_parse_clustering_prefix(self) -> None:
        """Test the parse_clustering_prefix method."""
        # Valid feature names
        algorithm, k_value = ClusteringFeatureGroup.parse_clustering_prefix("customer_behavior__cluster_kmeans_5")
        assert algorithm == "kmeans"
        assert k_value == "5"

        algorithm, k_value = ClusteringFeatureGroup.parse_clustering_prefix("sensor_readings__cluster_dbscan_auto")
        assert algorithm == "dbscan"
        assert k_value == "auto"

        # Invalid feature names
        with pytest.raises(ValueError):
            ClusteringFeatureGroup.parse_clustering_prefix("customer_behavior__kmeans_cluster_5")

        with pytest.raises(ValueError):
            ClusteringFeatureGroup.parse_clustering_prefix("customer_behavior__cluster_invalid_5")

        with pytest.raises(ValueError):
            ClusteringFeatureGroup.parse_clustering_prefix("customer_behavior__cluster_kmeans_invalid")

        with pytest.raises(ValueError):
            ClusteringFeatureGroup.parse_clustering_prefix("customer_behavior_cluster_kmeans_5")

    @pytest.mark.parametrize(
        ("feature_name", "algorithm", "k_value"),
        [
            ("a__standard_scaled__cluster_kmeans_3", "kmeans", "3"),
            ("sensor__normalized__cluster_dbscan_auto", "dbscan", "auto"),
        ],
    )
    def test_parse_clustering_prefix_with_chained_source(
        self, feature_name: str, algorithm: str, k_value: str
    ) -> None:
        """The clustering suffix follows the final chain separator, not one in its source."""
        assert ClusteringFeatureGroup.parse_clustering_prefix(feature_name) == (algorithm, k_value)

    @pytest.mark.parametrize(
        "feature_name",
        [
            "a__standard_scaled__cluster_kmeans_3",
            "sensor__normalized__cluster_dbscan_auto",
        ],
    )
    def test_match_feature_group_criteria_with_chained_source(self, feature_name: str) -> None:
        """A clustering operation can consume the output of an earlier feature-chain operation."""
        assert ClusteringFeatureGroup.match_feature_group_criteria(feature_name, Options())

    def test_get_algorithm(self) -> None:
        """Test extracting algorithm from feature names using parse_clustering_prefix."""
        algorithm, _ = ClusteringFeatureGroup.parse_clustering_prefix("customer_behavior__cluster_kmeans_5")
        assert algorithm == "kmeans"

        algorithm, _ = ClusteringFeatureGroup.parse_clustering_prefix("sensor_readings__cluster_dbscan_auto")
        assert algorithm == "dbscan"

        algorithm, _ = ClusteringFeatureGroup.parse_clustering_prefix("transaction_patterns__cluster_hierarchical_3")
        assert algorithm == "hierarchical"

    def test_get_k_value(self) -> None:
        """Test the get_k_value method."""
        assert ClusteringFeatureGroup.get_k_value("customer_behavior__cluster_kmeans_5") == 5
        assert ClusteringFeatureGroup.get_k_value("sensor_readings__cluster_dbscan_auto") == "auto"
        assert ClusteringFeatureGroup.get_k_value("transaction_patterns__cluster_hierarchical_3") == 3

    def test_input_features(self) -> None:
        """Test the input_features method."""
        feature_group = PandasClusteringFeatureGroup()

        # Single source feature
        input_features = feature_group.input_features(Options(), FeatureName("customer_behavior__cluster_kmeans_5"))
        assert input_features is not None
        assert len(input_features) == 1
        assert Feature("customer_behavior") in input_features

        # Multiple source features (ampersand-separated)
        input_features = feature_group.input_features(
            Options(), FeatureName("feature1&feature2&feature3__cluster_kmeans_5")
        )
        assert input_features is not None
        assert len(input_features) == 3
        assert Feature("feature1") in input_features
        assert Feature("feature2") in input_features
        assert Feature("feature3") in input_features
