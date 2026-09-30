"""
Tests for the base ClusteringFeatureGroup class.
"""

import pytest

from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
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

    def test_chained_source_name_matches_and_extracts_the_chained_source(self) -> None:
        """The pattern parses from the last suffix, so a chained source is the whole prefix."""
        name = "s__mean_imputed__cluster_kmeans_5"
        assert ClusteringFeatureGroup.match_feature_group_criteria(name, Options())
        assert ClusteringFeatureGroup._extract_source_features(Feature(name)) == ["s__mean_imputed"]

    def test_zero_k_value_is_a_recorded_non_match_naming_k_value(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        assert not ClusteringFeatureGroup.match_feature_group_criteria("x__cluster_kmeans_0", Options())
        assert "k_value" in rejection_window["ClusteringFeatureGroup"].reason

    def test_auto_k_value_still_matches_and_records_nothing(self, rejection_window: dict[str, MatchRejection]) -> None:
        assert ClusteringFeatureGroup.match_feature_group_criteria("x__cluster_kmeans_auto", Options())
        assert rejection_window == {}

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("x__cluster_kmeans_5", ("kmeans", 5)),
            ("x__cluster_dbscan_auto", ("dbscan", "auto")),
            ("s__mean_imputed__cluster_kmeans_5", ("kmeans", 5)),
        ],
    )
    def test_extract_clustering_params_reads_the_named_captures(
        self, name: str, expected: tuple[str, int | str]
    ) -> None:
        assert ClusteringFeatureGroup._extract_clustering_params(Feature(name)) == expected

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

    def test_public_helpers_accept_chained_names(self) -> None:
        chained = "s__mean_imputed__cluster_kmeans_5"

        assert ClusteringFeatureGroup.get_k_value(chained) == 5
        assert ClusteringFeatureGroup.parse_clustering_prefix(chained) == ("kmeans", "5")

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
