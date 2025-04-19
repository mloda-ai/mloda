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
        """Test the get_algorithm method."""
        assert ClusteringFeatureGroup.get_algorithm("cluster_kmeans_5__customer_behavior") == "kmeans"
        assert ClusteringFeatureGroup.get_algorithm("cluster_dbscan_auto__sensor_readings") == "dbscan"
        assert ClusteringFeatureGroup.get_algorithm("cluster_hierarchical_3__transaction_patterns") == "hierarchical"

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

    def test_feature_chain_parser_configuration(self) -> None:
        """Test the configurable_feature_chain_parser method."""
        parser_config = ClusteringFeatureGroup.configurable_feature_chain_parser()
        assert parser_config is not None

    def test_feature_chain_parser_integration(self) -> None:
        """Test integration with FeatureChainParser."""
        # Create a feature with options
        feature = Feature(
            "placeholder",
            Options(
                {
                    ClusteringFeatureGroup.ALGORITHM: "kmeans",
                    ClusteringFeatureGroup.K_VALUE: 5,
                    DefaultOptionKeys.mloda_source_feature: "customer_behavior",
                }
            ),
        )

        # Parse the feature using the parser configuration
        parser_config = ClusteringFeatureGroup.configurable_feature_chain_parser()
        assert parser_config is not None

        # Create a feature without options
        parsed_feature = parser_config.create_feature_without_options(feature)
        assert parsed_feature is not None
        assert parsed_feature.name.name == "cluster_kmeans_5__customer_behavior"

        # Check that the options were removed
        assert ClusteringFeatureGroup.ALGORITHM not in parsed_feature.options.data
        assert ClusteringFeatureGroup.K_VALUE not in parsed_feature.options.data
        assert DefaultOptionKeys.mloda_source_feature not in parsed_feature.options.data

    def test_parse_from_options(self) -> None:
        """Test the parse_from_options method of the configurable feature chain parser."""
        parser_config = ClusteringFeatureGroup.configurable_feature_chain_parser()
        assert parser_config is not None

        # Valid options
        options = Options(
            {
                ClusteringFeatureGroup.ALGORITHM: "kmeans",
                ClusteringFeatureGroup.K_VALUE: 5,
                DefaultOptionKeys.mloda_source_feature: "customer_behavior",
            }
        )
        feature_name = parser_config.parse_from_options(options)
        assert feature_name == "cluster_kmeans_5__customer_behavior"

        # Auto k_value
        options = Options(
            {
                ClusteringFeatureGroup.ALGORITHM: "dbscan",
                ClusteringFeatureGroup.K_VALUE: "auto",
                DefaultOptionKeys.mloda_source_feature: "sensor_readings",
            }
        )
        feature_name = parser_config.parse_from_options(options)
        assert feature_name == "cluster_dbscan_auto__sensor_readings"

        # Multiple source features
        options = Options(
            {
                ClusteringFeatureGroup.ALGORITHM: "kmeans",
                ClusteringFeatureGroup.K_VALUE: 5,
                DefaultOptionKeys.mloda_source_feature: ["feature1", "feature2", "feature3"],
            }
        )
        feature_name = parser_config.parse_from_options(options)
        assert feature_name == "cluster_kmeans_5__feature1,feature2,feature3"

        # Missing options
        options = Options(
            {
                ClusteringFeatureGroup.ALGORITHM: "kmeans",
                # Missing K_VALUE
                DefaultOptionKeys.mloda_source_feature: "customer_behavior",
            }
        )
        feature_name = parser_config.parse_from_options(options)
        assert feature_name is None

        # Invalid algorithm
        options = Options(
            {
                ClusteringFeatureGroup.ALGORITHM: "invalid",
                ClusteringFeatureGroup.K_VALUE: 5,
                DefaultOptionKeys.mloda_source_feature: "customer_behavior",
            }
        )
        with pytest.raises(ValueError):
            parser_config.parse_from_options(options)

        # Invalid k_value
        options = Options(
            {
                ClusteringFeatureGroup.ALGORITHM: "kmeans",
                ClusteringFeatureGroup.K_VALUE: -1,
                DefaultOptionKeys.mloda_source_feature: "customer_behavior",
            }
        )
        with pytest.raises(ValueError):
            parser_config.parse_from_options(options)
