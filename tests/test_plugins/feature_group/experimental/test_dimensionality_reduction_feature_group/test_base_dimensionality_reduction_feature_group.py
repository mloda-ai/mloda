"""
Tests for the base DimensionalityReductionFeatureGroup class.
"""

import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import DimensionalityReductionFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class TestDimensionalityReductionFeatureGroup:
    """Tests for the DimensionalityReductionFeatureGroup class."""

    def test_match_feature_group_criteria(self) -> None:
        """Test the match_feature_group_criteria method."""
        # Valid feature names
        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("pca_2d__customer_metrics", Options())
        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("tsne_3d__product_features", Options())
        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("isomap_5d__sensor_readings", Options())

        # Invalid feature names
        assert not DimensionalityReductionFeatureGroup.match_feature_group_criteria(
            "invalid_2d__customer_metrics", Options()
        )
        assert not DimensionalityReductionFeatureGroup.match_feature_group_criteria(
            "pca_invalid__customer_metrics", Options()
        )
        assert not DimensionalityReductionFeatureGroup.match_feature_group_criteria(
            "pca_2d_customer_metrics", Options()
        )

    def test_parse_reduction_prefix(self) -> None:
        """Test the parse_reduction_prefix method."""
        # Valid feature names
        algorithm, dimension = DimensionalityReductionFeatureGroup.parse_reduction_prefix("pca_2d__customer_metrics")
        assert algorithm == "pca"
        assert dimension == 2

        algorithm, dimension = DimensionalityReductionFeatureGroup.parse_reduction_prefix("tsne_3d__product_features")
        assert algorithm == "tsne"
        assert dimension == 3

        # Invalid feature names
        with pytest.raises(ValueError):
            DimensionalityReductionFeatureGroup.parse_reduction_prefix("invalid_2d__customer_metrics")

        with pytest.raises(ValueError):
            DimensionalityReductionFeatureGroup.parse_reduction_prefix("pca_invalid__customer_metrics")

        with pytest.raises(ValueError):
            DimensionalityReductionFeatureGroup.parse_reduction_prefix("pca_2d_customer_metrics")

    def test_get_algorithm(self) -> None:
        """Test the get_algorithm method."""
        assert DimensionalityReductionFeatureGroup.get_algorithm("pca_2d__customer_metrics") == "pca"
        assert DimensionalityReductionFeatureGroup.get_algorithm("tsne_3d__product_features") == "tsne"
        assert DimensionalityReductionFeatureGroup.get_algorithm("isomap_5d__sensor_readings") == "isomap"

    def test_get_dimension(self) -> None:
        """Test the get_dimension method."""
        assert DimensionalityReductionFeatureGroup.get_dimension("pca_2d__customer_metrics") == 2
        assert DimensionalityReductionFeatureGroup.get_dimension("tsne_3d__product_features") == 3
        assert DimensionalityReductionFeatureGroup.get_dimension("isomap_5d__sensor_readings") == 5

    def test_input_features(self) -> None:
        """Test the input_features method."""
        feature_group = DimensionalityReductionFeatureGroup()

        # Single source feature
        input_features = feature_group.input_features(Options(), FeatureName("pca_2d__customer_metrics"))
        assert input_features is not None
        assert len(input_features) == 1
        assert Feature("customer_metrics") in input_features

        # Multiple source features (comma-separated)
        input_features = feature_group.input_features(Options(), FeatureName("pca_2d__feature1,feature2,feature3"))
        assert input_features is not None
        assert len(input_features) == 3
        assert Feature("feature1") in input_features
        assert Feature("feature2") in input_features
        assert Feature("feature3") in input_features

    def test_feature_chain_parser_configuration(self) -> None:
        """Test the configurable_feature_chain_parser method."""
        parser_config = DimensionalityReductionFeatureGroup.configurable_feature_chain_parser()
        assert parser_config is not None

    def test_feature_chain_parser_integration(self) -> None:
        """Test integration with FeatureChainParser."""
        # Create a feature with options
        feature = Feature(
            "placeholder",
            Options(
                {
                    DimensionalityReductionFeatureGroup.ALGORITHM: "pca",
                    DimensionalityReductionFeatureGroup.DIMENSION: 2,
                    DefaultOptionKeys.mloda_source_feature: "customer_metrics",
                }
            ),
        )

        # Parse the feature using the parser configuration
        parser_config = DimensionalityReductionFeatureGroup.configurable_feature_chain_parser()
        assert parser_config is not None

        # Create a feature without options
        parsed_feature = parser_config.create_feature_without_options(feature)
        assert parsed_feature is not None
        assert parsed_feature.name.name == "pca_2d__customer_metrics"

        # Check that the options were removed
        assert DimensionalityReductionFeatureGroup.ALGORITHM not in parsed_feature.options.data
        assert DimensionalityReductionFeatureGroup.DIMENSION not in parsed_feature.options.data
        assert DefaultOptionKeys.mloda_source_feature not in parsed_feature.options.data

    def test_parse_from_options(self) -> None:
        """Test the parse_from_options method of the configurable feature chain parser."""
        parser_config = DimensionalityReductionFeatureGroup.configurable_feature_chain_parser()
        assert parser_config is not None

        # Valid options
        options = Options(
            {
                DimensionalityReductionFeatureGroup.ALGORITHM: "pca",
                DimensionalityReductionFeatureGroup.DIMENSION: 2,
                DefaultOptionKeys.mloda_source_feature: "customer_metrics",
            }
        )
        feature_name = parser_config.parse_from_options(options)
        assert feature_name == "pca_2d__customer_metrics"

        # Multiple source features
        options = Options(
            {
                DimensionalityReductionFeatureGroup.ALGORITHM: "tsne",
                DimensionalityReductionFeatureGroup.DIMENSION: 3,
                DefaultOptionKeys.mloda_source_feature: ["feature1", "feature2", "feature3"],
            }
        )
        feature_name = parser_config.parse_from_options(options)
        assert feature_name == "tsne_3d__feature1,feature2,feature3"

        # Missing options
        options = Options(
            {
                DimensionalityReductionFeatureGroup.ALGORITHM: "pca",
                # Missing DIMENSION
                DefaultOptionKeys.mloda_source_feature: "customer_metrics",
            }
        )
        feature_name = parser_config.parse_from_options(options)
        assert feature_name is None

        # Invalid algorithm
        options = Options(
            {
                DimensionalityReductionFeatureGroup.ALGORITHM: "invalid",
                DimensionalityReductionFeatureGroup.DIMENSION: 2,
                DefaultOptionKeys.mloda_source_feature: "customer_metrics",
            }
        )
        with pytest.raises(ValueError):
            parser_config.parse_from_options(options)

        # Invalid dimension
        options = Options(
            {
                DimensionalityReductionFeatureGroup.ALGORITHM: "pca",
                DimensionalityReductionFeatureGroup.DIMENSION: -1,
                DefaultOptionKeys.mloda_source_feature: "customer_metrics",
            }
        )
        with pytest.raises(ValueError):
            parser_config.parse_from_options(options)
