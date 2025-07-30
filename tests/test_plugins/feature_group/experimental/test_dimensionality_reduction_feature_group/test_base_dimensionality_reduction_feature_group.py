"""
Tests for the base DimensionalityReductionFeatureGroup class.
"""

import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import DimensionalityReductionFeatureGroup


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
