"""
Tests for the base DimensionalityReductionFeatureGroup class.
"""

import pytest

from mloda import Feature
from mloda.user import FeatureName
from mloda import Options
from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import DimensionalityReductionFeatureGroup
from mloda_plugins.feature_group.experimental.dimensionality_reduction.pandas import (
    PandasDimensionalityReductionFeatureGroup,
)


class TestDimensionalityReductionFeatureGroup:
    """Tests for the DimensionalityReductionFeatureGroup class."""

    def test_match_feature_group_criteria(self) -> None:
        """Test the match_feature_group_criteria method."""
        # Valid feature names
        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("customer_metrics__pca_2d", Options())
        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("product_features__tsne_3d", Options())
        assert DimensionalityReductionFeatureGroup.match_feature_group_criteria("sensor_readings__isomap_5d", Options())

        # Invalid feature names
        assert not DimensionalityReductionFeatureGroup.match_feature_group_criteria(
            "customer_metrics__invalid_2d", Options()
        )
        assert not DimensionalityReductionFeatureGroup.match_feature_group_criteria(
            "customer_metrics__pca_invalid", Options()
        )
        assert not DimensionalityReductionFeatureGroup.match_feature_group_criteria(
            "customer_metrics_pca_2d", Options()
        )

    def test_parse_reduction_suffix(self) -> None:
        """Test the parse_reduction_suffix method."""
        # Valid feature names
        algorithm, dimension = DimensionalityReductionFeatureGroup.parse_reduction_suffix("customer_metrics__pca_2d")
        assert algorithm == "pca"
        assert dimension == 2

        algorithm, dimension = DimensionalityReductionFeatureGroup.parse_reduction_suffix("product_features__tsne_3d")
        assert algorithm == "tsne"
        assert dimension == 3

        # Invalid feature names
        with pytest.raises(ValueError):
            DimensionalityReductionFeatureGroup.parse_reduction_suffix("customer_metrics__invalid_2d")

        with pytest.raises(ValueError):
            DimensionalityReductionFeatureGroup.parse_reduction_suffix("customer_metrics__pca_invalid")

        with pytest.raises(ValueError):
            DimensionalityReductionFeatureGroup.parse_reduction_suffix("customer_metrics_pca_2d")

    def test_input_features(self) -> None:
        """Test the input_features method."""
        feature_group = PandasDimensionalityReductionFeatureGroup()

        # Single source feature
        input_features = feature_group.input_features(Options(), FeatureName("customer_metrics__pca_2d"))
        assert input_features is not None
        assert len(input_features) == 1
        assert Feature("customer_metrics") in input_features

        # Multiple source features (comma-separated)
        input_features = feature_group.input_features(Options(), FeatureName("feature1,feature2,feature3__pca_2d"))
        assert input_features is not None
        assert len(input_features) == 3
        assert Feature("feature1") in input_features
        assert Feature("feature2") in input_features
        assert Feature("feature3") in input_features
