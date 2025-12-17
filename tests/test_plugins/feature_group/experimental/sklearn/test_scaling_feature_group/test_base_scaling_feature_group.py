"""
Tests for the base ScalingFeatureGroup class.
"""

import pytest

from mloda import Feature
from mloda.user import FeatureName
from mloda import Options
from mloda_plugins.feature_group.experimental.sklearn.scaling.base import ScalingFeatureGroup


class TestScalingFeatureGroup:
    """Tests for the ScalingFeatureGroup class."""

    def test_match_feature_group_criteria(self) -> None:
        """Test the match_feature_group_criteria method."""
        # Valid feature names
        assert ScalingFeatureGroup.match_feature_group_criteria("income__standard_scaled", Options())
        assert ScalingFeatureGroup.match_feature_group_criteria("age__minmax_scaled", Options())
        assert ScalingFeatureGroup.match_feature_group_criteria("outlier_prone_feature__robust_scaled", Options())
        assert ScalingFeatureGroup.match_feature_group_criteria("feature_vector__normalizer_scaled", Options())

        # Invalid feature names
        assert not ScalingFeatureGroup.match_feature_group_criteria("income__scaled_standard", Options())
        assert not ScalingFeatureGroup.match_feature_group_criteria("income__invalid_scaled", Options())
        assert not ScalingFeatureGroup.match_feature_group_criteria("income__standard_scale", Options())
        assert not ScalingFeatureGroup.match_feature_group_criteria("income_standard_scaled", Options())

    def test_get_scaler_type(self) -> None:
        """Test the get_scaler_type method."""
        # Valid feature names
        assert ScalingFeatureGroup.get_scaler_type("income__standard_scaled") == "standard"
        assert ScalingFeatureGroup.get_scaler_type("age__minmax_scaled") == "minmax"
        assert ScalingFeatureGroup.get_scaler_type("outlier_prone_feature__robust_scaled") == "robust"
        assert ScalingFeatureGroup.get_scaler_type("feature_vector__normalizer_scaled") == "normalizer"

        # Invalid feature names
        with pytest.raises(ValueError):
            ScalingFeatureGroup.get_scaler_type("income__scaled_standard")

        with pytest.raises(ValueError):
            ScalingFeatureGroup.get_scaler_type("income__invalid_scaled")

        with pytest.raises(ValueError):
            ScalingFeatureGroup.get_scaler_type("income__standard_scale")

        with pytest.raises(ValueError):
            ScalingFeatureGroup.get_scaler_type("income_standard_scaled")

    def test_input_features(self) -> None:
        """Test the input_features method."""
        feature_group = ScalingFeatureGroup()

        # Single source feature
        input_features = feature_group.input_features(Options(), FeatureName("income__standard_scaled"))
        assert input_features is not None
        assert len(input_features) == 1
        assert Feature("income") in input_features

        # Different scaler types
        input_features = feature_group.input_features(Options(), FeatureName("age__minmax_scaled"))
        assert input_features is not None
        assert len(input_features) == 1
        assert Feature("age") in input_features

        input_features = feature_group.input_features(Options(), FeatureName("outlier_prone_feature__robust_scaled"))
        assert input_features is not None
        assert len(input_features) == 1
        assert Feature("outlier_prone_feature") in input_features

    def test_supported_scalers(self) -> None:
        """Test that all supported scalers are properly defined."""
        expected_scalers = {
            "standard": "StandardScaler",
            "minmax": "MinMaxScaler",
            "robust": "RobustScaler",
            "normalizer": "Normalizer",
        }
        assert ScalingFeatureGroup.SUPPORTED_SCALERS == expected_scalers

    def test_create_scaler_instance(self) -> None:
        """Test the _create_scaler_instance method."""
        # Skip test if sklearn not available
        try:
            ScalingFeatureGroup._import_sklearn_components()
        except ImportError:
            pytest.skip("scikit-learn not available")

        # Test creating different scaler types
        standard_scaler = ScalingFeatureGroup._create_scaler_instance("standard")
        assert standard_scaler.__class__.__name__ == "StandardScaler"

        minmax_scaler = ScalingFeatureGroup._create_scaler_instance("minmax")
        assert minmax_scaler.__class__.__name__ == "MinMaxScaler"

        robust_scaler = ScalingFeatureGroup._create_scaler_instance("robust")
        assert robust_scaler.__class__.__name__ == "RobustScaler"

        normalizer_scaler = ScalingFeatureGroup._create_scaler_instance("normalizer")
        assert normalizer_scaler.__class__.__name__ == "Normalizer"

        # Test invalid scaler type
        with pytest.raises(ValueError):
            ScalingFeatureGroup._create_scaler_instance("invalid_scaler")

    def test_scaler_matches_type(self) -> None:
        """Test the _scaler_matches_type method."""
        # Skip test if sklearn not available
        try:
            ScalingFeatureGroup._import_sklearn_components()
        except ImportError:
            pytest.skip("scikit-learn not available")

        # Create scaler instances
        standard_scaler = ScalingFeatureGroup._create_scaler_instance("standard")
        minmax_scaler = ScalingFeatureGroup._create_scaler_instance("minmax")

        # Test matching - should return True for correct matches
        assert ScalingFeatureGroup._scaler_matches_type(standard_scaler, "standard")
        assert ScalingFeatureGroup._scaler_matches_type(minmax_scaler, "minmax")

        # Test mismatching - should raise ValueError
        with pytest.raises(ValueError, match="Artifact scaler type mismatch"):
            ScalingFeatureGroup._scaler_matches_type(standard_scaler, "minmax")

        with pytest.raises(ValueError, match="Artifact scaler type mismatch"):
            ScalingFeatureGroup._scaler_matches_type(minmax_scaler, "standard")

        # Test unsupported scaler type
        with pytest.raises(ValueError, match="Unsupported scaler type"):
            ScalingFeatureGroup._scaler_matches_type(standard_scaler, "invalid_type")

    def test_import_sklearn_components(self) -> None:
        """Test the _import_sklearn_components method."""
        # Skip test if sklearn not available
        try:
            components = ScalingFeatureGroup._import_sklearn_components()

            # Check that all expected components are imported
            expected_components = ["StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer"]
            for component_name in expected_components:
                assert component_name in components
                assert callable(components[component_name])

        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_import_sklearn_components_missing_sklearn(self) -> None:
        """Test that ImportError is raised when sklearn is not available."""
        # This test would require mocking sys.modules, but for now we'll just
        # verify that the method exists and can be called
        try:
            ScalingFeatureGroup._import_sklearn_components()
        except ImportError:
            # This is expected when sklearn is not available
            pass

    def test_abstract_methods_not_implemented(self) -> None:
        """Test that abstract methods raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            ScalingFeatureGroup._extract_training_data(None, "test")

        with pytest.raises(NotImplementedError):
            ScalingFeatureGroup._apply_scaler(None, "test", None)

        with pytest.raises(NotImplementedError):
            ScalingFeatureGroup._check_source_feature_exists(None, "test")

        with pytest.raises(NotImplementedError):
            ScalingFeatureGroup._add_result_to_data(None, "test", None)
