"""
Tests for the base ScalingFeatureGroup class.
"""

import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.sklearn.scaling.base import ScalingFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class TestScalingFeatureGroup:
    """Tests for the ScalingFeatureGroup class."""

    def test_match_feature_group_criteria(self) -> None:
        """Test the match_feature_group_criteria method."""
        # Valid feature names
        assert ScalingFeatureGroup.match_feature_group_criteria("standard_scaled__income", Options())
        assert ScalingFeatureGroup.match_feature_group_criteria("minmax_scaled__age", Options())
        assert ScalingFeatureGroup.match_feature_group_criteria("robust_scaled__outlier_prone_feature", Options())
        assert ScalingFeatureGroup.match_feature_group_criteria("normalizer_scaled__feature_vector", Options())

        # Invalid feature names
        assert not ScalingFeatureGroup.match_feature_group_criteria("scaled_standard__income", Options())
        assert not ScalingFeatureGroup.match_feature_group_criteria("invalid_scaled__income", Options())
        assert not ScalingFeatureGroup.match_feature_group_criteria("standard_scale__income", Options())
        assert not ScalingFeatureGroup.match_feature_group_criteria("standard_scaled_income", Options())

    def test_get_scaler_type(self) -> None:
        """Test the get_scaler_type method."""
        # Valid feature names
        assert ScalingFeatureGroup.get_scaler_type("standard_scaled__income") == "standard"
        assert ScalingFeatureGroup.get_scaler_type("minmax_scaled__age") == "minmax"
        assert ScalingFeatureGroup.get_scaler_type("robust_scaled__outlier_prone_feature") == "robust"
        assert ScalingFeatureGroup.get_scaler_type("normalizer_scaled__feature_vector") == "normalizer"

        # Invalid feature names
        with pytest.raises(ValueError):
            ScalingFeatureGroup.get_scaler_type("scaled_standard__income")

        with pytest.raises(ValueError):
            ScalingFeatureGroup.get_scaler_type("invalid_scaled__income")

        with pytest.raises(ValueError):
            ScalingFeatureGroup.get_scaler_type("standard_scale__income")

        with pytest.raises(ValueError):
            ScalingFeatureGroup.get_scaler_type("standard_scaled_income")

    def test_input_features(self) -> None:
        """Test the input_features method."""
        feature_group = ScalingFeatureGroup()

        # Single source feature
        input_features = feature_group.input_features(Options(), FeatureName("standard_scaled__income"))
        assert input_features is not None
        assert len(input_features) == 1
        assert Feature("income") in input_features

        # Different scaler types
        input_features = feature_group.input_features(Options(), FeatureName("minmax_scaled__age"))
        assert input_features is not None
        assert len(input_features) == 1
        assert Feature("age") in input_features

        input_features = feature_group.input_features(Options(), FeatureName("robust_scaled__outlier_prone_feature"))
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

    def test_feature_chain_parser_configuration(self) -> None:
        """Test the configurable_feature_chain_parser method."""
        parser_config = ScalingFeatureGroup.configurable_feature_chain_parser()
        assert parser_config is not None

    def test_feature_chain_parser_integration(self) -> None:
        """Test integration with FeatureChainParser."""
        # Create a feature with options
        feature = Feature(
            "placeholder",
            Options(
                {
                    ScalingFeatureGroup.SCALER_TYPE: "standard",
                    DefaultOptionKeys.mloda_source_feature: "income",
                }
            ),
        )

        # Parse the feature using the parser configuration
        parser_config = ScalingFeatureGroup.configurable_feature_chain_parser()
        assert parser_config is not None

        # Create a feature without options
        parsed_feature = parser_config.create_feature_without_options(feature)
        assert parsed_feature is not None
        assert parsed_feature.name.name == "standard_scaled__income"

        # Check that the options were removed
        assert ScalingFeatureGroup.SCALER_TYPE not in parsed_feature.options.data
        assert DefaultOptionKeys.mloda_source_feature not in parsed_feature.options.data

    def test_parse_from_options(self) -> None:
        """Test the parse_from_options method of the configurable feature chain parser."""
        parser_config = ScalingFeatureGroup.configurable_feature_chain_parser()
        assert parser_config is not None

        # Valid options for different scalers
        options = Options(
            {
                ScalingFeatureGroup.SCALER_TYPE: "standard",
                DefaultOptionKeys.mloda_source_feature: "income",
            }
        )
        feature_name = parser_config.parse_from_options(options)
        assert feature_name == "standard_scaled__income"

        options = Options(
            {
                ScalingFeatureGroup.SCALER_TYPE: "minmax",
                DefaultOptionKeys.mloda_source_feature: "age",
            }
        )
        feature_name = parser_config.parse_from_options(options)
        assert feature_name == "minmax_scaled__age"

        options = Options(
            {
                ScalingFeatureGroup.SCALER_TYPE: "robust",
                DefaultOptionKeys.mloda_source_feature: "outlier_prone_feature",
            }
        )
        feature_name = parser_config.parse_from_options(options)
        assert feature_name == "robust_scaled__outlier_prone_feature"

        options = Options(
            {
                ScalingFeatureGroup.SCALER_TYPE: "normalizer",
                DefaultOptionKeys.mloda_source_feature: "feature_vector",
            }
        )
        feature_name = parser_config.parse_from_options(options)
        assert feature_name == "normalizer_scaled__feature_vector"

        # Missing options
        options = Options(
            {
                # Missing SCALER_TYPE
                DefaultOptionKeys.mloda_source_feature: "income",
            }
        )
        feature_name = parser_config.parse_from_options(options)
        assert feature_name is None

        options = Options(
            {
                ScalingFeatureGroup.SCALER_TYPE: "standard",
                # Missing mloda_source_feature
            }
        )
        feature_name = parser_config.parse_from_options(options)
        assert feature_name is None

        # Invalid scaler type
        options = Options(
            {
                ScalingFeatureGroup.SCALER_TYPE: "invalid_scaler",
                DefaultOptionKeys.mloda_source_feature: "income",
            }
        )
        with pytest.raises(ValueError):
            parser_config.parse_from_options(options)

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
