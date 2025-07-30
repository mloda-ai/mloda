"""
Unit tests for the base EncodingFeatureGroup class.
"""

from typing import Any
import pytest
from unittest.mock import Mock, patch

from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.sklearn.encoding.base import EncodingFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class TestEncodingFeatureGroup:
    """Test cases for the base EncodingFeatureGroup class."""

    def test_match_feature_group_criteria_valid_onehot(self) -> None:
        """Test that valid onehot encoding feature names match the criteria."""
        valid_names = [
            "onehot_encoded__category",
            "onehot_encoded__status",
            "onehot_encoded__product_type",
        ]

        for name in valid_names:
            assert EncodingFeatureGroup.match_feature_group_criteria(FeatureName(name), Options({})), (
                f"Feature name '{name}' should match criteria"
            )

    def test_match_feature_group_criteria_valid_label(self) -> None:
        """Test that valid label encoding feature names match the criteria."""
        valid_names = [
            "label_encoded__category",
            "label_encoded__status",
            "label_encoded__priority",
        ]

        for name in valid_names:
            assert EncodingFeatureGroup.match_feature_group_criteria(FeatureName(name), Options({})), (
                f"Feature name '{name}' should match criteria"
            )

    def test_match_feature_group_criteria_valid_ordinal(self) -> None:
        """Test that valid ordinal encoding feature names match the criteria."""
        valid_names = [
            "ordinal_encoded__category",
            "ordinal_encoded__grade",
            "ordinal_encoded__size",
        ]

        for name in valid_names:
            assert EncodingFeatureGroup.match_feature_group_criteria(FeatureName(name), Options({})), (
                f"Feature name '{name}' should match criteria"
            )

    def test_match_feature_group_criteria_invalid(self) -> None:
        """Test that invalid feature names do not match the criteria."""
        invalid_names = [
            "invalid_encoded__category",
            "onehot_scaled__category",  # Wrong suffix
            "category",  # No prefix
            "encoded__category",  # Missing encoder type
            "onehot__category",  # Missing 'encoded'
            "",  # Empty string
        ]

        for name in invalid_names:
            assert not EncodingFeatureGroup.match_feature_group_criteria(FeatureName(name), Options({})), (
                f"Feature name '{name}' should not match criteria"
            )

    def test_match_feature_group_criteria_unsupported_encoder(self) -> None:
        """Test that unsupported encoder types do not match the criteria."""
        unsupported_names = [
            "binary_encoded__category",
            "target_encoded__category",
            "frequency_encoded__category",
        ]

        for name in unsupported_names:
            assert not EncodingFeatureGroup.match_feature_group_criteria(FeatureName(name), Options({})), (
                f"Feature name '{name}' should not match criteria (unsupported encoder)"
            )

    def test_get_encoder_type_valid(self) -> None:
        """Test extraction of encoder type from valid feature names."""
        test_cases = [
            ("onehot_encoded__category", "onehot"),
            ("label_encoded__status", "label"),
            ("ordinal_encoded__priority", "ordinal"),
        ]

        for feature_name, expected_encoder in test_cases:
            encoder_type = EncodingFeatureGroup.get_encoder_type(feature_name)
            assert encoder_type == expected_encoder, f"Expected {expected_encoder}, got {encoder_type}"

    def test_get_encoder_type_invalid(self) -> None:
        """Test that invalid feature names raise ValueError when extracting encoder type."""
        invalid_names = [
            "invalid_encoded__category",
            "category",
            "encoded__category",
        ]

        for name in invalid_names:
            with pytest.raises(ValueError, match="Invalid encoding feature name format"):
                EncodingFeatureGroup.get_encoder_type(name)

    def test_parse_feature_prefix_valid(self) -> None:
        """Test parsing of feature prefix from valid feature names."""
        from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser

        test_cases = [
            ("onehot_encoded__category", "category"),
            ("label_encoded__customer_status", "customer_status"),
            ("ordinal_encoded__product_priority", "product_priority"),
        ]

        for feature_name, expected_source in test_cases:
            source_feature = FeatureChainParser.extract_source_feature(
                feature_name, EncodingFeatureGroup.PREFIX_PATTERN
            )
            assert source_feature == expected_source, f"Expected {expected_source}, got {source_feature}"

    def test_supported_encoders(self) -> None:
        """Test that all supported encoders are properly defined."""
        expected_encoders = {
            "onehot": "OneHotEncoder",
            "label": "LabelEncoder",
            "ordinal": "OrdinalEncoder",
        }

        assert EncodingFeatureGroup.SUPPORTED_ENCODERS == expected_encoders

    def test_encoder_type_option_key(self) -> None:
        """Test that encoder type option key is properly defined."""
        assert EncodingFeatureGroup.ENCODER_TYPE == "encoder_type"

    def test_prefix_pattern(self) -> None:
        """Test that prefix pattern is properly defined."""
        expected_pattern = r"^(onehot|label|ordinal)_encoded__"
        assert EncodingFeatureGroup.PREFIX_PATTERN == expected_pattern

    def test_input_features(self) -> None:
        """Test input_features method extracts correct source features."""
        feature_group = EncodingFeatureGroup()
        feature_name = FeatureName("onehot_encoded__category")
        options = Options({})

        input_features = feature_group.input_features(options, feature_name)
        assert input_features is not None
        assert len(input_features) == 1

        feature = next(iter(input_features))
        assert feature.name == "category"

    def test_artifact_type(self) -> None:
        """Test that artifact type is properly configured."""
        from mloda_plugins.feature_group.experimental.sklearn.sklearn_artifact import SklearnArtifact

        artifact_type = EncodingFeatureGroup.artifact()
        assert artifact_type == SklearnArtifact

    @patch(
        "mloda_plugins.feature_group.experimental.sklearn.encoding.base.EncodingFeatureGroup._import_sklearn_components"
    )
    def test_create_encoder_instance_onehot(self, mock_import: Any) -> None:
        """Test creation of OneHotEncoder instance with proper configuration."""
        # Mock sklearn components
        mock_onehot = Mock()
        mock_import.return_value = {"OneHotEncoder": mock_onehot}

        EncodingFeatureGroup._create_encoder_instance("onehot")

        # Verify OneHotEncoder was called with correct parameters
        mock_onehot.assert_called_once_with(handle_unknown="ignore", drop=None)

    @patch(
        "mloda_plugins.feature_group.experimental.sklearn.encoding.base.EncodingFeatureGroup._import_sklearn_components"
    )
    def test_create_encoder_instance_ordinal(self, mock_import: Any) -> None:
        """Test creation of OrdinalEncoder instance with proper configuration."""
        # Mock sklearn components
        mock_ordinal = Mock()
        mock_import.return_value = {"OrdinalEncoder": mock_ordinal}

        EncodingFeatureGroup._create_encoder_instance("ordinal")

        # Verify OrdinalEncoder was called with correct parameters
        mock_ordinal.assert_called_once_with(handle_unknown="use_encoded_value", unknown_value=-1)

    @patch(
        "mloda_plugins.feature_group.experimental.sklearn.encoding.base.EncodingFeatureGroup._import_sklearn_components"
    )
    def test_create_encoder_instance_label(self, mock_import: Any) -> None:
        """Test creation of LabelEncoder instance with default configuration."""
        # Mock sklearn components
        mock_label = Mock()
        mock_import.return_value = {"LabelEncoder": mock_label}

        EncodingFeatureGroup._create_encoder_instance("label")

        # Verify LabelEncoder was called with default parameters
        mock_label.assert_called_once_with()

    def test_create_encoder_instance_invalid(self) -> None:
        """Test that invalid encoder type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported encoder type: invalid_encoder"):
            EncodingFeatureGroup._create_encoder_instance("invalid_encoder")

    def test_encoder_matches_type_valid(self) -> None:
        """Test encoder type validation with matching types."""
        # Mock encoder with correct class name
        mock_encoder = Mock()
        mock_encoder.__class__.__name__ = "OneHotEncoder"

        result = EncodingFeatureGroup._encoder_matches_type(mock_encoder, "onehot")
        assert result is True

    def test_encoder_matches_type_mismatch(self) -> None:
        """Test encoder type validation with mismatched types."""
        # Mock encoder with wrong class name
        mock_encoder = Mock()
        mock_encoder.__class__.__name__ = "LabelEncoder"

        with pytest.raises(ValueError, match="Artifact encoder type mismatch"):
            EncodingFeatureGroup._encoder_matches_type(mock_encoder, "onehot")

    def test_encoder_matches_type_unsupported(self) -> None:
        """Test encoder type validation with unsupported encoder type."""
        mock_encoder = Mock()
        mock_encoder.__class__.__name__ = "SomeEncoder"

        with pytest.raises(ValueError, match="Unsupported encoder type: invalid_encoder"):
            EncodingFeatureGroup._encoder_matches_type(mock_encoder, "invalid_encoder")

    def test_abstract_methods_not_implemented(self) -> None:
        """Test that abstract methods raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            EncodingFeatureGroup._extract_training_data(None, "test")

        with pytest.raises(NotImplementedError):
            EncodingFeatureGroup._apply_encoder(None, "test", None)

        with pytest.raises(NotImplementedError):
            EncodingFeatureGroup._check_source_feature_exists(None, "test")

        with pytest.raises(NotImplementedError):
            EncodingFeatureGroup._add_result_to_data(None, "test", None, "onehot")
