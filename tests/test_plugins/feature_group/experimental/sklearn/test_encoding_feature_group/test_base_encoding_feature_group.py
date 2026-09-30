"""
Unit tests for the base EncodingFeatureGroup class.
"""

from typing import Any
import pytest
from unittest.mock import Mock, patch

from mloda.user import Feature
from mloda.user import FeatureName
from mloda.provider import DefaultOptionKeys
from mloda.user import Options
from mloda_plugins.feature_group.experimental.sklearn.encoding.base import EncodingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.encoding.pandas import PandasEncodingFeatureGroup


class TestEncodingFeatureGroup:
    """Test cases for the base EncodingFeatureGroup class."""

    def test_extract_config_feature_with_double_underscore_name(self) -> None:
        options = Options(
            context={
                EncodingFeatureGroup.ENCODER_TYPE: "label",
                DefaultOptionKeys.in_features: frozenset([Feature("category")]),
            }
        )
        result = EncodingFeatureGroup._extract_encoder_type_and_source_feature(Feature("a__b", options=options))
        assert result == ("label", "category")

    @pytest.mark.parametrize(
        "valid_names",
        [
            ["category__onehot_encoded", "status__onehot_encoded", "product_type__onehot_encoded"],
            ["category__label_encoded", "status__label_encoded", "priority__label_encoded"],
            ["category__ordinal_encoded", "grade__ordinal_encoded", "size__ordinal_encoded"],
        ],
        ids=["onehot", "label", "ordinal"],
    )
    def test_match_feature_group_criteria_valid(self, valid_names: list[str]) -> None:
        for name in valid_names:
            assert EncodingFeatureGroup.match_feature_group_criteria(FeatureName(name), Options({})), (
                f"Feature name '{name}' should match criteria"
            )

    def test_match_feature_group_criteria_invalid(self) -> None:
        """Test that invalid feature names do not match the criteria."""
        invalid_names = [
            "category__invalid_encoded",
            "category__onehot_scaled",  # Wrong suffix
            "category",  # No suffix
            "category__encoded",  # Missing encoder type
            "category__onehot",  # Missing 'encoded'
            "",  # Empty string
        ]

        for name in invalid_names:
            assert not EncodingFeatureGroup.match_feature_group_criteria(FeatureName(name), Options({})), (
                f"Feature name '{name}' should not match criteria"
            )

    def test_column_suffixed_name_matches_declared_base_in_features(self) -> None:
        """The declared source is the column base, as input_features builds it, not the ~suffixed name."""
        options = Options(context={DefaultOptionKeys.in_features: ["x"]})

        assert EncodingFeatureGroup.match_feature_group_criteria(FeatureName("x~1__onehot_encoded"), options) is True

    def test_column_suffixed_name_returns_the_declared_base_feature(self) -> None:
        declared = Feature("x", options=Options(context={"some_key": 1}), feature_group="SomeReader")
        options = Options(context={DefaultOptionKeys.in_features: [declared]})

        result = PandasEncodingFeatureGroup().input_features(options, FeatureName("x~1__onehot_encoded"))

        assert result is not None
        [returned] = result
        assert returned.name == "x"
        assert returned.options.get("some_key") == 1
        assert returned.feature_group_scope == "SomeReader"

    def test_column_suffixed_name_aborts_on_other_in_features(self) -> None:
        options = Options(context={DefaultOptionKeys.in_features: ["y"]})

        with pytest.raises(ValueError, match="in_features"):
            EncodingFeatureGroup.match_feature_group_criteria(FeatureName("x~1__onehot_encoded"), options)

    def test_match_feature_group_criteria_unsupported_encoder(self) -> None:
        """Test that unsupported encoder types do not match the criteria."""
        unsupported_names = [
            "category__binary_encoded",
            "category__target_encoded",
            "category__frequency_encoded",
        ]

        for name in unsupported_names:
            assert not EncodingFeatureGroup.match_feature_group_criteria(FeatureName(name), Options({})), (
                f"Feature name '{name}' should not match criteria (unsupported encoder)"
            )

    def test_get_encoder_type_valid(self) -> None:
        """Test extraction of encoder type from valid feature names."""
        test_cases = [
            ("category__onehot_encoded", "onehot"),
            ("status__label_encoded", "label"),
            ("priority__ordinal_encoded", "ordinal"),
        ]

        for feature_name, expected_encoder in test_cases:
            encoder_type = EncodingFeatureGroup.get_encoder_type(feature_name)
            assert encoder_type == expected_encoder, f"Expected {expected_encoder}, got {encoder_type}"

    def test_get_encoder_type_invalid(self) -> None:
        """Test that invalid feature names raise ValueError when extracting encoder type."""
        invalid_names = [
            "category__invalid_encoded",
            "category",
            "category__encoded",
        ]

        for name in invalid_names:
            with pytest.raises(ValueError, match="Invalid encoding feature name format"):
                EncodingFeatureGroup.get_encoder_type(name)

    def test_parse_feature_suffix_valid(self) -> None:
        """Test parsing of feature suffix from valid feature names."""
        from mloda.provider import FeatureChainParser

        test_cases = [
            ("category__onehot_encoded", "category"),
            ("customer_status__label_encoded", "customer_status"),
            ("product_priority__ordinal_encoded", "product_priority"),
        ]

        for feature_name, expected_source in test_cases:
            source_feature = FeatureChainParser.extract_in_feature(feature_name, EncodingFeatureGroup.PREFIX_PATTERN)
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
        expected_pattern = r".*__(?P<encoder_type>onehot|label|ordinal)_encoded(?:~\d+)?$"
        assert EncodingFeatureGroup.PREFIX_PATTERN == expected_pattern

    @pytest.mark.parametrize("name", ["x__onehot_encoded~1", "x__onehot_encoded"])
    def test_input_features_read_the_source_through_the_column_base(self, name: str) -> None:
        input_features = PandasEncodingFeatureGroup().input_features(Options({}), FeatureName(name))
        assert input_features is not None
        assert {f.name for f in input_features} == {"x"}

    def test_input_features(self) -> None:
        """Test input_features method extracts correct source features."""
        feature_group = PandasEncodingFeatureGroup()
        feature_name = FeatureName("category__onehot_encoded")
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
        """Test that abstract base class cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract method"):
            EncodingFeatureGroup()  # type: ignore[abstract]
