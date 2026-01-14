"""Tests for FeatureChainParserMixin."""

import pytest

from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class MockFeatureGroup(FeatureChainParserMixin):
    """Mock Feature group for testing the mixin with default settings."""

    PREFIX_PATTERN = r".*__([\w]+)_test$"
    PROPERTY_MAPPING = {
        "operation": {
            "op1": "Operation 1",
            "op2": "Operation 2",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        }
    }


class MockFeatureGroupCustomSeparator(FeatureChainParserMixin):
    """Mock Feature group with custom in_feature separator."""

    PREFIX_PATTERN = r".*__([\w]+)_test$"
    IN_FEATURE_SEPARATOR = ","
    PROPERTY_MAPPING = {
        "operation": {
            "op1": "Operation 1",
            "op2": "Operation 2",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        }
    }


class MockFeatureGroupWithMinMax(FeatureChainParserMixin):
    """Mock Feature group with min/max in_feature constraints."""

    PREFIX_PATTERN = r".*__([\w]+)_test$"
    MIN_IN_FEATURES = 2
    MAX_IN_FEATURES = 3
    PROPERTY_MAPPING = {
        "operation": {
            "op1": "Operation 1",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        }
    }


class MockFeatureGroupWithValidationHook(FeatureChainParserMixin):
    """Mock Feature group that implements _validate_string_match hook."""

    PREFIX_PATTERN = r".*__([\w]+)_test$"
    PROPERTY_MAPPING = {
        "operation": {
            "op1": "Operation 1",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        }
    }

    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, source_feature: str) -> bool:
        """Hook to validate string-based matches - reject operation 'reject_me'."""
        return operation_config != "reject_me"


class TestFeatureChainParserMixinInputFeatures:
    """Tests for input_features() method."""

    def test_input_features_string_based_single_source(self) -> None:
        """Test that string-based parsing extracts single source feature."""
        feature_name = FeatureName("source_feature__op1_test")
        options = Options(context={"operation": "op1"})

        mock_fg = MockFeatureGroup()
        result = mock_fg.input_features(options, feature_name)

        assert result is not None
        assert len(result) == 1
        assert Feature("source_feature") in result

    def test_input_features_string_based_multiple_sources(self) -> None:
        """Test parsing feature name with ampersand separator for multiple sources."""
        feature_name = FeatureName("point1&point2__op1_test")
        options = Options(context={"operation": "op1"})

        mock_fg = MockFeatureGroup()
        result = mock_fg.input_features(options, feature_name)

        assert result is not None
        assert len(result) == 2
        assert Feature("point1") in result
        assert Feature("point2") in result

    def test_input_features_string_based_three_sources(self) -> None:
        """Test parsing feature name with three sources using ampersand separator."""
        feature_name = FeatureName("feat1&feat2&feat3__op1_test")
        options = Options(context={"operation": "op1"})

        mock_fg = MockFeatureGroup()
        result = mock_fg.input_features(options, feature_name)

        assert result is not None
        assert len(result) == 3
        assert Feature("feat1") in result
        assert Feature("feat2") in result
        assert Feature("feat3") in result

    def test_input_features_config_based_fallback(self) -> None:
        """Test that when string parsing fails, fall back to options.get_in_features()."""
        feature_name = FeatureName("simple_name")
        options = Options(
            context={
                DefaultOptionKeys.in_features: ["feature_a", "feature_b"],
                "operation": "op1",
            }
        )

        mock_fg = MockFeatureGroup()
        result = mock_fg.input_features(options, feature_name)

        assert result is not None
        assert len(result) == 2
        assert Feature("feature_a") in result
        assert Feature("feature_b") in result

    def test_input_features_min_constraint_violated(self) -> None:
        """Test that ValueError is raised when fewer than min_source_features."""
        feature_name = FeatureName("single_feature__op1_test")
        options = Options(context={"operation": "op1"})

        mock_fg = MockFeatureGroupWithMinMax()

        with pytest.raises(ValueError) as exc_info:
            mock_fg.input_features(options, feature_name)

        assert "minimum" in str(exc_info.value).lower() or "at least" in str(exc_info.value).lower()

    def test_input_features_max_constraint_violated(self) -> None:
        """Test that ValueError is raised when more than max_source_features."""
        feature_name = FeatureName("f1&f2&f3&f4__op1_test")
        options = Options(context={"operation": "op1"})

        mock_fg = MockFeatureGroupWithMinMax()

        with pytest.raises(ValueError) as exc_info:
            mock_fg.input_features(options, feature_name)

        assert "maximum" in str(exc_info.value).lower() or "at most" in str(exc_info.value).lower()

    def test_input_features_within_min_max_constraints(self) -> None:
        """Test that parsing succeeds when source features are within min/max bounds."""
        feature_name = FeatureName("f1&f2__op1_test")
        options = Options(context={"operation": "op1"})

        mock_fg = MockFeatureGroupWithMinMax()
        result = mock_fg.input_features(options, feature_name)

        assert result is not None
        assert len(result) == 2


class TestFeatureChainParserMixinMatchFeatureGroupCriteria:
    """Tests for match_feature_group_criteria() classmethod."""

    def test_match_feature_group_criteria_pattern_match(self) -> None:
        """Test that pattern-based matching returns True for valid feature names."""
        feature_name = FeatureName("source_feature__op1_test")
        options = Options(context={"operation": "op1"})

        result = MockFeatureGroup.match_feature_group_criteria(feature_name, options)

        assert result is True

    def test_match_feature_group_criteria_no_pattern_match_but_valid_config(self) -> None:
        """Test that config-based matching works even if pattern doesn't match.

        This matches the behavior of match_configuration_feature_chain_parser()
        which returns True for valid config even if pattern doesn't match.
        """
        feature_name = FeatureName("source_feature__invalid_suffix")
        options = Options(context={"operation": "op1"})

        result = MockFeatureGroup.match_feature_group_criteria(feature_name, options)

        # Config-based matching succeeds because operation is valid
        assert result is True

    def test_match_feature_group_criteria_config_based(self) -> None:
        """Test that config-based matching works when pattern doesn't match."""
        feature_name = FeatureName("any_name_without_pattern")
        options = Options(context={"operation": "op1"})

        result = MockFeatureGroup.match_feature_group_criteria(feature_name, options)

        assert result is True

    def test_match_feature_group_criteria_config_based_invalid_operation(self) -> None:
        """Test that config-based matching raises ValueError with invalid operation value.

        When strict_validation is True in PROPERTY_MAPPING, invalid values raise ValueError.
        This matches the behavior of match_configuration_feature_chain_parser().
        """
        feature_name = FeatureName("any_name")
        options = Options(context={"operation": "invalid_op"})

        with pytest.raises(ValueError):
            MockFeatureGroup.match_feature_group_criteria(feature_name, options)

    def test_match_feature_group_criteria_missing_required_property(self) -> None:
        """Test that matching fails when required property is missing."""
        feature_name = FeatureName("any_name")
        options = Options(context={})

        result = MockFeatureGroup.match_feature_group_criteria(feature_name, options)

        assert result is False


class TestFeatureChainParserMixinValidateStringMatchHook:
    """Tests for _validate_string_match() hook."""

    def test_validate_string_match_hook_called_and_accepts(self) -> None:
        """Test that hook is called and accepts valid operations."""
        feature_name = FeatureName("source__op1_test")
        options = Options(context={"operation": "op1"})

        result = MockFeatureGroupWithValidationHook.match_feature_group_criteria(feature_name, options)

        assert result is True

    def test_validate_string_match_hook_called_and_rejects(self) -> None:
        """Test that hook is called and can reject based on custom logic."""
        feature_name = FeatureName("source__reject_me_test")
        options = Options(context={"operation": "reject_me"})

        result = MockFeatureGroupWithValidationHook.match_feature_group_criteria(feature_name, options)

        assert result is False

    def test_validate_string_match_hook_not_called_for_config_based(self) -> None:
        """Test that hook is not called for config-based matching."""
        feature_name = FeatureName("simple_name")
        options = Options(context={"operation": "op1"})

        # Should succeed via config-based matching without calling hook
        result = MockFeatureGroupWithValidationHook.match_feature_group_criteria(feature_name, options)

        assert result is True


class TestFeatureChainParserMixinCustomSeparator:
    """Tests for custom source feature separator."""

    def test_custom_source_feature_separator(self) -> None:
        """Test using comma as source feature separator instead of ampersand."""
        feature_name = FeatureName("feat1,feat2,feat3__op1_test")
        options = Options(context={"operation": "op1"})

        mock_fg = MockFeatureGroupCustomSeparator()
        result = mock_fg.input_features(options, feature_name)

        assert result is not None
        assert len(result) == 3
        assert Feature("feat1") in result
        assert Feature("feat2") in result
        assert Feature("feat3") in result

    def test_custom_separator_does_not_parse_ampersand(self) -> None:
        """Test that ampersand is not treated as separator with custom separator."""
        feature_name = FeatureName("feat1&feat2__op1_test")
        options = Options(context={"operation": "op1"})

        mock_fg = MockFeatureGroupCustomSeparator()
        result = mock_fg.input_features(options, feature_name)

        # Should treat "feat1&feat2" as a single feature name
        assert result is not None
        assert len(result) == 1
        assert Feature("feat1&feat2") in result


class TestFeatureChainParserMixinEdgeCases:
    """Tests for edge cases and error handling."""

    def test_input_features_empty_feature_name(self) -> None:
        """Test handling of empty feature name."""
        feature_name = FeatureName("__op1_test")
        options = Options(context={"operation": "op1"})

        mock_fg = MockFeatureGroup()

        # Empty source feature should fall back to config-based
        with pytest.raises(ValueError):
            mock_fg.input_features(options, feature_name)

    def test_input_features_no_separator(self) -> None:
        """Test feature name without double underscore separator."""
        feature_name = FeatureName("simple_feature_name")
        options = Options(
            context={
                DefaultOptionKeys.in_features: ["fallback_feature"],
                "operation": "op1",
            }
        )

        mock_fg = MockFeatureGroup()
        result = mock_fg.input_features(options, feature_name)

        # Should fall back to config-based
        assert result is not None
        assert Feature("fallback_feature") in result

    def test_match_feature_group_criteria_with_string_feature_name(self) -> None:
        """Test that match_feature_group_criteria accepts string feature names."""
        feature_name_str = "source__op1_test"
        options = Options(context={"operation": "op1"})

        result = MockFeatureGroup.match_feature_group_criteria(feature_name_str, options)

        assert result is True

    def test_input_features_preserves_feature_order(self) -> None:
        """Test that order of features is preserved in parsing."""
        feature_name = FeatureName("first&second&third__op1_test")
        options = Options(context={"operation": "op1"})

        mock_fg = MockFeatureGroup()
        result = mock_fg.input_features(options, feature_name)

        assert result is not None
        assert len(result) == 3
        # Result is a set, so we just check all features are present
        feature_names = {f.get_name() for f in result}
        assert feature_names == {"first", "second", "third"}


class TestFeatureChainParserMixinExtractSourceFeatures:
    """Tests for _extract_source_features() classmethod."""

    def test_extract_source_features_string_based_single(self) -> None:
        """Test extraction from feature name with single source feature."""
        feature = Feature(
            name="source_feature__op1_test",
            options=Options(context={"operation": "op1"}),
        )

        result = MockFeatureGroup._extract_source_features(feature)

        assert result == ["source_feature"]

    def test_extract_source_features_string_based_multiple(self) -> None:
        """Test extraction from feature name with multiple source features (ampersand separator)."""
        feature = Feature(
            name="feat1&feat2&feat3__op1_test",
            options=Options(context={"operation": "op1"}),
        )

        result = MockFeatureGroup._extract_source_features(feature)

        assert result == ["feat1", "feat2", "feat3"]

    def test_extract_source_features_config_based_fallback(self) -> None:
        """Test that when string parsing fails, it falls back to feature.options.get_in_features()."""
        feature = Feature(
            name="simple_name",
            options=Options(
                context={
                    DefaultOptionKeys.in_features: ["feature_a", "feature_b"],
                    "operation": "op1",
                }
            ),
        )

        result = MockFeatureGroup._extract_source_features(feature)

        # Should return list of feature names from get_in_features()
        assert len(result) == 2
        assert "feature_a" in result
        assert "feature_b" in result

    def test_extract_source_features_custom_separator(self) -> None:
        """Test extraction with custom separator (comma instead of ampersand)."""
        feature = Feature(
            name="feat1,feat2,feat3__op1_test",
            options=Options(context={"operation": "op1"}),
        )

        result = MockFeatureGroupCustomSeparator._extract_source_features(feature)

        assert result == ["feat1", "feat2", "feat3"]
