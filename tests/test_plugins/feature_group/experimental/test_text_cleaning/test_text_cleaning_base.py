"""
Tests for the TextCleaningFeatureGroup base class.
"""

import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.text_cleaning.base import (
    TextCleaningFeatureGroup,
    TextCleaningFeatureChainParserConfiguration,
)


class TestTextCleaningFeatureGroupBase:
    """Tests for the TextCleaningFeatureGroup base class."""

    def test_match_feature_group_criteria_valid(self) -> None:
        """Test that valid feature names are accepted."""
        # Valid feature name
        feature_name = "cleaned_text__review"
        options = Options()

        # Test with string
        assert TextCleaningFeatureGroup.match_feature_group_criteria(feature_name, options)

        # Test with FeatureName
        feature_name_obj = FeatureName(feature_name)
        assert TextCleaningFeatureGroup.match_feature_group_criteria(feature_name_obj, options)

    def test_match_feature_group_criteria_invalid(self) -> None:
        """Test that invalid feature names are rejected."""
        options = Options()

        # Invalid feature names
        invalid_names = [
            "clean_text__review",  # Wrong prefix
            "cleaned_text_review",  # Missing double underscore
            "cleaned_text__",  # Missing source feature
            "cleaned__text__review",  # Extra double underscore in wrong place
        ]

        for name in invalid_names:
            assert not TextCleaningFeatureGroup.match_feature_group_criteria(name, options)

    def test_input_features(self) -> None:
        """Test that the feature group correctly extracts source features."""
        feature_name = FeatureName("cleaned_text__review")
        options = Options()

        feature_group = TextCleaningFeatureGroup()
        input_features = feature_group.input_features(options, feature_name)

        assert input_features is not None
        assert len(input_features) == 1
        assert next(iter(input_features)).name.name == "review"

    def test_feature_chaining(self) -> None:
        """Test that the feature group works with chained features."""
        # Chained feature name (cleaned_text applied to an aggregated feature)
        feature_name = FeatureName("cleaned_text__sum_aggr__sales")
        options = Options()

        feature_group = TextCleaningFeatureGroup()
        input_features = feature_group.input_features(options, feature_name)

        assert input_features is not None
        assert len(input_features) == 1
        assert next(iter(input_features)).name.name == "sum_aggr__sales"


class TestTextCleaningFeatureChainParserConfiguration:
    """Tests for the TextCleaningFeatureChainParserConfiguration class."""

    def test_parse_keys(self) -> None:
        """Test that the configuration returns the correct keys."""
        keys = TextCleaningFeatureChainParserConfiguration.parse_keys()
        # Only mloda_source_feature is in parse_keys, CLEANING_OPERATIONS is preserved
        assert TextCleaningFeatureGroup.CLEANING_OPERATIONS not in keys
        assert DefaultOptionKeys.mloda_source_feature in keys

    def test_parse_from_options_valid(self) -> None:
        """Test that the configuration correctly parses valid options."""
        # Valid options
        options = Options(
            {
                TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize", "remove_stopwords"),
                DefaultOptionKeys.mloda_source_feature: "review",
            }
        )

        feature_name = TextCleaningFeatureChainParserConfiguration.parse_from_options(options)
        assert feature_name == "cleaned_text__review"

    def test_parse_from_options_missing_operations(self) -> None:
        """Test that the configuration returns None when operations are missing."""
        # Missing operations
        options = Options({DefaultOptionKeys.mloda_source_feature: "review"})

        feature_name = TextCleaningFeatureChainParserConfiguration.parse_from_options(options)
        assert feature_name is None

    def test_parse_from_options_missing_source_feature(self) -> None:
        """Test that the configuration returns None when source feature is missing."""
        # Missing source feature
        options = Options({TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize", "remove_stopwords")})

        feature_name = TextCleaningFeatureChainParserConfiguration.parse_from_options(options)
        assert feature_name is None

    def test_parse_from_options_invalid_operation(self) -> None:
        """Test that the configuration raises an error for invalid operations."""
        # Invalid operation
        options = Options(
            {
                TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("invalid_operation",),
                DefaultOptionKeys.mloda_source_feature: "review",
            }
        )

        with pytest.raises(ValueError) as excinfo:
            TextCleaningFeatureChainParserConfiguration.parse_from_options(options)

        assert "Unsupported cleaning operation" in str(excinfo.value)

    def test_create_feature_without_options(self) -> None:
        """Test that the configuration correctly creates a feature without options."""
        # Create a feature with options
        feature = Feature(
            "PlaceHolder",
            Options(
                {
                    TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize", "remove_stopwords"),
                    DefaultOptionKeys.mloda_source_feature: "review",
                }
            ),
        )

        # Create a feature without options
        result = TextCleaningFeatureChainParserConfiguration.create_feature_without_options(feature)

        assert result is not None
        assert result.name.name == "cleaned_text__review"
        # CLEANING_OPERATIONS is preserved for use in calculate_feature
        assert TextCleaningFeatureGroup.CLEANING_OPERATIONS in result.options.data
        assert result.options.data[TextCleaningFeatureGroup.CLEANING_OPERATIONS] == ("normalize", "remove_stopwords")
        # Only mloda_source_feature is removed
        assert DefaultOptionKeys.mloda_source_feature not in result.options.data
