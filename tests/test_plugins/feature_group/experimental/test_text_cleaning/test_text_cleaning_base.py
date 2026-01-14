"""
Tests for the TextCleaningFeatureGroup base class.
"""

from mloda.user import FeatureName
from mloda.user import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.text_cleaning.base import TextCleaningFeatureGroup


class TestTextCleaningFeatureGroupBase:
    """Tests for the TextCleaningFeatureGroup base class."""

    def test_match_feature_group_criteria_valid(self) -> None:
        """Test that valid feature names are accepted."""
        # Valid feature name
        feature_name = "review__cleaned_text"
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
            "cleaned__text__review",  # Extra double underscore in wrong place
        ]

        for name in invalid_names:
            assert not TextCleaningFeatureGroup.match_feature_group_criteria(name, options)

    def test_input_features(self) -> None:
        """Test that the feature group correctly extracts source features."""
        feature_name = FeatureName("review__cleaned_text")
        options = Options()

        feature_group = TextCleaningFeatureGroup()
        input_features = feature_group.input_features(options, feature_name)

        assert input_features is not None
        assert len(input_features) == 1
        assert next(iter(input_features)).name.name == "review"

    def test_feature_chaining(self) -> None:
        """Test that the feature group works with chained features."""
        # Chained feature name (cleaned_text applied to an aggregated feature)
        feature_name = FeatureName("sum_aggr__sales__cleaned_text")
        options = Options()

        feature_group = TextCleaningFeatureGroup()
        input_features = feature_group.input_features(options, feature_name)

        assert input_features is not None
        assert len(input_features) == 1
        assert next(iter(input_features)).name.name == "sum_aggr__sales"


class TestTextCleaningFeatureChainParser:
    """Tests for the TextCleaningFeatureGroup's modernized configuration-based features."""

    def test_configuration_based_matching(self) -> None:
        """Test that configuration-based features are properly matched."""
        # Test configuration-based feature creation
        options = Options(
            context={
                TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize", "remove_stopwords"),
                DefaultOptionKeys.in_features: "review",
            }
        )

        # Configuration-based features should match with placeholder names
        assert TextCleaningFeatureGroup.match_feature_group_criteria("placeholder", options)

        # Test with group/context separation - operations in context should match
        options_with_group = Options(
            group={"some_group_param": "value"},
            context={
                TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize", "remove_punctuation"),
                DefaultOptionKeys.in_features: "description",
            },
        )
        assert TextCleaningFeatureGroup.match_feature_group_criteria("placeholder", options_with_group)

    def test_configuration_based_input_features(self) -> None:
        """Test that configuration-based features extract correct input features."""
        # Configuration-based feature with source feature in options
        options = Options(
            context={
                TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize", "remove_stopwords"),
                DefaultOptionKeys.in_features: "review",
            }
        )

        feature_group = TextCleaningFeatureGroup()
        feature_name = FeatureName("placeholder")

        input_features = feature_group.input_features(options, feature_name)

        assert input_features is not None
        assert len(input_features) == 1
        source_feature = next(iter(input_features))
        assert source_feature.name.name == "review"
