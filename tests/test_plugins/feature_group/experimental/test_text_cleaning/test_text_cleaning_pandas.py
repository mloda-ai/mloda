"""
Tests for the PandasTextCleaningFeatureGroup implementation.
"""

import pandas as pd
import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.text_cleaning.base import TextCleaningFeatureGroup
from mloda_plugins.feature_group.experimental.text_cleaning.pandas import PandasTextCleaningFeatureGroup


class TestPandasTextCleaningFeatureGroup:
    """Tests for the PandasTextCleaningFeatureGroup implementation."""

    def setup_method(self) -> None:
        """Set up test data."""
        # Create test data
        self.df = pd.DataFrame(
            {
                "text": [
                    "Hello World!",
                    "TESTING text normalization",
                    "This is a test with punctuation, special chars, and extra   spaces.",
                    "Visit https://example.com or email test@example.com",
                    "This contains stopwords like the and is and a",
                ]
            }
        )

    def test_check_source_feature_exists(self) -> None:
        """Test that the source feature existence check works correctly."""
        # Feature exists
        PandasTextCleaningFeatureGroup._check_source_feature_exists(self.df, "text")

        # Feature doesn't exist
        with pytest.raises(ValueError) as excinfo:
            PandasTextCleaningFeatureGroup._check_source_feature_exists(self.df, "nonexistent")

        assert "not found in the data" in str(excinfo.value)

    def test_get_source_text(self) -> None:
        """Test that the source text is correctly retrieved."""
        result = PandasTextCleaningFeatureGroup._get_source_text(self.df, "text")

        assert isinstance(result, pd.Series)
        assert len(result) == len(self.df)
        assert result.iloc[0] == "Hello World!"

    def test_add_result_to_data(self) -> None:
        """Test that the result is correctly added to the data."""
        result = pd.Series(["test1", "test2", "test3", "test4", "test5"])

        df_result = PandasTextCleaningFeatureGroup._add_result_to_data(self.df, "new_column", result)

        assert "new_column" in df_result.columns
        assert df_result["new_column"].iloc[0] == "test1"
        assert df_result["new_column"].iloc[4] == "test5"

    def test_normalize_operation(self) -> None:
        """Test the normalize operation."""
        text = self.df["text"]
        result = PandasTextCleaningFeatureGroup._normalize_text(text)

        assert result.iloc[0] == "hello world!"
        assert result.iloc[1] == "testing text normalization"

    def test_remove_punctuation_operation(self) -> None:
        """Test the remove_punctuation operation."""
        text = self.df["text"]
        result = PandasTextCleaningFeatureGroup._remove_punctuation(text)

        assert "!" not in result.iloc[0]
        assert "," not in result.iloc[2]
        assert "." not in result.iloc[2]

    def test_remove_special_chars_operation(self) -> None:
        """Test the remove_special_chars operation."""
        text = self.df["text"]
        result = PandasTextCleaningFeatureGroup._remove_special_chars(text)

        assert "!" not in result.iloc[0]
        assert "," not in result.iloc[2]
        assert "@" not in result.iloc[3]

    def test_normalize_whitespace_operation(self) -> None:
        """Test the normalize_whitespace operation."""
        text = self.df["text"]
        result = PandasTextCleaningFeatureGroup._normalize_whitespace(text)

        assert "   " not in result.iloc[2]
        assert result.iloc[2].count(" ") < text.iloc[2].count(" ")

    def test_remove_urls_operation(self) -> None:
        """Test the remove_urls operation."""
        text = self.df["text"]
        result = PandasTextCleaningFeatureGroup._remove_urls(text)

        assert "https://example.com" not in result.iloc[3]
        assert "test@example.com" not in result.iloc[3]

    def test_apply_operation(self) -> None:
        """Test the apply_operation method."""
        text = self.df["text"]

        # Test normalize
        result = PandasTextCleaningFeatureGroup._apply_operation(self.df, text, "normalize")
        assert result.iloc[0] == "hello world!"

        # Test invalid operation
        with pytest.raises(ValueError) as excinfo:
            PandasTextCleaningFeatureGroup._apply_operation(self.df, text, "invalid_operation")

        assert "Unsupported cleaning operation" in str(excinfo.value)

    def test_calculate_feature_single_operation(self) -> None:
        """Test calculate_feature with a single operation."""
        # Create feature with normalize operation
        feature = Feature(
            FeatureName("cleaned_text__text"),
            Options(
                {
                    TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize",),
                }
            ),
        )
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Apply feature
        result = PandasTextCleaningFeatureGroup.calculate_feature(self.df.copy(), feature_set)

        # Check results
        assert "cleaned_text__text" in result.columns
        assert result["cleaned_text__text"].iloc[0] == "hello world!"
        assert result["cleaned_text__text"].iloc[1] == "testing text normalization"

    def test_calculate_feature_multiple_operations(self) -> None:
        """Test calculate_feature with multiple operations."""
        # Create feature with multiple operations
        feature = Feature(
            FeatureName("cleaned_text__text"),
            Options(
                {
                    TextCleaningFeatureGroup.CLEANING_OPERATIONS: (
                        "normalize",
                        "remove_punctuation",
                        "normalize_whitespace",
                    ),
                }
            ),
        )
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Apply feature
        result = PandasTextCleaningFeatureGroup.calculate_feature(self.df.copy(), feature_set)

        # Check results
        assert "cleaned_text__text" in result.columns
        assert result["cleaned_text__text"].iloc[0] == "hello world"  # Lowercase and no punctuation
        assert "," not in result["cleaned_text__text"].iloc[2]  # No punctuation
        assert "   " not in result["cleaned_text__text"].iloc[2]  # No extra spaces

    def test_calculate_feature_missing_source(self) -> None:
        """Test calculate_feature with a missing source feature."""
        # Create feature with nonexistent source
        feature = Feature(
            FeatureName("cleaned_text__nonexistent"),
            Options(
                {
                    TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize",),
                }
            ),
        )
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Apply feature
        with pytest.raises(ValueError) as excinfo:
            PandasTextCleaningFeatureGroup.calculate_feature(self.df.copy(), feature_set)

        assert "not found in the data" in str(excinfo.value)

    def test_calculate_feature_invalid_operation(self) -> None:
        """Test calculate_feature with an invalid operation."""
        # Create feature with invalid operation
        feature = Feature(
            FeatureName("cleaned_text__text"),
            Options(
                {
                    TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("invalid_operation",),
                }
            ),
        )
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Apply feature
        with pytest.raises(ValueError) as excinfo:
            PandasTextCleaningFeatureGroup.calculate_feature(self.df.copy(), feature_set)

        assert "Unsupported cleaning operation" in str(excinfo.value)

    def test_integration_with_configuration(self) -> None:
        """Test integration with configuration-based feature creation."""
        # Test configuration-based feature creation using new Options structure
        feature = Feature(
            "placeholder",  # Placeholder name for configuration-based feature
            Options(
                context={
                    TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize", "remove_punctuation"),
                    DefaultOptionKeys.mloda_source_features: "text",
                }
            ),
        )

        # Test that the feature matches the criteria for text cleaning
        assert TextCleaningFeatureGroup.match_feature_group_criteria(feature.name, feature.options)

        # Test input features extraction
        feature_group = TextCleaningFeatureGroup()
        input_features = feature_group.input_features(feature.options, feature.name)
        assert input_features is not None
        assert len(input_features) == 1
        source_feature = next(iter(input_features))
        assert source_feature.name.name == "text"
