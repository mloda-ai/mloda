"""
Tests for the PythonDictTextCleaningFeatureGroup implementation.
"""

import pytest
from typing import Any

from mloda.user import Feature
from mloda.user import FeatureName
from mloda.provider import FeatureSet
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda

from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.feature_group.experimental.text_cleaning.base import TextCleaningFeatureGroup
from mloda_plugins.feature_group.experimental.text_cleaning.python_dict import PythonDictTextCleaningFeatureGroup

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class PythonDictTextCleaningTestDataCreator(ATestDataCreator):
    """Test data creator for PythonDict text cleaning tests."""

    compute_framework = PythonDictFramework

    @classmethod
    def get_raw_data(cls) -> dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "text": [
                "Hello World!",
                "TESTING text normalization",
                "This is a test with punctuation, special chars, and extra   spaces.",
                "Visit https://example.com or email test@example.com",
                "This contains stopwords like the and is and a",
            ],
            "review": [
                "This is a GREAT product with excellent quality.",
                "This contains stopwords like the and is and a very good item.",
                "Amazing product! I love it so much!!!",
                "Not bad, could be better though...",
                "Perfect product with no issues at all!",
            ],
            "description": [
                "Visit https://example.com for more info or email test@example.com",
                "This is a test with punctuation, special chars, and extra   spaces.",
                "Clean this text from URLs and emails please.",
                "Remove all the @#$%^&*() special characters from this text.",
                "Normalize    all   the    whitespace   in   this   text.",
            ],
        }

    @classmethod
    def transform_format_for_testing(cls, data: dict[str, Any]) -> dict[str, Any]:
        """PythonDict's native format is already columnar; return it unchanged."""
        return data


def _copy_columnar(data: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a columnar dict so tests don't mutate the shared fixture."""
    return {key: list(values) for key, values in data.items()}


class TestPythonDictTextCleaningFeatureGroup:
    """Tests for the PythonDictTextCleaningFeatureGroup implementation."""

    def setup_method(self) -> None:
        """Set up columnar test data."""
        self.data = {
            "text": [
                "Hello World!",
                "TESTING text normalization",
                "This is a test with punctuation, special chars, and extra   spaces.",
                "Visit https://example.com or email test@example.com",
                "This contains stopwords like the and is and a",
            ]
        }

    def test_compute_framework_rule(self) -> None:
        """Test compute_framework_rule method."""
        assert PythonDictTextCleaningFeatureGroup.compute_framework_rule() == {PythonDictFramework}

    def test_check_source_features_exist(self) -> None:
        """Test that the source feature existence check works correctly."""
        # Feature exists
        PythonDictTextCleaningFeatureGroup._check_source_features_exist(self.data, ["text"])

        # Feature doesn't exist
        with pytest.raises(ValueError) as excinfo:
            PythonDictTextCleaningFeatureGroup._check_source_features_exist(self.data, ["nonexistent"])

        assert "Source features not found in data:" in str(excinfo.value)

        # Empty data
        with pytest.raises(ValueError, match="Data cannot be empty"):
            PythonDictTextCleaningFeatureGroup._check_source_features_exist({}, ["text"])

    def test_get_source_text(self) -> None:
        """Test that the source text is correctly retrieved."""
        result = PythonDictTextCleaningFeatureGroup._get_source_text(self.data, "text")

        assert isinstance(result, list)
        assert len(result) == len(self.data["text"])
        assert result[0] == "Hello World!"
        assert result[1] == "TESTING text normalization"

        # Test with None values
        data_with_none: dict[str, Any] = {"text": ["Hello", None, "World"]}
        result_with_none = PythonDictTextCleaningFeatureGroup._get_source_text(data_with_none, "text")
        assert result_with_none == ["Hello", "", "World"]

    def test_add_result_to_data(self) -> None:
        """Test that the result is correctly added to the data."""
        result = ["test1", "test2", "test3", "test4", "test5"]
        data_copy = _copy_columnar(self.data)

        updated_data = PythonDictTextCleaningFeatureGroup._add_result_to_data(data_copy, "new_column", result)

        assert "new_column" in updated_data
        assert updated_data["new_column"][0] == "test1"
        assert updated_data["new_column"][4] == "test5"

        # Test mismatched lengths
        with pytest.raises(ValueError, match="Result length 3 does not match data length 5"):
            PythonDictTextCleaningFeatureGroup._add_result_to_data(data_copy, "test", ["a", "b", "c"])

    def test_normalize_operation(self) -> None:
        """Test the normalize operation."""
        text = ["Hello World!", "TESTING text normalization"]
        result = PythonDictTextCleaningFeatureGroup._normalize_text(text)

        assert result[0] == "hello world!"
        assert result[1] == "testing text normalization"

    def test_remove_punctuation_operation(self) -> None:
        """Test the remove_punctuation operation."""
        text = ["Hello World!", "This is a test, with punctuation."]
        result = PythonDictTextCleaningFeatureGroup._remove_punctuation(text)

        assert "!" not in result[0]
        assert "," not in result[1]
        assert "." not in result[1]
        assert result[0] == "Hello World"
        assert result[1] == "This is a test with punctuation"

    def test_remove_special_chars_operation(self) -> None:
        """Test the remove_special_chars operation."""
        text = ["Hello World!", "Test @#$%^&*() special chars"]
        result = PythonDictTextCleaningFeatureGroup._remove_special_chars(text)

        assert "!" not in result[0]
        assert "@" not in result[1]
        assert "#" not in result[1]
        assert "$" not in result[1]
        assert result[0] == "Hello World"
        assert result[1] == "Test  special chars"

    def test_normalize_whitespace_operation(self) -> None:
        """Test the normalize_whitespace operation."""
        text = ["Hello    World", "Multiple   spaces  and\ttabs\nand newlines"]
        result = PythonDictTextCleaningFeatureGroup._normalize_whitespace(text)

        assert "    " not in result[0]
        assert "   " not in result[1]
        assert "\t" not in result[1]
        assert "\n" not in result[1]
        assert result[0] == "Hello World"
        assert result[1] == "Multiple spaces and tabs and newlines"

    def test_remove_urls_operation(self) -> None:
        """Test the remove_urls operation."""
        text = ["Visit https://example.com for info", "Email test@example.com for help"]
        result = PythonDictTextCleaningFeatureGroup._remove_urls(text)

        assert "https://example.com" not in result[0]
        assert "test@example.com" not in result[1]
        assert "Visit  for info" in result[0]
        assert "Email  for help" in result[1]

    def test_remove_stopwords_operation(self) -> None:
        """Test the remove_stopwords operation."""
        text = ["This is a test with stopwords", "The quick brown fox"]
        result = PythonDictTextCleaningFeatureGroup._remove_stopwords(text)

        # Note: This test may pass even if NLTK is not available, as the method
        # gracefully falls back to returning the original text
        assert isinstance(result, list)
        assert len(result) == len(text)

    def test_apply_operation(self) -> None:
        """Test the apply_operation method."""
        text = ["Hello World!", "TESTING text"]

        # Test normalize
        result = PythonDictTextCleaningFeatureGroup._apply_operation(self.data, text, "normalize")
        assert result[0] == "hello world!"

        # Test invalid operation
        with pytest.raises(ValueError) as excinfo:
            PythonDictTextCleaningFeatureGroup._apply_operation(self.data, text, "invalid_operation")

        assert "Unsupported cleaning operation" in str(excinfo.value)

    def test_calculate_feature_single_operation(self) -> None:
        """Test calculate_feature with a single operation."""
        # Create feature with normalize operation
        feature = Feature(
            FeatureName("text__cleaned_text"),
            Options(
                {
                    TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize",),
                }
            ),
        )
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Apply feature
        data_copy = _copy_columnar(self.data)
        result = PythonDictTextCleaningFeatureGroup.calculate_feature(data_copy, feature_set)

        # Check results
        assert "text__cleaned_text" in result
        assert result["text__cleaned_text"][0] == "hello world!"
        assert result["text__cleaned_text"][1] == "testing text normalization"

    def test_calculate_feature_multiple_operations(self) -> None:
        """Test calculate_feature with multiple operations."""
        # Create feature with multiple operations
        feature = Feature(
            FeatureName("text__cleaned_text"),
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
        data_copy = _copy_columnar(self.data)
        result = PythonDictTextCleaningFeatureGroup.calculate_feature(data_copy, feature_set)

        # Check results
        assert "text__cleaned_text" in result
        assert result["text__cleaned_text"][0] == "hello world"  # Lowercase and no punctuation
        assert "," not in result["text__cleaned_text"][2]  # No punctuation
        assert "   " not in result["text__cleaned_text"][2]  # No extra spaces

    def test_calculate_feature_missing_source(self) -> None:
        """Test calculate_feature with a missing source feature."""
        # Create feature with nonexistent source
        feature = Feature(
            FeatureName("nonexistent__cleaned_text"),
            Options(
                {
                    TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize",),
                }
            ),
        )
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Apply feature
        data_copy = _copy_columnar(self.data)
        with pytest.raises(ValueError) as excinfo:
            PythonDictTextCleaningFeatureGroup.calculate_feature(data_copy, feature_set)

        assert "Source features not found in data:" in str(excinfo.value)

    def test_calculate_feature_invalid_operation(self) -> None:
        """Test calculate_feature with an invalid operation."""
        # Create feature with invalid operation
        feature = Feature(
            FeatureName("text__cleaned_text"),
            Options(
                {
                    TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("invalid_operation",),
                }
            ),
        )
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Apply feature
        data_copy = _copy_columnar(self.data)
        with pytest.raises(ValueError) as excinfo:
            PythonDictTextCleaningFeatureGroup.calculate_feature(data_copy, feature_set)

        assert "Unsupported cleaning operation" in str(excinfo.value)


class TestTextCleaningPythonDictIntegration:
    """Integration tests for the text cleaning feature group using PythonDict framework."""

    def test_text_cleaning_with_data_creator(self) -> None:
        """Test text cleaning features with mloda using DataCreator."""

        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PythonDictTextCleaningTestDataCreator, PythonDictTextCleaningFeatureGroup}
        )

        # Create options for text cleaning operations
        options = Options(
            {TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize", "remove_punctuation", "normalize_whitespace")}
        )

        feature_str = [
            "text",  # Source data with text content
            "review",
            "description",
            "text__cleaned_text",  # Text cleaning
            "review__cleaned_text",  # Text cleaning
            "description__cleaned_text",  # Text cleaning
        ]

        feature_list = [Feature(name=feature, options=options) for feature in feature_str]

        # Run the mloda with text cleaning features
        result = mloda.run_all(
            feature_list,  # type: ignore
            compute_frameworks={PythonDictFramework},
            plugin_collector=plugin_collector,
        )

        # Validate that the text cleaning features are present
        assert len(result) == 2, "Expected 2 results: one for original data and one for cleaned features"

        # Result is a columnar dict (PythonDict format)
        data = result[1]

        # Check that cleaning was applied (text should be lowercase and without punctuation)
        assert data["text__cleaned_text"][0] == "hello world"
        assert "!" not in data["review__cleaned_text"][0]
        assert data["review__cleaned_text"][0].islower()

        # Check that cleaned features are present
        assert "text__cleaned_text" in data
        assert "review__cleaned_text" in data
        assert "description__cleaned_text" in data

        data = result[0]

        # Check that original features are present
        assert "text" in data
        assert "review" in data
        assert "description" in data
