"""
Integration tests for the TextCleaningFeatureGroup with mlodaAPI.
"""

from typing import Any, Dict
import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.text_cleaning.base import TextCleaningFeatureGroup
from mloda_plugins.feature_group.experimental.text_cleaning.pandas import PandasTextCleaningFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class TextCleaningTestDataCreator(ATestDataCreator):
    """Test data creator for text cleaning tests."""

    compute_framework = PandasDataframe

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "text1": [
                "Hello World!",
                "TESTING text normalization",
                "This is a test with punctuation, special chars, and extra   spaces.",
                "Visit https://example.com or email test@example.com",
                "This contains stopwords like the and is and a",
            ],
            "text2": [
                "Hello World!",
                "TESTING text normalization",
                "This is a test with punctuation, special chars, and extra   spaces.",
                "Visit https://example.com or email test@example.com",
                "This contains stopwords like the and is and a",
            ],
        }


class TestTextCleaningFeatureGroupIntegration:
    """Integration tests for the TextCleaningFeatureGroup with mlodaAPI."""

    def test_integration_with_feature_parser(self) -> None:
        """Test integration with mlodaAPI using the parser."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {TextCleaningTestDataCreator, PandasTextCleaningFeatureGroup}
        )

        parser = TextCleaningFeatureGroup.configurable_feature_chain_parser()
        if parser is None:
            raise ValueError("Feature chain parser is not available.")

        # Create features with options
        f1 = Feature(
            "x",
            Options(
                {
                    TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize",),
                    DefaultOptionKeys.mloda_source_feature: "text1",
                }
            ),
        )

        f2 = Feature(
            "y",
            Options(
                {
                    TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize", "remove_punctuation"),
                    DefaultOptionKeys.mloda_source_feature: "text2",
                }
            ),
        )

        # Create features using the parser
        feature1 = parser.create_feature_without_options(f1)
        feature2 = parser.create_feature_without_options(f2)

        if feature1 is None or feature2 is None:
            raise ValueError("Failed to create features using the parser.")

        # Test with pre-parsed features
        results = mlodaAPI.run_all(
            [feature1, feature2],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Different options create different feature sets
        assert len(results) == 2, "Expected 2 results, one for each feature with different options"

        # Find the DataFrames with the cleaned text features
        cleaned_df1 = None
        cleaned_df2 = None
        for df in results:
            if "cleaned_text__text1" in df.columns:
                cleaned_df1 = df
            if "cleaned_text__text2" in df.columns:
                cleaned_df2 = df

        assert cleaned_df1 is not None, "DataFrame with cleaned_text__text1 not found"
        assert cleaned_df2 is not None, "DataFrame with cleaned_text__text2 not found"

        # Print the DataFrame for debugging
        print("DataFrame1 columns:", cleaned_df1.columns.tolist())
        print("Text1 value:", cleaned_df1["cleaned_text__text1"].iloc[0])
        print("DataFrame2 columns:", cleaned_df2.columns.tolist())
        print("Text2 value:", cleaned_df2["cleaned_text__text2"].iloc[0])

        # Check that normalization was applied to text1
        assert cleaned_df1["cleaned_text__text1"].iloc[0] == "hello world!"

        # Check that both normalization and punctuation removal were applied to text2
        assert cleaned_df2["cleaned_text__text2"].iloc[0].lower() == "hello world"
        assert "!" not in cleaned_df2["cleaned_text__text2"].iloc[0]

        # Test with mloda parsing the features
        results2 = mlodaAPI.run_all(
            [f1, f2],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Different options create different feature sets
        assert len(results2) == 2, "Expected 2 results, one for each feature with different options"

        # Find the DataFrames with the cleaned text features
        cleaned_df1_2 = None
        cleaned_df2_2 = None
        for df in results2:
            if "cleaned_text__text1" in df.columns:
                cleaned_df1_2 = df
            if "cleaned_text__text2" in df.columns:
                cleaned_df2_2 = df

        assert cleaned_df1_2 is not None, "DataFrame with cleaned_text__text1 not found in results2"
        assert cleaned_df2_2 is not None, "DataFrame with cleaned_text__text2 not found in results2"

        # Check that the results are the same
        assert cleaned_df1.sort_index(axis=1).equals(cleaned_df1_2.sort_index(axis=1))
        assert cleaned_df2.sort_index(axis=1).equals(cleaned_df2_2.sort_index(axis=1))
