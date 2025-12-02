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
        """Test integration with mlodaAPI using configuration-based features."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {TextCleaningTestDataCreator, PandasTextCleaningFeatureGroup}
        )

        # Create features with configuration-based options using new Options structure
        f1 = Feature(
            "placeholder1",  # Placeholder name for configuration-based feature
            Options(
                context={
                    TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize",),
                    DefaultOptionKeys.in_features: "text1",
                }
            ),
        )

        f2 = Feature(
            "placeholder2",  # Placeholder name for configuration-based feature
            Options(
                context={
                    TextCleaningFeatureGroup.CLEANING_OPERATIONS: ("normalize", "remove_punctuation"),
                    DefaultOptionKeys.in_features: "text2",
                }
            ),
        )

        # Test with configuration-based features directly
        results = mlodaAPI.run_all(
            [f1, f2],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Different options create different feature sets
        assert len(results) == 1, "Expected 1 result"

        # Find the DataFrames with the cleaned text features
        cleaned_df1 = None
        cleaned_df2 = None
        for df in results:
            if "placeholder1" in df.columns:
                cleaned_df1 = df
            if "placeholder2" in df.columns:
                cleaned_df2 = df

        assert cleaned_df1 is not None, "DataFrame with placeholder1 not found"
        assert cleaned_df2 is not None, "DataFrame with placeholder2 not found"

        # Check that normalization was applied to text1
        assert cleaned_df1["placeholder1"].iloc[0] == "hello world!"

        # Check that both normalization and punctuation removal were applied to text2
        assert cleaned_df2["placeholder2"].iloc[0].lower() == "hello world"
        assert "!" not in cleaned_df2["placeholder2"].iloc[0]

        # Test with mloda parsing the features
        results2 = mlodaAPI.run_all(
            [f1, f2],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Different options create different feature sets
        assert len(results2) == 1, "Expected 1 result"

        # Find the DataFrames with the cleaned text features
        cleaned_df1_2 = None
        cleaned_df2_2 = None
        for df in results2:
            if "placeholder1" in df.columns:
                cleaned_df1_2 = df
            if "placeholder2" in df.columns:
                cleaned_df2_2 = df

        assert cleaned_df1_2 is not None, "DataFrame with placeholder1 not found in results2"
        assert cleaned_df2_2 is not None, "DataFrame with placeholder2 not found in results2"

        # Check that the results are the same
        assert cleaned_df1.sort_index(axis=1).equals(cleaned_df1_2.sort_index(axis=1))
        assert cleaned_df2.sort_index(axis=1).equals(cleaned_df2_2.sort_index(axis=1))
