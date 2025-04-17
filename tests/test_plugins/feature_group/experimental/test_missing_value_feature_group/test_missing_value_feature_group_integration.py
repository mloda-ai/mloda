"""
Integration tests for the MissingValueFeatureGroup with mlodaAPI.
"""

from typing import Any, Dict

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.data_quality.missing_value.pandas import PandasMissingValueFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class MissingValueParserTestDataCreator(ATestDataCreator):
    """Test data creator for missing value parser tests."""

    compute_framework = PandasDataframe

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            "income": [50000, None, 75000, None, 60000],
            "category": ["A", None, "B", "A", None],
        }


class TestMissingValueFeatureGroupIntegration:
    def test_integration_with_feature_parser(self) -> None:
        """Test integration with mlodaAPI using the parser."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {MissingValueParserTestDataCreator, PandasMissingValueFeatureGroup}
        )

        parser = MissingValueFeatureGroup.configurable_feature_chain_parser()
        if parser is None:
            raise ValueError("Feature chain parser is not available.")

        f1 = Feature(
            "x",
            Options(
                {
                    MissingValueFeatureGroup.IMPUTATION_METHOD: "mean",
                    DefaultOptionKeys.mloda_source_feature: "income",
                }
            ),
        )

        f2 = Feature(
            "x",
            Options(
                {
                    MissingValueFeatureGroup.IMPUTATION_METHOD: "mode",
                    DefaultOptionKeys.mloda_source_feature: "category",
                }
            ),
        )

        feature1 = parser.create_feature_without_options(f1)
        feature2 = parser.create_feature_without_options(f2)

        if feature1 is None or feature2 is None:
            raise ValueError("Failed to create features using the parser.")

        # test with pre parsing the features
        results = mlodaAPI.run_all(
            [feature1, feature2],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        assert len(results) == 1

        # Find the DataFrame with the imputed features
        imputed_df = None
        for df in results:
            if "mean_imputed__income" in df.columns and "mode_imputed__category" in df.columns:
                imputed_df = df
                break

        assert imputed_df is not None, "DataFrame with imputed features not found"

        # Verify that missing values are imputed
        assert abs(imputed_df["mean_imputed__income"].iloc[1] - 61666.67) < 1.0
        assert abs(imputed_df["mean_imputed__income"].iloc[3] - 61666.67) < 1.0

        assert imputed_df["mode_imputed__category"].iloc[1] == "A"
        assert imputed_df["mode_imputed__category"].iloc[4] == "A"

        # test with mloda parsing the features
        results2 = mlodaAPI.run_all(
            [f1, f2],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        assert len(results2) == 1
        assert results[0].sort_index(axis=1).equals(results2[0].sort_index(axis=1))

    def test_integration_with_constant_imputation(self) -> None:
        """Test integration with mlodaAPI using constant imputation."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {MissingValueParserTestDataCreator, PandasMissingValueFeatureGroup}
        )

        parser = MissingValueFeatureGroup.configurable_feature_chain_parser()
        if parser is None:
            raise ValueError("Feature chain parser is not available.")

        # Create a feature with constant imputation
        f1 = Feature(
            "x",
            Options(
                {
                    MissingValueFeatureGroup.IMPUTATION_METHOD: "constant",
                    DefaultOptionKeys.mloda_source_feature: "category",
                    "constant_value": "Unknown",
                }
            ),
        )

        feature1 = parser.create_feature_without_options(f1)

        if feature1 is None:
            raise ValueError("Failed to create feature using the parser.")

        # test with pre parsing the features
        results = mlodaAPI.run_all([feature1], compute_frameworks={PandasDataframe}, plugin_collector=plugin_collector)

        assert len(results) == 1

        # Find the DataFrame with the imputed features
        imputed_df = None
        for df in results:
            if "constant_imputed__category" in df.columns:
                imputed_df = df
                break

        assert imputed_df is not None, "DataFrame with imputed features not found"

        # Verify that missing values are imputed with the constant value
        assert imputed_df["constant_imputed__category"].iloc[1] == "Unknown"
        assert imputed_df["constant_imputed__category"].iloc[4] == "Unknown"

        # test with mloda parsing the features
        results2 = mlodaAPI.run_all([f1], compute_frameworks={PandasDataframe}, plugin_collector=plugin_collector)

        assert len(results2) == 1
        assert results[0].sort_index(axis=1).equals(results2[0].sort_index(axis=1))
