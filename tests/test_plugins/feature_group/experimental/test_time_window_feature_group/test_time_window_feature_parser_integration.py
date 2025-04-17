"""
Integration tests for the TimeWindowFeatureGroup with FeatureChainParserConfiguration.
"""

from typing import Any, Dict

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindowFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

import pandas as pd
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class TimeWindowParserTestDataCreator(ATestDataCreator):
    """Test data creator for time window parser tests."""

    compute_framework = PandasDataframe

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        return {
            "temperature": [20, 22, 19, 23, 25, 21, 18, 20, 22, 24],
            "humidity": [65, 70, 75, 60, 55, 65, 70, 75, 60, 55],
            DefaultOptionKeys.reference_time: dates,
        }


class TestTimeWindowFeatureParserIntegration:
    def test_integration_with_feature_parser(self) -> None:
        """Test integration with mlodaAPI using the parser."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {TimeWindowParserTestDataCreator, PandasTimeWindowFeatureGroup}
        )

        parser = TimeWindowFeatureGroup.configurable_feature_chain_parser()
        if parser is None:
            raise ValueError("Feature chain parser is not available.")

        f1 = Feature(
            "x",
            Options(
                {
                    TimeWindowFeatureGroup.WINDOW_FUNCTION: "avg",
                    TimeWindowFeatureGroup.WINDOW_SIZE: 3,
                    TimeWindowFeatureGroup.TIME_UNIT: "day",
                    DefaultOptionKeys.mloda_source_feature: "temperature",
                }
            ),
        )

        f2 = Feature(
            "x",
            Options(
                {
                    TimeWindowFeatureGroup.WINDOW_FUNCTION: "max",
                    TimeWindowFeatureGroup.WINDOW_SIZE: 5,
                    TimeWindowFeatureGroup.TIME_UNIT: "day",
                    DefaultOptionKeys.mloda_source_feature: "humidity",
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

        # Find the DataFrame with the time window features
        window_df = None
        for df in results:
            if "avg_3_day_window__temperature" in df.columns and "max_5_day_window__humidity" in df.columns:
                window_df = df
                break

        assert window_df is not None, "DataFrame with time window features not found"

        # Verify that the time window features exist
        assert "avg_3_day_window__temperature" in window_df.columns
        assert "max_5_day_window__humidity" in window_df.columns

        # test with mloda parsing the features
        results2 = mlodaAPI.run_all(
            [f1, f2],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        assert len(results2) == 1
        assert results[0].sort_index(axis=1).equals(results2[0].sort_index(axis=1))

    def test_integration_with_different_time_units(self) -> None:
        """Test integration with mlodaAPI using different time units."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {TimeWindowParserTestDataCreator, PandasTimeWindowFeatureGroup}
        )

        parser = TimeWindowFeatureGroup.configurable_feature_chain_parser()
        if parser is None:
            raise ValueError("Feature chain parser is not available.")

        # Create features with different time units
        f1 = Feature(
            "x",
            Options(
                {
                    TimeWindowFeatureGroup.WINDOW_FUNCTION: "sum",
                    TimeWindowFeatureGroup.WINDOW_SIZE: 2,
                    TimeWindowFeatureGroup.TIME_UNIT: "day",
                    DefaultOptionKeys.mloda_source_feature: "temperature",
                }
            ),
        )

        f2 = Feature(
            "x",
            Options(
                {
                    TimeWindowFeatureGroup.WINDOW_FUNCTION: "min",
                    TimeWindowFeatureGroup.WINDOW_SIZE: 1,
                    TimeWindowFeatureGroup.TIME_UNIT: "week",
                    DefaultOptionKeys.mloda_source_feature: "humidity",
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

        # Find the DataFrame with the time window features
        window_df = None
        for df in results:
            if "sum_2_day_window__temperature" in df.columns and "min_1_week_window__humidity" in df.columns:
                window_df = df
                break

        assert window_df is not None, "DataFrame with time window features not found"

        # Verify that the time window features exist
        assert "sum_2_day_window__temperature" in window_df.columns
        assert "min_1_week_window__humidity" in window_df.columns

        # test with mloda parsing the features
        results2 = mlodaAPI.run_all(
            [f1, f2],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        assert len(results2) == 1
        assert results[0].sort_index(axis=1).equals(results2[0].sort_index(axis=1))
