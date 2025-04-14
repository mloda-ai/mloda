"""
Integration tests for time window feature groups.
"""

from typing import Any, Optional
import pandas as pd

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindowFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pyarrow import PyArrowTimeWindowFeatureGroup


class TestTimeWindowPandasIntegration:
    """Integration tests for the time window feature group using Pandas."""

    def test_time_window_with_data_creator(self) -> None:
        """Test time window features with mlodaAPI using DataCreator."""

        # Create a feature group that uses DataCreator to provide test data
        class TestTimeDataCreator(AbstractFeatureGroup):
            @classmethod
            def input_data(cls) -> Optional[BaseInputData]:
                return DataCreator(
                    {
                        "temperature",
                        "humidity",
                        "pressure",
                        "wind_speed",
                        DefaultOptionKeys.reference_time,
                    }
                )

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                # Create a DataFrame with the time filter feature
                dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
                return pd.DataFrame(
                    {
                        "temperature": [20, 22, 19, 23, 25, 21, 18, 20, 22, 24],
                        "humidity": [65, 70, 75, 60, 55, 65, 70, 75, 60, 55],
                        "pressure": [1012, 1010, 1008, 1007, 1009, 1011, 1013, 1012, 1010, 1009],
                        "wind_speed": [5, 7, 10, 8, 6, 4, 9, 11, 7, 5],
                        DefaultOptionKeys.reference_time: dates,
                    }
                )

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups({TestTimeDataCreator, PandasTimeWindowFeatureGroup})

        # Run the API with multiple time window features
        result = mlodaAPI.run_all(
            [
                "temperature",  # Source data
                "avg_3_day_window_temperature",  # 3-day average temperature
                "max_5_day_window_humidity",  # 5-day maximum humidity
                "min_2_day_window_pressure",  # 2-day minimum pressure
                "sum_4_day_window_wind_speed",  # 4-day sum of wind speed
            ],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) == 2  # Two DataFrames: one for source data, one for time window features

        # Find the DataFrame with the time window features
        window_df = None
        for df in result:
            if "avg_3_day_window_temperature" in df.columns:
                window_df = df
                break

        assert window_df is not None, "DataFrame with time window features not found"

        # Verify the time window features exist
        assert "avg_3_day_window_temperature" in window_df.columns
        assert "max_5_day_window_humidity" in window_df.columns
        assert "min_2_day_window_pressure" in window_df.columns
        assert "sum_4_day_window_wind_speed" in window_df.columns


class TestTimeWindowPyArrowIntegration:
    """Integration tests for the time window feature group using PyArrow."""

    def test_time_window_with_data_creator(self) -> None:
        """Test time window features with mlodaAPI using DataCreator."""

        # Create a feature group that uses DataCreator to provide test data
        class TestTimeDataCreator(AbstractFeatureGroup):
            @classmethod
            def input_data(cls) -> Optional[BaseInputData]:
                return DataCreator(
                    {
                        "temperature",
                        "humidity",
                        "pressure",
                        "wind_speed",
                        DefaultOptionKeys.reference_time,
                    }
                )

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                # Create a dictionary that will be converted to a PyArrow Table
                dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
                return {
                    "temperature": [20, 22, 19, 23, 25, 21, 18, 20, 22, 24],
                    "humidity": [65, 70, 75, 60, 55, 65, 70, 75, 60, 55],
                    "pressure": [1012, 1010, 1008, 1007, 1009, 1011, 1013, 1012, 1010, 1009],
                    "wind_speed": [5, 7, 10, 8, 6, 4, 9, 11, 7, 5],
                    DefaultOptionKeys.reference_time: dates,
                }

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups({TestTimeDataCreator, PyArrowTimeWindowFeatureGroup})

        # Run the API with multiple time window features
        result = mlodaAPI.run_all(
            [
                "temperature",  # Source data
                "avg_3_day_window_temperature",  # 3-day average temperature
                "max_5_day_window_humidity",  # 5-day maximum humidity
                "min_2_day_window_pressure",  # 2-day minimum pressure
                "sum_4_day_window_wind_speed",  # 4-day sum of wind speed
            ],
            compute_frameworks={PyarrowTable},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) == 2  # Two Tables: one for source data, one for time window features

        # Find the Table with the time window features
        window_table = None
        for table in result:
            if "avg_3_day_window_temperature" in table.schema.names:
                window_table = table
                break

        assert window_table is not None, "Table with time window features not found"

        # Verify the time window features exist
        assert "avg_3_day_window_temperature" in window_table.schema.names
        assert "max_5_day_window_humidity" in window_table.schema.names
        assert "min_2_day_window_pressure" in window_table.schema.names
        assert "sum_4_day_window_wind_speed" in window_table.schema.names
