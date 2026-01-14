"""
Integration tests for time window feature groups.
"""

from typing import List
from mloda.user import mloda
from mloda.user import Feature
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindowFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pyarrow import PyArrowTimeWindowFeatureGroup

from tests.test_plugins.feature_group.experimental.test_time_window_feature_group.test_time_window_utils import (
    PandasTimeWindowTestDataCreator,
    PyArrowTimeWindowTestDataCreator,
    validate_time_window_features,
)


# List of time window features to test
TIME_WINDOW_FEATURES: List[Feature | str] = [
    "temperature__avg_3_day_window",  # 3-day average temperature
    "humidity__max_5_day_window",  # 5-day maximum humidity
    "pressure__min_2_day_window",  # 2-day minimum pressure
    "wind_speed__sum_4_day_window",  # 4-day sum of wind speed
]


class TestTimeWindowPandasIntegration:
    """Integration tests for the time window feature group using Pandas."""

    def test_time_window_with_data_creator(self) -> None:
        """Test time window features with mloda using DataCreator."""

        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PandasTimeWindowTestDataCreator, PandasTimeWindowFeatureGroup}
        )

        # Run the mloda with multiple time window features
        result = mloda.run_all(
            [
                "temperature",  # Source data
                "temperature__avg_3_day_window",  # 3-day average temperature
                "humidity__max_5_day_window",  # 5-day maximum humidity
                "pressure__min_2_day_window",  # 2-day minimum pressure
                "wind_speed__sum_4_day_window",  # 4-day sum of wind speed
            ],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) == 2  # Two DataFrames: one for source data, one for time window features

        # Find the DataFrame with the time window features
        window_df = None
        for df in result:
            if "temperature__avg_3_day_window" in df.columns:
                window_df = df
                break

        assert window_df is not None, "DataFrame with time window features not found"

        # Validate the time window features
        validate_time_window_features(window_df, TIME_WINDOW_FEATURES)


class TestTimeWindowPyArrowIntegration:
    """Integration tests for the time window feature group using PyArrow."""

    def test_time_window_with_data_creator(self) -> None:
        """Test time window features with mloda using DataCreator."""

        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowTimeWindowTestDataCreator, PyArrowTimeWindowFeatureGroup}
        )

        # Run the mloda with multiple time window features
        result = mloda.run_all(
            [
                "temperature",  # Source data
                "temperature__avg_3_day_window",  # 3-day average temperature
                "humidity__max_5_day_window",  # 5-day maximum humidity
                "pressure__min_2_day_window",  # 2-day minimum pressure
                "wind_speed__sum_4_day_window",  # 4-day sum of wind speed
            ],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) == 2  # Two Tables: one for source data, one for time window features

        # Find the Table with the time window features
        window_table = None
        for table in result:
            if "temperature__avg_3_day_window" in table.schema.names:
                window_table = table
                break

        assert window_table is not None, "Table with time window features not found"

        # Convert PyArrow Table to Pandas DataFrame for validation
        window_df = window_table.to_pandas()

        # Validate the time window features
        validate_time_window_features(window_df, TIME_WINDOW_FEATURES)
