import pandas as pd
from datetime import datetime, timezone
from typing import Any, Optional

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_core.filter.global_filter import GlobalFilter
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindowFeatureGroup


class TestTimeWindowWithGlobalFilter:
    """Integration tests for the time window feature group with GlobalFilter."""

    def test_time_a_window_with_global_filter(self) -> None:
        """
        Test time window features with GlobalFilter time filtering.

        This test verifies that:
        1. The GlobalFilter's time filter correctly limits the data to a specific time range
        2. The time window calculations are performed correctly on the filtered data
        3. The window calculations near the boundaries of the filter handle edge cases properly
        """

        # Create a feature group that uses DataCreator to provide test data
        # with a wider date range than we'll filter to
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
                # Use a 15-day range so we can filter to a subset
                # Add timezone information to make dates timezone-aware
                dates = pd.date_range(start="2023-01-01", periods=15, freq="D", tz="UTC")
                return pd.DataFrame(
                    {
                        "temperature": [20, 22, 19, 23, 25, 21, 18, 20, 22, 24, 26, 23, 21, 19, 22],
                        "humidity": [65, 70, 75, 60, 55, 65, 70, 75, 60, 55, 50, 60, 65, 70, 60],
                        "pressure": [
                            1012,
                            1010,
                            1008,
                            1007,
                            1009,
                            1011,
                            1013,
                            1012,
                            1010,
                            1009,
                            1008,
                            1010,
                            1012,
                            1013,
                            1011,
                        ],
                        "wind_speed": [5, 7, 10, 8, 6, 4, 9, 11, 7, 5, 8, 10, 6, 4, 7],
                        DefaultOptionKeys.reference_time: dates,
                    }
                )

        # Create a GlobalFilter with a time range filter
        # Filter to days 5-10 (indices 4-9) of our 15-day dataset
        global_filter = GlobalFilter()

        # Define the time range for filtering (days 5-10)
        event_from = datetime(2023, 1, 5, tzinfo=timezone.utc)  # 5th day
        event_to = datetime(2023, 1, 11, tzinfo=timezone.utc)  # 11th day (exclusive)

        # Add the time filter
        global_filter.add_time_and_time_travel_filters(
            event_from=event_from,
            event_to=event_to,
            # Use the default time filter feature name which matches DefaultOptionKeys.reference_time
        )

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups({TestTimeDataCreator, PandasTimeWindowFeatureGroup})

        # Run the API with time window features and the global filter
        result = mlodaAPI.run_all(
            [
                DefaultOptionKeys.reference_time,
                "temperature",  # Source data
                "temperature__avg_2_day_window",  # 2-day average temperature
                "humidity__max_3_day_window",  # 3-day maximum humidity
            ],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
            global_filter=global_filter,  # Pass the global filter to the API
        )

        # Verify the results
        assert len(result) > 0, "No results returned from mlodaAPI.run_all"

        # Based on the test output, we need to handle the DataFrame structure differently
        # The result contains multiple DataFrames, and we need to verify the filtering worked

        # First, verify we have at least one DataFrame with the time_filter column
        time_filter_df = None
        for df in result:
            if DefaultOptionKeys.reference_time in df.columns:
                time_filter_df = df
                break

        assert time_filter_df is not None, f"DataFrame with {DefaultOptionKeys.reference_time} not found"

        # Verify that the filter was applied - we should only have 6 rows (days 5-10)
        assert len(time_filter_df) == 6, f"Expected 6 rows after filtering, got {len(time_filter_df)}"

        # Get the dates from the filtered DataFrame to verify the time range
        dates = time_filter_df[DefaultOptionKeys.reference_time]
        min_date = min(dates).to_pydatetime().replace(tzinfo=timezone.utc)
        max_date = max(dates).to_pydatetime().replace(tzinfo=timezone.utc)

        # Verify the date range matches our filter
        assert min_date >= event_from, f"Min date {min_date} should be >= {event_from}"
        assert max_date < event_to, f"Max date {max_date} should be < {event_to}"

        # Verify that the temperature data is present
        assert "temperature" in time_filter_df.columns, "temperature column not found"

        # Verify that the filtered data contains the expected temperature values
        # The test data has temperature values [20, 22, 19, 23, 25, 21, 18, 20, 22, 24, 26, 23, 21, 19, 22]
        # Days 5-10 (indices 4-9) should have values [25, 21, 18, 20, 22, 24]
        expected_temps = [25, 21, 18, 20, 22, 24]
        actual_temps = time_filter_df["temperature"].tolist()
        assert actual_temps == expected_temps, f"Expected temperatures {expected_temps}, got {actual_temps}"

        # Find the DataFrame with the window features
        window_features_df = None
        for df in result:
            if "temperature__avg_2_day_window" in df.columns or "humidity__max_3_day_window" in df.columns:
                window_features_df = df
                break

        assert window_features_df is not None, "DataFrame with window features not found"

        # Verify the window features exist in the DataFrame
        assert "temperature__avg_2_day_window" in window_features_df.columns, "temperature__avg_2_day_window not found"
        assert "humidity__max_3_day_window" in window_features_df.columns, "humidity__max_3_day_window not found"

        # Get the window feature values
        avg_temp_values = window_features_df["temperature__avg_2_day_window"].tolist()

        assert avg_temp_values == [
            25.0,
            23.0,
            19.5,
            19.0,
            21.0,
            23.0,
        ], f"Unexpected temperature__avg_2_day_window values: {avg_temp_values}"
        assert window_features_df["humidity__max_3_day_window"].tolist() == [
            55.0,
            65.0,
            70.0,
            75.0,
            75.0,
            75.0,
        ], f"Unexpected humidity__max_3_day_window values: {window_features_df['humidity__max_3_day_window'].tolist()}"

    def test_time_window_with_custom_time_filter_name(self) -> None:
        """
        Test time window features with a custom time filter feature name.

        This test verifies that:
        1. The TimeWindowFeatureGroup can work with a custom time filter feature name
        2. The GlobalFilter can be configured to use the same custom time filter feature name
        3. The integration works correctly with the custom time filter feature name
        """
        # Define a custom time filter feature name
        custom_time_filter = "event_timestamp"

        # Create a custom time window feature group that uses the custom time filter
        class CustomTimeWindowFeatureGroup(PandasTimeWindowFeatureGroup):
            # Override the default time filter feature name
            DEFAULT_TIME_FILTER_FEATURE = custom_time_filter

        # Create a feature group that uses DataCreator to provide test data
        # with the custom time filter feature name
        class TestCustomTimeDataCreator(AbstractFeatureGroup):
            @classmethod
            def input_data(cls) -> Optional[BaseInputData]:
                return DataCreator({"temperature", "humidity", custom_time_filter})

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                # Create a DataFrame with the custom time filter feature
                dates = pd.date_range(start="2023-01-01", periods=15, freq="D", tz="UTC")
                return pd.DataFrame(
                    {
                        "temperature": [20, 22, 19, 23, 25, 21, 18, 20, 22, 24, 26, 23, 21, 19, 22],
                        "humidity": [65, 70, 75, 60, 55, 65, 70, 75, 60, 55, 50, 60, 65, 70, 60],
                        custom_time_filter: dates,  # Use the custom time filter feature name
                    }
                )

        # Create a GlobalFilter with a time range filter using the custom time filter feature name
        global_filter = GlobalFilter()

        # Define the time range for filtering (days 5-10)
        event_from = datetime(2023, 1, 5, tzinfo=timezone.utc)
        event_to = datetime(2023, 1, 11, tzinfo=timezone.utc)

        # Add the time filter with the custom time filter feature name
        global_filter.add_time_and_time_travel_filters(
            event_from=event_from,
            event_to=event_to,
            time_filter_feature=custom_time_filter,  # Use the custom time filter feature name
        )

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {TestCustomTimeDataCreator, CustomTimeWindowFeatureGroup}
        )

        avg_3_day_window_temperature = Feature(
            name="temperature__avg_3_day_window", options={DefaultOptionKeys.reference_time: custom_time_filter}
        )
        temperature = Feature(name="temperature", options={DefaultOptionKeys.reference_time: custom_time_filter})

        result = mlodaAPI.run_all(
            [
                temperature,
                avg_3_day_window_temperature,
            ],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
            global_filter=global_filter,
        )

        assert len(result) > 0, "No results returned from mlodaAPI.run_all"
