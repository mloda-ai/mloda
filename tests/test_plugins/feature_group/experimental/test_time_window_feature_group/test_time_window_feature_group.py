import pandas as pd
import pytest
from typing import Any, Optional

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.time_window.base import BaseTimeWindowFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindowFeatureGroup


@pytest.fixture
def sample_time_dataframe() -> pd.DataFrame:
    """Create a sample pandas DataFrame with time series data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "temperature": [20, 22, 19, 23, 25, 21, 18, 20, 22, 24],
            "humidity": [65, 70, 75, 60, 55, 65, 70, 75, 60, 55],
            "pressure": [1012, 1010, 1008, 1007, 1009, 1011, 1013, 1012, 1010, 1009],
            "wind_speed": [5, 7, 10, 8, 6, 4, 9, 11, 7, 5],
            DefaultOptionKeys.reference_time: dates,
        }
    )
    return df


@pytest.fixture
def feature_set_avg_window() -> FeatureSet:
    """Create a feature set with an average time window feature."""
    feature_set = FeatureSet()
    feature_set.add(Feature("avg_3_day_window_temperature"))
    return feature_set


@pytest.fixture
def feature_set_multiple_windows() -> FeatureSet:
    """Create a feature set with multiple time window features."""
    feature_set = FeatureSet()
    feature_set.add(Feature("avg_3_day_window_temperature"))
    feature_set.add(Feature("max_5_day_window_humidity"))
    feature_set.add(Feature("min_2_day_window_pressure"))
    feature_set.add(Feature("sum_4_day_window_wind_speed"))
    return feature_set


class TestBaseTimeWindowFeatureGroup:
    """Tests for the BaseTimeWindowFeatureGroup class."""

    def test_parse_feature_name(self) -> None:
        """Test parsing of feature name into components."""
        window_function, window_size, time_unit, source_feature = BaseTimeWindowFeatureGroup.parse_feature_name(
            "avg_3_day_window_temperature"
        )
        assert window_function == "avg"
        assert window_size == 3
        assert time_unit == "day"
        assert source_feature == "temperature"

        window_function, window_size, time_unit, source_feature = BaseTimeWindowFeatureGroup.parse_feature_name(
            "max_7_hour_window_humidity"
        )
        assert window_function == "max"
        assert window_size == 7
        assert time_unit == "hour"
        assert source_feature == "humidity"

        # Test with invalid feature names
        with pytest.raises(ValueError):
            BaseTimeWindowFeatureGroup.parse_feature_name("invalid_feature_name")

        with pytest.raises(ValueError):
            BaseTimeWindowFeatureGroup.parse_feature_name("avg_day_window_temperature")

        with pytest.raises(ValueError):
            BaseTimeWindowFeatureGroup.parse_feature_name("avg_3_invalid_window_temperature")

        with pytest.raises(ValueError):
            BaseTimeWindowFeatureGroup.parse_feature_name("invalid_3_day_window_temperature")

        with pytest.raises(ValueError):
            BaseTimeWindowFeatureGroup.parse_feature_name("avg_-1_day_window_temperature")

    def test_get_window_function(self) -> None:
        """Test extraction of window function from feature name."""
        assert BaseTimeWindowFeatureGroup.get_window_function("avg_3_day_window_temperature") == "avg"
        assert BaseTimeWindowFeatureGroup.get_window_function("max_7_hour_window_humidity") == "max"
        assert BaseTimeWindowFeatureGroup.get_window_function("min_2_day_window_pressure") == "min"
        assert BaseTimeWindowFeatureGroup.get_window_function("sum_4_day_window_wind_speed") == "sum"

    def test_get_window_size(self) -> None:
        """Test extraction of window size from feature name."""
        assert BaseTimeWindowFeatureGroup.get_window_size("avg_3_day_window_temperature") == 3
        assert BaseTimeWindowFeatureGroup.get_window_size("max_7_hour_window_humidity") == 7
        assert BaseTimeWindowFeatureGroup.get_window_size("min_2_day_window_pressure") == 2
        assert BaseTimeWindowFeatureGroup.get_window_size("sum_4_day_window_wind_speed") == 4

    def test_get_time_unit(self) -> None:
        """Test extraction of time unit from feature name."""
        assert BaseTimeWindowFeatureGroup.get_time_unit("avg_3_day_window_temperature") == "day"
        assert BaseTimeWindowFeatureGroup.get_time_unit("max_7_hour_window_humidity") == "hour"
        assert BaseTimeWindowFeatureGroup.get_time_unit("min_2_minute_window_pressure") == "minute"
        assert BaseTimeWindowFeatureGroup.get_time_unit("sum_4_second_window_wind_speed") == "second"

    def test_get_source_feature(self) -> None:
        """Test extraction of source feature from feature name."""
        assert BaseTimeWindowFeatureGroup.get_source_feature("avg_3_day_window_temperature") == "temperature"
        assert BaseTimeWindowFeatureGroup.get_source_feature("max_7_hour_window_humidity") == "humidity"
        assert BaseTimeWindowFeatureGroup.get_source_feature("min_2_day_window_pressure") == "pressure"
        assert BaseTimeWindowFeatureGroup.get_source_feature("sum_4_day_window_wind_speed") == "wind_speed"

    def test_match_feature_group_criteria(self) -> None:
        """Test match_feature_group_criteria method."""
        options = Options()

        # Test with valid feature names
        assert BaseTimeWindowFeatureGroup.match_feature_group_criteria("avg_3_day_window_temperature", options)
        assert BaseTimeWindowFeatureGroup.match_feature_group_criteria("max_7_hour_window_humidity", options)
        assert BaseTimeWindowFeatureGroup.match_feature_group_criteria("min_2_day_window_pressure", options)
        assert BaseTimeWindowFeatureGroup.match_feature_group_criteria("sum_4_day_window_wind_speed", options)

        # Test with FeatureName objects
        assert BaseTimeWindowFeatureGroup.match_feature_group_criteria(
            FeatureName("avg_3_day_window_temperature"), options
        )
        assert BaseTimeWindowFeatureGroup.match_feature_group_criteria(
            FeatureName("max_7_hour_window_humidity"), options
        )

        # Test with invalid feature names
        assert not BaseTimeWindowFeatureGroup.match_feature_group_criteria("invalid_feature_name", options)
        assert not BaseTimeWindowFeatureGroup.match_feature_group_criteria("avg_day_window_temperature", options)
        assert not BaseTimeWindowFeatureGroup.match_feature_group_criteria("avg_3_invalid_window_temperature", options)

    def test_input_features(self) -> None:
        """Test input_features method."""
        options = Options()
        feature_group = BaseTimeWindowFeatureGroup()

        # Test with valid feature names
        input_features = feature_group.input_features(options, FeatureName("avg_3_day_window_temperature"))
        assert input_features == {
            Feature("temperature"),
            Feature(DefaultOptionKeys.reference_time),
        }

        input_features = feature_group.input_features(options, FeatureName("max_7_hour_window_humidity"))
        assert input_features == {Feature("humidity"), Feature(DefaultOptionKeys.reference_time)}

        input_features = feature_group.input_features(options, FeatureName("min_2_day_window_pressure"))
        assert input_features == {Feature("pressure"), Feature(DefaultOptionKeys.reference_time)}

        input_features = feature_group.input_features(options, FeatureName("sum_4_day_window_wind_speed"))
        assert input_features == {
            Feature("wind_speed"),
            Feature(DefaultOptionKeys.reference_time),
        }


class TestPandasTimeWindowFeatureGroup:
    """Tests for the PandasTimeWindowFeatureGroup class."""

    def test_compute_framework_rule(self) -> None:
        """Test compute_framework_rule method."""
        assert PandasTimeWindowFeatureGroup.compute_framework_rule() == {PandasDataframe}

    def test_get_pandas_freq(self) -> None:
        """Test _get_pandas_freq method."""
        assert PandasTimeWindowFeatureGroup._get_pandas_freq(3, "day") == "3D"
        assert PandasTimeWindowFeatureGroup._get_pandas_freq(7, "hour") == "7H"
        assert PandasTimeWindowFeatureGroup._get_pandas_freq(2, "minute") == "2T"
        assert PandasTimeWindowFeatureGroup._get_pandas_freq(4, "second") == "4S"
        assert PandasTimeWindowFeatureGroup._get_pandas_freq(1, "week") == "1W"
        assert PandasTimeWindowFeatureGroup._get_pandas_freq(6, "month") == "6M"
        assert PandasTimeWindowFeatureGroup._get_pandas_freq(2, "year") == "2Y"

        # Test with invalid time unit
        with pytest.raises(ValueError):
            PandasTimeWindowFeatureGroup._get_pandas_freq(3, "invalid")

    def test_perform_window_operation_avg(self, sample_time_dataframe: pd.DataFrame) -> None:
        """Test _perform_window_operation method with avg window function."""
        result = PandasTimeWindowFeatureGroup._perform_window_operation(
            sample_time_dataframe,
            "avg",
            3,
            "day",
            "temperature",
            DefaultOptionKeys.reference_time,
        )
        # First value should be the temperature itself (20) since window size is 1
        # Second value should be average of first two days (20+22)/2 = 21
        # Third value should be average of first three days (20+22+19)/3 = 20.33
        assert result[0] == 20
        assert abs(result[1] - 21) < 0.1
        assert abs(result[2] - 20.33) < 0.1

    def test_perform_window_operation_max(self, sample_time_dataframe: pd.DataFrame) -> None:
        """Test _perform_window_operation method with max window function."""
        result = PandasTimeWindowFeatureGroup._perform_window_operation(
            sample_time_dataframe,
            "max",
            3,
            "day",
            "temperature",
            DefaultOptionKeys.reference_time,
        )
        # First value should be the temperature itself (20)
        # Second value should be max of first two days max(20, 22) = 22
        # Third value should be max of first three days max(20, 22, 19) = 22
        assert result[0] == 20
        assert result[1] == 22
        assert result[2] == 22

    def test_perform_window_operation_min(self, sample_time_dataframe: pd.DataFrame) -> None:
        """Test _perform_window_operation method with min window function."""
        result = PandasTimeWindowFeatureGroup._perform_window_operation(
            sample_time_dataframe,
            "min",
            3,
            "day",
            "temperature",
            DefaultOptionKeys.reference_time,
        )
        # First value should be the temperature itself (20)
        # Second value should be min of first two days min(20, 22) = 20
        # Third value should be min of first three days min(20, 22, 19) = 19
        assert result[0] == 20
        assert result[1] == 20
        assert result[2] == 19

    def test_perform_window_operation_sum(self, sample_time_dataframe: pd.DataFrame) -> None:
        """Test _perform_window_operation method with sum window function."""
        result = PandasTimeWindowFeatureGroup._perform_window_operation(
            sample_time_dataframe,
            "sum",
            3,
            "day",
            "temperature",
            DefaultOptionKeys.reference_time,
        )
        # First value should be the temperature itself (20)
        # Second value should be sum of first two days (20+22) = 42
        # Third value should be sum of first three days (20+22+19) = 61
        assert result[0] == 20
        assert result[1] == 42
        assert result[2] == 61

    def test_perform_window_operation_invalid(self, sample_time_dataframe: pd.DataFrame) -> None:
        """Test _perform_window_operation method with invalid window function."""
        with pytest.raises(ValueError):
            PandasTimeWindowFeatureGroup._perform_window_operation(
                sample_time_dataframe,
                "invalid",
                3,
                "day",
                "temperature",
                DefaultOptionKeys.reference_time,
            )

    def test_calculate_feature_single(
        self, sample_time_dataframe: pd.DataFrame, feature_set_avg_window: FeatureSet
    ) -> None:
        """Test calculate_feature method with a single time window feature."""
        result = PandasTimeWindowFeatureGroup.calculate_feature(sample_time_dataframe, feature_set_avg_window)

        # Check that the result contains the original data plus the time window feature
        assert "avg_3_day_window_temperature" in result.columns

        # Check the values of the time window feature
        # First value should be the temperature itself (20)
        # Second value should be average of first two days (20+22)/2 = 21
        # Third value should be average of first three days (20+22+19)/3 = 20.33
        assert result["avg_3_day_window_temperature"].iloc[0] == 20
        assert abs(result["avg_3_day_window_temperature"].iloc[1] - 21) < 0.1
        assert abs(result["avg_3_day_window_temperature"].iloc[2] - 20.33) < 0.1

        # Check that the original data is preserved
        assert "temperature" in result.columns
        assert "humidity" in result.columns
        assert "pressure" in result.columns
        assert "wind_speed" in result.columns

    def test_calculate_feature_multiple(
        self, sample_time_dataframe: pd.DataFrame, feature_set_multiple_windows: FeatureSet
    ) -> None:
        """Test calculate_feature method with multiple time window features."""
        result = PandasTimeWindowFeatureGroup.calculate_feature(sample_time_dataframe, feature_set_multiple_windows)

        # Check that the result contains all time window features
        assert "avg_3_day_window_temperature" in result.columns
        assert "max_5_day_window_humidity" in result.columns
        assert "min_2_day_window_pressure" in result.columns
        assert "sum_4_day_window_wind_speed" in result.columns

        # Check that the original data is preserved
        assert "temperature" in result.columns
        assert "humidity" in result.columns
        assert "pressure" in result.columns
        assert "wind_speed" in result.columns

    def test_calculate_feature_missing_source(self, sample_time_dataframe: pd.DataFrame) -> None:
        """Test calculate_feature method with missing source feature."""
        feature_set = FeatureSet()
        feature_set.add(Feature("avg_3_day_window_missing"))

        with pytest.raises(ValueError, match="Source feature 'missing' not found in data"):
            PandasTimeWindowFeatureGroup.calculate_feature(sample_time_dataframe, feature_set)

    def test_calculate_feature_missing_time_filter(self) -> None:
        """Test calculate_feature method with DataFrame that has no time filter feature."""
        # Create a DataFrame without the time filter feature
        df = pd.DataFrame(
            {
                "temperature": [20, 22, 19, 23, 25],
                "humidity": [65, 70, 75, 60, 55],
            }
        )

        feature_set = FeatureSet()
        feature_set.add(Feature("avg_3_day_window_temperature"))

        with pytest.raises(
            ValueError,
            match=f"Time filter feature '{DefaultOptionKeys.reference_time}' not found in data",
        ):
            PandasTimeWindowFeatureGroup.calculate_feature(df, feature_set)


class TestTimeWindowPandasIntegration:
    """Integration tests for the time window feature group using DataCreator."""

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
