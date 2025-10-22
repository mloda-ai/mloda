"""
Tests for the PandasTimeWindowFeatureGroup class.
"""

from typing import Any
import pandas as pd
import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindowFeatureGroup
from tests.test_plugins.feature_group.experimental.test_time_window_feature_group.conftest import EXPECTED_VALUES


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

    @pytest.mark.parametrize(
        "window_function,expected_values",
        [
            ("avg", EXPECTED_VALUES["avg"]),
            ("sum", EXPECTED_VALUES["sum"]),
            ("min", EXPECTED_VALUES["min"]),
            ("max", EXPECTED_VALUES["max"]),
        ],
    )
    def test_perform_window_operation(
        self, sample_time_dataframe: pd.DataFrame, window_function: str, expected_values: Any
    ) -> None:
        """Test _perform_window_operation method with different window functions."""
        result = PandasTimeWindowFeatureGroup._perform_window_operation(
            sample_time_dataframe,
            window_function,
            3,
            "day",
            ["temperature"],
            DefaultOptionKeys.reference_time,
        )

        # Check the first few values
        for i in range(3):
            if isinstance(expected_values[i], float):
                assert abs(result[i] - expected_values[i]) < 0.1
            else:
                assert result[i] == expected_values[i]

    def test_perform_window_operation_invalid(self, sample_time_dataframe: pd.DataFrame) -> None:
        """Test _perform_window_operation method with invalid window function."""
        with pytest.raises(ValueError):
            PandasTimeWindowFeatureGroup._perform_window_operation(
                sample_time_dataframe,
                "invalid",
                3,
                "day",
                ["temperature"],
                DefaultOptionKeys.reference_time,
            )

    def test_calculate_feature_single(
        self, sample_time_dataframe: pd.DataFrame, feature_set_avg_window: FeatureSet
    ) -> None:
        """Test calculate_feature method with a single time window feature."""
        result = PandasTimeWindowFeatureGroup.calculate_feature(sample_time_dataframe, feature_set_avg_window)

        # Check that the result contains the original data plus the time window feature
        assert "avg_3_day_window__temperature" in result.columns

        # Check the values of the time window feature
        # First value should be the temperature itself (20)
        # Second value should be average of first two days (20+22)/2 = 21
        # Third value should be average of first three days (20+22+19)/3 = 20.33
        assert result["avg_3_day_window__temperature"].iloc[0] == 20
        assert abs(result["avg_3_day_window__temperature"].iloc[1] - 21) < 0.1
        assert abs(result["avg_3_day_window__temperature"].iloc[2] - 20.33) < 0.1

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
        assert "avg_3_day_window__temperature" in result.columns
        assert "max_5_day_window__humidity" in result.columns
        assert "min_2_day_window__pressure" in result.columns
        assert "sum_4_day_window__wind_speed" in result.columns

        # Check that the original data is preserved
        assert "temperature" in result.columns
        assert "humidity" in result.columns
        assert "pressure" in result.columns
        assert "wind_speed" in result.columns

    def test_calculate_feature_missing_source(self, sample_time_dataframe: pd.DataFrame) -> None:
        """Test calculate_feature method with missing source feature."""
        feature_set = FeatureSet()
        feature_set.add(Feature("avg_3_day_window__missing"))

        with pytest.raises(ValueError, match="None of the source features .* found in data"):
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
            match=f"Time filter feature '{DefaultOptionKeys.reference_time.value}' not found in data.*",
        ):
            PandasTimeWindowFeatureGroup.calculate_feature(df, feature_set)
