"""
Utility functions and data creators for time window feature tests.
"""

from typing import Any, Dict, List
import pandas as pd

from mloda.user import Feature
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class TimeWindowTestDataCreator(ATestDataCreator):
    """Base class for time window test data creators."""

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        return {
            "temperature": [20, 22, 19, 23, 25, 21, 18, 20, 22, 24],
            "humidity": [65, 70, 75, 60, 55, 65, 70, 75, 60, 55],
            "pressure": [1012, 1010, 1008, 1007, 1009, 1011, 1013, 1012, 1010, 1009],
            "wind_speed": [5, 7, 10, 8, 6, 4, 9, 11, 7, 5],
            DefaultOptionKeys.reference_time: dates,
        }


class PandasTimeWindowTestDataCreator(TimeWindowTestDataCreator):
    compute_framework = PandasDataFrame


class PyArrowTimeWindowTestDataCreator(TimeWindowTestDataCreator):
    compute_framework = PyArrowTable


def validate_time_window_features(window_df: pd.DataFrame, expected_features: List[Feature | str]) -> None:
    """
    Validate time window features in a Pandas DataFrame.

    Args:
        window_df: DataFrame containing time window features
        expected_features: List of expected feature names

    Raises:
        AssertionError: If validation fails
    """
    # Verify all expected features exist
    for feature in expected_features:
        # Get the feature name if it's a Feature object, otherwise use it directly
        feature_name = feature.name if isinstance(feature, Feature) else feature
        assert feature_name in window_df.columns, f"Expected feature '{feature_name}' not found"
