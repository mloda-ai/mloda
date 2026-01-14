"""
Common fixtures and utilities for time window feature group tests.
"""

import pandas as pd
import pyarrow as pa
import pytest
from typing import Any, Dict, List

from mloda.user import Feature
from mloda.provider import FeatureSet
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


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
            DefaultOptionKeys.reference_time.value: dates,
        }
    )
    return df


@pytest.fixture
def sample_time_table() -> pa.Table:
    """Create a sample PyArrow Table with time series data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    # Create a DataFrame with the time_filter column explicitly named
    df = pd.DataFrame(
        {
            "temperature": [20, 22, 19, 23, 25, 21, 18, 20, 22, 24],
            "humidity": [65, 70, 75, 60, 55, 65, 70, 75, 60, 55],
            "pressure": [1012, 1010, 1008, 1007, 1009, 1011, 1013, 1012, 1010, 1009],
            "wind_speed": [5, 7, 10, 8, 6, 4, 9, 11, 7, 5],
            DefaultOptionKeys.reference_time.value: dates,
        }
    )
    return pa.Table.from_pandas(df)


@pytest.fixture
def feature_set_avg_window() -> FeatureSet:
    """Create a feature set with an average time window feature."""
    feature_set = FeatureSet()
    feature_set.add(Feature("temperature__avg_3_day_window"))
    return feature_set


@pytest.fixture
def feature_set_multiple_windows() -> FeatureSet:
    """Create a feature set with multiple time window features."""
    feature_set = FeatureSet()
    feature_set.add(Feature("temperature__avg_3_day_window"))
    feature_set.add(Feature("humidity__max_5_day_window"))
    feature_set.add(Feature("pressure__min_2_day_window"))
    feature_set.add(Feature("wind_speed__sum_4_day_window"))
    return feature_set


# Expected values for different window functions
# These are the expected results for the sample data with a 3-day window
EXPECTED_VALUES: Dict[str, List[Any]] = {
    "avg": [20.0, 21.0, 20.33, 21.33, 22.33, 23.0, 21.33, 19.67, 20.0, 22.0],
    "sum": [20, 42, 61, 64, 67, 69, 64, 59, 60, 66],
    "min": [20, 20, 19, 19, 19, 21, 18, 18, 18, 20],
    "max": [20, 22, 22, 23, 25, 25, 25, 21, 22, 24],
}
