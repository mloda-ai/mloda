"""
Pandas implementation for time window feature groups.
"""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union

import pandas as pd

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.time_window.base import BaseTimeWindowFeatureGroup


class PandasTimeWindowFeatureGroup(BaseTimeWindowFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PandasDataframe}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Perform time window operations using Pandas.

        Processes all requested features, determining the window function, window size,
        time unit, and source feature from each feature name.

        Adds the time window results directly to the input DataFrame.
        """
        time_filter_feature = cls.get_time_filter_feature(features.options)

        # Check if the time filter feature exists in the DataFrame
        if time_filter_feature not in data.columns:
            raise ValueError(
                f"Time filter feature '{time_filter_feature}' not found in data. "
                f"Please ensure the DataFrame contains this column."
            )

        # Check if the time filter feature is a datetime column
        if not pd.api.types.is_datetime64_any_dtype(data[time_filter_feature]):
            raise ValueError(
                f"Time filter feature '{time_filter_feature}' must be a datetime column. "
                f"Current dtype: {data[time_filter_feature].dtype}"
            )

        # Process each requested feature
        for feature_name in features.get_all_names():
            window_function, window_size, time_unit, source_feature = cls.parse_feature_name(feature_name)

            if source_feature not in data.columns:
                raise ValueError(f"Source feature '{source_feature}' not found in data")

            data[feature_name] = cls._perform_window_operation(
                data, window_function, window_size, time_unit, source_feature, time_filter_feature
            )

        return data

    @classmethod
    def _perform_window_operation(
        cls,
        data: pd.DataFrame,
        window_function: str,
        window_size: int,
        time_unit: str,
        mloda_source_feature: str,
        time_filter_feature: Optional[str] = None,
    ) -> pd.Series:
        """
        Perform the time window operation using Pandas rolling window functions.

        Args:
            data: The Pandas DataFrame
            window_function: The type of window function to perform
            window_size: The size of the window
            time_unit: The time unit for the window
            mloda_source_feature: The name of the source feature
            time_filter_feature: The name of the time filter feature to use for time-based operations.
                                If None, uses the value from get_time_filter_feature().

        Returns:
            The result of the window operation as a Pandas Series
        """
        # Use the default time filter feature if none is provided
        if time_filter_feature is None:
            time_filter_feature = cls.get_time_filter_feature()

        # Create a copy of the DataFrame with the time filter feature as the index
        # This is necessary for time-based rolling operations
        df_with_time_index = data.set_index(time_filter_feature).sort_index()

        rolling_window = df_with_time_index[mloda_source_feature].rolling(window=window_size, min_periods=1)

        if window_function == "sum":
            result = rolling_window.sum()
        elif window_function == "min":
            result = rolling_window.min()
        elif window_function == "max":
            result = rolling_window.max()
        elif window_function in ["avg", "mean"]:
            result = rolling_window.mean()
        elif window_function == "count":
            result = rolling_window.count()
        elif window_function == "std":
            result = rolling_window.std()
        elif window_function == "var":
            result = rolling_window.var()
        elif window_function == "median":
            result = rolling_window.median()
        elif window_function == "first":
            result = rolling_window.apply(lambda x: x.iloc[0] if len(x) > 0 else None, raw=False)
        elif window_function == "last":
            result = rolling_window.apply(lambda x: x.iloc[-1] if len(x) > 0 else None, raw=False)
        else:
            raise ValueError(f"Unsupported window function: {window_function}")

        return result.values

    @classmethod
    def _get_pandas_freq(cls, window_size: int, time_unit: str) -> str:
        """
        Convert window size and time unit to a pandas-compatible frequency string.

        Args:
            window_size: The size of the window
            time_unit: The time unit for the window

        Returns:
            A pandas-compatible frequency string
        """
        # Map time units to pandas frequency aliases
        time_unit_map = {
            "second": "S",
            "minute": "T",
            "hour": "H",
            "day": "D",
            "week": "W",
            "month": "M",
            "year": "Y",
        }

        if time_unit not in time_unit_map:
            raise ValueError(f"Unsupported time unit: {time_unit}")

        # Construct the frequency string
        return f"{window_size}{time_unit_map[time_unit]}"
