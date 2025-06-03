"""
Pandas implementation for time window feature groups.
"""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup


try:
    import pandas as pd
except ImportError:
    pd = None


class PandasTimeWindowFeatureGroup(TimeWindowFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PandasDataframe}

    @classmethod
    def _check_time_filter_feature_exists(cls, data: pd.DataFrame, time_filter_feature: str) -> None:
        """Check if the time filter feature exists in the DataFrame."""
        if time_filter_feature not in data.columns:
            raise ValueError(
                f"Time filter feature '{time_filter_feature}' not found in data. "
                f"Please ensure the DataFrame contains this column."
            )

    @classmethod
    def _check_time_filter_feature_is_datetime(cls, data: pd.DataFrame, time_filter_feature: str) -> None:
        """Check if the time filter feature is a datetime column."""
        if not pd.api.types.is_datetime64_any_dtype(data[time_filter_feature]):
            raise ValueError(
                f"Time filter feature '{time_filter_feature}' must be a datetime column. "
                f"Current dtype: {data[time_filter_feature].dtype}"
            )

    @classmethod
    def _check_source_feature_exists(cls, data: pd.DataFrame, mloda_source_feature: str) -> None:
        """Check if the source feature exists in the DataFrame."""
        if mloda_source_feature not in data.columns:
            raise ValueError(f"Source feature '{mloda_source_feature}' not found in data")

    @classmethod
    def _add_result_to_data(cls, data: pd.DataFrame, feature_name: str, result: Any) -> pd.DataFrame:
        """Add the result to the DataFrame."""
        data[feature_name] = result
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
    ) -> Any:
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
            The result of the window operation
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

        # Convert to numpy array to avoid type issues
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
