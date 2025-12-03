"""
Pandas implementation for time window feature groups.
"""

from __future__ import annotations

from typing import Any, List, Optional, Set, Type, Union

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup


try:
    import pandas as pd
except ImportError:
    pd = None


class PandasTimeWindowFeatureGroup(TimeWindowFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PandasDataFrame}

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
    def _get_available_columns(cls, data: pd.DataFrame) -> Set[str]:
        """Get the set of available column names from the DataFrame."""
        return set(data.columns)

    @classmethod
    def _check_source_features_exist(cls, data: pd.DataFrame, feature_names: List[str]) -> None:
        """
        Check if the resolved features exist in the DataFrame.

        Args:
            data: The Pandas DataFrame
            feature_names: List of resolved feature names (may contain ~N suffixes)

        Raises:
            ValueError: If none of the resolved features exist in the data
        """
        missing_features = [name for name in feature_names if name not in data.columns]
        if len(missing_features) == len(feature_names):
            raise ValueError(
                f"None of the source features {feature_names} found in data. Available columns: {list(data.columns)}"
            )

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
        in_features: List[str],
        time_filter_feature: Optional[str] = None,
    ) -> Any:
        """
        Perform the time window operation using Pandas rolling window functions.

        Supports both single-column and multi-column window operations:
        - Single column: aggregates values within the column over time
        - Multi-column: aggregates across columns for each time window row

        Args:
            data: The Pandas DataFrame
            window_function: The type of window function to perform
            window_size: The size of the window
            time_unit: The time unit for the window
            in_features: List of source feature names (may be single or multiple columns)
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

        # Select the columns to perform window operation on
        if len(in_features) == 1:
            # Single column: extract as Series for simpler window operation
            selected_data = df_with_time_index[in_features[0]]
        else:
            # Multiple columns: keep as DataFrame
            selected_data = df_with_time_index[in_features]

        rolling_window = selected_data.rolling(window=window_size, min_periods=1)

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

        # For multi-column, aggregate across columns (axis=1) after rolling window
        if len(in_features) > 1:
            if window_function == "sum":
                result = result.sum(axis=1)
            elif window_function == "min":
                result = result.min(axis=1)
            elif window_function == "max":
                result = result.max(axis=1)
            elif window_function in ["avg", "mean"]:
                result = result.mean(axis=1)
            elif window_function == "count":
                result = result.count(axis=1)
            elif window_function == "std":
                result = result.std(axis=1)
            elif window_function == "var":
                result = result.var(axis=1)
            elif window_function == "median":
                result = result.median(axis=1)
            elif window_function in ["first", "last"]:
                # For first/last, already computed on each column, now aggregate across columns
                result = result.mean(axis=1)  # Use mean as aggregation for first/last

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
