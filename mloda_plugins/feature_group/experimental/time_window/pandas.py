"""
Pandas implementation for time window feature groups.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas import pandas_type_semantics
from mloda.user.pandas import PandasDataFrame
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup


try:
    import pandas as pd
except ImportError:
    pd = None


class PandasTimeWindowFeatureGroup(TimeWindowFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def _check_reference_time_column_exists(cls, data: pd.DataFrame, reference_time_column: str) -> None:
        """Check if the reference time column exists in the DataFrame."""
        if reference_time_column not in data.columns:
            raise ValueError(
                f"Reference time column '{reference_time_column}' not found in data. "
                f"Please ensure the DataFrame contains this column."
            )

    @classmethod
    def _check_reference_time_column_is_datetime(cls, data: pd.DataFrame, reference_time_column: str) -> None:
        """Check if the reference time column is a datetime column."""
        semantics = pandas_type_semantics.column_semantics(data, reference_time_column)
        cls._validate_reference_time_column(semantics, reference_time_column)

    @classmethod
    def _get_available_columns(cls, data: pd.DataFrame) -> set[str]:
        """Get the set of available column names from the DataFrame."""
        return set(data.columns)

    @classmethod
    def _check_source_features_exist(cls, data: pd.DataFrame, feature_names: list[str]) -> None:
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
        in_features: list[str],
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
                                If None, uses the value from get_reference_time_column().

        Returns:
            The result of the window operation
        """
        # Use the default time filter feature if none is provided
        if time_filter_feature is None:
            time_filter_feature = cls.get_reference_time_column()

        # Coerce the reference column to a proper datetime index (handles arrow
        # date32/date64 columns and keeps tz-awareness). A null/NaT reference time
        # has no position on the timeline, so fail explicitly.
        times = pd.to_datetime(data[time_filter_feature])
        if times.isna().any():
            cls._raise_null_reference_time(time_filter_feature)

        # Sort STABLY by absolute time, remembering original positions, so the
        # rolled result can be scattered back to the ORIGINAL (possibly unsorted)
        # row order. to_numpy() on a tz-aware column yields absolute UTC instants.
        order = np.argsort(times.to_numpy(), kind="stable")
        sorted_index = pd.DatetimeIndex(times.to_numpy()[order])

        # Select the columns to perform window operation on, in time-sorted order.
        if len(in_features) == 1:
            # Single column: extract as Series for simpler window operation
            selected_data = pd.Series(data[in_features[0]].to_numpy()[order], index=sorted_index)
        else:
            # Multiple columns: keep as DataFrame
            selected_data = pd.DataFrame({c: data[c].to_numpy()[order] for c in in_features}, index=sorted_index)

        # Time-based (not row-count) window: include rows in (t - span, t].
        # Pass a Timedelta, not an offset string: aliases like "3M"/"3Y" raise
        # "passed window 3M is not compatible with a datetimelike index".
        span = pd.Timedelta(cls._get_time_delta(window_size, time_unit))
        rolling_window = selected_data.rolling(window=span, min_periods=1, closed="right")

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

        # Scatter the time-sorted results back to the ORIGINAL row order.
        sorted_values = np.asarray(result)
        out = np.empty(len(sorted_values), dtype=sorted_values.dtype)
        out[order] = sorted_values
        return out
