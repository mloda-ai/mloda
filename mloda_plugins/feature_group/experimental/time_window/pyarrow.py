"""
PyArrow implementation for time window feature groups.
"""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union
import datetime

import pyarrow as pa
import pyarrow.compute as pc

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork

from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup


class PyArrowTimeWindowFeatureGroup(TimeWindowFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PyarrowTable}

    @classmethod
    def _check_time_filter_feature_exists(cls, data: pa.Table, time_filter_feature: str) -> None:
        """Check if the time filter feature exists in the Table."""
        if time_filter_feature not in data.schema.names:
            raise ValueError(
                f"Time filter feature '{time_filter_feature}' not found in data. "
                f"Please ensure the Table contains this column."
            )

    @classmethod
    def _check_time_filter_feature_is_datetime(cls, data: pa.Table, time_filter_feature: str) -> None:
        """Check if the time filter feature is a datetime column."""
        time_column = data.column(time_filter_feature)
        if not pa.types.is_timestamp(time_column.type):
            raise ValueError(
                f"Time filter feature '{time_filter_feature}' must be a timestamp column. "
                f"Current type: {time_column.type}"
            )

    @classmethod
    def _check_source_feature_exists(cls, data: pa.Table, mloda_source_feature: str) -> None:
        """Check if the source feature exists in the Table."""
        if mloda_source_feature not in data.schema.names:
            raise ValueError(f"Source feature '{mloda_source_feature}' not found in data")

    @classmethod
    def _add_result_to_data(cls, data: pa.Table, feature_name: str, result: Any) -> pa.Table:
        """Add the result to the Table."""
        return data.append_column(feature_name, result)

    @classmethod
    def _perform_window_operation(
        cls,
        data: pa.Table,
        window_function: str,
        window_size: int,
        time_unit: str,
        mloda_source_feature: str,
        time_filter_feature: Optional[str] = None,
    ) -> pa.Array:
        """
        Perform the time window operation using PyArrow compute functions.

        Args:
            data: The PyArrow Table
            window_function: The type of window function to perform
            window_size: The size of the window
            time_unit: The time unit for the window
            mloda_source_feature: The name of the source feature
            time_filter_feature: The name of the time filter feature to use for time-based operations.
                                If None, uses the value from get_time_filter_feature().

        Returns:
            The result of the window operation as a PyArrow Array
        """
        # Use the default time filter feature if none is provided
        if time_filter_feature is None:
            time_filter_feature = cls.get_time_filter_feature()

        # Get the time and source columns
        time_column = data.column(time_filter_feature)
        source_column = data.column(mloda_source_feature)

        # Sort the data by time
        # First create indices sorted by time
        sorted_indices = pc.sort_indices(time_column)

        # Get the sorted source values
        sorted_source = pc.take(source_column, sorted_indices)

        # Create a list to store the results
        results = []

        # For each row, calculate the window operation using a fixed-size window
        # This matches the pandas implementation which uses rolling(window=window_size, min_periods=1)
        for i in range(len(sorted_source)):
            # Get the window values (current and previous values up to window_size)
            start_idx = max(0, i - window_size + 1)
            window_indices = pa.array(range(start_idx, i + 1))
            window_values = pc.take(sorted_source, window_indices)

            # Apply the window function
            if len(window_values) == 0:
                # If no values in window, use the current value
                results.append(sorted_source[i].as_py())
            else:
                # Apply the appropriate window function
                if window_function == "sum":
                    results.append(pc.sum(window_values).as_py())
                elif window_function == "min":
                    results.append(pc.min(window_values).as_py())
                elif window_function == "max":
                    results.append(pc.max(window_values).as_py())
                elif window_function in ["avg", "mean"]:
                    results.append(pc.mean(window_values).as_py())
                elif window_function == "count":
                    results.append(pc.count(window_values).as_py())
                elif window_function == "std":
                    results.append(pc.stddev(window_values).as_py())
                elif window_function == "var":
                    results.append(pc.variance(window_values).as_py())
                elif window_function == "median":
                    # PyArrow doesn't have a direct median function
                    # We can approximate it using quantile with q=0.5
                    result = pc.quantile(window_values, q=0.5)
                    results.append(result[0].as_py())
                elif window_function == "first":
                    results.append(window_values[0].as_py())
                elif window_function == "last":
                    results.append(window_values[-1].as_py())
                else:
                    raise ValueError(f"Unsupported window function: {window_function}")

        # We need to reorder the results to match the original order
        # Create a mapping from sorted indices to original indices
        reordered_results = [results[sorted_indices.to_pylist().index(i)] for i in range(len(results))]

        # Convert the results to a PyArrow array
        return pa.array(reordered_results)

    @classmethod
    def _get_time_delta(cls, window_size: int, time_unit: str) -> datetime.timedelta:
        """
        Convert window size and time unit to a timedelta.

        Args:
            window_size: The size of the window
            time_unit: The time unit for the window

        Returns:
            A timedelta representing the window size
        """
        if time_unit == "second":
            return datetime.timedelta(seconds=window_size)
        elif time_unit == "minute":
            return datetime.timedelta(minutes=window_size)
        elif time_unit == "hour":
            return datetime.timedelta(hours=window_size)
        elif time_unit == "day":
            return datetime.timedelta(days=window_size)
        elif time_unit == "week":
            return datetime.timedelta(weeks=window_size)
        elif time_unit == "month":
            # Approximate a month as 30 days
            return datetime.timedelta(days=30 * window_size)
        elif time_unit == "year":
            # Approximate a year as 365 days
            return datetime.timedelta(days=365 * window_size)
        else:
            raise ValueError(f"Unsupported time unit: {time_unit}")
