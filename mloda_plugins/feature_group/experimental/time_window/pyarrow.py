"""
PyArrow implementation for time window feature groups.
"""

from __future__ import annotations

from typing import Any, List, Optional, Set, Type, Union
import datetime

import pyarrow as pa
import pyarrow.compute as pc

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork

from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup


class PyArrowTimeWindowFeatureGroup(TimeWindowFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PyArrowTable}

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
    def _get_available_columns(cls, data: pa.Table) -> Set[str]:
        """Get the set of available column names from the Table schema."""
        return set(data.schema.names)

    @classmethod
    def _check_source_features_exist(cls, data: pa.Table, feature_names: List[str]) -> None:
        """
        Check if the resolved features exist in the Table.

        Args:
            data: The PyArrow Table
            feature_names: List of resolved feature names (may contain ~N suffixes)

        Raises:
            ValueError: If none of the resolved features exist in the data
        """
        schema_names = set(data.schema.names)
        missing_features = [name for name in feature_names if name not in schema_names]
        if len(missing_features) == len(feature_names):
            raise ValueError(
                f"None of the source features {feature_names} found in data. Available columns: {list(schema_names)}"
            )

    @classmethod
    def _add_result_to_data(cls, data: pa.Table, feature_name: str, result: Any) -> pa.Table:
        """Add the result to the Table."""
        # Check if column already exists
        if feature_name in data.schema.names:
            # Column exists, replace it by removing the old one and adding the new one
            column_index = data.schema.names.index(feature_name)
            # Remove the existing column
            data = data.remove_column(column_index)
            # Add the new column
            return data.append_column(feature_name, result)
        else:
            # Column doesn't exist, add it normally
            return data.append_column(feature_name, result)

    @classmethod
    def _perform_window_operation(
        cls,
        data: pa.Table,
        window_function: str,
        window_size: int,
        time_unit: str,
        in_features: List[str],
        time_filter_feature: Optional[str] = None,
    ) -> pa.Array:
        """
        Perform the time window operation using PyArrow compute functions.

        Supports both single-column and multi-column window operations:
        - Single column: aggregates values within the column over time
        - Multi-column: aggregates across columns for each time window row

        Args:
            data: The PyArrow Table
            window_function: The type of window function to perform
            window_size: The size of the window
            time_unit: The time unit for the window
            in_features: List of source feature names (may be single or multiple columns)
            time_filter_feature: The name of the time filter feature to use for time-based operations.
                                If None, uses the value from get_time_filter_feature().

        Returns:
            The result of the window operation as a PyArrow Array
        """
        # Use the default time filter feature if none is provided
        if time_filter_feature is None:
            time_filter_feature = cls.get_time_filter_feature()

        # Get the time column
        time_column = data.column(time_filter_feature)

        # Get the source columns
        source_columns = [data.column(name) for name in in_features]

        # Sort the data by time
        # First create indices sorted by time
        sorted_indices = pc.sort_indices(time_column)

        # Get the sorted source values for each column
        sorted_sources = [pc.take(col, sorted_indices) for col in source_columns]

        # Create a list to store the results
        results = []

        # For each row, calculate the window operation using a fixed-size window
        # This matches the pandas implementation which uses rolling(window=window_size, min_periods=1)
        for i in range(len(sorted_sources[0])):
            # Get the window values (current and previous values up to window_size)
            start_idx = max(0, i - window_size + 1)
            window_indices = pa.array(range(start_idx, i + 1))

            # Get window values for all columns
            all_window_values = [pc.take(col, window_indices) for col in sorted_sources]

            # Apply the window function
            if len(all_window_values[0]) == 0:
                # If no values in window, use the current value
                results.append(sorted_sources[0][i].as_py())
            else:
                # For multi-column, first compute rolling window per column, then aggregate across columns
                column_results = []
                for window_values in all_window_values:
                    if window_function == "sum":
                        column_results.append(pc.sum(window_values).as_py())
                    elif window_function == "min":
                        column_results.append(pc.min(window_values).as_py())
                    elif window_function == "max":
                        column_results.append(pc.max(window_values).as_py())
                    elif window_function in ["avg", "mean"]:
                        column_results.append(pc.mean(window_values).as_py())
                    elif window_function == "count":
                        column_results.append(pc.count(window_values).as_py())
                    elif window_function == "std":
                        column_results.append(pc.stddev(window_values).as_py())
                    elif window_function == "var":
                        column_results.append(pc.variance(window_values).as_py())
                    elif window_function == "median":
                        # PyArrow doesn't have a direct median function
                        # We can approximate it using quantile with q=0.5
                        result = pc.quantile(window_values, q=0.5)
                        column_results.append(result[0].as_py())
                    elif window_function == "first":
                        column_results.append(window_values[0].as_py())
                    elif window_function == "last":
                        column_results.append(window_values[-1].as_py())
                    else:
                        raise ValueError(f"Unsupported window function: {window_function}")

                # If multi-column, aggregate across columns
                if len(in_features) > 1:
                    import numpy as np

                    column_array = np.array(column_results)
                    if window_function == "sum":
                        results.append(np.sum(column_array))
                    elif window_function == "min":
                        results.append(np.min(column_array))
                    elif window_function == "max":
                        results.append(np.max(column_array))
                    elif window_function in ["avg", "mean"]:
                        results.append(np.mean(column_array))
                    elif window_function == "count":
                        results.append(np.sum(~np.isnan(column_array)))
                    elif window_function == "std":
                        results.append(np.std(column_array))
                    elif window_function == "var":
                        results.append(np.var(column_array))
                    elif window_function == "median":
                        results.append(np.median(column_array))
                    elif window_function in ["first", "last"]:
                        results.append(np.mean(column_array))  # Use mean as aggregation for first/last
                else:
                    # Single column: use the result directly
                    results.append(column_results[0])

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
