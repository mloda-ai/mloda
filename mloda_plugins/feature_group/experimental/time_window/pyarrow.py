"""PyArrow time window."""

from __future__ import annotations

import bisect
import datetime
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from mloda.core.optional_dependency import require
from mloda.provider import FeatureSet, Implementation

from mloda_plugins.compute_framework.base_implementations.pyarrow import pyarrow_type_semantics
from mloda.user.pyarrow import PyArrowTable
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup, raise_null_reference_time
from mloda_plugins.feature_group.experimental.nan_safe_reduction import sample_variance_skipping_nan

_NUMPY_REASON = "multi-column time-window aggregation"


class PyArrowTimeWindow(Implementation):
    feature_group = TimeWindowFeatureGroup
    compute_framework = PyArrowTable

    @classmethod
    def calculate(cls, group: type[TimeWindowFeatureGroup], data: Any, features: FeatureSet) -> Any:
        return group.window(
            data,
            features,
            perform=cls._perform_window_operation,
            add_column=cls._add_result_to_data,
            columns=cls.compute_framework.extract_column_names,
            column_semantics=pyarrow_type_semantics.column_semantics,
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
        span: datetime.timedelta,
        in_features: list[str],
        time_filter_feature: str,
    ) -> pa.Array:
        """
        Perform the time window operation using PyArrow compute functions.

        Supports both single-column and multi-column window operations:
        - Single column: aggregates values within the column over time
        - Multi-column: aggregates across columns for each time window row

        Args:
            data: The PyArrow Table
            window_function: The type of window function to perform
            span: The window span
            in_features: List of source feature names (may be single or multiple columns)
            time_filter_feature: The name of the reference time column

        Returns:
            The result of the window operation as a PyArrow Array
        """
        # Get the time column
        time_column = data.column(time_filter_feature)

        # A null/NaT reference time has no position on the timeline; fail explicitly.
        if time_column.null_count > 0:
            raise_null_reference_time(time_filter_feature)

        # Get the source columns
        source_columns = [data.column(name) for name in in_features]

        # Sort the data by time
        # First create indices sorted by time
        sorted_indices = pc.sort_indices(time_column)

        # Get the sorted source values for each column
        sorted_sources = [pc.take(col, sorted_indices) for col in source_columns]

        # Time-based (not row-count) window: include rows in (t - span, t].
        # Bound each window by ABSOLUTE integer nanoseconds (not Python datetimes):
        # wall-clock datetime subtraction is wrong across a DST transition, whereas
        # int64 ns since epoch is DST-safe, unit-uniform, and works for tz-aware
        # timestamps and date32/date64 too.
        sorted_times_ns = pc.take(time_column, sorted_indices).cast(pa.timestamp("ns")).cast(pa.int64()).to_pylist()
        span_ns = span.days * 86_400_000_000_000 + span.seconds * 1_000_000_000 + span.microseconds * 1_000

        # Create a list to store the results
        results = []

        numpy = require("numpy", _NUMPY_REASON) if len(in_features) > 1 else None

        # For each row, calculate the window operation over the time-based window
        for i in range(len(sorted_sources[0])):
            # Rows are time-sorted ascending, so the window is a contiguous slice
            # ending at i. bisect_right of (t_i - span) gives the first index strictly
            # greater than the lower bound, implementing the left-open boundary.
            start_idx = bisect.bisect_right(sorted_times_ns, sorted_times_ns[i] - span_ns)
            length = i - start_idx + 1

            # Zero-copy slice of each sorted source column for the current window.
            all_window_values = [col.slice(start_idx, length) for col in sorted_sources]

            # Apply the window function
            if len(all_window_values[0]) == 0:
                # If no values in window, use the current value
                results.append(sorted_sources[0][i].as_py())
            else:
                # For multi-column, first compute rolling window per column, then aggregate across columns
                column_results = [
                    cls._window_scalar(window_values, window_function).as_py() for window_values in all_window_values
                ]

                # If multi-column, aggregate across columns
                if len(in_features) > 1:
                    assert numpy is not None
                    np = numpy

                    if any(v is None for v in column_results):
                        column_array = np.array([np.nan if v is None else v for v in column_results], dtype=float)
                    else:
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
                    elif window_function in ["std", "var"]:
                        variance = float(sample_variance_skipping_nan(column_array.astype(float)[np.newaxis, :])[0])
                        results.append(variance if window_function == "var" else np.sqrt(variance))
                    elif window_function == "median":
                        results.append(np.median(column_array))
                    elif window_function in ["first", "last"]:
                        results.append(np.mean(column_array))  # Use mean as aggregation for first/last
                else:
                    # Single column: use the result directly
                    results.append(column_results[0])

        # Scatter results back to original row order via the inverse permutation (O(n)).
        sorted_idx_list = sorted_indices.to_pylist()
        reordered_results: list[Any] = [None] * len(results)
        for k, orig_pos in enumerate(sorted_idx_list):
            reordered_results[orig_pos] = results[k]

        if not reordered_results:
            # pa.array([]) infers the null type; take the type the window function yields on the empty source.
            source = sorted_sources[0]
            if window_function in ["first", "last"]:
                return pa.array([], type=source.type)
            return pa.array([], type=cls._window_scalar(source, window_function).type)

        # Convert the results to a PyArrow array
        return pa.array(reordered_results)

    @classmethod
    def _window_scalar(cls, window_values: pa.ChunkedArray, window_function: str) -> pa.Scalar:
        """Apply the window function to one window of a single column, keeping the pyarrow result type."""
        if window_function == "sum":
            return pc.sum(window_values)
        elif window_function == "min":
            return pc.min(window_values)
        elif window_function == "max":
            return pc.max(window_values)
        elif window_function in ["avg", "mean"]:
            return pc.mean(window_values)
        elif window_function == "count":
            return pc.count(window_values)
        elif window_function == "std":
            return pc.stddev(window_values, ddof=1)
        elif window_function == "var":
            return pc.variance(window_values, ddof=1)
        elif window_function == "median":
            # PyArrow doesn't have a direct median function
            # We can approximate it using quantile with q=0.5
            return pc.quantile(window_values, q=0.5)[0]
        elif window_function == "first":
            return window_values[0]
        elif window_function == "last":
            return window_values[-1]
        else:
            raise ValueError(f"Unsupported window function: {window_function}")
