"""Pandas time window."""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pandas import pandas_type_semantics
from mloda.user.pandas import PandasDataFrame
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup, raise_null_reference_time


try:
    import pandas as pd
except ImportError:
    pd = None


class PandasTimeWindow(Implementation[TimeWindowFeatureGroup]):
    feature_group = TimeWindowFeatureGroup
    compute_framework = PandasDataFrame

    @classmethod
    def calculate(
        cls, group: type[TimeWindowFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.window(
            data,
            features,
            perform=cls._perform_window_operation,
            add_column=cls._add_result_to_data,
            columns=cls.compute_framework.extract_column_names,
            column_semantics=pandas_type_semantics.column_semantics,
        )

    @classmethod
    def _add_result_to_data(cls, data: pd.DataFrame, feature_name: str, result: Any) -> pd.DataFrame:
        """Add the result to the DataFrame."""
        data[feature_name] = result
        return data

    @classmethod
    def _roll_column(cls, col: pd.Series, window_function: str, span: pd.Timedelta, starts: np.ndarray) -> pd.Series:
        """Rolling aggregate of one column; unlike pandas rolling, +-inf stay values (only NaN is missing)."""
        values = col.to_numpy(dtype="float64", na_value=np.nan)
        if window_function in ("first", "last"):
            picked = values[starts] if window_function == "first" else values
            return pd.Series(picked, index=col.index)

        rolling = col.rolling(window=span, min_periods=0 if window_function == "sum" else 1, closed="right")
        if window_function == "count":
            return rolling.count()
        result = getattr(rolling, {"avg": "mean"}.get(window_function, window_function))()

        def window_count(flags: np.ndarray) -> np.ndarray:
            cum = np.concatenate(([0], np.cumsum(flags)))
            counts: np.ndarray = cum[1:] - cum[starts]
            return counts

        pos = window_count(values == np.inf)
        neg = window_count(values == -np.inf)
        affected = (pos + neg) > 0
        if not affected.any():
            return result

        fixed = result.to_numpy(dtype="float64", na_value=np.nan, copy=True)
        has_pos = pos > 0
        has_neg = neg > 0
        if window_function in ("sum", "avg", "mean"):
            inf_result = np.where(has_pos & ~has_neg, np.inf, np.where(has_neg & ~has_pos, -np.inf, np.nan))
            fixed = np.where(affected, inf_result, fixed)
        elif window_function == "min":
            fixed = np.where(has_pos & np.isnan(fixed), np.inf, fixed)
            fixed = np.where(has_neg, -np.inf, fixed)
        elif window_function == "max":
            fixed = np.where(has_neg & np.isnan(fixed), -np.inf, fixed)
            fixed = np.where(has_pos, np.inf, fixed)
        elif window_function in ("std", "var"):
            n = window_count(~np.isnan(values))
            fixed = np.where(affected & (n >= 2), np.nan, fixed)
        else:
            # median recomputes inf windows per row
            with np.errstate(invalid="ignore"):
                for i in np.flatnonzero(affected):
                    window = values[starts[i] : i + 1]
                    fixed[i] = np.median(window[~np.isnan(window)])
        return pd.Series(fixed, index=col.index)

    @classmethod
    def _perform_window_operation(
        cls,
        data: pd.DataFrame,
        window_function: str,
        span: datetime.timedelta,
        in_features: list[str],
        time_filter_feature: str,
    ) -> Any:
        """
        Perform the time window operation using Pandas rolling window functions.

        Supports both single-column and multi-column window operations:
        - Single column: aggregates values within the column over time
        - Multi-column: aggregates across columns for each time window row

        Args:
            data: The Pandas DataFrame
            window_function: The type of window function to perform
            span: The window span
            in_features: List of source feature names (may be single or multiple columns)
            time_filter_feature: The name of the reference time column

        Returns:
            The result of the window operation
        """
        # Coerce the reference column to a proper datetime index (handles arrow
        # date32/date64 columns and keeps tz-awareness). A null/NaT reference time
        # has no position on the timeline, so fail explicitly.
        times = pd.to_datetime(data[time_filter_feature])
        if times.isna().any():
            raise_null_reference_time(time_filter_feature)

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
        span_delta = pd.Timedelta(span)
        if window_function not in TimeWindowFeatureGroup.WINDOW_FUNCTIONS:
            raise ValueError(f"Unsupported window function: {window_function}")

        # Window start per sorted row; the window ends at the current row (duplicates included).
        ns = sorted_index.as_unit("ns").asi8
        starts = np.searchsorted(ns, ns - span_delta.value, side="right")

        if isinstance(selected_data, pd.Series):
            result = cls._roll_column(selected_data, window_function, span_delta, starts)
        else:
            result = pd.DataFrame(
                {c: cls._roll_column(selected_data[c], window_function, span_delta, starts) for c in selected_data}
            )

        # For multi-column, aggregate across columns (axis=1) after rolling window
        if len(in_features) > 1:
            with np.errstate(invalid="ignore"):
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
