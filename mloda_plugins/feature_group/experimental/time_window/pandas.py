"""Pandas time window."""

from __future__ import annotations

import datetime
import zoneinfo
from datetime import tzinfo
from typing import Any

import numpy as np
from pandas.api.indexers import BaseIndexer

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pandas import pandas_type_semantics
from mloda.user.pandas import PandasDataFrame
from mloda_plugins.feature_group.experimental.time_window.base import (
    TimeWindowFeatureGroup,
    WindowSpan,
    raise_null_reference_time,
)


try:
    import pandas as pd
except ImportError:
    pd = None


_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)
_DAY_NS = 86_400_000_000_000
_EDGE_NS = 3 * _DAY_NS  # margin for the +-1 day probes plus slack near the int64 limits
_FLOOR_DAYS = _INT64_MIN // _DAY_NS + 3  # earliest day count whose ns value cannot overflow, with slack
_NS_RANGE_DAYS = 213_504  # days spanned by the int64 ns range (about 584.5 years)
_PANDAS_TZ_RULES_END_NS = 4_102_444_800_000_000_000  # pandas tz_convert follows a zone's DST rules only before 2100
_MAX_DAYS_PER_UNIT = {"day": 1, "week": 7, "month": 31, "year": 366}


def _wall_ns(utc_ns: np.ndarray, zone: tzinfo) -> np.ndarray:
    """Wall clock reading of each instant in zone, as ns."""
    wall: np.ndarray = (
        pd.DatetimeIndex(utc_ns.view("M8[ns]")).tz_localize("UTC").tz_convert(zone).tz_localize(None).asi8
    )
    return wall


class _SpanIndexer(BaseIndexer):  # type: ignore[misc]
    """Rolling window bounds given explicitly per row."""

    def __init__(self, starts: np.ndarray, ends: np.ndarray) -> None:
        super().__init__()
        self._starts = starts
        self._ends = ends

    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._starts.astype(np.int64), self._ends.astype(np.int64)


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
    def _roll_column(cls, col: pd.Series, window_function: str, starts: np.ndarray) -> pd.Series:
        """Rolling aggregate of one column; unlike pandas rolling, +-inf stay values (only NaN is missing)."""
        values = col.to_numpy(dtype="float64", na_value=np.nan)
        if window_function in ("first", "last"):
            picked = values[starts] if window_function == "first" else values
            return pd.Series(picked, index=col.index)

        name = {"avg": "mean"}.get(window_function, window_function)
        min_periods = 0 if window_function == "sum" else 1
        if window_function == "median":
            finite = values[np.isfinite(values)]
            bound = float(np.abs(finite).max()) if len(finite) else 0.0
            capped = np.where(np.isposinf(values), bound, np.where(np.isneginf(values), -bound, values))
            source = pd.Series(capped, index=col.index)
        else:
            source = col
        result = pd.Series(cls._rolled(source, name, starts, min_periods), index=col.index)
        if window_function == "count":
            return result

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
        if window_function == "median":
            q, f, p = neg, window_count(np.isfinite(values)), pos
            total = q + f + p
            lo, hi = (total - 1) // 2, total // 2
            any_neg = lo < q
            any_pos = hi >= q + f
            mixed = np.where(any_neg & any_pos, np.nan, np.where(any_neg, -np.inf, np.where(any_pos, np.inf, fixed)))
            fixed = np.where(affected, mixed, fixed)
        elif window_function in ("sum", "avg", "mean"):
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
        return pd.Series(fixed, index=col.index)

    @classmethod
    def _rolled(cls, col: pd.Series, name: str, starts: np.ndarray, min_periods: int) -> np.ndarray:
        """Rolling aggregate for arbitrary starts, in layers of monotone starts (one pass per nesting level)."""
        n = len(starts)
        ends = np.arange(1, n + 1)
        # pandas rolls incrementally only on non-decreasing starts; floor[j] is at most every later start
        floor = np.minimum.accumulate(starts[::-1])[::-1]
        out = np.full(n, np.nan)
        pending = np.ones(n, dtype=bool)
        while pending.any():
            monotone = np.maximum.accumulate(np.where(pending, starts, floor))
            done = pending & (monotone == starts)
            rolled = getattr(col.rolling(window=_SpanIndexer(monotone, ends), min_periods=min_periods), name)()
            out[done] = rolled.to_numpy(dtype="float64", na_value=np.nan)[done]
            pending &= ~done
        return out

    @classmethod
    def _lower_bounds(cls, ns: np.ndarray, span: WindowSpan, zone: tzinfo | None) -> np.ndarray:
        """Window start per sorted instant; calendar spans are vectorized (mirrors TimeRule._resolve)."""
        width = span.fixed_ns(zone)
        if width is not None:
            if width > _INT64_MAX:
                return np.full(len(ns), _INT64_MIN, dtype=np.int64)
            return np.where(ns < _INT64_MIN + width, _INT64_MIN, ns - width)

        def per_row(i: int) -> int:
            return max(span.lower_ns(int(ns[i]), zone), _INT64_MIN)

        vectorizable = zone is None or isinstance(zone, (datetime.timezone, zoneinfo.ZoneInfo))
        if not vectorizable or span.size * _MAX_DAYS_PER_UNIT[span.unit] > _NS_RANGE_DAYS:
            return np.fromiter((per_row(i) for i in range(len(ns))), dtype=np.int64, count=len(ns))

        rule = span.rule
        slow = (ns < _INT64_MIN + _EDGE_NS) | (ns > _INT64_MAX - _EDGE_NS)
        if isinstance(zone, zoneinfo.ZoneInfo):
            slow = slow | (ns >= _PANDAS_TZ_RULES_END_NS - _EDGE_NS)
        safe = np.where(slow, 0, ns)
        wall = safe if zone is None else _wall_ns(safe, zone)
        days, time_of_day = np.divmod(wall, _DAY_NS)
        if span.unit in ("day", "week"):
            new_days = days - span.size * (7 if span.unit == "week" else 1)
        else:
            day = days.astype("M8[D]")
            month = day.astype("M8[M]")
            day_of_month = (day - month.astype("M8[D]")).astype(np.int64)
            target = month - np.timedelta64(span.size * (12 if span.unit == "year" else 1), "M")
            first = target.astype("M8[D]")
            length = ((target + 1).astype("M8[D]") - first).astype(np.int64)
            new_days = first.astype(np.int64) + np.minimum(day_of_month, length - 1)
            if rule.month_end == "raise":
                slow = slow | (day_of_month >= length)
        slow = slow | (new_days < _FLOOR_DAYS)
        guess = np.maximum(new_days, _FLOOR_DAYS) * _DAY_NS + time_of_day
        if zone is None:
            lower = guess
        else:
            offset_before = _wall_ns(guess - _DAY_NS, zone) - (guess - _DAY_NS)
            offset_after = _wall_ns(guess + _DAY_NS, zone) - (guess + _DAY_NS)
            before = guess - offset_before
            after = guess - offset_after
            before_valid = _wall_ns(before, zone) - before == offset_before
            after_valid = _wall_ns(after, zone) - after == offset_after
            gap = ~before_valid & ~after_valid
            repeated = before_valid & after_valid & (offset_before != offset_after)
            prefer_after = (rule.repeated == "later") | ((rule.repeated == "source") & (wall - safe == offset_after))
            lower = np.where(np.where(repeated, prefer_after, after_valid & ~before_valid), after, before)
            if rule.gap == "raise":
                slow = slow | gap
            if rule.repeated == "raise":
                slow = slow | repeated
        lower = np.asarray(lower, dtype=np.int64)
        for i in np.flatnonzero(slow):
            lower[i] = per_row(int(i))
        return lower

    @classmethod
    def _perform_window_operation(
        cls,
        data: pd.DataFrame,
        window_function: str,
        span: WindowSpan,
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
        # row order. asi8 holds UTC instants (tz-aware) or wall clock (naive).
        instants = pd.DatetimeIndex(times).as_unit("ns").asi8
        order = np.argsort(instants, kind="stable")
        ns = instants[order]
        sorted_index = pd.DatetimeIndex(ns.view("M8[ns]"))

        # Select the columns to perform window operation on, in time-sorted order.
        if len(in_features) == 1:
            # Single column: extract as Series for simpler window operation
            selected_data = pd.Series(data[in_features[0]].to_numpy()[order], index=sorted_index)
        else:
            # Multiple columns: keep as DataFrame
            selected_data = pd.DataFrame({c: data[c].to_numpy()[order] for c in in_features}, index=sorted_index)

        # Time-based (not row-count) window: include rows in (lower, t], lower being t minus the span.
        if window_function not in TimeWindowFeatureGroup.WINDOW_FUNCTIONS:
            raise ValueError(f"Unsupported window function: {window_function}")

        # Window start per sorted row; the window ends at the current row (duplicates included).
        zone = times.dt.tz
        lower = cls._lower_bounds(ns, span, zone)
        starts = np.searchsorted(ns, lower, side="right")

        if isinstance(selected_data, pd.Series):
            result = cls._roll_column(selected_data, window_function, starts)
        else:
            result = pd.DataFrame(
                {c: cls._roll_column(selected_data[c], window_function, starts) for c in selected_data}
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
