"""Regression tests pinning defects found by two reviews of the time-window fix.

The window for a row at reference time ``t`` is the half-open interval
``(t - span, t]`` (right-closed, left-open) with ``span = window_size * time_unit``.
Second, minute and hour are elapsed time; day, week, month and year are calendar
units on the column's own wall clock. Elapsed-time expectations come from a
brute-force reference (``_reference``); calendar ones are pinned literals.

Defects pinned:
- Pandas mis-aligns time-sorted results back to original (unsorted) rows.
- PyArrow computes windows with wall-clock (not absolute) datetime arithmetic,
  so it is wrong across a DST transition.
- A null/NaT reference time must be an explicit ``ValueError`` on BOTH backends;
  PyArrow currently raises ``TypeError`` (``None - timedelta``).
"""

from __future__ import annotations

import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pytest

from mloda.user import Feature
from mloda.provider import FeatureSet
from mloda.provider import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindow
from mloda_plugins.feature_group.experimental.time_window.pyarrow import PyArrowTimeWindow


REFERENCE_TIME = DefaultOptionKeys.reference_time


def _reference(times: list[datetime.datetime], values: list[int], span: datetime.timedelta, fn: str) -> list[float]:
    """Brute-force window reference aligned to the INPUT row order.

    For each row ``i`` the window is every value ``j`` with
    ``times[i] - span < times[j] <= times[i]`` (right-closed, left-open).
    ``times`` must be absolute instants (compare/subtract in absolute time).
    """
    out: list[float] = []
    for i in range(len(times)):
        lower = times[i] - span
        window = [values[j] for j in range(len(times)) if lower < times[j] <= times[i]]
        if fn == "sum":
            out.append(float(sum(window)))
        elif fn == "avg":
            out.append(sum(window) / len(window))
        elif fn == "min":
            out.append(float(min(window)))
        elif fn == "max":
            out.append(float(max(window)))
        else:  # pragma: no cover - guard against typos in tests
            raise ValueError(f"Unsupported reference function: {fn}")
    return out


def _as_floats(result: Any) -> list[float]:
    """Normalize pandas (numpy) / pyarrow window results to a plain float list."""
    if hasattr(result, "to_pylist"):
        return [float(v) for v in result.to_pylist()]
    return [float(v) for v in list(result)]


# --------------------------------------------------------------------------- #
# 1. Unsorted-input alignment + cross-backend agreement (the headline bug).    #
# --------------------------------------------------------------------------- #

# Rows are deliberately NOT in ascending time order.
_UNSORTED_VALUES = [300, 1, 20]
_UNSORTED_TIMES = [
    pd.Timestamp("2023-01-01 00:02:00"),
    pd.Timestamp("2023-01-01 00:00:00"),
    pd.Timestamp("2023-01-01 01:00:00"),
]
_UNSORTED_SPAN = datetime.timedelta(hours=3)


def _unsorted_dataframe() -> pd.DataFrame:
    return pd.DataFrame({"value": list(_UNSORTED_VALUES), REFERENCE_TIME: list(_UNSORTED_TIMES)})


def _unsorted_table() -> pa.Table:
    return pa.Table.from_pandas(_unsorted_dataframe())


@pytest.mark.parametrize("window_function", ["sum", "avg"])
def test_unsorted_input_end_to_end_alignment(window_function: str) -> None:
    """End-to-end result must stay aligned to ORIGINAL row order on both backends.

    Pins the pandas defect: it assigns time-sorted window results back to the
    original (unsorted) rows, so the produced column is permuted. PyArrow is
    correct here, so pandas != pyarrow as well.
    """
    feature_name = f"value__{window_function}_3_hour_window"
    expected = _reference(
        [t.to_pydatetime() for t in _UNSORTED_TIMES], _UNSORTED_VALUES, _UNSORTED_SPAN, window_function
    )

    pandas_fs = FeatureSet()
    pandas_fs.add(Feature(feature_name))
    pandas_out = PandasTimeWindow.calculate(TimeWindowFeatureGroup, _unsorted_dataframe(), pandas_fs, None)
    pandas_result = _as_floats(pandas_out[feature_name].tolist())

    pyarrow_fs = FeatureSet()
    pyarrow_fs.add(Feature(feature_name))
    pyarrow_out = PyArrowTimeWindow.calculate(TimeWindowFeatureGroup, _unsorted_table(), pyarrow_fs, None)
    pyarrow_result = _as_floats(pyarrow_out.column(feature_name))

    assert pandas_result == pytest.approx(expected)
    assert pyarrow_result == pytest.approx(expected)
    assert pandas_result == pytest.approx(pyarrow_result)


# --------------------------------------------------------------------------- #
# 2. DST / timezone: windows must use ABSOLUTE time, not wall clock.           #
# --------------------------------------------------------------------------- #

# 2023-03-12 is the US spring-forward: 02:00 -> 03:00 in America/New_York.
# 01:45 EST and 03:15 EDT are only 30 minutes apart in ABSOLUTE time, so a
# 1-hour window on the second row must include the first.
_DST_TZ = ZoneInfo("America/New_York")
_DST_TIMES = [
    pd.Timestamp("2023-03-12 01:45:00", tz=_DST_TZ),
    pd.Timestamp("2023-03-12 03:15:00", tz=_DST_TZ),
]
_DST_VALUES = [1, 10]
_DST_SPAN = datetime.timedelta(hours=1)


def _dst_dataframe() -> pd.DataFrame:
    return pd.DataFrame({"value": list(_DST_VALUES), REFERENCE_TIME: list(_DST_TIMES)})


def _dst_table() -> pa.Table:
    return pa.Table.from_pandas(_dst_dataframe())


def test_dst_transition_uses_absolute_time() -> None:
    """Both backends must compute the window in absolute time across DST.

    Reference is computed from UTC instants. Pins the pyarrow defect: it does
    wall-clock arithmetic, drops the first row from the second window, and so
    disagrees with both the reference and pandas.
    """
    utc_times = [t.tz_convert("UTC").to_pydatetime() for t in _DST_TIMES]
    expected = _reference(utc_times, _DST_VALUES, _DST_SPAN, "sum")

    pandas_result = _as_floats(
        PandasTimeWindow._perform_window_operation(
            _dst_dataframe(), "sum", TimeWindowFeatureGroup._window_span(1, "hour"), ["value"], REFERENCE_TIME
        )
    )
    pyarrow_result = _as_floats(
        PyArrowTimeWindow._perform_window_operation(
            _dst_table(), "sum", TimeWindowFeatureGroup._window_span(1, "hour"), ["value"], REFERENCE_TIME
        )
    )

    assert pandas_result == pytest.approx(expected)
    assert pyarrow_result == pytest.approx(expected)
    assert pandas_result == pytest.approx(pyarrow_result)


_NY = "America/New_York"


def _zoned_frame(instants: list[str], zone: str, values: list[int]) -> pd.DataFrame:
    times = pd.to_datetime(instants, utc=True).tz_convert(zone)
    return pd.DataFrame({"value": values, REFERENCE_TIME: times})


@pytest.mark.parametrize(
    ("zone", "instants", "values", "size", "unit", "expected"),
    [
        (_NY, ["2023-03-11 16:30", "2023-03-12 16:00"], [1, 10], 1, "day", [1, 10]),
        (_NY, ["2023-03-11 16:30", "2023-03-12 16:00"], [1, 10], 24, "hour", [1, 11]),
        (_NY, ["2023-11-04 16:30", "2023-11-05 17:00"], [1, 10], 1, "day", [1, 11]),
        (_NY, ["2023-11-04 16:30", "2023-11-05 17:00"], [1, 10], 24, "hour", [1, 10]),
        (_NY, ["2023-11-05 05:40", "2023-11-05 06:40", "2023-11-06 06:30"], [1, 10, 100], 1, "day", [1, 11, 110]),
        (_NY, ["2023-03-12 07:15", "2023-03-12 07:45", "2023-03-13 06:30"], [1, 10, 100], 1, "day", [1, 11, 110]),
        (
            "+05:30",
            ["2023-01-31 06:30", "2023-02-28 12:30", "2023-03-01 06:30"],
            [1, 10, 100],
            1,
            "month",
            [1, 11, 110],
        ),
        ("+05:30", ["2023-01-31 06:30", "2023-02-28 12:30", "2023-03-01 06:30"], [1, 10, 100], 1, "day", [1, 10, 110]),
    ],
    ids=[
        "dst_short_day_calendar",
        "dst_short_day_elapsed",
        "dst_long_day_calendar",
        "dst_long_day_elapsed",
        "bound_in_repeated_hour",
        "bound_in_gap",
        "fixed_offset_month",
        "fixed_offset_day",
    ],
)
def test_zoned_window_follows_the_column_wall_clock(
    zone: str, instants: list[str], values: list[int], size: int, unit: str, expected: list[int]
) -> None:
    frame = _zoned_frame(instants, zone, values)
    span = TimeWindowFeatureGroup._window_span(size, unit)

    pandas_result = _as_floats(
        PandasTimeWindow._perform_window_operation(frame.copy(), "sum", span, ["value"], REFERENCE_TIME)
    )
    pyarrow_result = _as_floats(
        PyArrowTimeWindow._perform_window_operation(pa.Table.from_pandas(frame), "sum", span, ["value"], REFERENCE_TIME)
    )

    assert pandas_result == [float(v) for v in expected]
    assert pyarrow_result == [float(v) for v in expected]


@pytest.mark.parametrize(
    ("size", "unit", "times", "expected"),
    [
        (
            1,
            "month",
            ["2023-01-31 12:00", "2023-02-28 12:00", "2023-03-01 12:00", "2023-03-31 12:00"],
            [1, 11, 110, 1100],
        ),
        (
            1,
            "year",
            ["2023-02-28 12:00", "2024-02-28 18:00", "2024-02-29 12:00", "2025-02-28 12:00"],
            [1, 10, 110, 1110],
        ),
    ],
)
def test_month_and_year_window_values(size: int, unit: str, times: list[str], expected: list[int]) -> None:
    """Month and year windows are calendar months with the month-end clamp, not 30 and 365 days."""
    df = pd.DataFrame({"value": [1, 10, 100, 1000], REFERENCE_TIME: pd.to_datetime(times)})
    span = TimeWindowFeatureGroup._window_span(size, unit)

    pandas_result = _as_floats(
        PandasTimeWindow._perform_window_operation(df.copy(), "sum", span, ["value"], REFERENCE_TIME)
    )
    pyarrow_result = _as_floats(
        PyArrowTimeWindow._perform_window_operation(pa.Table.from_pandas(df), "sum", span, ["value"], REFERENCE_TIME)
    )

    assert pandas_result == [float(v) for v in expected]
    assert pyarrow_result == [float(v) for v in expected]


# --------------------------------------------------------------------------- #
# 4. PyArrow narrow window value (mirrors the pandas-only narrow test).        #
# --------------------------------------------------------------------------- #

_NARROW_DATES = pd.date_range(start="2023-01-01 00:00:00", periods=6, freq="min")
_NARROW_VALUES = [1, 2, 3, 4, 5, 6]


def test_pyarrow_narrow_window_value() -> None:
    """A 3-second window over minute-spaced data leaves each row alone: [1..6]."""
    expected = _reference(
        [d.to_pydatetime() for d in _NARROW_DATES], _NARROW_VALUES, datetime.timedelta(seconds=3), "sum"
    )
    table = pa.Table.from_pandas(pd.DataFrame({"value": list(_NARROW_VALUES), REFERENCE_TIME: _NARROW_DATES}))

    pyarrow_result = _as_floats(
        PyArrowTimeWindow._perform_window_operation(
            table, "sum", TimeWindowFeatureGroup._window_span(3, "second"), ["value"], REFERENCE_TIME
        )
    )
    assert pyarrow_result == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# 5. Null / NaT reference time is invalid: BOTH backends must raise ValueError. #
# --------------------------------------------------------------------------- #


def _null_pandas() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": [1, 2, 3],
            REFERENCE_TIME: [pd.Timestamp("2023-01-01 00:00:00"), pd.NaT, pd.Timestamp("2023-01-01 02:00:00")],
        }
    )


def _null_pyarrow() -> pa.Table:
    return pa.Table.from_pandas(_null_pandas())


@pytest.mark.parametrize("feature_name", ["value__sum_3_hour_window", "value__sum_1_month_window"])
@pytest.mark.parametrize(
    "cls,data_factory",
    [
        (PandasTimeWindow, _null_pandas),
        (PyArrowTimeWindow, _null_pyarrow),
    ],
)
def test_null_reference_time_raises_value_error(cls: Any, data_factory: Callable[[], Any], feature_name: str) -> None:
    """A null/NaT reference time must raise a clear ValueError on both backends.

    Pins the pyarrow defect: it raises TypeError (``None - timedelta``) instead
    of a consistent ValueError.
    """
    feature_set = FeatureSet()
    feature_set.add(Feature(feature_name))
    with pytest.raises(ValueError):
        cls.calculate(TimeWindowFeatureGroup, data_factory(), feature_set, None)
