"""Regression tests pinning defects found by two reviews of the time-window fix.

The window for a row at reference time ``t`` is the half-open interval
``(t - span, t]`` (right-closed, left-open) with ``span = window_size * time_unit``
and week/month/year approximated as 7/30/365 days. All expected values are derived
by an independent brute-force reference (``_reference``) so the tests are
self-checking and never trust hand-computed numbers.

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
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindowFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pyarrow import PyArrowTimeWindowFeatureGroup


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
    pandas_out = PandasTimeWindowFeatureGroup.calculate_feature(_unsorted_dataframe(), pandas_fs)
    pandas_result = _as_floats(pandas_out[feature_name].tolist())

    pyarrow_fs = FeatureSet()
    pyarrow_fs.add(Feature(feature_name))
    pyarrow_out = PyArrowTimeWindowFeatureGroup.calculate_feature(_unsorted_table(), pyarrow_fs)
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
        PandasTimeWindowFeatureGroup._perform_window_operation(
            _dst_dataframe(), "sum", 1, "hour", ["value"], REFERENCE_TIME
        )
    )
    pyarrow_result = _as_floats(
        PyArrowTimeWindowFeatureGroup._perform_window_operation(
            _dst_table(), "sum", 1, "hour", ["value"], REFERENCE_TIME
        )
    )

    assert pandas_result == pytest.approx(expected)
    assert pyarrow_result == pytest.approx(expected)
    assert pandas_result == pytest.approx(pyarrow_result)


# --------------------------------------------------------------------------- #
# 3. month / year: assert exact values (span = 90 / 1095 days), not length.    #
# --------------------------------------------------------------------------- #

_MY_DATES = pd.date_range(start="2023-01-01", periods=6, freq="D")
_MY_VALUES = [1, 2, 3, 4, 5, 6]
_MY_SPAN_DAYS = {"month": 90, "year": 1095}


@pytest.mark.parametrize("time_unit", ["month", "year"])
def test_month_and_year_window_values(time_unit: str) -> None:
    """month/year windows must return the exact brute-force values on both backends."""
    span = datetime.timedelta(days=_MY_SPAN_DAYS[time_unit])
    expected = _reference([d.to_pydatetime() for d in _MY_DATES], _MY_VALUES, span, "sum")

    df = pd.DataFrame({"value": list(_MY_VALUES), REFERENCE_TIME: _MY_DATES})
    table = pa.Table.from_pandas(df)

    pandas_result = _as_floats(
        PandasTimeWindowFeatureGroup._perform_window_operation(
            df.copy(), "sum", 3, time_unit, ["value"], REFERENCE_TIME
        )
    )
    pyarrow_result = _as_floats(
        PyArrowTimeWindowFeatureGroup._perform_window_operation(table, "sum", 3, time_unit, ["value"], REFERENCE_TIME)
    )

    assert pandas_result == pytest.approx(expected)
    assert pyarrow_result == pytest.approx(expected)
    assert pandas_result == pytest.approx(pyarrow_result)


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
        PyArrowTimeWindowFeatureGroup._perform_window_operation(table, "sum", 3, "second", ["value"], REFERENCE_TIME)
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


@pytest.mark.parametrize(
    "cls,data_factory",
    [
        (PandasTimeWindowFeatureGroup, _null_pandas),
        (PyArrowTimeWindowFeatureGroup, _null_pyarrow),
    ],
)
def test_null_reference_time_raises_value_error(
    cls: type[TimeWindowFeatureGroup], data_factory: Callable[[], Any]
) -> None:
    """A null/NaT reference time must raise a clear ValueError on both backends.

    Pins the pyarrow defect: it raises TypeError (``None - timedelta``) instead
    of a consistent ValueError.
    """
    feature_set = FeatureSet()
    feature_set.add(Feature("value__sum_3_hour_window"))
    with pytest.raises(ValueError):
        cls.calculate_feature(data_factory(), feature_set)
