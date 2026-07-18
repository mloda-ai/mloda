"""Regression tests: time-window feature groups must respect ``time_unit``.

The window for a row at reference time ``t`` is the half-open interval
``(t - span, t]`` (right-closed, left-open) where ``span = window_size * time_unit``.
The buggy implementations ignore ``time_unit`` and compute a fixed ROW-COUNT
window instead, so ``sum_3_hour_window`` and ``sum_3_second_window`` produce the
same "last 3 rows" result on identical data. These tests assert the true
time-based values and therefore fail on the current code.

All input data is time-sorted ascending; sub-daily (minute) spacing makes the
time unit observable.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

from mloda.provider import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindowFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.pyarrow import PyArrowTimeWindowFeatureGroup


REFERENCE_TIME = DefaultOptionKeys.reference_time

# Minute-spaced data: 6 rows at 00:00..00:05, values 1..6.
_MINUTE_TIMES = pd.date_range(start="2023-01-01 00:00:00", periods=6, freq="min")
_MINUTE_VALUES = [1, 2, 3, 4, 5, 6]

# Ground-truth time-based results (verified with pandas rolling, closed="right").
# span=3h is wider than the 5-minute range -> cumulative over all prior rows.
EXPECTED_SUM_3_HOUR = [1.0, 3.0, 6.0, 10.0, 15.0, 21.0]
EXPECTED_AVG_3_HOUR = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
# span=3s is narrower than the 1-minute spacing -> each row's window is only itself.
EXPECTED_SUM_3_SECOND = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def _minute_dataframe() -> pd.DataFrame:
    return pd.DataFrame({"value": list(_MINUTE_VALUES), REFERENCE_TIME: _MINUTE_TIMES})


def _minute_table() -> pa.Table:
    return pa.Table.from_pandas(_minute_dataframe())


def _as_floats(result: Any) -> list[float]:
    """Normalize pandas (numpy) / pyarrow window results to a plain float list."""
    if hasattr(result, "to_pylist"):
        return [float(v) for v in result.to_pylist()]
    return [float(v) for v in list(result)]


def test_pandas_sum_hour_window_is_time_based_not_row_count() -> None:
    """sum_3_hour_window must sum all rows within 3h, not just the last 3 rows."""
    result = PandasTimeWindowFeatureGroup._perform_window_operation(
        _minute_dataframe(), "sum", 3, "hour", ["value"], REFERENCE_TIME
    )
    assert _as_floats(result) == pytest.approx(EXPECTED_SUM_3_HOUR)


def test_pandas_time_unit_changes_result() -> None:
    """3_second and 3_hour must differ on identical data (time_unit matters)."""
    hour = PandasTimeWindowFeatureGroup._perform_window_operation(
        _minute_dataframe(), "sum", 3, "hour", ["value"], REFERENCE_TIME
    )
    second = PandasTimeWindowFeatureGroup._perform_window_operation(
        _minute_dataframe(), "sum", 3, "second", ["value"], REFERENCE_TIME
    )
    assert _as_floats(hour) == pytest.approx(EXPECTED_SUM_3_HOUR)
    assert _as_floats(second) == pytest.approx(EXPECTED_SUM_3_SECOND)
    assert _as_floats(hour) != _as_floats(second)


def test_pyarrow_sum_hour_window_is_time_based_not_row_count() -> None:
    """PyArrow must match the pandas time-based result for sum_3_hour_window."""
    result = PyArrowTimeWindowFeatureGroup._perform_window_operation(
        _minute_table(), "sum", 3, "hour", ["value"], REFERENCE_TIME
    )
    assert _as_floats(result) == pytest.approx(EXPECTED_SUM_3_HOUR)


@pytest.mark.parametrize(
    "window_function,expected",
    [
        ("sum", EXPECTED_SUM_3_HOUR),
        ("avg", EXPECTED_AVG_3_HOUR),
    ],
)
def test_pandas_and_pyarrow_agree_time_based(window_function: str, expected: list[float]) -> None:
    """Both backends must produce the same time-based result on sub-daily data."""
    pandas_result = _as_floats(
        PandasTimeWindowFeatureGroup._perform_window_operation(
            _minute_dataframe(), window_function, 3, "hour", ["value"], REFERENCE_TIME
        )
    )
    pyarrow_result = _as_floats(
        PyArrowTimeWindowFeatureGroup._perform_window_operation(
            _minute_table(), window_function, 3, "hour", ["value"], REFERENCE_TIME
        )
    )
    assert pandas_result == pytest.approx(expected)
    assert pyarrow_result == pytest.approx(expected)
    assert pandas_result == pytest.approx(pyarrow_result)


@pytest.mark.parametrize("time_unit", ["month", "year"])
def test_month_and_year_units_run_to_completion(time_unit: str) -> None:
    """month/year must run without raising and return one value per row."""
    dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
    values = [1, 2, 3, 4, 5]
    df = pd.DataFrame({"value": values, REFERENCE_TIME: dates})
    table = pa.Table.from_pandas(df)

    pandas_result = PandasTimeWindowFeatureGroup._perform_window_operation(
        df.copy(), "sum", 3, time_unit, ["value"], REFERENCE_TIME
    )
    pyarrow_result = PyArrowTimeWindowFeatureGroup._perform_window_operation(
        table, "sum", 3, time_unit, ["value"], REFERENCE_TIME
    )

    assert len(_as_floats(pandas_result)) == len(values)
    assert len(_as_floats(pyarrow_result)) == len(values)
