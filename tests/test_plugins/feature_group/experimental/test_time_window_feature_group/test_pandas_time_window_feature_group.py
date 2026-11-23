"""
Tests for the PandasTimeWindow implementation.
"""

from collections.abc import Callable
from datetime import UTC, timedelta, timezone, tzinfo
from typing import Any
from zoneinfo import ZoneInfo

from dateutil import tz
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.contract.time_rule import TimeRule
from mloda_plugins.feature_group.experimental.time_window.base import WindowSpan

from tests.test_plugins.feature_group.experimental.core_input_check import assert_core_rejects_missing_input
from mloda.provider import FeatureSet
from mloda._testing.framework_adapter import PandasAdapter
from mloda._testing.compute_framework.time_window_suite import TimeWindowRuleSuite
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda.provider import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.time_window.pandas import PandasTimeWindow
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup
from tests.test_plugins.feature_group.experimental.test_time_window_feature_group.conftest import EXPECTED_VALUES
from tests.test_plugins.feature_group.experimental.test_time_window_feature_group.time_window_zero_row_test_mixin import (
    TimeWindowZeroRowTestMixin,
)
from tests.test_plugins.feature_group.experimental.zero_row_result_type_test_mixin import PandasZeroRowAdapter


class TestPandasTimeWindow:
    """Tests for the PandasTimeWindow implementation."""

    def test_compute_framework_rule(self) -> None:
        """Test compute_framework_rule method."""
        assert PandasTimeWindow.compute_framework is PandasDataFrame

    @pytest.mark.parametrize(
        "window_function,expected_values",
        [
            ("avg", EXPECTED_VALUES["avg"]),
            ("sum", EXPECTED_VALUES["sum"]),
            ("min", EXPECTED_VALUES["min"]),
            ("max", EXPECTED_VALUES["max"]),
        ],
    )
    def test_perform_window_operation(
        self, sample_time_dataframe: pd.DataFrame, window_function: str, expected_values: Any
    ) -> None:
        """Test _perform_window_operation method with different window functions."""
        result = PandasTimeWindow._perform_window_operation(
            sample_time_dataframe,
            window_function,
            TimeWindowFeatureGroup._window_span(3, "day"),
            ["temperature"],
            DefaultOptionKeys.reference_time,
        )

        # Check the first few values
        for i in range(3):
            if isinstance(expected_values[i], float):
                assert abs(result[i] - expected_values[i]) < 0.1
            else:
                assert result[i] == expected_values[i]

    def test_perform_window_operation_invalid(self, sample_time_dataframe: pd.DataFrame) -> None:
        """Test _perform_window_operation method with invalid window function."""
        with pytest.raises(ValueError):
            PandasTimeWindow._perform_window_operation(
                sample_time_dataframe,
                "invalid",
                TimeWindowFeatureGroup._window_span(3, "day"),
                ["temperature"],
                DefaultOptionKeys.reference_time,
            )

    def test_calculate_feature_single(
        self, sample_time_dataframe: pd.DataFrame, feature_set_avg_window: FeatureSet
    ) -> None:
        """Test calculate_feature method with a single time window feature."""
        result = PandasTimeWindow.calculate(TimeWindowFeatureGroup, sample_time_dataframe, feature_set_avg_window, None)

        # Check that the result contains the original data plus the time window feature
        assert "temperature__avg_3_day_window" in result.columns

        # Check the values of the time window feature
        # First value should be the temperature itself (20)
        # Second value should be average of first two days (20+22)/2 = 21
        # Third value should be average of first three days (20+22+19)/3 = 20.33
        assert result["temperature__avg_3_day_window"].iloc[0] == 20
        assert abs(result["temperature__avg_3_day_window"].iloc[1] - 21) < 0.1
        assert abs(result["temperature__avg_3_day_window"].iloc[2] - 20.33) < 0.1

        # Check that the original data is preserved
        assert "temperature" in result.columns
        assert "humidity" in result.columns
        assert "pressure" in result.columns
        assert "wind_speed" in result.columns

    def test_calculate_feature_multiple(
        self, sample_time_dataframe: pd.DataFrame, feature_set_multiple_windows: FeatureSet
    ) -> None:
        """Test calculate_feature method with multiple time window features."""
        result = PandasTimeWindow.calculate(
            TimeWindowFeatureGroup, sample_time_dataframe, feature_set_multiple_windows, None
        )

        # Check that the result contains all time window features
        assert "temperature__avg_3_day_window" in result.columns
        assert "humidity__max_5_day_window" in result.columns
        assert "pressure__min_2_day_window" in result.columns
        assert "wind_speed__sum_4_day_window" in result.columns

        # Check that the original data is preserved
        assert "temperature" in result.columns
        assert "humidity" in result.columns
        assert "pressure" in result.columns
        assert "wind_speed" in result.columns

    def test_calculate_feature_missing_source(self, sample_time_dataframe: pd.DataFrame) -> None:
        """Test calculate_feature method with missing source feature."""
        error = assert_core_rejects_missing_input(
            PandasTimeWindow, sample_time_dataframe, "missing__avg_3_day_window", ("missing",)
        )
        assert "temperature" in error.available_columns

    def test_calculate_feature_missing_time_filter(self) -> None:
        """Test calculate_feature method with DataFrame that has no time filter feature."""
        # Create a DataFrame without the time filter feature
        df = pd.DataFrame(
            {
                "temperature": [20, 22, 19, 23, 25],
                "humidity": [65, 70, 75, 60, 55],
            }
        )

        assert_core_rejects_missing_input(
            PandasTimeWindow,
            df,
            "temperature__avg_3_day_window",
            (DefaultOptionKeys.reference_time.value,),
            ("humidity", "temperature"),
        )


class TestPandasTimeWindowZeroRow(PandasZeroRowAdapter, TimeWindowZeroRowTestMixin):
    feature_group_class = PandasTimeWindow


class TestPandasTimeWindowRule(TimeWindowRuleSuite):
    adapter_class = PandasAdapter
    implementation = PandasTimeWindow


_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)
_DAY_NS = 86_400 * 10**9
_UNITS = ("day", "week", "month", "year")

_ZONES: dict[str, Callable[[], tzinfo | None]] = {
    "naive": lambda: None,
    "new_york": lambda: ZoneInfo("America/New_York"),
    "lord_howe": lambda: ZoneInfo("Australia/Lord_Howe"),
    "kolkata": lambda: ZoneInfo("Asia/Kolkata"),
    "berlin": lambda: ZoneInfo("Europe/Berlin"),
    "fixed_offset": lambda: timezone(timedelta(hours=5, minutes=30)),
    "utc": lambda: UTC,
    "dateutil_new_york": lambda: tz.gettz("America/New_York"),
}

_RULES: dict[str, TimeRule] = {
    "default": TimeRule(),
    "fixed_length": TimeRule(calendar_units="fixed_length"),
    "gap_raise": TimeRule(gap="raise"),
    "repeated_earlier": TimeRule(repeated="earlier"),
    "repeated_later": TimeRule(repeated="later"),
    "repeated_raise": TimeRule(repeated="raise"),
    "month_end_raise": TimeRule(month_end="raise"),
}

# (first instant, ns remainder) of short grids across DST changes, month ends and the 2024 leap day
_GRID_STARTS: tuple[tuple[str, int], ...] = (
    ("2024-03-09 12:00", 0),  # New York spring change on 03-10
    ("2024-11-02 12:00", 0),  # New York fall change on 11-03
    ("2024-04-05 12:00", 0),  # Lord Howe 30 minute change on 04-07
    ("2024-02-27 03:00", 0),  # leap day
    ("2023-01-30 03:00", 0),  # month ends
    ("2024-05-30 03:00", 0),  # month ends, long month
    ("2024-11-02 12:00", 999),  # sub-microsecond remainder
    ("2024-04-09 12:00", 0),  # month back from 04-10 lands in the 2024-03-10 gap
    ("2024-12-02 12:00", 0),  # month back from 12-03 lands in the 2024-11-03 repeated hour
    ("2025-03-09 12:00", 0),  # year back from 2025-03-10 lands in the 2024-03-10 gap
    ("2025-11-02 12:00", 0),  # year back from 2025-11-03 lands in the 2024-11-03 repeated hour
    ("2099-03-07 12:00", 0),  # last year before pandas stops following DST rules, still vectorized
    ("2100-03-13 12:00", 0),  # pandas stops following DST rules from 2100
)


def _grid(start: str, remainder: int) -> np.ndarray:
    first = int(pd.Timestamp(start).value) + remainder
    grid: np.ndarray = first + np.arange(110, dtype=np.int64) * (37 * 60 * 10**9)
    return grid


def _reference(ns: np.ndarray, span: WindowSpan, zone: tzinfo | None) -> np.ndarray:
    """Reference bounds: constant width when the span has one, else the per-row shift."""
    width = span.fixed_ns(zone)
    if width is None:
        return np.fromiter((max(span.lower_ns(int(v), zone), _INT64_MIN) for v in ns), dtype=np.int64, count=len(ns))
    if width > _INT64_MAX:
        return np.full(len(ns), _INT64_MIN, dtype=np.int64)
    return np.where(ns < _INT64_MIN + width, _INT64_MIN, ns - width)


def _outcome(ns: np.ndarray, span: WindowSpan, zone: tzinfo | None, use_reference: bool) -> Any:
    try:
        if use_reference:
            return _reference(ns, span, zone)
        return PandasTimeWindow._lower_bounds(ns, span, zone)
    except (ValueError, OverflowError) as exc:
        return (type(exc), str(exc))


def _assert_same_outcome(ns: np.ndarray, span: WindowSpan, zone: tzinfo | None) -> None:
    expected = _outcome(ns, span, zone, use_reference=True)
    actual = _outcome(ns, span, zone, use_reference=False)
    if isinstance(expected, tuple):
        assert actual == expected, f"{span}: expected the same error"
    else:
        assert not isinstance(actual, tuple), f"{span}: raised {actual}"
        np.testing.assert_array_equal(actual, expected, err_msg=str(span))


class TestPandasCalendarBounds:
    """_lower_bounds equals the per-row shift for every knob and zone, edges included."""

    @pytest.mark.parametrize("zone_id", list(_ZONES))
    @pytest.mark.parametrize("rule_id", list(_RULES))
    def test_matches_per_row_reference(self, rule_id: str, zone_id: str) -> None:
        zone = _ZONES[zone_id]()
        for start, remainder in _GRID_STARTS:
            ns = _grid(start, remainder)
            for unit in _UNITS:
                for size in (1, 2):
                    _assert_same_outcome(ns, WindowSpan(size, unit, _RULES[rule_id]), zone)

    @pytest.mark.parametrize("zone_id", ["naive", "new_york"])
    @pytest.mark.parametrize(
        "ns",
        [
            [_INT64_MIN + 1],
            [_INT64_MIN + _DAY_NS],
            [_INT64_MIN + 40 * _DAY_NS],
            [_INT64_MAX],
            [_INT64_MIN + 1, _INT64_MIN + _DAY_NS, 0, _INT64_MAX],
        ],
        ids=["floor", "floor_plus_day", "floor_plus_40_days", "ceiling", "floor_to_ceiling"],
    )
    def test_int64_edges(self, ns: list[int], zone_id: str) -> None:
        zone = _ZONES[zone_id]()
        for rule in (TimeRule(), TimeRule(gap="raise"), TimeRule(month_end="raise")):
            for unit in _UNITS:
                for size in (1, 2):
                    _assert_same_outcome(np.array(ns, dtype=np.int64), WindowSpan(size, unit, rule), zone)

    def test_floor_row_mixed_with_a_gap_raise_row(self) -> None:
        gap_row = pd.Timestamp("2023-03-13 06:30").value  # 02:30 EDT, one wall day earlier is in the gap
        ns = np.array([_INT64_MIN + 1, gap_row], dtype=np.int64)
        _assert_same_outcome(ns, WindowSpan(1, "day", TimeRule(gap="raise")), ZoneInfo("America/New_York"))

    @pytest.mark.parametrize("zone_id", ["naive", "new_york"])
    @pytest.mark.parametrize(
        ("size", "unit"), [(583, "year"), (584, "year"), (10**17, "year"), (10**18, "month"), (2**63, "day")]
    )
    def test_huge_sizes(self, size: int, unit: str, zone_id: str) -> None:
        ns = _grid("2024-03-09 12:00", 0)[:20]
        _assert_same_outcome(ns, WindowSpan(size, unit), _ZONES[zone_id]())

    @pytest.mark.parametrize(
        ("zone_id", "size", "unit"),
        [
            ("new_york", 1, "day"),
            ("new_york", 1, "month"),
            ("naive", 1, "month"),
            ("kolkata", 1, "day"),
            ("kolkata", 2, "week"),
        ],
    )
    def test_in_range_rows_make_no_per_row_call(
        self, monkeypatch: pytest.MonkeyPatch, zone_id: str, size: int, unit: str
    ) -> None:
        def no_per_row_call(*args: Any, **kwargs: Any) -> int:
            raise AssertionError("per-row shift_ns called")

        zone = _ZONES[zone_id]()
        ns = _grid("2023-03-09 12:00", 0)
        monkeypatch.setattr(TimeRule, "shift_ns", no_per_row_call)
        bounds = PandasTimeWindow._lower_bounds(ns, WindowSpan(size, unit), zone)
        assert len(bounds) == len(ns)
        if zone_id == "kolkata":
            width = {"day": _DAY_NS, "week": 7 * _DAY_NS}[unit]
            np.testing.assert_array_equal(bounds, ns - size * width)

    @pytest.mark.parametrize("zone_id", ["naive", "new_york"])
    def test_unsorted_zoned_microsecond_column_matches_sorted(self, zone_id: str) -> None:
        zone = "America/New_York" if zone_id == "new_york" else None
        base = pd.Timestamp("2024-03-09 12:00")
        times = [base + pd.Timedelta(hours=7 * i) for i in range(24)]
        times += [pd.Timestamp("2024-03-31 12:00") + pd.Timedelta(hours=5 * i) for i in range(8)]
        stamps = pd.DatetimeIndex(times)
        if zone is not None:
            stamps = stamps.tz_localize("UTC").tz_convert(zone)
        stamps = stamps.as_unit("us")
        order = np.random.default_rng(0).permutation(len(stamps))
        df = pd.DataFrame({"ts": stamps[order], "value": (2.0 ** (order % 20)).astype(float)})
        assert str(df["ts"].dtype) == ("datetime64[us, America/New_York]" if zone else "datetime64[us]")

        span = WindowSpan(1, "month")
        unsorted_result = np.asarray(PandasTimeWindow._perform_window_operation(df, "sum", span, ["value"], "ts"))
        by_time = df.sort_values("ts", kind="stable")
        sorted_result = np.asarray(PandasTimeWindow._perform_window_operation(by_time, "sum", span, ["value"], "ts"))
        expected = np.empty_like(sorted_result)
        expected[by_time.index.to_numpy()] = sorted_result
        np.testing.assert_array_equal(unsorted_result, expected)

    @pytest.mark.parametrize(
        ("zone", "instant", "size", "unit"),
        [
            (lambda: ZoneInfo("America/New_York"), "2100-03-15 12:00", 2, "day"),
            (lambda: ZoneInfo("Europe/Berlin"), "2100-04-01 12:00", 1, "month"),
            (lambda: tz.gettz("Europe/Berlin"), "1947-05-12 01:30", 1, "day"),
            (lambda: tz.gettz("Australia/Lord_Howe"), "2038-02-01 00:00", 1, "month"),
        ],
        ids=["zoneinfo_new_york_2100", "zoneinfo_berlin_2100", "dateutil_berlin_1947", "dateutil_lord_howe_2038"],
    )
    def test_offsets_where_pandas_and_python_disagree(
        self, zone: Callable[[], tzinfo | None], instant: str, size: int, unit: str
    ) -> None:
        ns = np.array([pd.Timestamp(instant).value], dtype=np.int64)
        _assert_same_outcome(ns, WindowSpan(size, unit), zone())

    @pytest.mark.parametrize(
        ("dtype", "plain"),
        [
            (pa.timestamp("ns"), "datetime64[ns]"),
            (pa.timestamp("us", tz="America/New_York"), "datetime64[ns, America/New_York]"),
        ],
        ids=["ns_naive", "us_new_york"],
    )
    def test_arrow_backed_timestamp_column(self, dtype: pa.DataType, plain: str) -> None:
        times = pd.to_datetime(
            ["2024-03-10 20:00", "2024-03-09 12:00", "2024-03-11 03:00", "2024-03-10 05:00", "2024-03-10 09:30"],
            utc=dtype.tz is not None,
        )
        if dtype.tz is not None:
            times = times.tz_convert(dtype.tz)
        arrow_ts = pd.Series(times).astype(pd.ArrowDtype(dtype))
        df = pd.DataFrame({"ts": arrow_ts, "value": [1.0, 2.0, 4.0, 8.0, 16.0]})
        assert isinstance(df["ts"].dtype, pd.ArrowDtype)
        span = WindowSpan(1, "day")
        actual = PandasTimeWindow._perform_window_operation(df, "sum", span, ["value"], "ts")
        expected_df = df.assign(ts=df["ts"].astype(plain))
        expected = PandasTimeWindow._perform_window_operation(expected_df, "sum", span, ["value"], "ts")
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
