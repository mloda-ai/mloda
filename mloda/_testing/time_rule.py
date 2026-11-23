"""Conformance kit for the time zone computation rule: pinned shift cases and the check a backend's shift must pass."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from mloda._testing.result import CheckResult
from mloda._testing.suite import assert_passed
from mloda.core.abstract_plugins.components.contract.time_rule import DEFAULT_TIME_RULE, TimeRule

__all__ = [
    "SHIFT_CASES",
    "Shift",
    "ShiftCase",
    "WINDOW_CASES",
    "WINDOW_FUNCTIONS",
    "WindowCase",
    "assert_time_rule",
    "check_time_rule",
]


@dataclass(frozen=True)
class ShiftCase:
    id: str
    value: datetime
    amount: int
    unit: str
    expected: str | None  # isoformat of the result; None: the rule raises ValueError
    rule: TimeRule = DEFAULT_TIME_RULE


_NY = ZoneInfo("America/New_York")
_NOON = datetime(2023, 3, 11, 12, tzinfo=_NY)
_SUNDAY = datetime(2023, 3, 12, 12, tzinfo=_NY)
_NAIVE_NOON = datetime(2023, 3, 11, 12)
_IST = timezone(timedelta(hours=5, minutes=30))
_GAP_WALL = datetime(2023, 3, 11, 2, 30, tzinfo=_NY)  # one day later is the nonexistent 2023-03-12 02:30
_TWICE_EARLY = datetime(2023, 11, 4, 1, 30, tzinfo=_NY)  # one day later is the repeated 2023-11-05 01:30
_TWICE_LATE = datetime(2023, 11, 6, 1, 30, tzinfo=_NY)  # one day earlier is the repeated 2023-11-05 01:30

_FIXED = TimeRule(calendar_units="fixed_length")
_GAP_RAISE = TimeRule(gap="raise")
_EARLIER = TimeRule(repeated="earlier")
_LATER = TimeRule(repeated="later")
_REPEATED_RAISE = TimeRule(repeated="raise")
_MONTH_END_RAISE = TimeRule(month_end="raise")

SHIFT_CASES: tuple[ShiftCase, ...] = (
    # exact units are elapsed, calendar units keep the wall time
    ShiftCase("day_keeps_the_wall_time", _NOON, 1, "day", "2023-03-12T12:00:00-04:00"),
    ShiftCase("24_hours_are_elapsed", _NOON, 24, "hour", "2023-03-12T13:00:00-04:00"),
    ShiftCase("1440_minutes_are_elapsed", _NOON, 1440, "minute", "2023-03-12T13:00:00-04:00"),
    ShiftCase("86400_seconds_are_elapsed", _NOON, 86400, "second", "2023-03-12T13:00:00-04:00"),
    ShiftCase("week_keeps_the_wall_time", _NOON, 1, "week", "2023-03-18T12:00:00-04:00"),
    ShiftCase("zero_days_keeps_the_offset", _NOON, 0, "day", "2023-03-11T12:00:00-05:00"),
    ShiftCase("back_one_day_over_the_change", _SUNDAY, -1, "day", "2023-03-11T12:00:00-05:00"),
    ShiftCase("back_24_hours_over_the_change", _SUNDAY, -24, "hour", "2023-03-11T11:00:00-05:00"),
    ShiftCase("naive_day", _NAIVE_NOON, 1, "day", "2023-03-12T12:00:00"),
    ShiftCase("naive_24_hours", _NAIVE_NOON, 24, "hour", "2023-03-12T12:00:00"),
    ShiftCase("naive_back_one_week", _NAIVE_NOON, -1, "week", "2023-03-04T12:00:00"),
    # month and year clamp to the month end
    ShiftCase("month_back_to_february", datetime(2023, 3, 31), -1, "month", "2023-02-28T00:00:00"),
    ShiftCase("month_forward_to_february", datetime(2023, 1, 31, 12), 1, "month", "2023-02-28T12:00:00"),
    ShiftCase("month_forward_to_march", datetime(2023, 1, 31), 2, "month", "2023-03-31T00:00:00"),
    ShiftCase("month_forward_to_march_at_noon", datetime(2023, 1, 31, 12), 2, "month", "2023-03-31T12:00:00"),
    ShiftCase("leap_day_back_one_year", datetime(2024, 2, 29), -1, "year", "2023-02-28T00:00:00"),
    ShiftCase("leap_day_forward_one_year", datetime(2024, 2, 29), 1, "year", "2025-02-28T00:00:00"),
    ShiftCase("leap_day_forward_four_years", datetime(2024, 2, 29), 4, "year", "2028-02-29T00:00:00"),
    ShiftCase("month_over_the_year_end", datetime(2023, 12, 15), 1, "month", "2024-01-15T00:00:00"),
    ShiftCase("month_back_over_the_year_start", datetime(2023, 1, 15), -1, "month", "2022-12-15T00:00:00"),
    ShiftCase("thirteen_months_back", datetime(2023, 1, 15), -13, "month", "2021-12-15T00:00:00"),
    ShiftCase(
        "month_back_in_berlin",
        datetime(2023, 3, 31, 12, tzinfo=ZoneInfo("Europe/Berlin")),
        -1,
        "month",
        "2023-02-28T12:00:00+01:00",
    ),
    # gaps and repeated times
    ShiftCase("gap_going_back", datetime(2023, 3, 13, 2, 30, tzinfo=_NY), -1, "day", "2023-03-12T03:30:00-04:00"),
    ShiftCase("gap_going_forward", _GAP_WALL, 1, "day", "2023-03-12T03:30:00-04:00"),
    ShiftCase(
        "thirty_minute_gap",
        datetime(2023, 9, 30, 2, 15, tzinfo=ZoneInfo("Australia/Lord_Howe")),
        1,
        "day",
        "2023-10-01T02:45:00+11:00",
    ),
    ShiftCase("repeated_source_on_earlier_offset", _TWICE_EARLY, 1, "day", "2023-11-05T01:30:00-04:00"),
    ShiftCase("repeated_source_on_later_offset", _TWICE_LATE, -1, "day", "2023-11-05T01:30:00-05:00"),
    ShiftCase("utc_zone", datetime(2023, 3, 11, 12, tzinfo=ZoneInfo("UTC")), 1, "day", "2023-03-12T12:00:00+00:00"),
    ShiftCase("fixed_offset", datetime(2023, 1, 31, 12, tzinfo=_IST), 1, "month", "2023-02-28T12:00:00+05:30"),
    # the forecast horizon steps
    ShiftCase("two_days_keep_the_wall_time", _NOON, 2, "day", "2023-03-13T12:00:00-04:00"),
    # calendar_units="fixed_length": day 24 hours, week 7 days, month 30 days, year 365 days
    ShiftCase("fixed_day", _NOON, 1, "day", "2023-03-12T13:00:00-04:00", _FIXED),  # 17:00Z + 24h = 13:00 EDT
    ShiftCase("fixed_day_back", _SUNDAY, -1, "day", "2023-03-11T11:00:00-05:00", _FIXED),  # 16:00Z - 24h = 11:00 EST
    ShiftCase("fixed_week", _NOON, 1, "week", "2023-03-18T13:00:00-04:00", _FIXED),  # 17:00Z + 168h = 13:00 EDT
    ShiftCase("fixed_month", _NOON, 1, "month", "2023-04-10T13:00:00-04:00", _FIXED),  # Mar 11 17:00Z + 30d = Apr 10
    ShiftCase("fixed_year", datetime(2024, 1, 1), 1, "year", "2024-12-31T00:00:00", _FIXED),  # 2024 has 366 days
    ShiftCase(
        "fixed_length_never_hits_a_gap",
        _GAP_WALL,
        1,
        "day",
        "2023-03-12T03:30:00-04:00",
        TimeRule(calendar_units="fixed_length", gap="raise"),  # 07:30Z + 24h = 03:30 EDT
    ),
    # gap="raise"
    ShiftCase("gap_raise_on_a_missing_wall_time", _GAP_WALL, 1, "day", None, _GAP_RAISE),
    ShiftCase("gap_raise_on_an_existing_wall_time", _NOON, 1, "day", "2023-03-12T12:00:00-04:00", _GAP_RAISE),
    # repeated="earlier" / "later" / "raise"
    ShiftCase("repeated_earlier_beats_a_later_source", _TWICE_LATE, -1, "day", "2023-11-05T01:30:00-04:00", _EARLIER),
    ShiftCase("repeated_later_beats_an_earlier_source", _TWICE_EARLY, 1, "day", "2023-11-05T01:30:00-05:00", _LATER),
    ShiftCase("repeated_later_from_a_later_source", _TWICE_LATE, -1, "day", "2023-11-05T01:30:00-05:00", _LATER),
    ShiftCase("repeated_raise_on_a_repeated_wall_time", _TWICE_EARLY, 1, "day", None, _REPEATED_RAISE),
    ShiftCase("repeated_raise_on_a_unique_wall_time", _NOON, 1, "day", "2023-03-12T12:00:00-04:00", _REPEATED_RAISE),
    # month_end="raise"
    ShiftCase("month_end_raise_on_a_missing_day", datetime(2023, 1, 31, 12), 1, "month", None, _MONTH_END_RAISE),
    ShiftCase("month_end_raise_on_a_leap_day", datetime(2024, 2, 29), 1, "year", None, _MONTH_END_RAISE),
    ShiftCase(
        "month_end_raise_on_an_existing_day", datetime(2023, 1, 31), 2, "month", "2023-03-31T00:00:00", _MONTH_END_RAISE
    ),
    ShiftCase(
        "month_end_raise_on_the_next_leap_day",
        datetime(2024, 2, 29),
        4,
        "year",
        "2028-02-29T00:00:00",
        _MONTH_END_RAISE,
    ),
    ShiftCase("repeated_earlier_from_an_earlier_source", _TWICE_EARLY, 1, "day", "2023-11-05T01:30:00-04:00", _EARLIER),
    # the wall time 02:30 does not exist; forward by the one hour gap
    ShiftCase(
        "month_step_into_a_gap",
        datetime(2023, 2, 12, 2, 30, tzinfo=_NY),
        1,
        "month",
        "2023-03-12T03:30:00-04:00",
    ),
    # the source offset -04:00 is one of the two
    ShiftCase(
        "month_step_into_a_repeated_hour",
        datetime(2023, 10, 5, 1, 30, tzinfo=_NY),
        1,
        "month",
        "2023-11-05T01:30:00-04:00",
    ),
    ShiftCase(
        "month_back_into_a_repeated_hour",
        datetime(2023, 12, 5, 1, 30, tzinfo=_NY),
        -1,
        "month",
        "2023-11-05T01:30:00-05:00",
    ),
)


Shift = Callable[[datetime, int, str, TimeRule], datetime]
"""Shifts a value by amount units under a rule."""

_UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86_400, "week": 604_800}
_UNIT_SECONDS |= {"month": 30 * 86_400, "year": 365 * 86_400}


def _default_shift(value: datetime, amount: int, unit: str, rule: TimeRule) -> datetime:
    return rule.shift(value, amount, unit)


def _case_failures(shift: Shift, case: ShiftCase) -> list[str]:
    try:
        result = shift(case.value, case.amount, case.unit, case.rule)
    except ValueError as exc:
        if case.expected is None:
            return []
        return [f"raises: {case.id} raised ValueError unexpectedly: {exc}"]
    except Exception as exc:  # the check must observe any failure of the shift under test
        return [f"raises: {case.id} raised {type(exc).__name__}: {exc}"]
    if case.expected is None:
        return [f"raises: {case.id} expected ValueError, got {result.isoformat()}"]
    failures: list[str] = []
    if result.isoformat() != case.expected:
        failures.append(f"value: {case.id} expected {case.expected}, got {result.isoformat()}")
    if result.tzinfo != case.value.tzinfo:
        failures.append(f"zone: {case.id} expected zone {case.value.tzinfo}, got {result.tzinfo}")
    if case.rule.clock(case.unit) == "elapsed" and (result.tzinfo is None) == (case.value.tzinfo is None):
        wanted = timedelta(seconds=case.amount * _UNIT_SECONDS[case.unit])
        moved = (
            result - case.value if case.value.tzinfo is None else result.astimezone(UTC) - case.value.astimezone(UTC)
        )
        if moved != wanted:
            failures.append(f"elapsed: {case.id} moved {moved}, expected exactly {wanted}")
    if result.tzinfo is not None and result.astimezone(UTC).astimezone(result.tzinfo).replace(
        tzinfo=None
    ) != result.replace(tzinfo=None):
        failures.append(f"wall_time: {case.id} {result.isoformat()} is not an existing wall time")
    return failures


def check_time_rule(shift: Shift | None = None, cases: Iterable[ShiftCase] = SHIFT_CASES) -> CheckResult:
    """Check a shift against the pinned cases, each under its own rule, and the zone, elapsed and wall time rules."""
    used = _default_shift if shift is None else shift
    failures = [f for case in cases for f in _case_failures(used, case)]
    subject = "TimeRule.shift" if shift is None else getattr(used, "__name__", type(used).__name__)
    return CheckResult(check="time_rule", subject=subject, failures=tuple(failures))


def assert_time_rule(shift: Shift | None = None, cases: Iterable[ShiftCase] = SHIFT_CASES) -> None:
    """Fail the running pytest test if the time rule check reports failures."""
    assert_passed(check_time_rule(shift, cases))


@dataclass(frozen=True)
class WindowCase:
    id: str
    zone: str | None  # zone of the reference column, None for a naive column
    instants: tuple[str, ...]  # ascending; UTC instants for a zoned column, wall times for a naive one
    size: int
    unit: str
    members: tuple[tuple[int, ...], ...] | None  # per row, the row values inside its window; None: ValueError
    rule: TimeRule = DEFAULT_TIME_RULE


WINDOW_FUNCTIONS: tuple[str, ...] = (
    "sum",
    "min",
    "max",
    "avg",
    "mean",
    "count",
    "std",
    "var",
    "median",
    "first",
    "last",
)

_NY_ZONE = "America/New_York"
_SHORT_DAY = ("2023-03-11 16:30", "2023-03-12 16:00")
_LONG_DAY = ("2023-11-04 16:30", "2023-11-05 17:00")
_REPEATED_HOUR = ("2023-11-05 05:40", "2023-11-05 06:40", "2023-11-06 06:30")
_GAP_HOUR = ("2023-03-12 07:15", "2023-03-12 07:45", "2023-03-13 06:30")
_IST_DAYS = ("2023-01-31 06:30", "2023-02-28 12:30", "2023-03-01 06:30")

# Row i holds the value 10 ** i, so each member tuple reads as the rows inside the window.
WINDOW_CASES: tuple[WindowCase, ...] = (
    WindowCase("dst_short_day_calendar", _NY_ZONE, _SHORT_DAY, 1, "day", ((1,), (10,))),
    WindowCase("dst_short_day_elapsed", _NY_ZONE, _SHORT_DAY, 24, "hour", ((1,), (1, 10))),
    WindowCase("dst_long_day_calendar", _NY_ZONE, _LONG_DAY, 1, "day", ((1,), (1, 10))),
    WindowCase("dst_long_day_elapsed", _NY_ZONE, _LONG_DAY, 24, "hour", ((1,), (10,))),
    WindowCase("bound_in_repeated_hour", _NY_ZONE, _REPEATED_HOUR, 1, "day", ((1,), (1, 10), (10, 100))),
    WindowCase("bound_in_gap", _NY_ZONE, _GAP_HOUR, 1, "day", ((1,), (1, 10), (10, 100))),
    WindowCase("fixed_offset_month", "+05:30", _IST_DAYS, 1, "month", ((1,), (1, 10), (10, 100))),
    WindowCase("fixed_offset_day", "+05:30", _IST_DAYS, 1, "day", ((1,), (10,), (10, 100))),
    WindowCase(
        "month_calendar_clamp",
        None,
        ("2023-01-31 12:00", "2023-02-28 12:00", "2023-03-01 12:00", "2023-03-31 12:00"),
        1,
        "month",
        ((1,), (1, 10), (10, 100), (100, 1000)),
    ),
    WindowCase(
        "year_calendar_clamp",
        None,
        ("2023-02-28 12:00", "2024-02-28 18:00", "2024-02-29 12:00", "2025-02-28 12:00"),
        1,
        "year",
        ((1,), (10,), (10, 100), (10, 100, 1000)),
    ),
    WindowCase(
        "starts_back_bound_in_gap",
        _NY_ZONE,
        ("2023-03-12 07:10", "2023-03-12 07:40", "2023-03-13 06:30", "2023-03-13 07:00"),
        1,
        "day",
        ((1,), (1, 10), (10, 100), (1, 10, 100, 1000)),
    ),
    WindowCase(
        "starts_back_source_in_repeated_hour",
        _NY_ZONE,
        ("2023-11-04 05:20", "2023-11-04 05:40", "2023-11-05 05:50", "2023-11-05 06:30"),
        1,
        "day",
        ((1,), (1, 10), (100,), (10, 100, 1000)),
    ),
    # fixed_length: the second row's lower bound is 16:00Z minus 24h = 03-11 16:00Z, so 16:30Z is inside
    WindowCase(
        "fixed_length_short_day",
        _NY_ZONE,
        _SHORT_DAY,
        1,
        "day",
        ((1,), (1, 10)),
        TimeRule(calendar_units="fixed_length"),
    ),
    # gap="raise": 2023-03-13 06:30Z is 02:30 EDT, one wall day earlier is 2023-03-12 02:30, which does not exist
    WindowCase(
        "gap_raise_lower_bound_in_the_gap",
        _NY_ZONE,
        ("2023-03-12 07:15", "2023-03-13 06:30"),
        1,
        "day",
        None,
        TimeRule(gap="raise"),
    ),
    # row 2 is 2023-03-13 12:00 EDT; two wall days earlier is 2023-03-11 12:00 EST = 17:00Z, so 16:30Z is outside
    WindowCase(
        "two_days_over_the_short_day",
        _NY_ZONE,
        ("2023-03-11 16:30", "2023-03-12 16:00", "2023-03-13 16:00"),
        2,
        "day",
        ((1,), (1, 10), (10, 100)),
    ),
    # row 2: three months before 2023-05-31 is 2023-02-31, clamped to 2023-02-28 12:00, the left-open bound
    WindowCase(
        "three_months_clamp",
        None,
        ("2023-02-28 12:00", "2023-03-01 12:00", "2023-05-31 12:00"),
        3,
        "month",
        ((1,), (1, 10), (10, 100)),
    ),
    # row 1 is 2023-03-15 12:00 EDT; one wall week earlier is 2023-03-08 12:00 EST = 17:00Z, so 16:30Z is outside
    WindowCase(
        "week_over_the_short_day",
        _NY_ZONE,
        ("2023-03-08 16:30", "2023-03-15 16:00"),
        1,
        "week",
        ((1,), (10,)),
    ),
    # row 2's bound is the repeated 2023-11-05 01:30; the earlier instant is 05:30Z, so 05:40Z is inside
    WindowCase(
        "repeated_earlier_bound_in_repeated_hour",
        _NY_ZONE,
        _REPEATED_HOUR,
        1,
        "day",
        ((1,), (1, 10), (1, 10, 100)),
        TimeRule(repeated="earlier"),
    ),
    # Asia/Kolkata is +05:30 without DST, so a wall day is 24h: row 1 (02-28 12:30Z) looks back to 02-27 12:30Z, past
    # row 0; row 2 (03-01 06:30Z) looks back to 02-28 06:30Z, which is before row 1
    WindowCase("named_zone_without_dst_day", "Asia/Kolkata", _IST_DAYS, 1, "day", ((1,), (10,), (10, 100))),
    # one month before 2023-03-31 is the missing 2023-02-31
    WindowCase(
        "month_end_raise_missing_day",
        None,
        ("2023-01-31 12:00", "2023-02-28 12:00", "2023-03-01 12:00", "2023-03-31 12:00"),
        1,
        "month",
        None,
        TimeRule(month_end="raise"),
    ),
)
