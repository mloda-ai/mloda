"""Tests for the calendar_shift helpers: elapsed units, calendar units, gap and repeated-time resolution."""

from datetime import UTC, datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo

import pytest

from mloda_plugins.feature_group.experimental.calendar_shift import is_fixed_zone, shift, shift_ns

NY = ZoneInfo("America/New_York")
NOON = datetime(2023, 3, 11, 12, tzinfo=NY)
SUNDAY = datetime(2023, 3, 12, 12, tzinfo=NY)
NAIVE_NOON = datetime(2023, 3, 11, 12)
IST = timezone(timedelta(hours=5, minutes=30))


@pytest.mark.parametrize(
    ("value", "amount", "unit", "expected"),
    [
        (NOON, 1, "day", "2023-03-12T12:00:00-04:00"),
        (NOON, 24, "hour", "2023-03-12T13:00:00-04:00"),
        (NOON, 1440, "minute", "2023-03-12T13:00:00-04:00"),
        (NOON, 86400, "second", "2023-03-12T13:00:00-04:00"),
        (NOON, 1, "week", "2023-03-18T12:00:00-04:00"),
        (NOON, 0, "day", "2023-03-11T12:00:00-05:00"),
        (SUNDAY, -1, "day", "2023-03-11T12:00:00-05:00"),
        (SUNDAY, -24, "hour", "2023-03-11T11:00:00-05:00"),
        (NAIVE_NOON, 1, "day", "2023-03-12T12:00:00"),
        (NAIVE_NOON, 24, "hour", "2023-03-12T12:00:00"),
        (NAIVE_NOON, -1, "week", "2023-03-04T12:00:00"),
    ],
)
def test_shift_exact_units_are_elapsed_and_calendar_units_keep_the_wall_time(
    value: datetime, amount: int, unit: str, expected: str
) -> None:
    assert shift(value, amount, unit).isoformat() == expected


@pytest.mark.parametrize(
    ("value", "amount", "unit", "expected"),
    [
        (datetime(2023, 3, 31), -1, "month", "2023-02-28T00:00:00"),
        (datetime(2023, 1, 31, 12), 1, "month", "2023-02-28T12:00:00"),
        (datetime(2023, 1, 31), 2, "month", "2023-03-31T00:00:00"),
        (datetime(2024, 2, 29), -1, "year", "2023-02-28T00:00:00"),
        (datetime(2024, 2, 29), 4, "year", "2028-02-29T00:00:00"),
        (datetime(2023, 12, 15), 1, "month", "2024-01-15T00:00:00"),
        (datetime(2023, 1, 15), -1, "month", "2022-12-15T00:00:00"),
        (datetime(2023, 1, 15), -13, "month", "2021-12-15T00:00:00"),
        (datetime(2023, 3, 31, 12, tzinfo=ZoneInfo("Europe/Berlin")), -1, "month", "2023-02-28T12:00:00+01:00"),
    ],
)
def test_shift_month_and_year_clamp_to_the_month_end(value: datetime, amount: int, unit: str, expected: str) -> None:
    assert shift(value, amount, unit).isoformat() == expected


def test_shift_rejects_an_unknown_unit() -> None:
    with pytest.raises(ValueError):
        shift(datetime(2023, 1, 1), 1, "fortnight")


@pytest.mark.parametrize(
    ("value", "amount", "unit", "expected"),
    [
        (datetime(2023, 3, 13, 2, 30, tzinfo=NY), -1, "day", "2023-03-12T03:30:00-04:00"),
        (datetime(2023, 3, 11, 2, 30, tzinfo=NY), 1, "day", "2023-03-12T03:30:00-04:00"),
        (
            datetime(2023, 9, 30, 2, 15, tzinfo=ZoneInfo("Australia/Lord_Howe")),
            1,
            "day",
            "2023-10-01T02:45:00+11:00",
        ),
        (datetime(2023, 11, 4, 1, 30, tzinfo=NY), 1, "day", "2023-11-05T01:30:00-04:00"),
        (datetime(2023, 11, 6, 1, 30, tzinfo=NY), -1, "day", "2023-11-05T01:30:00-05:00"),
        (datetime(2023, 3, 11, 12, tzinfo=ZoneInfo("UTC")), 1, "day", "2023-03-12T12:00:00+00:00"),
        (datetime(2023, 1, 31, 12, tzinfo=IST), 1, "month", "2023-02-28T12:00:00+05:30"),
    ],
    ids=[
        "gap_going_back",
        "gap_going_forward",
        "thirty_minute_gap",
        "repeated_source_on_earlier_offset",
        "repeated_source_on_later_offset",
        "utc_zone",
        "fixed_offset",
    ],
)
def test_shift_resolves_gaps_and_repeated_times(value: datetime, amount: int, unit: str, expected: str) -> None:
    assert shift(value, amount, unit).isoformat() == expected


@pytest.mark.parametrize(
    ("zone", "expected"),
    [
        (None, True),
        (UTC, True),
        (ZoneInfo("UTC"), True),
        (IST, True),
        (ZoneInfo("Etc/GMT-5"), True),
        (NY, False),
    ],
)
def test_is_fixed_zone(zone: tzinfo | None, expected: bool) -> None:
    assert is_fixed_zone(zone) is expected


def _ns(text: str) -> int:
    seconds = int(datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp())
    return seconds * 1_000_000_000


def test_shift_ns_aware_column_uses_its_zone() -> None:
    assert shift_ns(_ns("2023-03-11T17:00:00") + 1, 1, "day", NY) == _ns("2023-03-12T16:00:00") + 1


def test_shift_ns_naive_column_keeps_the_sub_microsecond_remainder() -> None:
    assert shift_ns(_ns("2023-01-31T12:00:00") + 999, 1, "month", None) == _ns("2023-02-28T12:00:00") + 999


def test_shift_ns_before_1970() -> None:
    assert shift_ns(-1, -1, "day", None) == -1 - 86_400_000_000_000
