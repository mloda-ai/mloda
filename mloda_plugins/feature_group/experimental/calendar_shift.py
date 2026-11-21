"""Shift a timestamp by calendar or elapsed units on the value's own zone, resolving gaps and repeated times."""

from __future__ import annotations

import calendar
from datetime import UTC, datetime, timedelta, tzinfo

_ELAPSED = {"second": "seconds", "minute": "minutes", "hour": "hours"}
_DAYS = {"day": 1, "week": 7}
_EPOCH = datetime(1970, 1, 1)
_MICRO = timedelta(microseconds=1)


def is_fixed_zone(zone: tzinfo | None) -> bool:
    """A zone without DST: none, or one whose offset does not depend on the date."""
    return zone is None or zone.utcoffset(None) is not None


def _resolve(wall: datetime, zone: tzinfo, prefer: timedelta | None) -> datetime:
    """Turn a wall time in zone into an instant: keep prefer where valid, gap moves forward, repeat takes earlier."""
    guess = wall.replace(tzinfo=UTC)
    day = timedelta(days=1)
    before_offset = (guess - day).astimezone(zone).utcoffset()
    after_offset = (guess + day).astimezone(zone).utcoffset()
    assert before_offset is not None and after_offset is not None
    before = guess - before_offset
    after = guess - after_offset
    before_valid = before.astimezone(zone).utcoffset() == before_offset
    after_valid = after.astimezone(zone).utcoffset() == after_offset
    if after_valid and (not before_valid or prefer == after_offset):
        chosen = after
    else:
        chosen = before
    return chosen.astimezone(zone)


def shift(value: datetime, amount: int, unit: str) -> datetime:
    """Return value plus amount units; second, minute, hour are elapsed, the others follow the wall clock."""
    if unit in _ELAPSED:
        delta = timedelta(**{_ELAPSED[unit]: amount})
        if value.tzinfo is None:
            return value + delta
        return (value.astimezone(UTC) + delta).astimezone(value.tzinfo)
    wall = value.replace(tzinfo=None)
    if unit in _DAYS:
        wall = wall + timedelta(days=amount * _DAYS[unit])
    elif unit in ("month", "year"):
        months = wall.year * 12 + wall.month - 1 + amount * (12 if unit == "year" else 1)
        year, month0 = divmod(months, 12)
        day = min(wall.day, calendar.monthrange(year, month0 + 1)[1])
        wall = wall.replace(year=year, month=month0 + 1, day=day)
    else:
        raise ValueError(f"Unsupported time unit: {unit}")
    if value.tzinfo is None:
        return wall
    return _resolve(wall, value.tzinfo, value.utcoffset())


def shift_ns(instant_ns: int, amount: int, unit: str, zone: tzinfo | None) -> int:
    """shift on integer nanoseconds since the epoch; UTC instants for a zone, wall clock when zone is None."""
    micros, remainder = divmod(instant_ns, 1000)
    base = _EPOCH if zone is None else _EPOCH.replace(tzinfo=UTC)
    value = base + timedelta(microseconds=micros)
    if zone is not None:
        value = value.astimezone(zone)
    shifted = shift(value, amount, unit)
    if zone is not None:
        shifted = shifted.astimezone(UTC)
    return (shifted - base) // _MICRO * 1000 + remainder
