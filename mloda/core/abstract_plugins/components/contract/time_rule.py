"""The time zone computation rule: shift a timestamp by elapsed or calendar units, with fixed knobs for the edges."""

from __future__ import annotations

import calendar
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta, tzinfo
from typing import ClassVar

_SECONDS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86_400,
    "week": 604_800,
    "month": 30 * 86_400,
    "year": 365 * 86_400,
}
_ELAPSED = ("second", "minute", "hour")
_DAYS = {"day": 1, "week": 7}
_EPOCH = datetime(1970, 1, 1)
_MICRO = timedelta(microseconds=1)


@dataclass(frozen=True)
class TimeRule:
    """How a time shift behaves: calendar or fixed-length units, a missing or repeated wall time, a missing day."""

    UNITS: ClassVar[dict[str, str]] = {
        "second": "Seconds",
        "minute": "Minutes",
        "hour": "Hours",
        "day": "Days",
        "week": "Weeks",
        "month": "Months",
        "year": "Years",
    }
    KNOBS: ClassVar[dict[str, tuple[str, ...]]] = {
        "calendar_units": ("wall_clock", "fixed_length"),
        "gap": ("forward", "raise"),
        "repeated": ("source", "earlier", "later", "raise"),
        "month_end": ("clamp", "raise"),
    }
    OPTION_KEY: ClassVar[str] = "time_rule"

    calendar_units: str = "wall_clock"
    gap: str = "forward"
    repeated: str = "source"
    month_end: str = "clamp"

    def __post_init__(self) -> None:
        for knob, allowed in self.KNOBS.items():
            if getattr(self, knob) not in allowed:
                raise ValueError(f"Invalid {knob}: {getattr(self, knob)!r}. Allowed: {', '.join(allowed)}")

    def clock(self, unit: str) -> str:
        """The clock a unit runs on under this rule: elapsed or calendar."""
        if unit not in self.UNITS:
            raise ValueError(f"Unsupported time unit: {unit}")
        if unit in _ELAPSED or self.calendar_units == "fixed_length":
            return "elapsed"
        return "calendar"

    def fixed_ns(self, size: int, unit: str, zone: tzinfo | None) -> int | None:
        """The span as constant nanoseconds, or None when it depends on the calendar."""
        if self.clock(unit) == "elapsed" or (unit in _DAYS and (zone is None or zone.utcoffset(None) is not None)):
            return size * _SECONDS[unit] * 1_000_000_000
        return None

    def shift(self, value: datetime, amount: int, unit: str) -> datetime:
        """Return value plus amount units on the value's own zone."""
        if self.clock(unit) == "elapsed":
            delta = timedelta(seconds=amount * _SECONDS[unit])
            if value.tzinfo is None:
                return value + delta
            return (value.astimezone(UTC) + delta).astimezone(value.tzinfo)
        wall = value.replace(tzinfo=None)
        if unit in _DAYS:
            wall = wall + timedelta(days=amount * _DAYS[unit])
        else:
            months = wall.year * 12 + wall.month - 1 + amount * (12 if unit == "year" else 1)
            year, month0 = divmod(months, 12)
            last = calendar.monthrange(year, month0 + 1)[1]
            if wall.day > last and self.month_end == "raise":
                raise ValueError(f"The date {year:04d}-{month0 + 1:02d}-{wall.day:02d} does not exist")
            wall = wall.replace(year=year, month=month0 + 1, day=min(wall.day, last))
        if value.tzinfo is None:
            return wall
        return self._resolve(wall, value.tzinfo, value.utcoffset())

    def _resolve(self, wall: datetime, zone: tzinfo, prefer: timedelta | None) -> datetime:
        """Turn a wall time in zone into an instant under the gap and repeated knobs."""
        guess = wall.replace(tzinfo=UTC)
        day = timedelta(days=1)
        before_offset = (guess - day).astimezone(zone).utcoffset()
        after_offset = (guess + day).astimezone(zone).utcoffset()
        assert before_offset is not None and after_offset is not None
        before = guess - before_offset
        after = guess - after_offset
        before_valid = before.astimezone(zone).utcoffset() == before_offset
        after_valid = after.astimezone(zone).utcoffset() == after_offset
        if not before_valid and not after_valid and self.gap == "raise":
            raise ValueError(f"The wall time {wall.isoformat()} does not exist in {zone}")
        if before_valid and after_valid and before_offset != after_offset:
            if self.repeated == "raise":
                raise ValueError(f"The wall time {wall.isoformat()} occurs twice in {zone}")
            use_after = self.repeated == "later" or (self.repeated == "source" and prefer == after_offset)
        else:
            use_after = after_valid and not before_valid
        return (after if use_after else before).astimezone(zone)

    def shift_ns(self, instant_ns: int, amount: int, unit: str, zone: tzinfo | None) -> int:
        """shift on integer nanoseconds since the epoch; UTC instants for a zone, wall clock when zone is None."""
        micros, remainder = divmod(instant_ns, 1000)
        base = _EPOCH if zone is None else _EPOCH.replace(tzinfo=UTC)
        value = base + timedelta(microseconds=micros)
        if zone is not None:
            value = value.astimezone(zone)
        shifted = self.shift(value, amount, unit)
        if zone is not None:
            shifted = shifted.astimezone(UTC)
        return (shifted - base) // _MICRO * 1000 + remainder

    def as_option(self) -> dict[str, str]:
        """The non-default knobs as a plain dict."""
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) != self.KNOBS[f.name][0]}

    @classmethod
    def from_option(cls, value: object) -> TimeRule:
        """Build a rule from an option value: None or a mapping of knob names to values."""
        if value is None:
            return DEFAULT_TIME_RULE
        if not isinstance(value, Mapping) or not set(value) <= set(cls.KNOBS):
            raise ValueError(f"The {cls.OPTION_KEY} option must be a mapping of the knobs {', '.join(cls.KNOBS)}")
        return cls(**value)


DEFAULT_TIME_RULE = TimeRule()
