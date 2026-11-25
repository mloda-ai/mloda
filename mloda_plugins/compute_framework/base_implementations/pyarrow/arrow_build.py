"""Arrow builders that keep the instants and zone of aware datetimes.

With PYARROW_IGNORE_TIMEZONE=1 plain pyarrow reads the wall time of an aware datetime as UTC.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from functools import cache
from typing import Any
from zoneinfo import ZoneInfo, available_timezones

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]


def _zone_name(tz: tzinfo, values: list[datetime], where: str) -> str:
    if pa is None:
        return _zone_name_without_pyarrow(tz, values, where)
    name = None
    resolved = None
    # pyarrow has no validating API, so its errors are translated into the rule's ValueError.
    try:
        name = pa.lib.tzinfo_to_string(tz)
        resolved = pa.lib.string_to_tzinfo(name)
    except (TypeError, NotImplementedError, KeyError, ValueError, AttributeError) as error:
        raise ValueError(_unnameable(tz, name, where)) from error
    if any(v.astimezone(resolved).utcoffset() != v.utcoffset() for v in values if v.tzinfo == tz):
        raise ValueError(_unnameable(tz, name, where))
    return str(name)


@cache
def _known_zones() -> frozenset[str]:
    return frozenset(available_timezones())


def _zone_name_without_pyarrow(tz: tzinfo, values: list[datetime], where: str) -> str:
    if tz is UTC:
        return "UTC"
    if isinstance(tz, ZoneInfo) and tz.key is not None:
        return tz.key
    if isinstance(tz, timezone):
        offset = tz.utcoffset(None)
        minutes, seconds = divmod(int(offset.total_seconds()), 60)
        if seconds == 0 and not offset.microseconds:
            hours, mins = divmod(abs(minutes), 60)
            return f"{'-' if minutes < 0 else '+'}{hours:02d}:{mins:02d}"
    elif tz.utcoffset(None) == timedelta(0) and tz.tzname(None) == "UTC":
        return "UTC"
    zone = getattr(tz, "zone", None)
    if isinstance(zone, str) and zone in _known_zones():
        resolved = ZoneInfo(zone)
        if all(v.astimezone(resolved).utcoffset() == v.utcoffset() for v in values if v.tzinfo == tz):
            return zone
    raise ValueError(_unnameable(tz, None, where))


def _unnameable(tz: tzinfo, name: str | None, where: str) -> str:
    detail = f"{tz!r}" if name is None else f"{tz!r} (name {name!r})"
    return f"Cannot name the time zone {detail}{where}; use a zone with a readable name."


def is_gap(v: Any) -> bool:
    """True for None and pd.NaT."""
    # NaT is the only datetime unequal to itself (no pandas import).
    return v is None or (isinstance(v, datetime) and v != v)


def _aware_values(values: Any, column: str | None = None) -> list[datetime] | None:
    """The non-gap values if all are aware datetimes, None if not datetimes; raises on a naive/aware mix."""
    present = []
    has_naive = has_aware = False
    for v in values:
        if is_gap(v):
            continue
        if not isinstance(v, datetime):
            return None
        present.append(v)
        if v.tzinfo is None:
            has_naive = True
        else:
            has_aware = True
    if not has_aware:
        return None
    if has_naive:
        subject = "Values mix" if column is None else f"Column {column!r} mixes"
        raise ValueError(f"{subject} naive and aware datetimes; make them all naive or all aware.")
    return present


def aware_zone(values: Any, column: str | None = None) -> str | None:
    """Return the single zone name of aware datetimes, else None; raise on mixed or unnameable zones."""
    present = _aware_values(values, column)
    if present is None:
        return None
    where = "" if column is None else f" in column {column!r}"
    zones: list[tzinfo] = []
    for v in present:
        if v.tzinfo not in zones:
            zones.append(v.tzinfo)  # type: ignore[arg-type]
    names = sorted({_zone_name(tz, present, where) for tz in zones})
    if len(names) > 1:
        raise ValueError(f"Mixed time zones {names}{where}; convert the values to one zone (astimezone).")
    return names[0]


def arrow_array(values: Any, type: pa.DataType | None = None, column: str | None = None) -> pa.Array:
    """Build an array; a list of aware datetimes keeps its instants and its zone."""
    if isinstance(values, (list, tuple)):
        if type is None:
            zone = aware_zone(values, column)
            aware = zone is not None
            if aware:
                type = pa.timestamp("us", tz=zone)
        else:
            aware = _aware_values(values, column) is not None
        if aware:
            utc_values = [None if is_gap(v) else v.astimezone(UTC) for v in values]
            return pa.array(utc_values, type=type)
    return pa.array(values, type=type)


def arrow_table(data: Mapping[str, Any]) -> pa.Table:
    """Build a table, converting list columns through ``arrow_array``."""
    return pa.table(
        {name: arrow_array(col, column=name) if isinstance(col, (list, tuple)) else col for name, col in data.items()}
    )
