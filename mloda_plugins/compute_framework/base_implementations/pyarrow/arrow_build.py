"""Arrow builders that keep the instants and zone of aware datetimes.

With PYARROW_IGNORE_TIMEZONE=1 plain pyarrow reads the wall time of an aware datetime as UTC.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, tzinfo
from typing import Any

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]


def _zone_name(tz: tzinfo, values: list[datetime], where: str) -> str:
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


def _unnameable(tz: tzinfo, name: str | None, where: str) -> str:
    detail = f"{tz!r}" if name is None else f"{tz!r} (name {name!r})"
    return f"Cannot name the time zone {detail}{where}; use a zone with a readable name."


def _aware_values(values: Any) -> list[datetime] | None:
    """The non-None values when there is at least one and all are aware datetimes, else None."""
    present = []
    for v in values:
        if v is None:
            continue
        if not (isinstance(v, datetime) and v.tzinfo is not None):
            return None
        present.append(v)
    return present or None


def aware_zone(values: Any, column: str | None = None) -> str | None:
    """Return the single zone name of aware datetimes, else None; raise on mixed or unnameable zones."""
    present = _aware_values(values)
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
            aware = _aware_values(values) is not None
        if aware:
            utc_values = [None if v is None else v.astimezone(UTC) for v in values]
            return pa.array(utc_values, type=type)
    return pa.array(values, type=type)


def arrow_table(data: Mapping[str, Any]) -> pa.Table:
    """Build a table, converting list columns through ``arrow_array``."""
    return pa.table(
        {name: arrow_array(col, column=name) if isinstance(col, (list, tuple)) else col for name, col in data.items()}
    )
