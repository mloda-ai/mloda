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


def _zone_name(tz: tzinfo, where: str) -> str:
    name = None
    try:
        name = pa.lib.tzinfo_to_string(tz)
        pa.lib.string_to_tzinfo(name)
    except (TypeError, NotImplementedError, KeyError, ValueError) as error:
        detail = f"{tz!r}" if name is None else f"{tz!r} (name {name!r})"
        raise ValueError(f"Cannot name the time zone {detail}{where}; use a zone with a readable name.") from error
    return str(name)


def _all_aware(values: Any) -> bool:
    present = [v for v in values if v is not None]
    return bool(present) and all(isinstance(v, datetime) and v.tzinfo is not None for v in present)


def aware_zone(values: Any, column: str | None = None) -> str | None:
    """Return the single zone name of aware datetimes, else None; raise on mixed or unnameable zones."""
    if not _all_aware(values):
        return None
    present = [v for v in values if v is not None]
    where = "" if column is None else f" in column {column!r}"
    zones: list[tzinfo] = []
    for v in present:
        if v.tzinfo not in zones:
            zones.append(v.tzinfo)
    names = sorted({_zone_name(tz, where) for tz in zones})
    if len(names) > 1:
        raise ValueError(f"Mixed time zones {names}{where}; convert the values to one zone (astimezone).")
    return names[0]


def arrow_array(values: Any, type: pa.DataType | None = None, column: str | None = None) -> pa.Array:
    """Build an array; a list of aware datetimes keeps its instants and its zone."""
    if isinstance(values, (list, tuple)):
        zone = aware_zone(values, column) if type is None else None
        if type is None and zone is not None:
            type = pa.timestamp("us", tz=zone)
        if type is not None and _all_aware(values):
            utc_values = [None if v is None else v.astimezone(UTC) for v in values]
            return pa.array(utc_values, type=type)
    return pa.array(values, type=type)


def arrow_table(data: Mapping[str, Any]) -> pa.Table:
    """Build a table, converting list columns through ``arrow_array``."""
    return pa.table(
        {name: arrow_array(col, column=name) if isinstance(col, (list, tuple)) else col for name, col in data.items()}
    )
