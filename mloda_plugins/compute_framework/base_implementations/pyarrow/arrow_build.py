"""Arrow builders that keep the instants and zone of aware datetimes.

With PYARROW_IGNORE_TIMEZONE=1 plain pyarrow reads the wall time of an aware datetime as UTC.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from functools import cache
from typing import Any
from zoneinfo import ZoneInfo, available_timezones

from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import (
    _PLAIN_TYPES,
    is_nat,
)

_FAST_TYPES = _PLAIN_TYPES | {float}

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
    """True for None, pd.NaT and NumPy NaT."""
    return v is None or (type(v) not in _FAST_TYPES and is_nat(v))


def has_gap(values: Any) -> bool:
    """True if a non-None gap sits at any list or tuple depth or in a dict value."""
    if isinstance(values, dict):
        values = values.values()
    return any(has_gap(v) if isinstance(v, (list, tuple, dict)) else v is not None and is_gap(v) for v in values)


def null_gaps(values: Any) -> Any:
    """Copy with every gap at any list depth and in dict values as None."""
    if isinstance(values, dict):
        return {
            k: null_gaps(v) if isinstance(v, (list, tuple, dict)) else None if is_gap(v) else v
            for k, v in values.items()
        }
    return [null_gaps(v) if isinstance(v, (list, tuple, dict)) else None if is_gap(v) else v for v in values]


def nested_leaves(values: Any) -> Iterator[Any] | None:
    """Lazy non-gap leaves at any list depth if the first non-gap value is a list or tuple, else None."""
    first = next((v for v in values if not is_gap(v)), None)
    return _leaves(values) if isinstance(first, (list, tuple)) else None


def _leaves(values: Any) -> Iterator[Any]:
    for v in values:
        if isinstance(v, (list, tuple)):
            yield from _leaves(v)
        elif not is_gap(v):
            yield v


def utc_copy(values: Any) -> Any:
    if isinstance(values, (list, tuple)):
        return [utc_copy(v) for v in values]
    if is_gap(values):
        return None
    return values.astimezone(UTC) if isinstance(values, datetime) else values


def _with_zone(dtype: pa.DataType, zone: str) -> pa.DataType:
    if pa.types.is_timestamp(dtype):
        return pa.timestamp(dtype.unit, tz=zone)
    if pa.types.is_list(dtype):
        return pa.list_(_with_zone(dtype.value_type, zone))
    return dtype


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
        leaves = nested_leaves(values)
        if leaves is not None and type is not None and _aware_values(leaves, column) is not None:
            return pa.array(utc_copy(values), type=type)
        zone = None if leaves is None or type is not None else aware_zone(leaves, column)
        if zone is not None:
            arr = pa.array(utc_copy(values))
            return arr.cast(_with_zone(arr.type, zone))
        if has_gap(values):
            values = null_gaps(values)
    return pa.array(values, type=type)


def arrow_table(data: Mapping[str, Any]) -> pa.Table:
    """Build a table, converting list columns through ``arrow_array``."""
    return pa.table(
        {name: arrow_array(col, column=name) if isinstance(col, (list, tuple)) else col for name, col in data.items()}
    )
