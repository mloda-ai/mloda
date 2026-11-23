"""Arrow builders that keep the instants and zone of aware datetimes.

With PYARROW_IGNORE_TIMEZONE=1 plain pyarrow reads the wall time of an aware datetime as UTC.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]


def arrow_array(values: Any, type: pa.DataType | None = None) -> pa.Array:
    """Build an array; a list of aware datetimes keeps its instants and the first value's zone."""
    if isinstance(values, (list, tuple)):
        present = [v for v in values if v is not None]
        if present and all(isinstance(v, datetime) and v.tzinfo is not None for v in present):
            if type is None:
                type = pa.timestamp("us", tz=pa.lib.tzinfo_to_string(present[0].tzinfo))
            utc_values = [None if v is None else v.astimezone(UTC) for v in values]
            return pa.array(utc_values, type=type)
    return pa.array(values, type=type)


def arrow_table(data: Mapping[str, Any]) -> pa.Table:
    """Build a table, converting list columns through ``arrow_array``."""
    return pa.table({name: arrow_array(col) if isinstance(col, (list, tuple)) else col for name, col in data.items()})
