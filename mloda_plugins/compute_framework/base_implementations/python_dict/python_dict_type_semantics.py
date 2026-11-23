"""Column-semantics introspector for python_dict data (epic #518, Phase 1).

python_dict carries no schema, so semantics are inferred from the first
non-null value. ``unit`` is always None (value-level resolution is deferred).
The native representation is columnar: a mapping of column name to a list of
values.
"""

from datetime import date, datetime, timedelta
from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics, zone_name
from mloda.core.abstract_plugins.components.contract.value_inspection import iso8601_string_semantics


def column_semantics(data: dict[str, list[Any]], column: str) -> ColumnSemantics:
    """Return the observed semantics of ``column`` inferred from the first non-null value."""
    value = None
    for candidate in data.get(column, []):
        if candidate is not None:
            value = candidate
            break

    if isinstance(value, str):
        inspected = iso8601_string_semantics(data.get(column, []))
        if inspected is not None:
            return inspected

    is_ordered = False
    is_temporal = False
    is_numeric = False
    tz: str | None = None

    if isinstance(value, bool):
        pass
    elif isinstance(value, (int, float)):
        is_numeric = True
        is_ordered = True
    elif isinstance(value, datetime):
        is_temporal = True
        is_ordered = True
        tz = zone_name(value.tzinfo) if value.tzinfo is not None else None
    elif isinstance(value, date):
        is_temporal = True
        is_ordered = True
    elif isinstance(value, timedelta):
        is_ordered = True

    return ColumnSemantics(
        is_ordered=is_ordered,
        is_temporal=is_temporal,
        is_numeric=is_numeric,
        unit=None,
        tz=tz,
    )
