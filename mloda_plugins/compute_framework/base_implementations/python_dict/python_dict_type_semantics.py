"""Column semantics for columnar Python dictionaries.
Classification uses the first non-null value; zone identity requires agreement across native datetimes.
``unit`` stays unknown. ISO strings use the shared bounded sample."""

from datetime import date, datetime, timedelta
from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda.core.abstract_plugins.components.contract.value_inspection import common_time_zone, iso8601_string_semantics


def column_semantics(data: dict[str, list[Any]], column: str) -> ColumnSemantics:
    """Classify from the first non-null value and retain a common native timezone when known."""
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
    is_tz_aware = False

    if isinstance(value, bool):
        pass
    elif isinstance(value, (int, float)):
        is_numeric = True
        is_ordered = True
    elif isinstance(value, datetime):
        is_temporal = True
        is_ordered = True
        is_tz_aware = value.tzinfo is not None
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
        is_tz_aware=is_tz_aware,
        tz=common_time_zone(data.get(column, [])) if isinstance(value, datetime) else None,
    )
