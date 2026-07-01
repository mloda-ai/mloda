"""Column-semantics introspector for python_dict rows (epic #518, Phase 1).

python_dict carries no schema, so semantics are inferred from the first
non-null value. ``unit`` is always None (value-level resolution is deferred).
"""

from datetime import date, datetime, timedelta
from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics


def column_semantics(rows: list[dict[str, Any]], column: str) -> ColumnSemantics:
    """Return the observed semantics of ``column`` inferred from the first non-null value."""
    value = None
    for row in rows:
        candidate = row.get(column)
        if candidate is not None:
            value = candidate
            break

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
    )
