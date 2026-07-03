"""Value-inspection helpers for dynamically typed backends (epic #518, follow-up).

Dynamically typed backends (e.g. sqlite storing datetimes as ISO TEXT) expose
temporal columns as plain strings under schema-only introspection. These helpers
sample the actual values and classify all-ISO-8601 date/datetime string columns
as temporal with the correct timezone awareness.
"""

from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics


def _parse_iso(value: str) -> date | datetime | None:
    # fromisoformat is lenient on 3.11+ (accepts separator-less/basic and week/ordinal forms)
    # but strict on 3.10. Require a dash-bearing extended calendar date and reject week dates
    # so classification is reproducible across the 3.10-3.14 CI matrix and bare-digit codes
    # (zip/product codes) never classify as temporal.
    date_part = value.split("T", 1)[0].split(" ", 1)[0]
    if "-" not in date_part or "w" in value.lower():
        return None
    # Python 3.10's fromisoformat rejects a trailing 'Z'; normalize it to '+00:00'.
    parse_input = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(parse_input)
    except ValueError:
        pass
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def iso8601_string_semantics(values: Iterable[Any], sample_size: int = 100) -> ColumnSemantics | None:
    """Classify a column of ISO-8601 date/datetime strings from a bounded value sample.

    Returns temporal :class:`ColumnSemantics` when every sampled non-None value is a
    parseable ISO-8601 date/datetime string, otherwise ``None``.
    """
    parsed: list[date | datetime] = []
    for value in values:
        if value is None:
            continue
        if not isinstance(value, str):
            return None
        result = _parse_iso(value)
        if result is None:
            return None
        parsed.append(result)
        if len(parsed) >= sample_size:
            break

    if not parsed:
        return None

    # A date-only value or a naive datetime both count as tz-naive. A sample mixing aware and
    # naive values is not confidently classifiable, so do not assert a timezone state.
    any_aware = any(isinstance(p, datetime) and p.tzinfo is not None for p in parsed)
    any_naive = any(not isinstance(p, datetime) or p.tzinfo is None for p in parsed)
    if any_aware and any_naive:
        return None

    is_tz_aware = all(isinstance(p, datetime) and p.tzinfo is not None for p in parsed)
    return ColumnSemantics(
        is_ordered=True,
        is_temporal=True,
        is_numeric=False,
        unit=None,
        is_tz_aware=is_tz_aware,
    )
