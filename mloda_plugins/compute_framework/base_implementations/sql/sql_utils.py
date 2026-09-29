"""SQL quoting utilities shared by all SQL-based compute frameworks.

SQL injection prevention follows two layers:

- Identifiers (column names, table names, aliases) go through ``quote_ident``
  which applies SQL-standard double-quote escaping.
- Literal values should use PEP 249 (DB-API 2.0) parameterized queries
  whenever the backend supports them. ``quote_value`` and ``inline_params``
  exist as a fallback for backends whose API lacks PEP 249 parameter binding.
  They accept primitive scalars (None, bool, int, float, str, Decimal) and
  datetime; unsupported types raise.
"""

import math
import string
from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

_ASCII_FOLD = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def is_ordered_arrow_type(t: "pa.DataType") -> bool:
    """Return True if a pyarrow DataType is ordered (integer/float/decimal/temporal)."""
    import pyarrow as pa

    return bool(pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t) or pa.types.is_temporal(t))


def null_or_nan_condition(quoted_column: str, nan_condition: str | None) -> str:
    """Return a SQL condition true when the column is NULL, plus ``nan_condition`` when given."""
    cond = f"{quoted_column} IS NULL"
    if nan_condition is not None:
        return f"({cond} OR {nan_condition})"
    return cond


def quote_ident(name: str) -> str:
    """Quote a SQL identifier with double-quote escaping (SQL standard)."""
    return f'"{name.replace(chr(34), chr(34) + chr(34))}"'


def quote_value(value: Any) -> str:
    """Quote a SQL literal value. Fallback for backends without PEP 249 parameter support."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"Cannot convert non-finite float to SQL: {value!r}")
        return str(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"Cannot convert non-finite Decimal to SQL: {value!r}")
        return format(value, "f")
    if isinstance(value, datetime):
        # ISO 8601 string literal. DuckDB / SQLite / most engines auto-cast against
        # a temporal column. isoformat() emits no embedded single quotes.
        return f"'{value.isoformat()}'"
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    raise TypeError(f"Unsupported type for SQL literal: {type(value).__name__}")


def fold_identifier(name: str) -> str:
    """Lowercase ASCII letters only, matching how DuckDB and SQLite compare identifiers."""
    return name.translate(_ASCII_FOLD)


def ensure_distinct_identifiers(columns: Iterable[str], operation: str) -> None:
    """Raise ``ValueError`` on the first pair of columns equal under ``fold_identifier``."""
    seen: dict[str, str] = {}
    for column in columns:
        folded = fold_identifier(column)
        if folded in seen:
            raise ValueError(
                f"{operation}: columns {seen[folded]!r} and {column!r} resolve to the same SQL column "
                "(names are compared ignoring ASCII case); rename one side"
            )
        seen[folded] = column


def pick_helper_column_name(taken: set[str], prefix: str = "__mloda_rn") -> str:
    """Return the lowest ``{prefix}{n}__`` name not present in ``taken`` (ASCII case-insensitive).

    The ``prefix`` is ASCII-lowercased, so a mixed-case ``prefix`` cannot return a name
    that collides with an entry in ``taken``.
    """
    prefix_cf = fold_identifier(prefix)
    taken_cf = {fold_identifier(t) for t in taken}
    n = 0
    while f"{prefix_cf}{n}__" in taken_cf:
        n += 1
    return f"{prefix_cf}{n}__"


def inline_params(condition: str, params: tuple[Any, ...]) -> str:
    """Replace ``?`` placeholders with ``quote_value`` output.

    Fallback for backends whose API lacks PEP 249 parameter binding.
    Backends that support native parameterized queries should use those instead.
    """
    parts = condition.split("?")
    if len(parts) != len(params) + 1:
        raise ValueError(f"Placeholder count ({len(parts) - 1}) != param count ({len(params)})")
    result = parts[0]
    for part, p in zip(parts[1:], params):
        result += quote_value(p) + part
    return result
