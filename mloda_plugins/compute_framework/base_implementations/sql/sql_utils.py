"""SQL quoting utilities shared by all SQL-based compute frameworks.

SQL injection prevention follows two layers:

- Identifiers (column names, table names, aliases) go through ``quote_ident``
  which applies SQL-standard double-quote escaping.
- Literal values should use PEP 249 (DB-API 2.0) parameterized queries
  whenever the backend supports them. ``quote_value`` and ``inline_params``
  exist as a fallback for backends whose API lacks PEP 249 parameter binding.
  They accept only (None, bool, int, float, str); unsupported types raise.
"""

import math
from typing import Any, Tuple


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
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    raise TypeError(f"Unsupported type for SQL literal: {type(value).__name__}")


def inline_params(condition: str, params: Tuple[Any, ...]) -> str:
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
