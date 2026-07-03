"""Bounded value sampling for sqlite ISO-TEXT datetime introspection (epic #518, follow-up).

sqlite stores datetimes as ISO-8601 TEXT, so schema-only introspection reports such
columns as plain strings. Sampling a bounded set of non-null values lets the filter/merge
engines classify ISO-TEXT datetime columns as temporal via value-inspection.
"""

from typing import Any

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

_SAMPLE_LIMIT = 100


def sample_string_values(data: Any, column: str) -> list[Any]:
    """Fetch up to ``_SAMPLE_LIMIT`` non-null values of ``column`` from the sqlite relation."""
    sql = (
        f"SELECT {quote_ident(column)} FROM {quote_ident(data.table_name)} "  # nosec
        f"WHERE {quote_ident(column)} IS NOT NULL LIMIT {_SAMPLE_LIMIT}"
    )
    rows = data.connection.execute(sql).fetchall()
    return [row[0] for row in rows]
