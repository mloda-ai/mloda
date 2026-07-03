"""Bounded value sampling for sqlite ISO-TEXT datetime introspection (epic #518, follow-up).

sqlite stores datetimes as ISO-8601 TEXT, so schema-only introspection reports such
columns as plain strings. Sampling a bounded set of non-null values lets the filter/merge
engines classify ISO-TEXT datetime columns as temporal via value-inspection. Sampling is
staged: a single LIMIT 1 probe fast-rejects non-ISO string columns before the full LIMIT 100
scan is issued.
"""

from typing import Any

from mloda.core.abstract_plugins.components.contract.value_inspection import is_iso8601_string
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

_SAMPLE_LIMIT = 100


def sample_string_values(data: Any, column: str) -> list[Any]:
    """Fetch non-null values of ``column``, probing with LIMIT 1 before the full LIMIT 100 scan.

    A non-ISO or empty probe returns after one query; only an ISO-8601 probe value issues the
    fuller sample query.
    """
    qc = quote_ident(column)
    table_ref = quote_ident(data.table_name)
    probe_sql = f"SELECT {qc} FROM {table_ref} WHERE {qc} IS NOT NULL LIMIT 1"  # nosec
    probe_rows = data.connection.execute(probe_sql).fetchall()
    if not probe_rows:
        return []
    first_value = probe_rows[0][0]
    if not is_iso8601_string(first_value):
        return []
    sql = f"SELECT {qc} FROM {table_ref} WHERE {qc} IS NOT NULL LIMIT {_SAMPLE_LIMIT}"  # nosec
    rows = data.connection.execute(sql).fetchall()
    return [row[0] for row in rows]
