"""Column-semantics introspector for duckdb relations (epic #518, Phase 3b).

Reads the duckdb ``LogicalType`` of a column (via ``data.types`` / ``data.columns``)
and derives :class:`ColumnSemantics` from its lowercased string/id, mirroring the
string-based detection already used by ``_asof_time_column_is_ordered``.
"""

from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics

_NUMERIC_TOKENS = ("int", "decimal", "float", "double", "real", "numeric", "hugeint", "bigint", "tinyint", "smallint")
_TEMPORAL_TOKENS = ("date", "time", "timestamp", "interval")


def column_semantics(data: Any, column: str) -> ColumnSemantics:
    """Return the observed semantics of ``column`` in a duckdb relation."""
    logical_type = data.types[data.columns.index(column)]
    type_id = getattr(logical_type, "id", str(logical_type)).lower()

    is_numeric = any(token in type_id for token in _NUMERIC_TOKENS)
    is_temporal = any(token in type_id for token in _TEMPORAL_TOKENS)
    is_ordered = is_numeric or is_temporal

    is_tz_aware = "with time zone" in type_id or "timestamp_tz" in type_id or "timestamptz" in type_id

    unit: str | None = None
    if "timestamp" in type_id:
        if "timestamp_ns" in type_id:
            unit = "ns"
        elif "timestamp_ms" in type_id:
            unit = "ms"
        elif "timestamp_s" in type_id:
            unit = "s"
        else:
            unit = "us"

    return ColumnSemantics(
        is_ordered=is_ordered,
        is_temporal=is_temporal,
        is_numeric=is_numeric,
        unit=unit,
        is_tz_aware=is_tz_aware,
    )
