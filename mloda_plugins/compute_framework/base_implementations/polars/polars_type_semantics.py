"""Column-semantics introspector for polars frames (epic #518, Phase 1).

Works for both ``pl.DataFrame`` and ``pl.LazyFrame`` via ``collect_schema``.
"""

from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics


def column_semantics(df: Any, column: str) -> ColumnSemantics:
    """Return the observed semantics of ``column`` in a polars DataFrame or LazyFrame."""
    import polars as pl

    dtype = df.collect_schema()[column]

    is_numeric = bool(dtype.is_numeric())
    is_temporal = bool(dtype.is_temporal())
    is_ordered = is_numeric or is_temporal

    unit: str | None = None
    is_tz_aware = False
    if isinstance(dtype, pl.Datetime):
        unit = dtype.time_unit
        is_tz_aware = dtype.time_zone is not None

    return ColumnSemantics(
        is_ordered=is_ordered,
        is_temporal=is_temporal,
        is_numeric=is_numeric,
        unit=unit,
        is_tz_aware=is_tz_aware,
    )
