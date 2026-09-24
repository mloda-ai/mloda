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


def nan_as_null(col: Any, dtype: Any) -> Any:
    """Turn NaN into null on a float column/expr (fill_nan raises on str and Decimal).

    A ``None`` dtype means the column is missing; skip fill_nan so polars raises its own error.
    """
    if dtype is not None and dtype.is_float():
        return col.fill_nan(None)
    return col


def null_or_nan(col: Any, dtype: Any) -> Any:
    """Return a null-or-NaN boolean mask for a column/expr, given its dtype.

    A ``None`` dtype means the column is missing; skip is_nan so polars raises its own error.
    """
    if dtype is not None and dtype.is_float():
        return col.is_null() | col.is_nan()
    return col.is_null()


def is_in_values(values: Any, dtype: Any) -> Any:
    """Wrap ``values`` for ``is_in`` against a ``pl.Decimal`` column; values the column cannot
    represent exactly never match. Pass through unchanged for every other dtype.
    """
    import polars as pl

    if isinstance(dtype, pl.Decimal):
        values = list(values)
        cast = pl.Series(values, dtype=dtype, strict=False)
        kept = [v for v, c in zip(values, cast.to_list()) if v is None or c == v]
        return pl.Series(kept, dtype=dtype).implode()
    return values
