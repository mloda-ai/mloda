"""Column-semantics introspector for pandas DataFrames (epic #518, Phase 1)."""

from typing import TYPE_CHECKING

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics

if TYPE_CHECKING:
    import pandas as pd


def column_semantics(df: "pd.DataFrame", column: str) -> ColumnSemantics:
    """Return the observed semantics of ``column`` in a pandas DataFrame."""
    import pandas as pd

    dtype = df[column].dtype

    from mloda_plugins.compute_framework.base_implementations.sql.sql_type_semantics import column_semantics_from_arrow

    if isinstance(dtype, pd.ArrowDtype):
        return column_semantics_from_arrow(dtype.pyarrow_dtype)

    is_bool = pd.api.types.is_bool_dtype(dtype)
    is_numeric = bool(pd.api.types.is_numeric_dtype(dtype)) and not is_bool
    is_datetime = bool(pd.api.types.is_datetime64_any_dtype(dtype))
    is_timedelta = bool(pd.api.types.is_timedelta64_dtype(dtype))

    is_temporal = is_datetime
    is_ordered = (is_numeric or is_temporal or is_timedelta) and not is_bool
    is_tz_aware = isinstance(dtype, pd.DatetimeTZDtype)

    unit: str | None = None
    if is_datetime:
        unit = getattr(dtype, "unit", None)
        if unit is None:
            text = str(dtype)
            for candidate in ("ns", "us", "ms", "s"):
                if f"[{candidate}" in text:
                    unit = candidate
                    break

    return ColumnSemantics(
        is_ordered=is_ordered,
        is_temporal=is_temporal,
        is_numeric=is_numeric,
        unit=unit,
        is_tz_aware=is_tz_aware,
    )
