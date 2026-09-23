from typing import Any

from mloda.core.abstract_plugins.components.mask.base_mask_engine import BaseMaskEngine
from mloda.core.abstract_plugins.components.mask.null_or_nan import is_null_or_nan, split_null_or_nan
from mloda.core.abstract_plugins.components.utils import require_value_collection
from mloda_plugins.compute_framework.base_implementations.polars.polars_type_semantics import (
    is_in_values,
    nan_as_null,
    null_or_nan,
)

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


def _require_polars() -> Any:
    if pl is None:
        raise ImportError("polars is required for PolarsExprMaskEngine")
    return pl


class PolarsExprMaskEngine(BaseMaskEngine):
    """Polars mask engine that returns pl.Expr boolean expressions.

    Returns lazy pl.Expr objects instead of materialized pl.Series, enabling
    use with pl.LazyFrame pipelines without forcing collection.
    data must be the LazyFrame, so a null/NaN-aware condition can read its schema.
    """

    @classmethod
    def supported_data_type(cls) -> type[Any]:
        return _require_polars().LazyFrame  # type: ignore[no-any-return]

    @classmethod
    def all_true(cls, data: Any) -> Any:
        _pl = _require_polars()
        return _pl.repeat(True, _pl.len())

    @classmethod
    def combine(cls, mask1: Any, mask2: Any) -> Any:
        return mask1 & mask2

    @classmethod
    def equal(cls, data: Any, column: str, value: Any) -> Any:
        if is_null_or_nan(value):
            dtype = data.collect_schema().get(column)
            return null_or_nan(_require_polars().col(column), dtype)
        return _require_polars().col(column).eq_missing(value)

    @classmethod
    def greater_equal(cls, data: Any, column: str, value: Any) -> Any:
        dtype = data.collect_schema().get(column)
        col = nan_as_null(_require_polars().col(column), dtype)
        return (col >= value).fill_null(False)

    @classmethod
    def less_equal(cls, data: Any, column: str, value: Any) -> Any:
        return (_require_polars().col(column) <= value).fill_null(False)

    @classmethod
    def less_than(cls, data: Any, column: str, value: Any) -> Any:
        return (_require_polars().col(column) < value).fill_null(False)

    @classmethod
    def greater_than(cls, data: Any, column: str, value: Any) -> Any:
        dtype = data.collect_schema().get(column)
        col = nan_as_null(_require_polars().col(column), dtype)
        return (col > value).fill_null(False)

    @classmethod
    def is_in(cls, data: Any, column: str, values: Any) -> Any:
        require_value_collection(values, "is_in values")
        dtype = data.collect_schema().get(column)
        present, has_null_or_nan = split_null_or_nan(values)
        mask = _require_polars().col(column).is_in(is_in_values(present, dtype)).fill_null(False)
        if has_null_or_nan:
            mask = mask | null_or_nan(_require_polars().col(column), dtype)
        return mask
