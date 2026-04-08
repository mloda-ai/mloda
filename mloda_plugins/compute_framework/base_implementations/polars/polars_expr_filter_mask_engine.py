from typing import Any

from mloda.core.filter.filter_mask_engine import BaseFilterMaskEngine

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


class PolarsExprFilterMaskEngine(BaseFilterMaskEngine):
    """Polars filter mask engine that returns pl.Expr boolean expressions.

    Returns lazy pl.Expr objects instead of materialized pl.Series, enabling
    use with pl.LazyFrame pipelines without forcing collection.
    """

    @classmethod
    def supported_data_type(cls) -> type[Any]:
        return pl.LazyFrame

    @classmethod
    def all_true(cls, data: Any) -> Any:
        return pl.repeat(True, pl.len())

    @classmethod
    def combine(cls, mask1: Any, mask2: Any) -> Any:
        return mask1 & mask2

    @classmethod
    def equal(cls, data: Any, column: str, value: Any) -> Any:
        return pl.col(column) == value

    @classmethod
    def greater_equal(cls, data: Any, column: str, value: Any) -> Any:
        return pl.col(column) >= value

    @classmethod
    def less_equal(cls, data: Any, column: str, value: Any) -> Any:
        return pl.col(column) <= value

    @classmethod
    def less_than(cls, data: Any, column: str, value: Any) -> Any:
        return pl.col(column) < value

    @classmethod
    def is_in(cls, data: Any, column: str, values: Any) -> Any:
        return pl.col(column).is_in(values)
