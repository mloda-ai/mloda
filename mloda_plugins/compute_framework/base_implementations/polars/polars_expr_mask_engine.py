from typing import Any

from mloda.core.abstract_plugins.components.mask.base_mask_engine import BaseMaskEngine

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
        return _require_polars().col(column) == value

    @classmethod
    def greater_equal(cls, data: Any, column: str, value: Any) -> Any:
        return _require_polars().col(column) >= value

    @classmethod
    def less_equal(cls, data: Any, column: str, value: Any) -> Any:
        return _require_polars().col(column) <= value

    @classmethod
    def less_than(cls, data: Any, column: str, value: Any) -> Any:
        return _require_polars().col(column) < value

    @classmethod
    def greater_than(cls, data: Any, column: str, value: Any) -> Any:
        return _require_polars().col(column) > value

    @classmethod
    def is_in(cls, data: Any, column: str, values: Any) -> Any:
        return _require_polars().col(column).is_in(values)
