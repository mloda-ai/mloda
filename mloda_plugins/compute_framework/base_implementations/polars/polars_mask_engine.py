from typing import Any

from mloda.core.abstract_plugins.components.mask.base_mask_engine import BaseMaskEngine

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


def _require_polars() -> Any:
    if pl is None:
        raise ImportError("polars is required for PolarsMaskEngine")
    return pl


class PolarsMaskEngine(BaseMaskEngine):
    @classmethod
    def supported_data_type(cls) -> type[Any]:
        return _require_polars().DataFrame  # type: ignore[no-any-return]

    @classmethod
    def all_true(cls, data: Any) -> Any:
        return _require_polars().Series([True] * data.height, dtype=_require_polars().Boolean)

    @classmethod
    def combine(cls, mask1: Any, mask2: Any) -> Any:
        return mask1 & mask2

    @classmethod
    def equal(cls, data: Any, column: str, value: Any) -> Any:
        return data[column] == value

    @classmethod
    def greater_equal(cls, data: Any, column: str, value: Any) -> Any:
        return data[column] >= value

    @classmethod
    def less_equal(cls, data: Any, column: str, value: Any) -> Any:
        return data[column] <= value

    @classmethod
    def less_than(cls, data: Any, column: str, value: Any) -> Any:
        return data[column] < value

    @classmethod
    def greater_than(cls, data: Any, column: str, value: Any) -> Any:
        return data[column] > value

    @classmethod
    def is_in(cls, data: Any, column: str, values: Any) -> Any:
        non_null = [v for v in values if v is not None]
        mask = data[column].is_in(non_null)
        if len(non_null) != len(values):
            mask = mask | data[column].is_null()
        return mask
