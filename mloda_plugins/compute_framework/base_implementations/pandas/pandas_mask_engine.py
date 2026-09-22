from typing import Any

from mloda.core.abstract_plugins.components.mask.base_mask_engine import BaseMaskEngine

try:
    import pandas as pd
except ImportError:
    pd = None


class PandasMaskEngine(BaseMaskEngine):
    @classmethod
    def supported_data_type(cls) -> type[Any]:
        return pd.DataFrame  # type: ignore[no-any-return]

    @classmethod
    def all_true(cls, data: Any) -> Any:
        return pd.Series(True, index=data.index, dtype=bool)

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
        mask = data[column].isin(non_null)
        if len(non_null) != len(values):
            mask = mask | data[column].isna()
        return mask
