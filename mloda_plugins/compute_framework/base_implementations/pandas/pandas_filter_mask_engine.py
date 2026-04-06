from typing import Any

import pandas as pd

from mloda.core.filter.filter_mask_engine import BaseFilterMaskEngine


class PandasFilterMaskEngine(BaseFilterMaskEngine):
    @classmethod
    def supported_data_type(cls) -> type[Any]:
        return pd.DataFrame  # type: ignore[no-any-return]

    @classmethod
    def all_true(cls, data: Any) -> Any:
        return pd.Series([True] * len(data), index=data.index)

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
    def is_in(cls, data: Any, column: str, values: Any) -> Any:
        return data[column].isin(values)
