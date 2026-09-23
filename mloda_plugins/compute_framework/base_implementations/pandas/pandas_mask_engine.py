from typing import Any

from mloda.core.abstract_plugins.components.mask.base_mask_engine import BaseMaskEngine
from mloda.core.abstract_plugins.components.mask.null_or_nan import is_null_or_nan, split_null_or_nan
from mloda.core.abstract_plugins.components.utils import require_value_collection
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_type_semantics import null_or_nan_mask

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
        if is_null_or_nan(value):
            return null_or_nan_mask(data[column])
        return (data[column] == value).fillna(False).astype(bool)

    @classmethod
    def greater_equal(cls, data: Any, column: str, value: Any) -> Any:
        return (data[column] >= value).fillna(False).astype(bool)

    @classmethod
    def less_equal(cls, data: Any, column: str, value: Any) -> Any:
        return (data[column] <= value).fillna(False).astype(bool)

    @classmethod
    def less_than(cls, data: Any, column: str, value: Any) -> Any:
        return (data[column] < value).fillna(False).astype(bool)

    @classmethod
    def greater_than(cls, data: Any, column: str, value: Any) -> Any:
        return (data[column] > value).fillna(False).astype(bool)

    @classmethod
    def is_in(cls, data: Any, column: str, values: Any) -> Any:
        require_value_collection(values, "is_in values")
        present, has_null_or_nan = split_null_or_nan(values)
        mask = data[column].isin(present).astype(bool)
        if has_null_or_nan:
            mask = mask | null_or_nan_mask(data[column])
        return mask
