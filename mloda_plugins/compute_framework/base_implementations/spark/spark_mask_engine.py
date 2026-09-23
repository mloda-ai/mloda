from typing import Any

from mloda.core.abstract_plugins.components.mask.base_mask_engine import BaseMaskEngine
from mloda.core.abstract_plugins.components.mask.null_or_nan import is_null_or_nan, split_null_or_nan
from mloda.core.abstract_plugins.components.utils import require_value_collection

try:
    from pyspark.sql import DataFrame
except ImportError:
    DataFrame = None


class SparkMaskEngine(BaseMaskEngine):
    @classmethod
    def supported_data_type(cls) -> type[Any]:
        return DataFrame  # type: ignore[no-any-return]

    @classmethod
    def all_true(cls, data: Any) -> list[Any]:
        return [True] * data.count()

    @classmethod
    def combine(cls, mask1: Any, mask2: Any) -> list[Any]:
        return [a and b for a, b in zip(mask1, mask2)]

    @classmethod
    def equal(cls, data: Any, column: str, value: Any) -> list[Any]:
        values = [row[column] for row in data.collect()]
        if is_null_or_nan(value):
            return [is_null_or_nan(v) for v in values]
        return [v == value for v in values]

    @classmethod
    def greater_equal(cls, data: Any, column: str, value: Any) -> list[Any]:
        values = [row[column] for row in data.collect()]
        return [v is not None and v >= value for v in values]

    @classmethod
    def less_equal(cls, data: Any, column: str, value: Any) -> list[Any]:
        values = [row[column] for row in data.collect()]
        return [v is not None and v <= value for v in values]

    @classmethod
    def less_than(cls, data: Any, column: str, value: Any) -> list[Any]:
        values = [row[column] for row in data.collect()]
        return [v is not None and v < value for v in values]

    @classmethod
    def greater_than(cls, data: Any, column: str, value: Any) -> list[Any]:
        values = [row[column] for row in data.collect()]
        return [v is not None and v > value for v in values]

    @classmethod
    def is_in(cls, data: Any, column: str, values: Any) -> list[Any]:
        require_value_collection(values, "is_in values")
        allowed = set(values)
        present, has_null_or_nan = split_null_or_nan(allowed)
        present_set = set(present)
        col_values = [row[column] for row in data.collect()]
        if has_null_or_nan:
            return [v in present_set or is_null_or_nan(v) for v in col_values]
        return [v in present_set for v in col_values]
