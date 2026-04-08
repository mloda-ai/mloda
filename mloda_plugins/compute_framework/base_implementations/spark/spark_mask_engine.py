from typing import Any

from mloda.core.abstract_plugins.components.mask.base_mask_engine import BaseMaskEngine

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
    def is_in(cls, data: Any, column: str, values: Any) -> list[Any]:
        allowed = set(values) if isinstance(values, (list, tuple)) else {values}
        col_values = [row[column] for row in data.collect()]
        return [v in allowed for v in col_values]
