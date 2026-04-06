from typing import Any

from mloda.core.filter.filter_mask_engine import BaseFilterMaskEngine


class PythonDictFilterMaskEngine(BaseFilterMaskEngine):
    @classmethod
    def supported_data_type(cls) -> type[Any]:
        return list

    @classmethod
    def all_true(cls, data: Any) -> list[Any]:
        return [True] * len(data)

    @classmethod
    def combine(cls, mask1: Any, mask2: Any) -> list[Any]:
        return [a and b for a, b in zip(mask1, mask2)]

    @classmethod
    def equal(cls, data: Any, column: str, value: Any) -> list[Any]:
        return [row.get(column) == value for row in data]

    @classmethod
    def greater_equal(cls, data: Any, column: str, value: Any) -> list[Any]:
        return [row.get(column) is not None and row.get(column) >= value for row in data]

    @classmethod
    def less_equal(cls, data: Any, column: str, value: Any) -> list[Any]:
        return [row.get(column) is not None and row.get(column) <= value for row in data]

    @classmethod
    def less_than(cls, data: Any, column: str, value: Any) -> list[Any]:
        return [row.get(column) is not None and row.get(column) < value for row in data]

    @classmethod
    def is_in(cls, data: Any, column: str, values: Any) -> list[Any]:
        allowed = set(values) if isinstance(values, (list, tuple)) else {values}
        return [row.get(column) in allowed for row in data]
