from typing import Any

from mloda.core.abstract_plugins.components.mask.base_mask_engine import BaseMaskEngine


class PythonDictMaskEngine(BaseMaskEngine):
    @classmethod
    def supported_data_type(cls) -> type[Any]:
        return dict

    @staticmethod
    def _row_count(data: dict[str, list[Any]]) -> int:
        for column in data.values():
            return len(column)
        return 0

    @classmethod
    def all_true(cls, data: Any) -> list[Any]:
        return [True] * cls._row_count(data)

    @classmethod
    def combine(cls, mask1: Any, mask2: Any) -> list[Any]:
        return [a and b for a, b in zip(mask1, mask2)]

    @classmethod
    def equal(cls, data: Any, column: str, value: Any) -> list[Any]:
        return [v == value for v in data.get(column, [])]

    @classmethod
    def greater_equal(cls, data: Any, column: str, value: Any) -> list[Any]:
        return [v is not None and v >= value for v in data.get(column, [])]

    @classmethod
    def less_equal(cls, data: Any, column: str, value: Any) -> list[Any]:
        return [v is not None and v <= value for v in data.get(column, [])]

    @classmethod
    def less_than(cls, data: Any, column: str, value: Any) -> list[Any]:
        return [v is not None and v < value for v in data.get(column, [])]

    @classmethod
    def greater_than(cls, data: Any, column: str, value: Any) -> list[Any]:
        return [v is not None and v > value for v in data.get(column, [])]

    @classmethod
    def is_in(cls, data: Any, column: str, values: Any) -> list[Any]:
        allowed = set(values) if isinstance(values, (list, tuple)) else {values}
        return [v in allowed for v in data.get(column, [])]
