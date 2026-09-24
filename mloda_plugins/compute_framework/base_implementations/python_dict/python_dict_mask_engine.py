from typing import Any, Callable

from mloda.core.abstract_plugins.components.mask.base_mask_engine import BaseMaskEngine
from mloda.core.abstract_plugins.components.mask.null_or_nan import is_null_or_nan, split_null_or_nan
from mloda.core.abstract_plugins.components.utils import require_value_collection
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import row_count


class PythonDictMaskEngine(BaseMaskEngine):
    """Mask engine for the columnar ``dict[str, list[Any]]`` data of the PythonDict framework.

    A mask over a column absent from the data is all-False at row-count length: a missing
    column never matches, regardless of the compared value.
    """

    @classmethod
    def supported_data_type(cls) -> type[Any]:
        return dict

    @classmethod
    def all_true(cls, data: Any) -> list[Any]:
        return [True] * row_count(data)

    @classmethod
    def combine(cls, mask1: Any, mask2: Any) -> list[Any]:
        return [a and b for a, b in zip(mask1, mask2)]

    @classmethod
    def _mask(cls, data: dict[str, list[Any]], column: str, predicate: Callable[[Any], bool]) -> list[Any]:
        """Apply ``predicate`` per value; a missing column yields all-False at row-count length,
        so it never matches (even ``None`` comparisons) and ``combine``'s ``zip`` cannot truncate.
        """
        if column not in data:
            return [False] * row_count(data)
        return [predicate(v) for v in data[column]]

    @classmethod
    def equal(cls, data: Any, column: str, value: Any) -> list[Any]:
        if is_null_or_nan(value):
            return cls._mask(data, column, is_null_or_nan)
        return cls._mask(data, column, lambda v: v == value)

    @classmethod
    def greater_equal(cls, data: Any, column: str, value: Any) -> list[Any]:
        return cls._mask(data, column, lambda v: v is not None and v >= value)

    @classmethod
    def less_equal(cls, data: Any, column: str, value: Any) -> list[Any]:
        return cls._mask(data, column, lambda v: v is not None and v <= value)

    @classmethod
    def less_than(cls, data: Any, column: str, value: Any) -> list[Any]:
        return cls._mask(data, column, lambda v: v is not None and v < value)

    @classmethod
    def greater_than(cls, data: Any, column: str, value: Any) -> list[Any]:
        return cls._mask(data, column, lambda v: v is not None and v > value)

    @classmethod
    def is_in(cls, data: Any, column: str, values: Any) -> list[Any]:
        require_value_collection(values, "is_in values")
        allowed = set(values)
        present, has_null_or_nan = split_null_or_nan(allowed)
        present_set = set(present)
        if has_null_or_nan:
            return cls._mask(data, column, lambda v: v in present_set or is_null_or_nan(v))
        return cls._mask(data, column, lambda v: v in present_set)
