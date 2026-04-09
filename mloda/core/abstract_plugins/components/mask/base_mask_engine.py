from abc import ABC, abstractmethod
from typing import Any


class BaseMaskEngine(ABC):
    """Abstract engine for building boolean masks over tabular data.

    Each compute framework provides a subclass implementing these primitives
    and wires it via ComputeFramework.mask_engine().
    """

    @classmethod
    @abstractmethod
    def supported_data_type(cls) -> type[Any]:
        """Return the data container type this engine handles (e.g. pa.Table, pd.DataFrame)."""
        ...

    @classmethod
    @abstractmethod
    def all_true(cls, data: Any) -> Any:
        """Return a mask of all True with length matching the number of rows in data."""
        ...

    @classmethod
    @abstractmethod
    def combine(cls, mask1: Any, mask2: Any) -> Any:
        """Combine two masks with logical AND."""
        ...

    @classmethod
    @abstractmethod
    def equal(cls, data: Any, column: str, value: Any) -> Any:
        """Return a boolean mask where data[column] == value."""
        ...

    @classmethod
    @abstractmethod
    def greater_equal(cls, data: Any, column: str, value: Any) -> Any:
        """Return a boolean mask where data[column] >= value."""
        ...

    @classmethod
    @abstractmethod
    def less_equal(cls, data: Any, column: str, value: Any) -> Any:
        """Return a boolean mask where data[column] <= value."""
        ...

    @classmethod
    @abstractmethod
    def less_than(cls, data: Any, column: str, value: Any) -> Any:
        """Return a boolean mask where data[column] < value."""
        ...

    @classmethod
    @abstractmethod
    def greater_than(cls, data: Any, column: str, value: Any) -> Any:
        """Return a boolean mask where data[column] > value."""
        ...

    @classmethod
    @abstractmethod
    def is_in(cls, data: Any, column: str, values: Any) -> Any:
        """Return a boolean mask where data[column] is in the values collection."""
        ...

    # -- Convenience methods (concrete, built from primitives above) ----------

    @classmethod
    def between(
        cls,
        data: Any,
        column: str,
        min_value: Any,
        max_value: Any,
        *,
        min_exclusive: bool = False,
        max_exclusive: bool = False,
    ) -> Any:
        """Return a boolean mask where data[column] is between min_value and max_value."""
        lo = cls.greater_than(data, column, min_value) if min_exclusive else cls.greater_equal(data, column, min_value)
        hi = cls.less_than(data, column, max_value) if max_exclusive else cls.less_equal(data, column, max_value)
        return cls.combine(lo, hi)

    @classmethod
    def all_of(cls, data: Any, masks: list[Any]) -> Any:
        """Combine a list of masks with logical AND. Returns all-True mask if list is empty."""
        if len(masks) == 0:
            return cls.all_true(data)
        out = masks[0]
        for m in masks[1:]:
            out = cls.combine(out, m)
        return out
