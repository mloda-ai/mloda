from abc import ABC, abstractmethod
from typing import Any


class BaseFilterMaskEngine(ABC):
    """Abstract engine for building boolean masks over tabular data.

    Each compute framework provides a subclass implementing these primitives
    and wires it via ComputeFramework.filter_mask_engine().
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
    def is_in(cls, data: Any, column: str, values: Any) -> Any:
        """Return a boolean mask where data[column] is in the values collection."""
        ...
