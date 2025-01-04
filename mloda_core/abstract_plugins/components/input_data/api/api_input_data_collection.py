from typing import Dict, Tuple, Type
from mloda_core.abstract_plugins.components.hashable_dict import HashableDict
from mloda_core.abstract_plugins.components.input_data.api.base_api_data import BaseApiData


class ApiInputDataCollection:
    """
    Manages a collection of API input data classes.

    This class maintains a registry of API input data subclasses, allowing for the registration
    and retrieval of API input data classes based on their names and associated column names.
    """

    def __init__(self, registry: Dict[str, Type[BaseApiData]] = {}) -> None:
        self._registry: Dict[str, Type[BaseApiData]] = registry

    def register(self, name: str, api_data_cls: Type[BaseApiData]) -> None:
        """Register additional ApiData subclass with a unique name."""
        if name in self._registry:
            raise ValueError(f"An ApiData with name '{name}' is already registered.")
        self._registry[name] = api_data_cls

    def get_column_names(self) -> HashableDict:
        """Get column names for all registered ApiData."""
        columns = HashableDict({})
        for name, cls in self._registry.items():
            columns.data[name] = tuple(cls.column_names())
        return columns

    def get_name_cls_by_matching_column_name(self, column_name: str) -> Tuple[str, Type[BaseApiData]]:
        """Get the ApiData class by matching column name."""
        for name, cls in self._registry.items():
            if column_name in cls.column_names():
                return name, cls
        raise ValueError(f"Column name {column_name} not found in any registered ApiData: {self._registry.keys()}.")
