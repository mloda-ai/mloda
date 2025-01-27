from typing import Any, Dict, List, Optional, Set, Tuple, Type
from mloda_core.abstract_plugins.components.hashable_dict import HashableDict
from mloda_core.abstract_plugins.components.input_data.api.base_api_data import BaseApiData


class ApiInputDataCollection:
    """
    Manages a collection of API input data classes.

    This class maintains a registry of API input data subclasses, allowing for the registration
    and retrieval of API input data classes based on their names and associated column names.
    """

    def __init__(self, registry: Optional[Dict[str, Type[BaseApiData]]] = None) -> None:
        self._registry: Dict[str, Type[BaseApiData]] = registry if registry is not None else {}

        # Keep track of used key names to avoid misalignment
        self._used_key_names: Set[str] = set()

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

    def setup_key_class(self, key_name: str, column_names: List[str]) -> None:
        """
        Setup a key class with the given column names.

        This convenience functionality creates automatically the matching BaseApiData class for the given key_name.
        """

        if key_name in self._used_key_names:
            raise ValueError(f"Key name '{key_name}' is already used. This Key name must be unique.")

        class_name = f"DynamicApiCls_{key_name}"

        DynamicApiCls = type(
            class_name,
            (BaseApiData,),
            {
                "column_names": classmethod(lambda cls: column_names),
            },
        )

        self.register(key_name, DynamicApiCls)

    def setup_key_api_data(self, key_name: str, api_input_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Setup a key API data with the given data.

        This convenience functionality creates automatically the matching BaseApiData class for the given key_name.

        We return the api_data.
        """

        self.setup_key_class(key_name, list(api_input_data.keys()))

        api_data = {
            key_name: api_input_data,
        }
        return api_data
