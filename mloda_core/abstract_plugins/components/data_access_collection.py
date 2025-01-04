from typing import Any, Dict, List, Set

from mloda_core.abstract_plugins.components.hashable_dict import HashableDict


class DataAccessCollection:
    """
    This object is used to collect all the data access objects that are used in the application.
    This interface enables easy access to all the data access objects that are used in the application.
    """

    def __init__(
        self,
        files: Set[str] = set(),
        folders: Set[str] = set(),
        credential_dicts: Dict[str, Any] = {},
        initialized_connection_objects: Set[Any] = set(),
        uninitialized_connection_objects: List[Any] = [],
    ) -> None:
        self.files: Set[str] = files
        self.folders: Set[str] = folders
        self.add_credential_dict(credential_dicts)
        self.initialized_connection_objects: Set[Any] = initialized_connection_objects
        self.uninitialized_connection_objects: List[Any] = uninitialized_connection_objects

    def add_file(self, file: str) -> None:
        self.files.add(file)

    def add_folder(self, folder: str) -> None:
        self.folders.add(folder)

    def add_credential_dict(self, credential_dict: Dict[str, Any]) -> None:
        self.credential_dicts = HashableDict(credential_dict)

    def add_initialized_connection_object(self, connection_object: Any) -> None:
        self.initialized_connection_objects.add(connection_object)

    def add_uninitialized_connection_object(self, connection_object: Any) -> None:
        self.uninitialized_connection_objects.append(connection_object)
