from multiprocessing.managers import BaseManager
from typing import Any, Dict, Optional, Set, Tuple, Type
from uuid import UUID

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.abstract_plugins.function_extender import WrapperFunctionExtender
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes

import logging


logger = logging.getLogger(__name__)


class MyManager(BaseManager):
    pass


class CfwManager:
    """
    Manages Compute Frameworks (CFWs) and related data.

    This class handles the registration, merging, and retrieval of Compute Frameworks,
    along with managing multiprocessing resources and error tracking.  It aims to
    centralize and simplify the management of CFWs within the mloda core.
    """

    def __init__(
        self,
        parallelization_modes: Set[ParallelizationModes],
        function_extender: Optional[Set[WrapperFunctionExtender]] = None,
    ) -> None:
        """
        Initializes the CfwManager.

        Args:
            parallelization_modes: The set of parallelization modes to use.
            function_extender: Optional set of function extenders.
        """
        self.parallelization_modes = parallelization_modes
        self.function_extender = function_extender

        self.compute_frameworks: Dict[
            UUID, Tuple[str, Set[UUID]]
        ] = {}  # cfw uuid -> (cfw class name, children_if_root)
        self.cfw_merge_relation: Dict[UUID, Tuple[UUID, str]] = {}  # merge relation

        self.location: Optional[str] = None  # multiprocessing location
        self.error = False  # multiprocessing error flag
        self.msg: Any = None
        self.exc_info: Any = None

        self.uuid_column_names: Dict[UUID, Set[str]] = {}  # We only set this in case of TransformFrameworkStep
        self.uuid_flyway_datasets: Dict[UUID, Set[UUID]] = {}

        self.artifact_to_save: Dict[str, Any] = {}

        self.api_data: Optional[Dict[str, Any]] = None

    def add_uuid_flyway_datasets(self, cf_uuid: UUID, object_ids: Set[UUID]) -> None:
        """Associates a set of Flyway dataset UUIDs with a Compute Framework UUID."""
        self.uuid_flyway_datasets[cf_uuid] = object_ids

    def get_uuid_flyway_datasets(self, cf_uuid: UUID) -> Optional[Set[UUID]]:
        """Retrieves the set of Flyway dataset UUIDs associated with a Compute Framework UUID."""
        return self.uuid_flyway_datasets.get(cf_uuid, None)

    def add_column_names_to_cf_uuid(self, cf_uuid: UUID, column_names: Set[str]) -> None:
        """Associates a set of column names with a Compute Framework UUID."""
        self.uuid_column_names[cf_uuid] = column_names

    def get_column_names(self, cf_uuid: UUID) -> Set[str]:
        """Retrieves the set of column names associated with a Compute Framework UUID."""
        return self.uuid_column_names[cf_uuid]

    def get_cfw_uuid(
        self,
        cf_class_name: str,
        feature_uuid: UUID,
    ) -> Optional[UUID]:
        """
        Retrieves the UUID of a Compute Framework based on its class name and a feature UUID.

        Usually, the feature UUID is a parent of the current feature.

        Args:
            cf_class_name: The class name of the Compute Framework.
            feature_uuid: The UUID of the feature.

        Returns:
            The UUID of the Compute Framework, or None if not found.
        """
        for cfw_uuid, value in self.compute_frameworks.items():
            cls_name, children_if_root = value
            if cf_class_name == cls_name and feature_uuid in children_if_root:
                cfw_uuid = self.find_leftmost(cfw_uuid, cls_name)
                return cfw_uuid
        return None

    def add_to_merge_relation(self, left_uuid: UUID, right_uuid: UUID, cls_name: str) -> None:
        """
        Adds a merge relation between two Compute Framework UUIDs.

        Args:
            left_uuid: The UUID of the left Compute Framework.
            right_uuid: The UUID of the right Compute Framework.
            cls_name: The class name of the Compute Framework.
        """
        self.cfw_merge_relation[right_uuid] = (left_uuid, cls_name)

        if left_uuid not in self.cfw_merge_relation:
            self.cfw_merge_relation[left_uuid] = (left_uuid, cls_name)

    def find_leftmost(self, uuid: UUID, cls_name: str) -> UUID:
        """
        Finds the leftmost UUID in a merge relation chain.

        Args:
            uuid: The starting UUID.
            cls_name: The class name of the Compute Framework.

        Returns:
            The leftmost UUID in the chain.
        """
        if uuid not in self.cfw_merge_relation:
            return uuid

        leftmost_uuid = uuid

        while self.cfw_merge_relation[uuid][0] != uuid:
            uuid = self.cfw_merge_relation[uuid][0]

            if self.cfw_merge_relation[uuid][1] == cls_name:
                leftmost_uuid = uuid
        return leftmost_uuid

    def add_cfw_to_compute_frameworks(self, uuid: UUID, cls_name: str, children_if_root: Set[UUID]) -> None:
        """
        Adds a Compute Framework to the registered frameworks.

        Args:
            uuid: The UUID of the Compute Framework.
            cls_name: The class name of the Compute Framework.
            children_if_root: The set of child UUIDs if the CFW is a root..
        """
        if self.compute_frameworks.get(uuid):
            raise ValueError(f"UUID {uuid} already exists in compute_frameworks")
        self.compute_frameworks[uuid] = (cls_name, children_if_root)

    def get_initialized_compute_framework_uuid(self, cf_class: Type[ComputeFrameWork], feature_uuid: UUID) -> UUID:
        """
        Retrieves the UUID of an initialized Compute Framework.

        Args:
            cf_class: The class of the Compute Framework.
            feature_uuid: The UUID of the feature.

        Returns:
            The UUID of the Compute Framework.
        """
        cfw_uuid = self.get_cfw_uuid(cf_class.get_class_name(), feature_uuid)

        if cfw_uuid is None:
            raise ValueError("No compute framework registered.")
        return cfw_uuid

    def set_location(self, location: str) -> None:
        """Sets the location for multiprocessing."""
        if not self.location:
            self.location = location

    def get_location(self) -> Optional[str]:
        """Retrieves the location for multiprocessing."""
        return self.location

    def get_parallelization_modes(self) -> Set[ParallelizationModes]:
        """Retrieves the set of parallelization modes."""
        return self.parallelization_modes

    def set_error(self, msg: Any, exc_info: Any) -> None:
        """Sets an error message and exception information."""
        self.error = True
        self.msg = msg
        self.exc_info = exc_info

    def get_error(self) -> bool:
        """Retrieves the error flag."""
        return self.error

    def get_error_msg(self) -> Any:
        """Retrieves the error message."""
        return self.msg

    def get_error_exc_info(self) -> Any:
        """Retrieves the exception information."""
        return self.exc_info

    def get_compute_frameworks(self) -> Dict[UUID, Tuple[str, Set[UUID]]]:
        """Retrieves the dictionary of compute frameworks."""
        return self.compute_frameworks

    def get_function_extender(self) -> Optional[Set[WrapperFunctionExtender]]:
        """Retrieves the optional set of function extenders."""
        return self.function_extender

    def set_artifact_to_save(self, artifact_name: str, artifact: Any) -> None:
        """
        Saves an artifact or meta-information to the artifact_to_save dictionary.

        Args:
            artifact_name: The name of the artifact.
            artifact: The artifact to save.
        """
        if artifact_name in self.artifact_to_save:
            raise ValueError(f"Artifact name {artifact_name} already exists.")

        self.artifact_to_save[artifact_name] = artifact

    def get_artifacts(self) -> Dict[str, Any]:
        """Retrieves the dictionary of saved artifacts."""
        return self.artifact_to_save

    def set_api_data(self, api_data: Dict[str, Any]) -> None:
        """Sets the API data."""
        self.api_data = api_data

    def get_api_data_by_name(self, key: str) -> Optional[Any]:
        """
        Retrieves API data by name.

        Args:
            key: The name of the API data.

        Returns:
            The API data, or None if not found.
        """
        if self.api_data is None:
            raise ValueError("No api data set.")

        api_data = self.api_data.get(key, None)

        if api_data is None:
            raise ValueError(f"Api data with key {key} not found.")

        return api_data
