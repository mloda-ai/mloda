from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import ComputeFrameworkTransformer
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.runtime.flight.flight_server import FlightServer


class DataLifecycleManager:
    """
    Manages data lifecycle including dropping, result collection, and artifacts.

    This class handles the lifecycle of data in compute frameworks (CFWs),
    including tracking data to drop, collecting results, and managing artifacts.
    """

    def __init__(self, transformer: Optional[ComputeFrameworkTransformer] = None) -> None:
        """
        Initializes DataLifecycleManager with empty state and transformer.

        Args:
            transformer: Optional transformer for CFW data conversion.
                        If None, a new ComputeFrameworkTransformer is created.
        """
        self.result_data_collection: Dict[UUID, Any] = {}
        self.track_data_to_drop: Dict[UUID, Set[UUID]] = {}
        self.artifacts: Dict[str, Any] = {}
        self.transformer = transformer if transformer is not None else ComputeFrameworkTransformer()

    def drop_data_for_finished_cfws(
        self, finished_ids: Set[UUID], cfw_collection: Dict[UUID, ComputeFramework], location: Optional[str] = None
    ) -> None:
        """
        Drops data for CFWs when all their dependent steps are finished.

        Args:
            finished_ids: Set of step UUIDs that have been completed.
            cfw_collection: Dictionary of CFWs keyed by UUID.
            location: Optional location string for remote data dropping.
        """
        if not finished_ids:
            return

        cfw_to_delete = set()
        for cfw_uuid, step_uuids in self.track_data_to_drop.items():
            if all(step_id in finished_ids for step_id in step_uuids):
                self.drop_cfw_data(cfw_uuid, cfw_collection, location)
                cfw_to_delete.add(cfw_uuid)

        for cfw_uuid in cfw_to_delete:
            del self.track_data_to_drop[cfw_uuid]

    def drop_cfw_data(
        self, cfw_uuid: UUID, cfw_collection: Dict[UUID, ComputeFramework], location: Optional[str] = None
    ) -> None:
        """
        Drops data associated with a specific CFW.

        Args:
            cfw_uuid: The UUID of the CFW to drop data for.
            cfw_collection: Dictionary of CFWs keyed by UUID.
            location: Optional location string for remote data dropping.
        """
        cfw = cfw_collection[cfw_uuid]
        if location:
            cfw.drop_last_data(location)
        else:
            cfw.drop_last_data(None)

    def track_flyway_datasets(self, cfw_uuid: UUID, datasets: Set[UUID]) -> None:
        """
        Stores flyway datasets for a CFW UUID for later dropping.

        Args:
            cfw_uuid: The UUID of the CFW.
            datasets: Set of dataset UUIDs to track for dropping.
        """
        self.track_data_to_drop[cfw_uuid] = datasets

    def add_to_result_data_collection(
        self, cfw: ComputeFramework, features: FeatureSet, step_uuid: UUID, location: Optional[str] = None
    ) -> None:
        """
        Adds result data to the collection if features are requested.

        Args:
            cfw: The compute framework containing the data.
            features: The feature set to extract from the CFW.
            step_uuid: The UUID of the step to associate with the result.
            location: Optional location string for remote data access.
        """
        initial_requested_features = features.get_initial_requested_features()
        if not initial_requested_features:
            return

        result = self.get_result_data(cfw, initial_requested_features, location)
        if result is not None:
            self.result_data_collection[step_uuid] = result

    def get_result_data(
        self, cfw: ComputeFramework, selected_feature_names: Set[FeatureName], location: Optional[str] = None
    ) -> Any:
        """
        Gets result data from the compute framework.

        Args:
            cfw: The compute framework containing the data.
            selected_feature_names: Set of feature names to select.
            location: Optional location string for remote data access.

        Returns:
            The selected data from the CFW.

        Raises:
            ValueError: If CFW has no data and no location is provided.
        """
        if cfw.data is not None:
            data = cfw.data
        elif location:
            data = FlightServer.download_table(location, str(cfw.uuid))
            data = cfw.convert_flyserver_data_back(data, self.transformer)
        else:
            raise ValueError("Not implemented")

        return cfw.select_data_by_column_names(data, selected_feature_names)

    def get_results(self) -> List[Any]:
        """
        Returns list of all collected results.

        Returns:
            List of all result data.

        Raises:
            ValueError: If no results have been collected.
        """
        if not self.result_data_collection:
            raise ValueError("No results found")

        return list(self.result_data_collection.values())

    def set_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """
        Stores artifacts dictionary.

        Args:
            artifacts: Dictionary of artifacts to store.
        """
        self.artifacts = artifacts

    def get_artifacts(self) -> Dict[str, Any]:
        """
        Returns stored artifacts.

        Returns:
            Dictionary of artifacts.
        """
        return self.artifacts
