# mypy: disable-error-code="arg-type, unused-ignore"
"""Tests for DataLifecycleManager class that manages data dropping, result collection, and artifacts."""

from typing import Dict
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

import pytest

from mloda.core.runtime.data_lifecycle_manager import DataLifecycleManager
from mloda import ComputeFramework
from mloda.provider import FeatureSet
from mloda.user import FeatureName
from mloda.provider import ComputeFrameworkTransformer


class TestDataLifecycleManagerInit:
    """Test DataLifecycleManager initialization."""

    def test_init_creates_empty_state_with_no_transformer(self) -> None:
        """DataLifecycleManager should initialize with empty collections and create transformer."""
        manager = DataLifecycleManager()

        assert manager.result_data_collection == {}
        assert manager.track_data_to_drop == {}
        assert manager.artifacts == {}
        assert isinstance(manager.transformer, ComputeFrameworkTransformer)

    def test_init_accepts_transformer_argument(self) -> None:
        """DataLifecycleManager should accept custom transformer on initialization."""
        custom_transformer = Mock(spec=ComputeFrameworkTransformer)
        manager = DataLifecycleManager(transformer=custom_transformer)

        assert manager.transformer == custom_transformer

    def test_init_creates_transformer_if_not_provided(self) -> None:
        """DataLifecycleManager should create transformer if none provided."""
        manager = DataLifecycleManager(transformer=None)

        assert isinstance(manager.transformer, ComputeFrameworkTransformer)


class TestDataLifecycleManagerDropDataForFinishedCfws:
    """Test dropping data for finished CFWs."""

    def test_drop_data_for_finished_cfws_drops_when_all_dependent_steps_finished(self) -> None:
        """Should drop CFW data when all dependent steps are finished."""
        manager = DataLifecycleManager()
        cfw_uuid = uuid4()
        step_uuid1 = uuid4()
        step_uuid2 = uuid4()

        manager.track_data_to_drop[cfw_uuid] = {step_uuid1, step_uuid2}

        mock_cfw = Mock(spec=ComputeFramework)
        mock_cfw.uuid = cfw_uuid
        cfw_collection = {cfw_uuid: mock_cfw}

        finished_ids = {step_uuid1, step_uuid2}

        manager.drop_data_for_finished_cfws(finished_ids, cfw_collection)

        # Should drop data for the CFW
        mock_cfw.drop_last_data.assert_called_once_with(None)
        # Should remove from tracking
        assert cfw_uuid not in manager.track_data_to_drop

    def test_drop_data_for_finished_cfws_does_not_drop_when_steps_incomplete(self) -> None:
        """Should not drop CFW data when not all dependent steps are finished."""
        manager = DataLifecycleManager()
        cfw_uuid = uuid4()
        step_uuid1 = uuid4()
        step_uuid2 = uuid4()

        manager.track_data_to_drop[cfw_uuid] = {step_uuid1, step_uuid2}

        mock_cfw = Mock(spec=ComputeFramework)
        mock_cfw.uuid = cfw_uuid
        cfw_collection = {cfw_uuid: mock_cfw}

        finished_ids = {step_uuid1}  # Only one step finished

        manager.drop_data_for_finished_cfws(finished_ids, cfw_collection)

        # Should not drop data
        mock_cfw.drop_last_data.assert_not_called()
        # Should keep in tracking
        assert cfw_uuid in manager.track_data_to_drop

    def test_drop_data_for_finished_cfws_handles_empty_finished_ids(self) -> None:
        """Should handle empty finished_ids set gracefully."""
        manager = DataLifecycleManager()
        cfw_uuid = uuid4()
        step_uuid1 = uuid4()

        manager.track_data_to_drop[cfw_uuid] = {step_uuid1}

        mock_cfw = Mock(spec=ComputeFramework)
        cfw_collection = {cfw_uuid: mock_cfw}

        manager.drop_data_for_finished_cfws(set(), cfw_collection)

        # Should not drop or modify anything
        mock_cfw.drop_last_data.assert_not_called()
        assert cfw_uuid in manager.track_data_to_drop

    def test_drop_data_for_finished_cfws_handles_multiple_cfws(self) -> None:
        """Should handle dropping multiple CFWs correctly."""
        manager = DataLifecycleManager()
        cfw_uuid1 = uuid4()
        cfw_uuid2 = uuid4()
        step_uuid1 = uuid4()
        step_uuid2 = uuid4()
        step_uuid3 = uuid4()

        manager.track_data_to_drop[cfw_uuid1] = {step_uuid1, step_uuid2}
        manager.track_data_to_drop[cfw_uuid2] = {step_uuid3}

        mock_cfw1 = Mock(spec=ComputeFramework)
        mock_cfw2 = Mock(spec=ComputeFramework)
        cfw_collection = {cfw_uuid1: mock_cfw1, cfw_uuid2: mock_cfw2}

        finished_ids = {step_uuid1, step_uuid2, step_uuid3}

        manager.drop_data_for_finished_cfws(finished_ids, cfw_collection)

        # Both should be dropped
        mock_cfw1.drop_last_data.assert_called_once_with(None)
        mock_cfw2.drop_last_data.assert_called_once_with(None)
        # Both should be removed from tracking
        assert cfw_uuid1 not in manager.track_data_to_drop
        assert cfw_uuid2 not in manager.track_data_to_drop

    def test_drop_data_for_finished_cfws_with_location(self) -> None:
        """Should pass location to drop_last_data when provided."""
        manager = DataLifecycleManager()
        cfw_uuid = uuid4()
        step_uuid = uuid4()
        location = "grpc://localhost:8815"

        manager.track_data_to_drop[cfw_uuid] = {step_uuid}

        mock_cfw = Mock(spec=ComputeFramework)
        cfw_collection = {cfw_uuid: mock_cfw}

        finished_ids = {step_uuid}

        manager.drop_data_for_finished_cfws(finished_ids, cfw_collection, location=location)

        mock_cfw.drop_last_data.assert_called_once_with(location)


class TestDataLifecycleManagerDropCfwData:
    """Test dropping data for a specific CFW."""

    def test_drop_cfw_data_calls_drop_last_data_without_location(self) -> None:
        """Should call drop_last_data on CFW when location is None."""
        manager = DataLifecycleManager()
        cfw_uuid = uuid4()

        mock_cfw = Mock(spec=ComputeFramework)
        mock_cfw.uuid = cfw_uuid
        cfw_collection = {cfw_uuid: mock_cfw}

        manager.drop_cfw_data(cfw_uuid, cfw_collection)

        mock_cfw.drop_last_data.assert_called_once_with(None)

    def test_drop_cfw_data_passes_location_to_drop_last_data(self) -> None:
        """Should pass location to drop_last_data when provided."""
        manager = DataLifecycleManager()
        cfw_uuid = uuid4()
        location = "grpc://localhost:8815"

        mock_cfw = Mock(spec=ComputeFramework)
        cfw_collection = {cfw_uuid: mock_cfw}

        manager.drop_cfw_data(cfw_uuid, cfw_collection, location=location)

        mock_cfw.drop_last_data.assert_called_once_with(location)

    def test_drop_cfw_data_handles_missing_cfw_gracefully(self) -> None:
        """Should handle missing CFW UUID in collection."""
        manager = DataLifecycleManager()
        cfw_uuid = uuid4()
        cfw_collection: Dict[UUID, ComputeFramework] = {}

        # Should raise KeyError when CFW not in collection
        with pytest.raises(KeyError):
            manager.drop_cfw_data(cfw_uuid, cfw_collection)


class TestDataLifecycleManagerTrackFlywayDatasets:
    """Test tracking flyway datasets for data dropping."""

    def test_track_flyway_datasets_stores_datasets_for_cfw(self) -> None:
        """Should store flyway datasets for a CFW UUID."""
        manager = DataLifecycleManager()
        cfw_uuid = uuid4()
        dataset_uuid1 = uuid4()
        dataset_uuid2 = uuid4()
        datasets = {dataset_uuid1, dataset_uuid2}

        manager.track_flyway_datasets(cfw_uuid, datasets)

        assert manager.track_data_to_drop[cfw_uuid] == datasets

    def test_track_flyway_datasets_overwrites_existing_tracking(self) -> None:
        """Should overwrite existing tracking for a CFW UUID."""
        manager = DataLifecycleManager()
        cfw_uuid = uuid4()
        old_dataset = uuid4()
        new_dataset1 = uuid4()
        new_dataset2 = uuid4()

        manager.track_data_to_drop[cfw_uuid] = {old_dataset}
        manager.track_flyway_datasets(cfw_uuid, {new_dataset1, new_dataset2})

        assert manager.track_data_to_drop[cfw_uuid] == {new_dataset1, new_dataset2}

    def test_track_flyway_datasets_handles_empty_set(self) -> None:
        """Should handle empty set of datasets."""
        manager = DataLifecycleManager()
        cfw_uuid = uuid4()

        manager.track_flyway_datasets(cfw_uuid, set())

        assert manager.track_data_to_drop[cfw_uuid] == set()


class TestDataLifecycleManagerAddToResultDataCollection:
    """Test adding result data to the collection."""

    def test_add_to_result_data_collection_stores_result_when_requested_features_exist(self) -> None:
        """Should store result data when CFW has requested features."""
        manager = DataLifecycleManager()
        step_uuid = uuid4()

        mock_cfw = Mock(spec=ComputeFramework)
        mock_cfw.uuid = uuid4()
        mock_cfw.data = {"feature1": [1, 2, 3]}
        mock_cfw.select_data_by_column_names.return_value = {"feature1": [1, 2, 3]}

        feature1 = Mock()
        feature1.name = Mock(spec=FeatureName)
        feature1.initial_requested_data = True

        mock_features = Mock(spec=FeatureSet)
        mock_features.get_initial_requested_features.return_value = {feature1.name}

        manager.add_to_result_data_collection(mock_cfw, mock_features, step_uuid)

        assert step_uuid in manager.result_data_collection
        assert manager.result_data_collection[step_uuid] == {"feature1": [1, 2, 3]}

    def test_add_to_result_data_collection_skips_when_no_requested_features(self) -> None:
        """Should not store result when no features are requested."""
        manager = DataLifecycleManager()
        step_uuid = uuid4()

        mock_cfw = Mock(spec=ComputeFramework)
        mock_features = Mock(spec=FeatureSet)
        mock_features.get_initial_requested_features.return_value = set()

        manager.add_to_result_data_collection(mock_cfw, mock_features, step_uuid)

        assert step_uuid not in manager.result_data_collection

    def test_add_to_result_data_collection_calls_get_result_data(self) -> None:
        """Should call get_result_data to retrieve data from CFW."""
        manager = DataLifecycleManager()
        step_uuid = uuid4()

        mock_cfw = Mock(spec=ComputeFramework)
        mock_cfw.uuid = uuid4()
        mock_cfw.data = {"feature1": [1, 2, 3]}

        feature_name = Mock(spec=FeatureName)
        mock_features = Mock(spec=FeatureSet)
        mock_features.get_initial_requested_features.return_value = {feature_name}

        with patch.object(manager, "get_result_data", return_value={"feature1": [1, 2, 3]}) as mock_get_result:
            manager.add_to_result_data_collection(mock_cfw, mock_features, step_uuid)

            mock_get_result.assert_called_once_with(mock_cfw, {feature_name}, None)

    def test_add_to_result_data_collection_passes_location(self) -> None:
        """Should pass location to get_result_data when provided."""
        manager = DataLifecycleManager()
        step_uuid = uuid4()
        location = "grpc://localhost:8815"

        mock_cfw = Mock(spec=ComputeFramework)
        feature_name = Mock(spec=FeatureName)
        mock_features = Mock(spec=FeatureSet)
        mock_features.get_initial_requested_features.return_value = {feature_name}

        with patch.object(manager, "get_result_data", return_value={"feature1": [1, 2, 3]}) as mock_get_result:
            manager.add_to_result_data_collection(mock_cfw, mock_features, step_uuid, location=location)

            mock_get_result.assert_called_once_with(mock_cfw, {feature_name}, location)

    def test_add_to_result_data_collection_handles_none_result(self) -> None:
        """Should not add to collection when get_result_data returns None."""
        manager = DataLifecycleManager()
        step_uuid = uuid4()

        mock_cfw = Mock(spec=ComputeFramework)
        feature_name = Mock(spec=FeatureName)
        mock_features = Mock(spec=FeatureSet)
        mock_features.get_initial_requested_features.return_value = {feature_name}

        with patch.object(manager, "get_result_data", return_value=None):
            manager.add_to_result_data_collection(mock_cfw, mock_features, step_uuid)

            assert step_uuid not in manager.result_data_collection


class TestDataLifecycleManagerGetResultData:
    """Test getting result data from CFW."""

    def test_get_result_data_returns_selected_data_when_cfw_has_data(self) -> None:
        """Should return selected data from CFW when data is available."""
        manager = DataLifecycleManager()

        mock_cfw = Mock(spec=ComputeFramework)
        mock_cfw.uuid = uuid4()
        mock_cfw.data = {"feature1": [1, 2, 3], "feature2": [4, 5, 6]}

        selected_data = {"feature1": [1, 2, 3]}
        mock_cfw.select_data_by_column_names.return_value = selected_data

        feature_name = Mock(spec=FeatureName)
        feature_name.name = "feature1"
        selected_feature_names = {feature_name}

        result = manager.get_result_data(mock_cfw, selected_feature_names)

        assert result == selected_data
        mock_cfw.select_data_by_column_names.assert_called_once_with(mock_cfw.data, selected_feature_names)

    def test_get_result_data_downloads_from_flight_server_when_location_provided(self) -> None:
        """Should download data from flight server when location is provided and cfw.data is None."""
        manager = DataLifecycleManager()
        location = "grpc://localhost:8815"

        mock_cfw = Mock(spec=ComputeFramework)
        mock_cfw.uuid = uuid4()
        mock_cfw.data = None

        downloaded_data = Mock()
        converted_data = {"feature1": [1, 2, 3]}
        selected_data = {"feature1": [1, 2, 3]}

        mock_cfw.convert_flyserver_data_back.return_value = converted_data
        mock_cfw.select_data_by_column_names.return_value = selected_data

        feature_name = Mock(spec=FeatureName)
        selected_feature_names = {feature_name}

        with patch("mloda.core.runtime.data_lifecycle_manager.FlightServer") as mock_flight_server:
            mock_flight_server.download_table.return_value = downloaded_data

            result = manager.get_result_data(mock_cfw, selected_feature_names, location=location)

            mock_flight_server.download_table.assert_called_once_with(location, str(mock_cfw.uuid))
            mock_cfw.convert_flyserver_data_back.assert_called_once_with(downloaded_data, manager.transformer)
            mock_cfw.select_data_by_column_names.assert_called_once_with(converted_data, selected_feature_names)
            assert result == selected_data

    def test_get_result_data_raises_error_when_no_data_and_no_location(self) -> None:
        """Should raise ValueError when CFW has no data and no location is provided."""
        manager = DataLifecycleManager()

        mock_cfw = Mock(spec=ComputeFramework)
        mock_cfw.data = None

        feature_name = Mock(spec=FeatureName)
        selected_feature_names = {feature_name}

        with pytest.raises(ValueError, match="Not implemented"):
            manager.get_result_data(mock_cfw, selected_feature_names, location=None)


class TestDataLifecycleManagerGetResults:
    """Test getting all collected results."""

    def test_get_results_returns_list_of_all_results(self) -> None:
        """Should return list of all result data."""
        manager = DataLifecycleManager()
        step_uuid1 = uuid4()
        step_uuid2 = uuid4()

        result1 = {"feature1": [1, 2, 3]}
        result2 = {"feature2": [4, 5, 6]}

        manager.result_data_collection[step_uuid1] = result1
        manager.result_data_collection[step_uuid2] = result2

        results = manager.get_results()

        assert len(results) == 2
        assert result1 in results
        assert result2 in results

    def test_get_results_returns_empty_list_when_no_results(self) -> None:
        """Should raise ValueError when no results collected."""
        manager = DataLifecycleManager()

        with pytest.raises(ValueError, match="No results found"):
            manager.get_results()

    def test_get_results_maintains_order_independence(self) -> None:
        """Should return results as a list regardless of insertion order."""
        manager = DataLifecycleManager()
        step_uuid1 = uuid4()
        step_uuid2 = uuid4()
        step_uuid3 = uuid4()

        manager.result_data_collection[step_uuid1] = "result1"
        manager.result_data_collection[step_uuid2] = "result2"
        manager.result_data_collection[step_uuid3] = "result3"

        results = manager.get_results()

        assert len(results) == 3
        assert "result1" in results
        assert "result2" in results
        assert "result3" in results


class TestDataLifecycleManagerArtifacts:
    """Test artifact management."""

    def test_set_artifacts_stores_artifacts(self) -> None:
        """Should store artifacts dictionary."""
        manager = DataLifecycleManager()
        artifacts = {"model": "trained_model", "scaler": "standard_scaler"}

        manager.set_artifacts(artifacts)

        assert manager.artifacts == artifacts

    def test_get_artifacts_returns_artifacts(self) -> None:
        """Should return stored artifacts."""
        manager = DataLifecycleManager()
        artifacts = {"model": "trained_model", "scaler": "standard_scaler"}
        manager.artifacts = artifacts

        result = manager.get_artifacts()

        assert result == artifacts

    def test_get_artifacts_returns_empty_dict_when_no_artifacts(self) -> None:
        """Should return empty dict when no artifacts set."""
        manager = DataLifecycleManager()

        result = manager.get_artifacts()

        assert result == {}

    def test_set_artifacts_overwrites_existing_artifacts(self) -> None:
        """Should overwrite existing artifacts."""
        manager = DataLifecycleManager()
        old_artifacts = {"old_model": "old"}
        new_artifacts = {"new_model": "new"}

        manager.artifacts = old_artifacts
        manager.set_artifacts(new_artifacts)

        assert manager.artifacts == new_artifacts


class TestDataLifecycleManagerIntegration:
    """Integration tests for DataLifecycleManager."""

    def test_complete_lifecycle_workflow(self) -> None:
        """Test complete workflow: add results, track drops, drop data, get results."""
        manager = DataLifecycleManager()

        # Setup CFWs
        cfw_uuid1 = uuid4()
        cfw_uuid2 = uuid4()
        step_uuid1 = uuid4()
        step_uuid2 = uuid4()

        mock_cfw1 = Mock(spec=ComputeFramework)
        mock_cfw1.uuid = cfw_uuid1
        mock_cfw1.data = {"feature1": [1, 2, 3]}
        mock_cfw1.select_data_by_column_names.return_value = {"feature1": [1, 2, 3]}

        mock_cfw2 = Mock(spec=ComputeFramework)
        mock_cfw2.uuid = cfw_uuid2
        mock_cfw2.data = {"feature2": [4, 5, 6]}
        mock_cfw2.select_data_by_column_names.return_value = {"feature2": [4, 5, 6]}

        cfw_collection = {cfw_uuid1: mock_cfw1, cfw_uuid2: mock_cfw2}

        # Add results
        feature_name1 = Mock(spec=FeatureName)
        features1 = Mock(spec=FeatureSet)
        features1.get_initial_requested_features.return_value = {feature_name1}

        feature_name2 = Mock(spec=FeatureName)
        features2 = Mock(spec=FeatureSet)
        features2.get_initial_requested_features.return_value = {feature_name2}

        manager.add_to_result_data_collection(mock_cfw1, features1, step_uuid1)
        manager.add_to_result_data_collection(mock_cfw2, features2, step_uuid2)

        # Track data to drop
        manager.track_flyway_datasets(cfw_uuid1, {step_uuid1})
        manager.track_flyway_datasets(cfw_uuid2, {step_uuid2})

        # Drop data for finished CFWs
        finished_ids = {step_uuid1, step_uuid2}
        manager.drop_data_for_finished_cfws(finished_ids, cfw_collection)

        # Verify drops occurred
        mock_cfw1.drop_last_data.assert_called_once()
        mock_cfw2.drop_last_data.assert_called_once()

        # Get results
        results = manager.get_results()
        assert len(results) == 2

    def test_artifact_workflow(self) -> None:
        """Test artifact storage and retrieval workflow."""
        manager = DataLifecycleManager()

        artifacts = {"model": "trained_model", "scaler": "standard_scaler", "encoder": "label_encoder"}

        manager.set_artifacts(artifacts)
        retrieved = manager.get_artifacts()

        assert retrieved == artifacts
        assert "model" in retrieved
        assert retrieved["model"] == "trained_model"

    def test_mixed_location_and_local_data_workflow(self) -> None:
        """Test workflow with both local data and flight server data."""
        manager = DataLifecycleManager()
        location = "grpc://localhost:8815"

        # CFW with local data
        cfw_uuid1 = uuid4()
        step_uuid1 = uuid4()
        mock_cfw1 = Mock(spec=ComputeFramework)
        mock_cfw1.uuid = cfw_uuid1
        mock_cfw1.data = {"feature1": [1, 2, 3]}
        mock_cfw1.select_data_by_column_names.return_value = {"feature1": [1, 2, 3]}

        # CFW with flight server data
        cfw_uuid2 = uuid4()
        step_uuid2 = uuid4()
        mock_cfw2 = Mock(spec=ComputeFramework)
        mock_cfw2.uuid = cfw_uuid2
        mock_cfw2.data = None
        mock_cfw2.convert_flyserver_data_back.return_value = {"feature2": [4, 5, 6]}
        mock_cfw2.select_data_by_column_names.return_value = {"feature2": [4, 5, 6]}

        cfw_collection = {cfw_uuid1: mock_cfw1, cfw_uuid2: mock_cfw2}

        feature_name1 = Mock(spec=FeatureName)
        features1 = Mock(spec=FeatureSet)
        features1.get_initial_requested_features.return_value = {feature_name1}

        feature_name2 = Mock(spec=FeatureName)
        features2 = Mock(spec=FeatureSet)
        features2.get_initial_requested_features.return_value = {feature_name2}

        # Add results with different locations
        manager.add_to_result_data_collection(mock_cfw1, features1, step_uuid1)

        with patch("mloda.core.runtime.data_lifecycle_manager.FlightServer") as mock_flight_server:
            mock_flight_server.download_table.return_value = Mock()
            manager.add_to_result_data_collection(mock_cfw2, features2, step_uuid2, location=location)

        results = manager.get_results()
        assert len(results) == 2
