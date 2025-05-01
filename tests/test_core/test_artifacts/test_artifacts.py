from typing import Any, Dict, List, Optional, Set, Tuple, Type
import pytest

from mloda_core.abstract_plugins.components.base_artifact import BaseArtifact
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.api.request import mlodaAPI
from mloda_core.runtime.flight.flight_server import FlightServer
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_collection import Features
from mloda_core.abstract_plugins.components.feature_set import FeatureSet


import logging

logger = logging.getLogger(__name__)


class BaseTestArtifactFeature(AbstractFeatureGroup):
    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @staticmethod
    def artifact() -> Type[BaseArtifact] | None:
        return BaseArtifact

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.artifact_to_save:
            cls.save_artifact_for_test(features)

        if features.artifact_to_load:
            cls.load_artifact_for_test(features)

        return {cls.get_class_name(): [1, 2, 3]}

    @classmethod
    def save_artifact_for_test(
        cls,
        features: FeatureSet,
    ) -> None:
        if features.artifact_to_save != "BaseTestArtifactFeature":
            raise ValueError("Artifact to save is missing.")
        if features.artifact_to_load is not None:
            raise ValueError(f"Artifact to load is not None: {features.artifact_to_load}")
        if features.save_artifact is not None:
            raise ValueError(f"Saver is not None: {features.save_artifact}")
        features.save_artifact = "DummyArtifact"

    @classmethod
    def load_artifact_for_test(
        cls,
        features: FeatureSet,
    ) -> None:
        if features.artifact_to_load != "BaseTestArtifactFeature":
            raise ValueError("Artifact to load is missing.")
        if features.artifact_to_save is not None:
            raise ValueError(f"Artifact to save is not None: {features.artifact_to_save}")
        if features.save_artifact is not None:
            raise ValueError(f"Saver is not None: {features.save_artifact}")

        if cls.load_artifact(features) != "DummyArtifact":
            raise ValueError("Artifact loading failed.")


@pytest.mark.parametrize(
    "modes",
    [
        ({ParallelizationModes.SYNC}),
        ({ParallelizationModes.THREADING}),
        ({ParallelizationModes.MULTIPROCESSING}),
    ],
)
class TestBaseArtifacts:
    def get_features(self, feature_list: List[str], options: Dict[str, Any] = {}) -> Features:
        return Features([Feature(name=f_name, options=options, initial_requested_data=True) for f_name in feature_list])

    def basic_runner(
        self,
        features: Features,
        parallelization_modes: Set[ParallelizationModes],
        flight_server: Any,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        api = mlodaAPI(features, {PyarrowTable})
        api._batch_run(parallelization_modes, flight_server)
        results = api.get_result()
        artifacts = api.get_artifacts()

        # make sure all datasets are dropped on server
        if ParallelizationModes.MULTIPROCESSING in parallelization_modes:
            flight_infos = FlightServer.list_flight_infos(flight_server.location)
            assert len(flight_infos) == 0

        return results, artifacts

    def test_basic_artifact_feature(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        _features = "BaseTestArtifactFeature"

        features = self.get_features([_features])

        _, artifacts = self.basic_runner(features, modes, flight_server)
        assert artifacts == {"BaseTestArtifactFeature": "DummyArtifact"}

        features = self.get_features([_features], options={"BaseTestArtifactFeature": "DummyArtifact"})
        _, artifacts = self.basic_runner(features, modes, flight_server)
        assert not artifacts
