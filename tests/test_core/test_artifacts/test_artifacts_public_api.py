"""Tests that artifact workflows work correctly through the public API.

Validates the patterns documented in docs/docs/in_depth/artifacts.md:
  1. prepare() + run() + get_artifacts() for saving artifacts
  2. run_all() for loading artifacts via options
"""

from typing import Any, Optional

from mloda.provider import BaseArtifact, BaseInputData, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class PublicApiArtifactFeature(FeatureGroup):
    """Artifact feature group mirroring the documentation example."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @staticmethod
    def artifact() -> type[BaseArtifact] | None:
        return BaseArtifact

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.artifact_to_save:
            features.save_artifact = "TestPublicApiArtifact"

        if features.artifact_to_load:
            result = cls.load_artifact(features)
            if result != "TestPublicApiArtifact":
                raise ValueError(f"Expected 'TestPublicApiArtifact', got '{result}'")

        return {cls.get_class_name(): [1, 2, 3]}


class TestArtifactsRunGetArtifacts:
    """Test the constructor + run() + get_artifacts() pattern for saving artifacts."""

    def test_run_returns_artifacts(self) -> None:
        api = mloda(["PublicApiArtifactFeature"], {PyArrowTable})
        api.run()
        artifacts = api.get_artifacts()

        assert artifacts == {"PublicApiArtifactFeature": "TestPublicApiArtifact"}


class TestArtifactsRunAll:
    """Test the run_all() pattern for loading artifacts via options."""

    def test_run_all_loads_artifact_via_options(self) -> None:
        artifacts = {"PublicApiArtifactFeature": "TestPublicApiArtifact"}
        feat = Feature(name="PublicApiArtifactFeature", options=artifacts)
        result = mloda.run_all([feat], {PyArrowTable})

        assert len(result) == 1
