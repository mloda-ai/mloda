"""Tests that artifact workflows work correctly through the public API.

Validates the patterns documented in docs/docs/in_depth/artifacts.md:
  1. prepare() + run() + result.artifacts for saving artifacts
  2. run_all() for loading artifacts via options
  3. a saver returning a dict yields one artifact per key; a FeatureSet binds one of them on load
"""

from typing import Any

import pytest

from mloda.provider import BaseArtifact, FeatureSet, DataCreatorFeatureGroup
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class PublicApiArtifactFeature(DataCreatorFeatureGroup):
    """Artifact feature group mirroring the documentation example."""

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
    """Test the constructor + run() + result.artifacts pattern for saving artifacts."""

    def test_run_returns_artifacts(self) -> None:
        api = mloda(["PublicApiArtifactFeature"], [PyArrowTable])
        result = api.run()

        assert result.artifacts == {"PublicApiArtifactFeature": "TestPublicApiArtifact"}


class TestArtifactsRunAll:
    """Test the run_all() pattern for loading artifacts via options."""

    def test_run_all_loads_artifact_via_options(self) -> None:
        artifacts = {"PublicApiArtifactFeature": "TestPublicApiArtifact"}
        feat = Feature(name="PublicApiArtifactFeature", options=artifacts)
        result = mloda.run_all([feat], [PyArrowTable])

        assert len(result) == 1


class MultiArtifactFeature(DataCreatorFeatureGroup):
    """Saves a dict artifact; output columns record what the FeatureSet bound on load."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"MultiArtifactKeyA", "MultiArtifactKeyB"}

    @staticmethod
    def artifact() -> type[BaseArtifact] | None:
        return BaseArtifact

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.artifact_to_save:
            features.save_artifact = {"MultiArtifactKeyA": "value-a", "MultiArtifactKeyB": "value-b"}
        seen = [str(features.artifact_to_load), repr(features.artifact_value)]
        return {name: [seen] for name in features.get_all_names()}


_multi_enabled = PluginCollector.enabled_feature_groups({MultiArtifactFeature})
_MULTI_SAVED = {"MultiArtifactKeyA": "value-a", "MultiArtifactKeyB": "value-b"}


class TestMultiArtifactSaver:
    """A dict saver result becomes one artifact per key; load binds one key per FeatureSet."""

    def test_a_dict_saver_yields_one_artifact_per_key(self) -> None:
        session = mloda.prepare(
            ["MultiArtifactKeyA", "MultiArtifactKeyB"], [PyArrowTable], plugin_collector=_multi_enabled
        )

        assert session.run().artifacts == _MULTI_SAVED

    @pytest.mark.parametrize("how", ["options", "runtime"])
    def test_load_binds_one_key_per_feature_set(self, how: str) -> None:
        if how == "options":
            features: list[Any] = [Feature(n, options=dict(_MULTI_SAVED)) for n in _MULTI_SAVED]
            result = mloda.prepare(features, [PyArrowTable], plugin_collector=_multi_enabled).run()
        else:
            session = mloda.prepare(list(_MULTI_SAVED), [PyArrowTable], plugin_collector=_multi_enabled)
            result = session.run(artifacts=dict(_MULTI_SAVED))

        columns = {c: list(t.to_pandas()[c]) for t in result for c in t.to_pandas().columns}
        for column in _MULTI_SAVED:
            assert list(columns[column][0]) == ["MultiArtifactKeyA", "'value-a'"]
