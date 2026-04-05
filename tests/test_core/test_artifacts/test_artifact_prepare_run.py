"""Tests for artifact semantics under the prepare/run split.

Validates that artifact save/load mode can be resolved at run-time,
enabling train-then-predict and cross-validation workflows in a
single prepare() session.

Covers issue #338: artifacts should behave like api_data and resolve
per run() call, not be frozen at prepare-time.
"""

from typing import Any, Optional, Set, Type

from mloda.provider import BaseArtifact, BaseInputData, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, mloda, mlodaAPI, PluginCollector
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class PrepareRunArtifactFeature(FeatureGroup):
    """Artifact-capable feature group for prepare/run tests."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @staticmethod
    def artifact() -> Type[BaseArtifact] | None:
        return BaseArtifact

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.artifact_to_save:
            features.save_artifact = "trained_model_v1"

        if features.artifact_to_load:
            loaded = cls.load_artifact(features)
            if loaded != "trained_model_v1":
                raise ValueError(f"Expected 'trained_model_v1', got '{loaded}'")

        return {cls.get_class_name(): [1, 2, 3]}


_enabled = PluginCollector.enabled_feature_groups({PrepareRunArtifactFeature})


class TestArtifactSaveWithPrepareRun:
    """Save flow using prepare() + run() produces artifacts."""

    def test_prepare_run_save(self) -> None:
        session = mloda.prepare(
            ["PrepareRunArtifactFeature"],
            {PyArrowTable},
            plugin_collector=_enabled,
        )
        result = session.run()
        artifacts = session.get_artifacts()

        assert len(result) == 1
        assert artifacts == {"PrepareRunArtifactFeature": "trained_model_v1"}


class TestArtifactLoadViaOptionsWithPrepareRun:
    """Load flow using prepare() with artifact options + run()."""

    def test_prepare_run_load_via_options(self) -> None:
        feat = Feature(
            name="PrepareRunArtifactFeature",
            options={"PrepareRunArtifactFeature": "trained_model_v1"},
        )
        session = mloda.prepare(
            [feat],
            {PyArrowTable},
            plugin_collector=_enabled,
        )
        result = session.run()

        assert len(result) == 1
        # Load mode produces no new artifacts
        assert session.get_artifacts() == {}


class TestArtifactSaveThenLoadSameSession:
    """The core fix: prepare() once, run() to save, run(artifacts=...) to load."""

    def test_save_then_load_same_session(self) -> None:
        session = mloda.prepare(
            ["PrepareRunArtifactFeature"],
            {PyArrowTable},
            plugin_collector=_enabled,
        )

        # First run: save mode (default, no artifacts param)
        save_result = session.run()
        saved_artifacts = session.get_artifacts()
        assert saved_artifacts == {"PrepareRunArtifactFeature": "trained_model_v1"}
        assert len(save_result) == 1

        # Second run: load mode (pass saved artifacts)
        load_result = session.run(artifacts=saved_artifacts)
        assert len(load_result) == 1


class TestMultipleSaveRunsIndependent:
    """Multiple save-mode runs produce independent artifacts each time."""

    def test_multiple_saves(self) -> None:
        session = mloda.prepare(
            ["PrepareRunArtifactFeature"],
            {PyArrowTable},
            plugin_collector=_enabled,
        )

        session.run()
        artifacts_first = session.get_artifacts()

        session.run()
        artifacts_second = session.get_artifacts()

        assert artifacts_first == {"PrepareRunArtifactFeature": "trained_model_v1"}
        assert artifacts_second == {"PrepareRunArtifactFeature": "trained_model_v1"}


class TestFullRoundTripSeparateSessions:
    """Save in one session, load in a separate session (existing pattern)."""

    def test_round_trip_separate_sessions(self) -> None:
        # Session 1: save
        save_session = mloda.prepare(
            ["PrepareRunArtifactFeature"],
            {PyArrowTable},
            plugin_collector=_enabled,
        )
        save_session.run()
        saved = save_session.get_artifacts()
        assert "PrepareRunArtifactFeature" in saved

        # Session 2: load via options (existing pattern)
        feat = Feature(name="PrepareRunArtifactFeature", options=saved)
        load_session = mloda.prepare(
            [feat],
            {PyArrowTable},
            plugin_collector=_enabled,
        )
        load_result = load_session.run()
        assert len(load_result) == 1
        assert load_session.get_artifacts() == {}


class TestAlternatingSaveLoadCrossValidation:
    """Alternating save/load in same session (cross-validation pattern)."""

    def test_alternating_save_load(self) -> None:
        session = mloda.prepare(
            ["PrepareRunArtifactFeature"],
            {PyArrowTable},
            plugin_collector=_enabled,
        )

        # Fold 1: save (train)
        session.run()
        saved = session.get_artifacts()
        assert saved == {"PrepareRunArtifactFeature": "trained_model_v1"}

        # Fold 2: load (predict using trained model)
        load_result = session.run(artifacts=saved)
        assert len(load_result) == 1

        # Fold 3: save again (retrain)
        session.run()
        saved_again = session.get_artifacts()
        assert saved_again == {"PrepareRunArtifactFeature": "trained_model_v1"}

        # Fold 4: load again (predict with retrained model)
        load_result_2 = session.run(artifacts=saved_again)
        assert len(load_result_2) == 1
