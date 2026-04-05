"""Tests for artifact semantics under the prepare/run split.

Validates that artifact save/load mode can be resolved at run-time,
enabling train-then-predict and cross-validation workflows in a
single prepare() session.
"""

import hashlib
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


class VerifiableArtifactFeature(FeatureGroup):
    """Artifact feature group that embeds the loaded artifact hash into its output.

    In save mode: stores a versioned model string as the artifact.
    In load mode: loads the artifact, verifies its structure, and writes a SHA-256
    hash of it into every output row. This lets the caller assert from outside
    which exact artifact was used during calculate_feature.

    The artifact is stored as a string (not a dict) to avoid triggering the
    multi-artifact save path in FeatureGroupStep.save_artifact.
    """

    ARTIFACT_VALUE = "model|v1-alpha|weights=0.1,0.2,0.3"

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @staticmethod
    def artifact() -> Type[BaseArtifact] | None:
        return BaseArtifact

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if features.artifact_to_save:
            features.save_artifact = cls.ARTIFACT_VALUE

        if features.artifact_to_load:
            loaded = cls.load_artifact(features)
            if not isinstance(loaded, str) or "|" not in loaded:
                raise ValueError(f"Artifact has unexpected structure: {loaded}")

            artifact_hash = hashlib.sha256(loaded.encode()).hexdigest()[:16]
            return {cls.get_class_name(): [artifact_hash, artifact_hash, artifact_hash]}

        return {cls.get_class_name(): ["no_artifact", "no_artifact", "no_artifact"]}


_enabled = PluginCollector.enabled_feature_groups({PrepareRunArtifactFeature})
_enabled_verifiable = PluginCollector.enabled_feature_groups({VerifiableArtifactFeature})


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


def _artifact_hash(artifact_value: str) -> str:
    return hashlib.sha256(artifact_value.encode()).hexdigest()[:16]


class TestArtifactValueAccessibleInCalculateFeature:
    """Integration test: verify the artifact is genuinely loaded and usable
    inside calculate_feature by checking a hash derived from its content."""

    def test_save_produces_no_artifact_hash_in_output(self) -> None:
        """Save mode: output contains 'no_artifact' sentinel, not a hash."""
        session = mloda.prepare(
            ["VerifiableArtifactFeature"],
            {PyArrowTable},
            plugin_collector=_enabled_verifiable,
        )
        result = session.run()
        df = result[0].to_pandas()
        assert list(df["VerifiableArtifactFeature"]) == ["no_artifact"] * 3

        artifacts = session.get_artifacts()
        assert artifacts["VerifiableArtifactFeature"] == VerifiableArtifactFeature.ARTIFACT_VALUE

    def test_load_embeds_correct_artifact_hash_in_output(self) -> None:
        """Load mode: output contains a hash derived from the artifact content,
        proving calculate_feature actually accessed the artifact."""
        session = mloda.prepare(
            ["VerifiableArtifactFeature"],
            {PyArrowTable},
            plugin_collector=_enabled_verifiable,
        )

        # Save
        session.run()
        saved = session.get_artifacts()

        # Load: the hash in the output must match the artifact we saved
        result = session.run(artifacts=saved)
        df = result[0].to_pandas()
        expected_hash = _artifact_hash(VerifiableArtifactFeature.ARTIFACT_VALUE)
        assert list(df["VerifiableArtifactFeature"]) == [expected_hash] * 3

    def test_different_artifacts_produce_different_hashes(self) -> None:
        """Pass two different artifact values to consecutive run() calls.
        Each must produce output with a hash matching its own artifact,
        proving the runtime artifact is truly swapped between runs."""
        session = mloda.prepare(
            ["VerifiableArtifactFeature"],
            {PyArrowTable},
            plugin_collector=_enabled_verifiable,
        )

        artifact_a = {"VerifiableArtifactFeature": "model|v2-beta|weights=0.4,0.5"}
        artifact_b = {"VerifiableArtifactFeature": "model|v3-release|weights=0.9"}

        # Run with artifact A
        result_a = session.run(artifacts=artifact_a)
        df_a = result_a[0].to_pandas()
        expected_a = _artifact_hash("model|v2-beta|weights=0.4,0.5")
        assert list(df_a["VerifiableArtifactFeature"]) == [expected_a] * 3

        # Run with artifact B
        result_b = session.run(artifacts=artifact_b)
        df_b = result_b[0].to_pandas()
        expected_b = _artifact_hash("model|v3-release|weights=0.9")
        assert list(df_b["VerifiableArtifactFeature"]) == [expected_b] * 3

        # The two hashes must differ
        assert expected_a != expected_b

    def test_save_then_load_then_different_artifact(self) -> None:
        """Full flow: save -> load saved -> load a completely different artifact.
        Verifies the output changes to match whatever artifact was provided."""
        session = mloda.prepare(
            ["VerifiableArtifactFeature"],
            {PyArrowTable},
            plugin_collector=_enabled_verifiable,
        )

        # Step 1: save
        save_result = session.run()
        df_save = save_result[0].to_pandas()
        assert list(df_save["VerifiableArtifactFeature"]) == ["no_artifact"] * 3
        saved = session.get_artifacts()

        # Step 2: load the saved artifact
        load_result = session.run(artifacts=saved)
        df_load = load_result[0].to_pandas()
        expected_saved = _artifact_hash(VerifiableArtifactFeature.ARTIFACT_VALUE)
        assert list(df_load["VerifiableArtifactFeature"]) == [expected_saved] * 3

        # Step 3: load a different artifact entirely
        other_artifact = {"VerifiableArtifactFeature": "model|custom-42|weights=99.0"}
        other_result = session.run(artifacts=other_artifact)
        df_other = other_result[0].to_pandas()
        expected_other = _artifact_hash("model|custom-42|weights=99.0")
        assert list(df_other["VerifiableArtifactFeature"]) == [expected_other] * 3

        # All three outputs must be distinct
        assert len({"no_artifact", expected_saved, expected_other}) == 3
