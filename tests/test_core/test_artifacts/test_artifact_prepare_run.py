"""Tests for artifact semantics under the prepare/run split.

Validates that artifact save/load mode can be resolved at run-time,
enabling train-then-predict and cross-validation workflows in a
single prepare() session.
"""

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.provider import BaseArtifact, FeatureSet, DataCreatorFeatureGroup
from mloda.user import Feature, Options, mloda, PluginCollector
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class PrepareRunArtifactFeature(DataCreatorFeatureGroup):
    """Artifact-capable feature group for prepare/run tests."""

    @staticmethod
    def artifact() -> type[BaseArtifact] | None:
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


class VerifiableArtifactFeature(DataCreatorFeatureGroup):
    """Artifact feature group that embeds the loaded artifact hash into its output.

    In save mode: stores a versioned model string as the artifact.
    In load mode: loads the artifact, verifies its structure, and writes a SHA-256
    hash of it into every output row. This lets the caller assert from outside
    which exact artifact was used during calculate_feature.

    The artifact is stored as a string (not a dict) to avoid triggering the
    multi-artifact save path in FeatureGroupStep.save_artifact.
    """

    ARTIFACT_VALUE = "model|v1-alpha|weights=0.1,0.2,0.3"

    @staticmethod
    def artifact() -> type[BaseArtifact] | None:
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


class CountingArtifactFeature(PrepareRunArtifactFeature):
    """Counts calculate_feature calls so refusals can be shown to happen before any step."""

    calls: list[int] = []

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cls.calls.append(1)
        if features.artifact_to_save:
            features.save_artifact = "trained_model_v1"
        return {cls.get_class_name(): [1, 2, 3]}


class IdentityArtifactFeature(PrepareRunArtifactFeature):
    """Records the FeatureSet.features each run's calculate_feature sees."""

    seen: list[Any] = []

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cls.seen.append(features.features)
        if features.artifact_to_save:
            features.save_artifact = "trained_model_v1"
        return {cls.get_class_name(): [1, 2, 3]}


class VerifiableArtifactFeatureB(VerifiableArtifactFeature):
    """Second artifact group in the same request."""


_enabled = PluginCollector.enabled_feature_groups({PrepareRunArtifactFeature})
_enabled_verifiable = PluginCollector.enabled_feature_groups({VerifiableArtifactFeature})
_enabled_counting = PluginCollector.enabled_feature_groups({CountingArtifactFeature})
_enabled_two = PluginCollector.enabled_feature_groups({PrepareRunArtifactFeature, VerifiableArtifactFeatureB})


class TestArtifactSaveWithPrepareRun:
    """Save flow using prepare() + run() produces artifacts."""

    def test_prepare_run_save(self) -> None:
        session = mloda.prepare(
            ["PrepareRunArtifactFeature"],
            [PyArrowTable],
            plugin_collector=_enabled,
        )
        result = session.run()

        assert len(result) == 1
        assert result.artifacts == {"PrepareRunArtifactFeature": "trained_model_v1"}


class TestArtifactSaveWithStreamRun:
    """stream_run exposes saved artifacts once the stream is exhausted or closed."""

    @pytest.mark.parametrize("exit_style", ["exhaust", "next_then_close"])
    def test_stream_artifacts(self, exit_style: str) -> None:
        session = mloda.prepare(["PrepareRunArtifactFeature"], [PyArrowTable], plugin_collector=_enabled)
        stream = session.stream_run()
        assert stream.artifacts == {}

        if exit_style == "exhaust":
            assert len(list(stream)) == 1
        else:
            next(stream)
            stream.close()

        assert stream.artifacts == {"PrepareRunArtifactFeature": "trained_model_v1"}


class TestArtifactLoadViaOptionsWithPrepareRun:
    """Load flow using prepare() with artifact options + run()."""

    def test_prepare_run_load_via_options(self) -> None:
        feat = Feature(
            name="PrepareRunArtifactFeature",
            options={"PrepareRunArtifactFeature": "trained_model_v1"},
        )
        session = mloda.prepare(
            [feat],
            [PyArrowTable],
            plugin_collector=_enabled,
        )
        result = session.run()

        assert len(result) == 1
        # Load mode produces no new artifacts
        assert result.artifacts == {}


class TestArtifactSaveThenLoadSameSession:
    """The core fix: prepare() once, run() to save, run(artifacts=...) to load."""

    def test_save_then_load_same_session(self) -> None:
        session = mloda.prepare(
            ["PrepareRunArtifactFeature"],
            [PyArrowTable],
            plugin_collector=_enabled,
        )

        # First run: save mode (default, no artifacts param)
        save_result = session.run()
        saved_artifacts = save_result.artifacts
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
            [PyArrowTable],
            plugin_collector=_enabled,
        )

        artifacts_first = session.run().artifacts
        artifacts_second = session.run().artifacts

        assert artifacts_first == {"PrepareRunArtifactFeature": "trained_model_v1"}
        assert artifacts_second == {"PrepareRunArtifactFeature": "trained_model_v1"}


class TestFullRoundTripSeparateSessions:
    """Save in one session, load in a separate session (existing pattern)."""

    def test_round_trip_separate_sessions(self) -> None:
        # Session 1: save
        save_session = mloda.prepare(
            ["PrepareRunArtifactFeature"],
            [PyArrowTable],
            plugin_collector=_enabled,
        )
        saved = save_session.run().artifacts
        assert "PrepareRunArtifactFeature" in saved

        # Session 2: load via options (existing pattern)
        feat = Feature(name="PrepareRunArtifactFeature", options=saved)
        load_session = mloda.prepare(
            [feat],
            [PyArrowTable],
            plugin_collector=_enabled,
        )
        load_result = load_session.run()
        assert len(load_result) == 1
        assert load_result.artifacts == {}


class TestAlternatingSaveLoadCrossValidation:
    """Alternating save/load in same session (cross-validation pattern)."""

    def test_alternating_save_load(self) -> None:
        session = mloda.prepare(
            ["PrepareRunArtifactFeature"],
            [PyArrowTable],
            plugin_collector=_enabled,
        )

        # Fold 1: save (train)
        saved = session.run().artifacts
        assert saved == {"PrepareRunArtifactFeature": "trained_model_v1"}

        # Fold 2: load (predict using trained model)
        load_result = session.run(artifacts=saved)
        assert len(load_result) == 1

        # Fold 3: save again (retrain)
        saved_again = session.run().artifacts
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
            [PyArrowTable],
            plugin_collector=_enabled_verifiable,
        )
        result = session.run()
        df = result[0].to_pandas()
        assert list(df["VerifiableArtifactFeature"]) == ["no_artifact"] * 3

        artifacts = result.artifacts
        assert artifacts["VerifiableArtifactFeature"] == VerifiableArtifactFeature.ARTIFACT_VALUE

    def test_load_embeds_correct_artifact_hash_in_output(self) -> None:
        """Load mode: output contains a hash derived from the artifact content,
        proving calculate_feature actually accessed the artifact."""
        session = mloda.prepare(
            ["VerifiableArtifactFeature"],
            [PyArrowTable],
            plugin_collector=_enabled_verifiable,
        )

        # Save
        saved = session.run().artifacts

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
            [PyArrowTable],
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
            [PyArrowTable],
            plugin_collector=_enabled_verifiable,
        )

        # Step 1: save
        save_result = session.run()
        df_save = save_result[0].to_pandas()
        assert list(df_save["VerifiableArtifactFeature"]) == ["no_artifact"] * 3
        saved = save_result.artifacts

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


def _plan_options_context(session: Any) -> list[tuple[Any, dict[str, Any]]]:
    """Every plan Feature's Options object paired with a copy of its context."""
    return [
        (f.options, dict(f.options.context))
        for step in session.engine.execution_planner
        for f in getattr(getattr(step, "features", None), "features", [])
    ]


class TestArtifactRunsLeavePlanUntouched:
    """Save, load and save runs on one session, serial or threaded, never write into the plan."""

    @staticmethod
    def _assert_plan_untouched(session: Any, before: list[tuple[Any, dict[str, Any]]]) -> None:
        after = _plan_options_context(session)
        assert len(after) == len(before) > 0
        for (options_before, context_before), (options_after, context_after) in zip(before, after):
            assert options_after is options_before
            assert context_after == context_before
        for step in session.engine.execution_planner:
            assert "step_is_done" not in vars(step)
            assert "location" not in vars(step)

    def test_save_load_save_serial(self) -> None:
        session = mloda.prepare(["PrepareRunArtifactFeature"], [PyArrowTable], plugin_collector=_enabled)
        before = _plan_options_context(session)

        saved = session.run().artifacts
        assert len(session.run(artifacts=saved)) == 1
        assert session.run().artifacts == saved

        self._assert_plan_untouched(session, before)

    def test_save_load_save_concurrent(self) -> None:
        session = mloda.prepare(["PrepareRunArtifactFeature"], [PyArrowTable], plugin_collector=_enabled)
        saved = session.run().artifacts
        before = _plan_options_context(session)
        barrier = threading.Barrier(6, timeout=5)

        def worker(i: int) -> Any:
            barrier.wait()
            return session.run().artifacts if i % 2 else len(session.run(artifacts=saved))

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = [future.result(timeout=8) for future in [pool.submit(worker, i) for i in range(6)]]

        assert results == [1, saved, 1, saved, 1, saved]
        self._assert_plan_untouched(session, before)

    @staticmethod
    @staticmethod
    def _plan_artifact_state(session: Any) -> list[tuple[Any, ...]]:
        return [
            (
                step.features.artifact_to_load,
                step.features.artifact_to_save,
                step.features.artifact_value,
                step.features.save_artifact,
            )
            for step in session.engine.execution_planner
            if isinstance(getattr(step, "features", None), FeatureSet)
        ]

    def test_plan_feature_set_unchanged_after_save_and_load_runs(self) -> None:
        session = mloda.prepare(["VerifiableArtifactFeature"], [PyArrowTable], plugin_collector=_enabled_verifiable)
        before = self._plan_artifact_state(session)
        assert before

        saved = session.run().artifacts
        assert self._plan_artifact_state(session) == before
        session.run(artifacts=saved)
        assert self._plan_artifact_state(session) == before

    def test_calculate_feature_sees_the_plan_feature_objects(self) -> None:
        IdentityArtifactFeature.seen.clear()
        session = mloda.prepare(
            ["IdentityArtifactFeature"],
            [PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({IdentityArtifactFeature}),
        )
        assert session.engine is not None
        plan_features = [
            f
            for step in session.engine.execution_planner
            for f in getattr(getattr(step, "features", None), "features", [])
        ]
        session.run()
        session.run(artifacts={"IdentityArtifactFeature": "x"})

        assert len(IdentityArtifactFeature.seen) == 2
        for run_features in IdentityArtifactFeature.seen:
            assert all(any(f is p for p in plan_features) for f in run_features)

    def test_concurrent_predict_runs_with_different_artifacts(self) -> None:
        session = mloda.prepare(["VerifiableArtifactFeature"], [PyArrowTable], plugin_collector=_enabled_verifiable)
        values = [f"model|v{i}|weights={i}" for i in range(6)]
        barrier = threading.Barrier(6, timeout=5)

        def worker(value: str) -> list[str]:
            barrier.wait()
            result = session.run(artifacts={"VerifiableArtifactFeature": value})
            return list(result[0].to_pandas()["VerifiableArtifactFeature"])

        with ThreadPoolExecutor(max_workers=6) as pool:
            outputs = [future.result(timeout=8) for future in [pool.submit(worker, v) for v in values]]

        for value, output in zip(values, outputs):
            assert output == [_artifact_hash(value)] * 3

    def test_train_run_concurrent_with_predict_runs(self) -> None:
        session = mloda.prepare(["VerifiableArtifactFeature"], [PyArrowTable], plugin_collector=_enabled_verifiable)
        values = [f"model|v{i}|weights={i}" for i in range(5)]
        barrier = threading.Barrier(6, timeout=5)

        def predict(value: str) -> list[str]:
            barrier.wait()
            result = session.run(artifacts={"VerifiableArtifactFeature": value})
            return list(result[0].to_pandas()["VerifiableArtifactFeature"])

        def train() -> tuple[list[str], dict[str, Any]]:
            barrier.wait()
            result = session.run()
            return list(result[0].to_pandas()["VerifiableArtifactFeature"]), result.artifacts

        with ThreadPoolExecutor(max_workers=6) as pool:
            train_future = pool.submit(train)
            predict_futures = [pool.submit(predict, v) for v in values]
            outputs = [f.result(timeout=8) for f in predict_futures]
            trained, trained_artifacts = train_future.result(timeout=8)

        for value, output in zip(values, outputs):
            assert output == [_artifact_hash(value)] * 3
        assert trained == ["no_artifact"] * 3
        assert trained_artifacts == {"VerifiableArtifactFeature": VerifiableArtifactFeature.ARTIFACT_VALUE}


class TestRuntimeArtifactSurface:
    """run(artifacts=...) keys are validated against session.artifact_inputs before any step runs."""

    def test_artifact_inputs_lists_the_artifact_capable_names(self) -> None:
        session = mloda.prepare(
            ["PrepareRunArtifactFeature", "VerifiableArtifactFeatureB"],
            [PyArrowTable],
            plugin_collector=_enabled_two,
        )
        assert session.artifact_inputs == frozenset({"PrepareRunArtifactFeature", "VerifiableArtifactFeatureB"})

    @staticmethod
    def _call(session: Any, how: str, artifacts: dict[str, Any]) -> None:
        if how == "run":
            session.run(artifacts=artifacts)
        else:
            session.stream_run(artifacts=artifacts)

    @pytest.mark.parametrize("how", ["run", "stream_run"])
    def test_unknown_key_raises_at_the_call_before_any_step(self, how: str) -> None:
        CountingArtifactFeature.calls.clear()
        session = mloda.prepare(["CountingArtifactFeature"], [PyArrowTable], plugin_collector=_enabled_counting)

        with pytest.raises(ValueError, match="not_an_input") as exc_info:
            self._call(session, how, {"not_an_input": "x"})

        assert "CountingArtifactFeature" in str(exc_info.value)
        assert CountingArtifactFeature.calls == []

    @pytest.mark.parametrize("how", ["run", "stream_run"])
    def test_group_conflict_raises_at_the_call_before_any_step(self, how: str) -> None:
        feat = Feature("PrepareRunArtifactFeature", options={"PrepareRunArtifactFeature": "trained_model_v1"})
        session = mloda.prepare([feat], [PyArrowTable], plugin_collector=_enabled)

        with pytest.raises(ValueError, match="already stored in Options.group"):
            self._call(session, how, {"PrepareRunArtifactFeature": "other"})

    @staticmethod
    def _columns(result: Any) -> dict[str, list[Any]]:
        out: dict[str, list[Any]] = {}
        for table in result:
            frame = table.to_pandas()
            for column in frame.columns:
                out[column] = list(frame[column])
        return out

    def test_options_loaded_group_keeps_loading_when_other_group_binds_runtime(self) -> None:
        loaded = Feature("PrepareRunArtifactFeature", options={"PrepareRunArtifactFeature": "trained_model_v1"})
        session = mloda.prepare([loaded, "VerifiableArtifactFeatureB"], [PyArrowTable], plugin_collector=_enabled_two)
        value = "model|runtime|weights=7"

        result = session.run(artifacts={"VerifiableArtifactFeatureB": value})

        columns = self._columns(result)
        assert columns["PrepareRunArtifactFeature"] == [1, 2, 3]
        assert columns["VerifiableArtifactFeatureB"] == [_artifact_hash(value)] * 3
        assert result.artifacts == {}

    def test_save_mode_group_still_saves_when_other_group_binds_runtime(self) -> None:
        session = mloda.prepare(
            ["PrepareRunArtifactFeature", "VerifiableArtifactFeatureB"], [PyArrowTable], plugin_collector=_enabled_two
        )
        value = "model|runtime|weights=7"

        result = session.run(artifacts={"VerifiableArtifactFeatureB": value})

        assert result.artifacts == {"PrepareRunArtifactFeature": "trained_model_v1"}
        assert self._columns(result)["VerifiableArtifactFeatureB"] == [_artifact_hash(value)] * 3

    def test_runtime_value_wins_over_context_artifact_at_prepare(self) -> None:
        feat = Feature(
            "VerifiableArtifactFeature",
            options=Options(group={}, context={"VerifiableArtifactFeature": "model|prepared|weights=1"}),
        )
        session = mloda.prepare([feat], [PyArrowTable], plugin_collector=_enabled_verifiable)
        value = "model|runtime|weights=2"

        result = session.run(artifacts={"VerifiableArtifactFeature": value})

        assert list(result[0].to_pandas()["VerifiableArtifactFeature"]) == [_artifact_hash(value)] * 3

    @pytest.mark.parametrize("mode", [ParallelizationMode.THREADING, ParallelizationMode.MULTIPROCESSING])
    def test_save_then_load_in_parallel_mode(self, mode: ParallelizationMode, flight_server: Any) -> None:
        session = mloda.prepare(["VerifiableArtifactFeature"], [PyArrowTable], plugin_collector=_enabled_verifiable)
        saved = session.run(parallelization_modes={mode}, flight_server=flight_server).artifacts
        assert saved == {"VerifiableArtifactFeature": VerifiableArtifactFeature.ARTIFACT_VALUE}

        result = session.run(parallelization_modes={mode}, flight_server=flight_server, artifacts=saved)

        expected = _artifact_hash(VerifiableArtifactFeature.ARTIFACT_VALUE)
        assert list(result[0].to_pandas()["VerifiableArtifactFeature"]) == [expected] * 3
