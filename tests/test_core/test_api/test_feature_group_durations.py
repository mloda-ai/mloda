"""Per-run calculation timings and a public level-3 plugin regression example."""

import math
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import UUID

import pytest

from mloda.provider import ComputeFramework, DataCreatorFeatureGroup, FeatureGroup, FeatureSet
from mloda.steward import Extender, ExtenderHook, HookContext
from mloda.user import Feature, FeatureName, Options, ParallelizationMode, PluginCollector, RunResult, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework


class DurationSource(DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"duration_source"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        assert features.options is not None
        size = features.options.get("size") or 3
        return [{"duration_source": value} for value in range(size)]


class DurationDouble(FeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"duration_double"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PythonDictFramework}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("duration_source", options=options)}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"duration_double": [value * 2 for value in data["duration_source"]]}


class DurationRecorder(Extender):
    def __init__(self, clock: list[float] | None = None) -> None:
        self.clock = clock
        self.contexts: list[HookContext] = []

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        if self.clock is not None:
            self.clock[0] += 100.0
        result = func(*args, **kwargs)
        context = HookContext.current()
        assert context is not None
        self.contexts.append(context)
        if self.clock is not None:
            self.clock[0] += 100.0
        return result


_PLUGINS = PluginCollector.enabled_feature_groups({DurationSource, DurationDouble})


def _assert_durations_cover_compute_steps(result: RunResult) -> None:
    expected = {step.step_uuid for step in result.plan if step.step_kind == "compute"}
    durations = result.feature_group_durations
    assert expected
    assert set(durations) == expected
    assert all(isinstance(step_uuid, UUID) for step_uuid in durations)
    assert all(isinstance(seconds, float) and math.isfinite(seconds) and seconds >= 0 for seconds in durations.values())


def _recorded_durations(recorder: DurationRecorder, result: RunResult) -> dict[UUID, float]:
    durations: dict[UUID, float] = {}
    for context in recorder.contexts:
        if context.run_id == result.run_id:
            assert context.step_uuid is not None
            assert context.duration_seconds is not None
            durations[context.step_uuid] = context.duration_seconds
    return durations


def test_manually_constructed_run_result_defaults_to_empty_durations() -> None:
    assert RunResult([], []).feature_group_durations == {}


@pytest.mark.parametrize("mode", list(ParallelizationMode))
def test_run_all_times_dependencies_without_extenders(mode: ParallelizationMode, flight_server: Any) -> None:
    result = mloda.run_all(
        ["duration_double"],
        compute_frameworks=[PythonDictFramework],
        plugin_collector=_PLUGINS,
        parallelization_modes={mode},
        flight_server=flight_server if mode is ParallelizationMode.MULTIPROCESSING else None,
    )

    assert result.frame_for("duration_double") == {"duration_double": [0, 2, 4]}
    assert len(result) == 1
    assert {step.feature_group for step in result.plan if step.step_kind == "compute"} == {
        DurationSource,
        DurationDouble,
    }
    _assert_durations_cover_compute_steps(result)


def test_same_group_with_different_options_has_separate_durations() -> None:
    result = mloda.run_all(
        [Feature("duration_source", options={"size": size}) for size in (2, 4)],
        compute_frameworks=[PythonDictFramework],
        plugin_collector=_PLUGINS,
    )

    assert len(result.plan) == 2
    assert all(step.feature_group is DurationSource for step in result.plan)
    _assert_durations_cover_compute_steps(result)
    assert len(result.feature_group_durations) == 2
    for step, frame in result.frames():
        assert step.feature_set_options is not None
        assert len(frame["duration_source"]) == step.feature_set_options.get("size")


@pytest.mark.parametrize("concurrent", [False, True])
def test_prepared_session_keeps_each_runs_durations(concurrent: bool) -> None:
    recorder = DurationRecorder()
    session = mloda.prepare(
        ["duration_double"],
        compute_frameworks=[PythonDictFramework],
        plugin_collector=_PLUGINS,
        function_extender={recorder},
    )

    if concurrent:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(session.run, parallelization_modes={ParallelizationMode.SYNC}) for _ in range(2)]
            first, second = [future.result() for future in pending]
    else:
        first = session.run(parallelization_modes={ParallelizationMode.SYNC})
        snapshot = first.feature_group_durations.copy()
        second = session.run(parallelization_modes={ParallelizationMode.SYNC})
        assert first.feature_group_durations == snapshot

    assert first.run_id != second.run_id
    assert first.feature_group_durations is not second.feature_group_durations
    for result in (first, second):
        _assert_durations_cover_compute_steps(result)
        assert result.feature_group_durations == _recorded_durations(recorder, result)


@pytest.mark.parametrize("with_extender", [False, True])
def test_timing_excludes_validation_transform_and_extender_overhead(
    monkeypatch: pytest.MonkeyPatch, with_extender: bool
) -> None:
    clock = [0.0]
    recorder = DurationRecorder(clock)
    original_transform = PythonDictFramework.transform

    def validate(cls: type[FeatureGroup], data: Any, features: FeatureSet) -> None:
        clock[0] += 10.0

    def transform(self: PythonDictFramework, data: Any, feature_names: Sequence[str]) -> dict[str, list[Any]]:
        clock[0] += 20.0
        return original_transform(self, data, feature_names)

    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(DurationDouble, "validate_input_features", classmethod(validate))
    monkeypatch.setattr(DurationDouble, "validate_output_features", classmethod(validate))
    monkeypatch.setattr(PythonDictFramework, "transform", transform)

    result = mloda.run_all(
        ["duration_double"],
        compute_frameworks=[PythonDictFramework],
        plugin_collector=_PLUGINS,
        function_extender={recorder} if with_extender else None,
    )

    assert clock[0] >= 40.0
    _assert_durations_cover_compute_steps(result)
    assert set(result.feature_group_durations.values()) == {0.0}
    if with_extender:
        assert result.feature_group_durations == _recorded_durations(recorder, result)


def test_level_3_plugin_calculation_budget() -> None:
    """Check a plugin's output and its calculation budget through the public run result."""
    size = 10_000
    result = mloda.run_all(
        [Feature("duration_double", options={"size": size})],
        compute_frameworks=[PythonDictFramework],
        plugin_collector=PluginCollector.enabled_feature_groups({DurationSource, DurationDouble}),
    )

    assert result.frame_for("duration_double") == {"duration_double": [value * 2 for value in range(size)]}
    step = next(step for step in result.plan if step.feature_group is DurationDouble)
    assert step.step_uuid is not None
    assert result.feature_group_durations[step.step_uuid] < 1.0
