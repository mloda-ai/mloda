"""Step-completion snapshots include saver reads before a run's records are sealed."""

import json
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import pyarrow as pa
import pytest

from mloda.provider import (
    BaseArtifact,
    ComputeFramework,
    CredentialReadRefused,
    DataCreatorFeatureGroup,
    FeatureSet,
    credential_handle_spec,
    credentials_scope,
    read_credential,
)
from mloda.steward import Extender, ExtenderHook, HookContext, LifecycleOutcome, RunContext, verified_context
from mloda.user import DataAccessCollection, Feature, ParallelizationMode, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

_COLUMN = "step_complete_value"
_NO_ARTIFACT_COLUMN = "step_complete_without_artifact"
_FIRST = "step-calculate-first"
_SECOND = "step-calculate-second"
_SAVER = "step-saver"
_UNUSED = "step-unused"
_CALLBACK = "step-observer-only"
_OPTIONS = {"first_handle": _FIRST, "second_handle": _SECOND, "saver_handle": _SAVER, "unused_handle": _UNUSED}
_CREDENTIALS = {handle: {"token": f"step-complete-secret-{index}"} for index, handle in enumerate(_OPTIONS.values())}
_MODES = [ParallelizationMode.SYNC, ParallelizationMode.THREADING]
_STAGES = ["calculate", "transform", "validate", "save"]


@dataclass
class _StepState:
    events: list[str] = field(default_factory=list)
    calculate_calls: int = 0
    saves: list[Any] = field(default_factory=list)
    fail_at: str | None = None
    failure: BaseException | None = None
    business_thread: int | None = None

    def enter(self, stage: str) -> None:
        self.events.append(stage)
        self.business_thread = threading.get_ident()
        if self.fail_at == stage:
            assert self.failure is not None
            raise self.failure


class _StepArtifact(BaseArtifact):
    state = _StepState()

    @classmethod
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Any | None:
        # Repeat a calculate read and the saver read: observations are a set of names.
        for key in ("first_handle", "saver_handle", "saver_handle"):
            assert read_credential(features.get_options_key(key))["token"]
        cls.state.saves.append(artifact)
        cls.state.enter("save")
        return "saved-step-artifact"


class _StepFeatureGroup(DataCreatorFeatureGroup):
    state = _StepState()
    PROPERTY_MAPPING = {key: credential_handle_spec("Step-completion fixture handle.") for key in _OPTIONS}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {_COLUMN}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PyArrowTable}

    @staticmethod
    def artifact() -> type[BaseArtifact] | None:
        return _StepArtifact

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cls.state.calculate_calls += 1
        key = "first_handle" if cls.state.calculate_calls == 1 else "second_handle"
        for _ in range(2):
            assert read_credential(features.get_options_key(key))["token"]
        cls.state.enter("calculate")
        if features.artifact_to_save:
            features.save_artifact = "step-artifact"
        value = cls.state.calculate_calls * 10
        return {name: [value, value + 1] for name in features.get_all_names()}

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> None:
        assert isinstance(data, pa.Table)
        cls.state.enter("validate")


class _NoArtifactStepFeatureGroup(_StepFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {_NO_ARTIFACT_COLUMN}

    @staticmethod
    def artifact() -> type[BaseArtifact] | None:
        return None


def _snapshot(context: HookContext, outcome: LifecycleOutcome | None = None) -> dict[str, Any]:
    # Copy values now; a retained HookContext can be changed after an audit record was written.
    return {
        "credentials_read": tuple(context.credentials_read),
        "credential_handles": tuple(context.credential_handles),
        "feature_names": tuple(context.feature_names),
        "feature_group_class": context.feature_group_class,
        "compute_framework_name": context.compute_framework_name,
        "hook": context.hook.name,
        "plan_id": context.plan_id,
        "run_id": context.run_id,
        "step_uuid": str(context.step_uuid),
        "tenant_id": context.tenant_id,
        "project_id": context.project_id,
        "principal": context.principal,
        "rows_out": context.rows_out,
        "output_schema": context.output_schema,
        "duration_seconds": context.duration_seconds,
        "status": context.status,
        "outcome": asdict(outcome) if outcome is not None else None,
        "thread": threading.get_ident(),
        "context_repr": repr(context),
    }


class _StepObserver(Extender):
    def __init__(
        self,
        state: _StepState,
        *,
        wrap_calculate: bool = True,
        calls: int = 1,
        callback_error: Exception | None = None,
        read_in_callback: bool = False,
    ) -> None:
        self.state = state
        self.wrap_calculate = wrap_calculate
        self.calls = calls
        self.callback_error = callback_error
        self.read_in_callback = read_in_callback
        self.calculates: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []
        self.denied: list[str] = []
        self.own_reads = 0
        self.sealed = ""

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE} if self.wrap_calculate else set()

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        result = None
        for _ in range(self.calls):
            try:
                result = func(*args, **kwargs)
            finally:
                context = HookContext.current()
                assert context is not None
                self.calculates.append(_snapshot(context))
        return result

    def on_step_complete(self, context: HookContext, outcome: LifecycleOutcome) -> None:
        self.state.events.append("step-complete")
        self.steps.append(_snapshot(context, outcome))
        for handle in (_FIRST, _SAVER):
            with pytest.raises(CredentialReadRefused):
                read_credential(handle)
            self.denied.append(handle)
        if self.read_in_callback:
            with credentials_scope({_CALLBACK: {"token": "observer-local-secret"}}):
                assert read_credential(_CALLBACK)["token"] == "observer-local-secret"
                self.own_reads += 1
        if self.callback_error is not None:
            raise self.callback_error

    def on_run_complete(self, run: RunContext, outcome: LifecycleOutcome) -> None:
        self.state.events.append("run-complete")
        self.sealed = json.dumps(
            {"run_id": run.run_id, "steps": self.steps, "outcome": asdict(outcome)}, sort_keys=True
        )


class _PassiveExtender(Extender):
    def wraps(self) -> set[ExtenderHook]:
        return set()

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


@pytest.fixture
def state(monkeypatch: pytest.MonkeyPatch) -> _StepState:
    current = _StepState()
    monkeypatch.setattr(_StepFeatureGroup, "state", current)
    monkeypatch.setattr(_StepArtifact, "state", current)
    original_transform = PyArrowTable.transform

    def transform(framework: PyArrowTable, data: Any, feature_names: Sequence[str]) -> Any:
        current.enter("transform")
        return original_transform(framework, data, feature_names)

    monkeypatch.setattr(PyArrowTable, "transform", transform)
    return current


def _prepare(
    observers: set[Extender],
    *,
    group: type[_StepFeatureGroup] = _StepFeatureGroup,
    saver_handle: str = _SAVER,
) -> Any:
    name = next(iter(group.feature_names_supported()))
    with verified_context(tenant_id="step-tenant", project_id="step-project", principal="step-service"):
        return mloda.prepare(
            [Feature(name, options={**_OPTIONS, "saver_handle": saver_handle})],
            compute_frameworks=[PyArrowTable],
            plugin_collector=PluginCollector.enabled_feature_groups({group}),
            parallelization_modes={ParallelizationMode.SYNC},
            data_access_collection=DataAccessCollection(credentials=_CREDENTIALS),
            function_extender=observers,
        )


def _assert_complete(
    observer: _StepObserver, expected_reads: tuple[str, ...], outcome: LifecycleOutcome
) -> dict[str, Any]:
    assert len(observer.steps) == 1, "exactly one step-completion notification must precede run sealing"
    snapshot = observer.steps[0]
    assert snapshot["credentials_read"] == tuple(sorted(expected_reads))
    assert snapshot["outcome"] == asdict(outcome)
    assert snapshot["thread"] == observer.state.business_thread
    assert observer.denied == [_FIRST, _SAVER]
    sealed = json.loads(observer.sealed)
    assert sealed["steps"] == json.loads(json.dumps(observer.steps))
    assert sealed["run_id"] == snapshot["run_id"]
    assert sealed["outcome"] == asdict(outcome)
    for credential in _CREDENTIALS.values():
        assert credential["token"] not in observer.sealed
    return snapshot


def test_extenders_inherit_a_noop_step_completion_notification() -> None:
    callback = getattr(_PassiveExtender(), "on_step_complete", None)
    assert callable(callback), "Extender must expose the step-completion lifecycle callback"
    context = HookContext(
        hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
        feature_group_class="fixture",
        feature_group_version=None,
        compute_framework_name="PyArrowTable",
    )
    assert callback(context, LifecycleOutcome("succeeded")) is None


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("wrap_calculate", [True, False], ids=["calculate-wrapper", "lifecycle-only"])
def test_step_snapshot_contains_saver_reads_before_run_sealing(
    state: _StepState, mode: ParallelizationMode, wrap_calculate: bool, caplog: pytest.LogCaptureFixture
) -> None:
    observer = _StepObserver(state, wrap_calculate=wrap_calculate)
    session = _prepare({observer})

    result = session.run(parallelization_modes={mode})

    assert result[0].to_pydict() == {_COLUMN: [10, 11]}
    assert result.artifacts == {_COLUMN: "saved-step-artifact"}
    assert state.saves == ["step-artifact"]
    snapshot = _assert_complete(observer, (_FIRST, _SAVER), LifecycleOutcome("succeeded"))
    assert state.events == [*_STAGES, "step-complete", "run-complete"]
    assert snapshot["credential_handles"] == tuple(sorted(_CREDENTIALS))
    assert snapshot["feature_names"] == (_COLUMN,)
    assert snapshot["feature_group_class"] == f"{_StepFeatureGroup.__module__}.{_StepFeatureGroup.__qualname__}"
    assert snapshot["compute_framework_name"] == "PyArrowTable"
    assert snapshot["hook"] == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE.name
    assert snapshot["plan_id"] == session.plan_id
    assert snapshot["run_id"] and snapshot["step_uuid"] != "None"
    assert (snapshot["tenant_id"], snapshot["project_id"], snapshot["principal"]) == (
        "step-tenant",
        "step-project",
        "step-service",
    )
    if wrap_calculate:
        (raw,) = observer.calculates
        assert raw["credentials_read"] == (_FIRST,)
        assert raw["rows_out"] == 2 and raw["output_schema"] is not None
        assert raw["duration_seconds"] is not None
        for attribute in ("rows_out", "output_schema", "duration_seconds", "status"):
            assert snapshot[attribute] == raw[attribute]
    else:
        assert observer.calculates == []
    for credential in _CREDENTIALS.values():
        assert credential["token"] not in caplog.text


def test_a_handle_read_in_both_calculation_and_saver_is_reported_once(state: _StepState) -> None:
    observer = _StepObserver(state)

    _prepare({observer}, saver_handle=_FIRST).run(parallelization_modes={ParallelizationMode.SYNC})

    snapshot = _assert_complete(observer, (_FIRST,), LifecycleOutcome("succeeded"))
    assert snapshot["credential_handles"] == tuple(sorted((_FIRST, _SECOND, _UNUSED)))
    assert state.saves == ["step-artifact"]


@pytest.mark.parametrize("mode", _MODES)
def test_repeated_calculate_calls_union_their_reads_but_save_only_once(
    state: _StepState, mode: ParallelizationMode
) -> None:
    observer = _StepObserver(state, calls=2)

    result = _prepare({observer}).run(parallelization_modes={mode})

    assert result[0].to_pydict() == {_COLUMN: [10, 11]}, "the first successful calculation still supplies the data"
    assert state.calculate_calls == 2
    assert state.saves == ["step-artifact"]
    assert [snapshot["credentials_read"] for snapshot in observer.calculates] == [(_FIRST,), (_SECOND,)]
    _assert_complete(observer, (_FIRST, _SECOND, _SAVER), LifecycleOutcome("succeeded"))
    assert state.events == ["calculate", "calculate", "transform", "validate", "save", "step-complete", "run-complete"]


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("stage", _STAGES)
@pytest.mark.parametrize("callback_raises", [False, True], ids=["observer-succeeds", "observer-fails"])
def test_a_failed_step_notifies_once_without_masking_the_business_error(
    state: _StepState, mode: ParallelizationMode, stage: str, callback_raises: bool
) -> None:
    state.fail_at = stage
    state.failure = RuntimeError(f"business {stage} failed")
    observer = _StepObserver(state, callback_error=ValueError("observer failed") if callback_raises else None)
    session = _prepare({observer})

    with pytest.raises(RuntimeError, match=f"business {stage} failed") as error:
        session.run(parallelization_modes={mode})

    assert error.value is state.failure
    assert state.saves == (["step-artifact"] if stage == "save" else [])
    expected_reads = (_FIRST, _SAVER) if stage == "save" else (_FIRST,)
    snapshot = _assert_complete(observer, expected_reads, LifecycleOutcome("failed", "builtins.RuntimeError"))
    assert state.events == [*_STAGES[: _STAGES.index(stage) + 1], "step-complete", "run-complete"]
    assert snapshot["status"] == ("error" if stage == "calculate" else "success")
    (raw,) = observer.calculates
    assert raw["credentials_read"] == (_FIRST,)
    for attribute in ("rows_out", "output_schema", "duration_seconds", "status"):
        assert snapshot[attribute] == raw[attribute]


class _StepCancelled(BaseException):
    pass


@pytest.mark.parametrize("stage", ["calculate", "save"])
def test_step_cancellation_notifies_once_and_survives_a_failing_observer(state: _StepState, stage: str) -> None:
    state.fail_at = stage
    state.failure = _StepCancelled("step cancelled")
    observer = _StepObserver(state, callback_error=RuntimeError("observer failed"))
    session = _prepare({observer})

    with pytest.raises(_StepCancelled) as error:
        session.run(parallelization_modes={ParallelizationMode.SYNC})

    assert error.value is state.failure
    expected_reads = (_FIRST, _SAVER) if stage == "save" else (_FIRST,)
    _assert_complete(
        observer,
        expected_reads,
        LifecycleOutcome("cancelled", f"{_StepCancelled.__module__}.{_StepCancelled.__qualname__}"),
    )
    assert state.events == [*_STAGES[: _STAGES.index(stage) + 1], "step-complete", "run-complete"]


@pytest.mark.parametrize("mode", _MODES)
def test_steps_without_artifacts_still_notify_before_run_completion(
    state: _StepState, mode: ParallelizationMode
) -> None:
    observer = _StepObserver(state, wrap_calculate=False)

    result = _prepare({observer}, group=_NoArtifactStepFeatureGroup).run(parallelization_modes={mode})

    assert result[0].to_pydict() == {_NO_ARTIFACT_COLUMN: [10, 11]}
    assert result.artifacts == {} and state.saves == []
    _assert_complete(observer, (_FIRST,), LifecycleOutcome("succeeded"))
    assert state.events == ["calculate", "transform", "validate", "step-complete", "run-complete"]


def test_observers_own_reads_cannot_pollute_a_later_observers_step_snapshot(state: _StepState) -> None:
    first = _StepObserver(state, wrap_calculate=False, read_in_callback=True)
    first.priority = 10
    second = _StepObserver(state, wrap_calculate=False)
    second.priority = 20

    _prepare({first, second}).run(parallelization_modes={ParallelizationMode.SYNC})

    _assert_complete(first, (_FIRST, _SAVER), LifecycleOutcome("succeeded"))
    _assert_complete(second, (_FIRST, _SAVER), LifecycleOutcome("succeeded"))
    assert first.own_reads == 1
    assert "observer-local-secret" not in first.sealed + second.sealed
    assert _CALLBACK not in first.steps[0]["credentials_read"] + second.steps[0]["credentials_read"]


@pytest.mark.parametrize("mode", _MODES)
def test_saving_without_any_observer_keeps_the_existing_stage_order(
    state: _StepState, mode: ParallelizationMode
) -> None:
    result = _prepare(set()).run(parallelization_modes={mode})

    assert result[0].to_pydict() == {_COLUMN: [10, 11]}
    assert result.artifacts == {_COLUMN: "saved-step-artifact"}
    assert state.events == _STAGES
    assert state.saves == ["step-artifact"]
