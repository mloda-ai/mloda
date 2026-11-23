"""Completion observers keep their failure policy and cannot inspect unredacted business errors."""

import json
import sys
from dataclasses import asdict

import pytest

from mloda.steward import HookContext, LifecycleOutcome
from mloda.user import ParallelizationMode
from tests.test_plugins.extender.test_step_complete import (
    _COLUMN,
    _CREDENTIALS,
    _FIRST,
    _MODES,
    _SAVER,
    _STAGES,
    _StepObserver,
    _StepState,
    _prepare,
    state as state,
)


def _assert_every_observer_notified(observers: list[_StepObserver], run_error: Exception | None) -> None:
    outcome = LifecycleOutcome("succeeded")
    if run_error is not None:
        outcome = LifecycleOutcome("failed", f"{type(run_error).__module__}.{type(run_error).__qualname__}")
    for observer in observers:
        assert len(observer.steps) == 1
        snapshot = observer.steps[0]
        assert snapshot["outcome"] == asdict(LifecycleOutcome("succeeded")), "the business step itself succeeded"
        assert snapshot["credentials_read"] == (_FIRST, _SAVER)
        sealed = json.loads(observer.sealed)
        assert sealed["outcome"] == asdict(outcome)
        assert sealed["steps"] == json.loads(json.dumps(observer.steps))


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize(
    ("raise_on_error", "never_fall_back", "must_raise"),
    [(True, False, True), (False, False, False), (False, True, True)],
    ids=["default-strict", "warning-only", "never-fall-back-overrides-warning-only"],
)
def test_successful_step_completion_applies_error_policy_after_notifying_every_observer(
    state: _StepState, mode: ParallelizationMode, raise_on_error: bool, never_fall_back: bool, must_raise: bool
) -> None:
    failure = RuntimeError("first completion observer failed")
    first = _StepObserver(state, wrap_calculate=False, callback_error=failure)
    first.priority = 10
    first.raise_on_error = raise_on_error
    first.never_fall_back = never_fall_back
    second = _StepObserver(state, wrap_calculate=False)
    second.priority = 20
    session = _prepare({first, second})

    if must_raise:
        with pytest.raises(RuntimeError, match="first completion observer failed") as error:
            session.run(parallelization_modes={mode})
        assert error.value is failure
    else:
        result = session.run(parallelization_modes={mode})
        assert result[0].to_pydict() == {_COLUMN: [10, 11]}
        assert result.artifacts == {_COLUMN: "saved-step-artifact"}

    _assert_every_observer_notified([first, second], failure if must_raise else None)
    assert state.calculate_calls == 1 and state.saves == ["step-artifact"]
    assert state.events == [*_STAGES, "step-complete", "step-complete", "run-complete", "run-complete"]


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("first_is_strict", [True, False], ids=["first-error-is-strict", "first-error-is-warning-only"])
def test_completion_raises_the_first_strict_error_after_all_subscribers_receive_the_success(
    state: _StepState, mode: ParallelizationMode, first_is_strict: bool
) -> None:
    earlier = RuntimeError("earlier completion observer failed")
    later = ValueError("later completion observer failed")
    first = _StepObserver(state, wrap_calculate=False, callback_error=earlier)
    first.priority = 10
    first.raise_on_error = first_is_strict
    second = _StepObserver(state, wrap_calculate=False, callback_error=later)
    second.priority = 20
    last = _StepObserver(state, wrap_calculate=False)
    last.priority = 30
    session = _prepare({first, second, last})
    expected = earlier if first_is_strict else later

    with pytest.raises(type(expected)) as error:
        session.run(parallelization_modes={mode})

    assert error.value is expected
    _assert_every_observer_notified([first, second, last], expected)
    assert state.calculate_calls == 1 and state.saves == ["step-artifact"]
    assert state.events == [*_STAGES, *("step-complete" for _ in range(3)), *("run-complete" for _ in range(3))]


class _ExceptionSnapshotObserver(_StepObserver):
    def __init__(self, state: _StepState) -> None:
        super().__init__(state, wrap_calculate=False)
        self.exception_text: list[str] = []

    def on_step_complete(self, context: HookContext, outcome: LifecycleOutcome) -> None:
        active = sys.exception()
        self.exception_text.append(f"{active!s} | {active!r} | {getattr(active, 'args', ())!r}")
        super().on_step_complete(context, outcome)


@pytest.mark.parametrize("stage", ["transform", "validate", "save"])
def test_business_exception_is_redacted_before_step_completion_observers_can_inspect_it(
    state: _StepState, stage: str, caplog: pytest.LogCaptureFixture
) -> None:
    secret = _CREDENTIALS[_FIRST]["token"]
    state.fail_at = stage
    state.failure = RuntimeError(f"business {stage} failed with {secret}")
    observer = _ExceptionSnapshotObserver(state)
    session = _prepare({observer})

    with pytest.raises(RuntimeError, match=f"business {stage} failed") as error:
        session.run(parallelization_modes={ParallelizationMode.SYNC})

    assert error.value is state.failure
    assert len(observer.steps) == len(observer.exception_text) == 1
    assert observer.steps[0]["outcome"] == asdict(LifecycleOutcome("failed", "builtins.RuntimeError"))
    assert secret not in str(error.value)
    assert secret not in observer.exception_text[0], (
        "redact the escaping business exception before observer notification"
    )
    assert secret not in observer.sealed + caplog.text
