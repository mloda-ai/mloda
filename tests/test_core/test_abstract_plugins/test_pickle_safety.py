"""Pins pickle_failure_reason, is_picklable, and WarnOncePerInstance: a thread-safe,
double-checked-locking once-guard whose pickled/copied instances are always fresh and unwarned,
across every pickle protocol.
"""

from __future__ import annotations

import copy
import pickle  # nosec B403
import threading
from dataclasses import dataclass
from typing import Any

import pytest

from mloda.core.abstract_plugins.pickle_safety import (
    WarnOncePerInstance,
    is_picklable,
    pickle_failure_reason,
)


@dataclass
class _Point:
    x: int
    y: int


class TestPickleFailureReason:
    def test_returns_none_for_a_picklable_int(self) -> None:
        assert pickle_failure_reason(42) is None

    def test_returns_none_for_a_picklable_dataclass(self) -> None:
        assert pickle_failure_reason(_Point(1, 2)) is None

    def test_returns_the_caught_exception_type_name_for_an_unpicklable_value(self) -> None:
        reason = pickle_failure_reason(threading.Lock())
        assert reason == "TypeError", f"expected the caught exception's type name, got: {reason!r}"


class _Node:
    """Linked-list node; built iteratively so constructing the fixture itself never recurses."""

    def __init__(self, next_node: "_Node | None") -> None:
        self.next = next_node


class TestPickleFailureReasonDoesNotSwallowRecursionError:
    def test_recursion_error_propagates_instead_of_being_reported_as_a_reason_string(self) -> None:
        head: _Node | None = None
        for _ in range(50000):
            head = _Node(head)

        with pytest.raises(RecursionError):
            pickle_failure_reason(head)


class TestIsPicklable:
    def test_true_for_a_picklable_value(self) -> None:
        assert is_picklable(_Point(1, 2)) is True

    def test_false_for_an_unpicklable_value(self) -> None:
        assert is_picklable(threading.Lock()) is False


class TestWarnOncePerInstanceFiresAtMostOnce:
    def test_emit_called_exactly_once_across_five_sequential_calls(self) -> None:
        guard = WarnOncePerInstance()
        calls: list[int] = []

        for _ in range(5):
            guard.warn_once(lambda: calls.append(1))

        assert len(calls) == 1, f"emit must fire exactly once, fired: {len(calls)}"


class TestWarnOncePerInstanceMarksFiredBeforeEmitRuns:
    def test_second_call_does_not_retry_emit_after_the_first_raised(self) -> None:
        guard = WarnOncePerInstance()
        calls: list[int] = []

        def raising_emit() -> None:
            calls.append(1)
            raise RuntimeError("emit boom")

        with pytest.raises(RuntimeError, match="emit boom"):
            guard.warn_once(raising_emit)
        assert len(calls) == 1, "the raising emit must still run on the first call"

        guard.warn_once(raising_emit)
        assert len(calls) == 1, "the guard must be marked fired even though emit raised, so no retry"


class TestWarnOncePerInstanceHoldsItsLockDuringEmit:
    """A lock-free warn_once still passes a 16-thread-plus-barrier race test 100% of the time
    under the GIL (0/500 failures across two mutation runs), so it doesn't bite. This test
    instead probes the lock's held/free state from a second thread while emit is running."""

    def test_warn_once_holds_its_lock_while_emit_runs(self) -> None:
        guard = WarnOncePerInstance()
        acquired: list[bool] = []

        def emit() -> None:
            def probe() -> None:
                got = guard._lock.acquire(timeout=0.2)
                acquired.append(got)
                if got:
                    guard._lock.release()

            thread = threading.Thread(target=probe)
            thread.start()
            thread.join()

        guard.warn_once(emit)
        assert acquired == [False], "warn_once must hold its lock across the check-and-set"


class TestWarnOncePerInstancePickleRoundTrip:
    """An earlier __getstate__-returning-{} version silently skipped __setstate__ on protocols
    0/1 (empty dict is falsy), leaving the restored instance unusable. Cover every protocol and
    prove the restored instance actually works, not just that loads() didn't raise."""

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_restored_instance_is_usable_on_every_protocol(self, protocol: int) -> None:
        guard = WarnOncePerInstance()
        payload = pickle.dumps(guard, protocol=protocol)
        restored = pickle.loads(payload)  # nosec B301

        calls: list[int] = []
        restored.warn_once(lambda: calls.append(1))
        assert len(calls) == 1, f"restored instance must be usable on protocol {protocol}"


class _LabeledGuard(WarnOncePerInstance):
    """Subclass with a required constructor argument, as a real Extender's guard subclass might have."""

    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label


class TestWarnOncePerInstanceSubclassSurvivesRoundTrip:
    """dump time never calls __init__, so pickle.dumps succeeds even when __reduce__ targets a
    subclass requiring a constructor argument; only pickle.loads (e.g. in a spawned
    MULTIPROCESSING worker) hits the missing-argument TypeError."""

    def test_pickle_round_trip_of_a_subclass_with_a_required_init_argument_does_not_raise(self) -> None:
        guard = _LabeledGuard("sink-drop")

        restored = pickle.loads(pickle.dumps(guard))  # nosec B301

        calls: list[int] = []
        restored.warn_once(lambda: calls.append(1))
        assert len(calls) == 1, "the restored instance must still be usable"


class TestWarnOncePerInstanceCopyResetsWarnedState:
    def test_pickle_copy_of_an_already_warned_instance_fires_emit_again(self) -> None:
        original = WarnOncePerInstance()
        original.warn_once(lambda: None)

        restored = pickle.loads(pickle.dumps(original))  # nosec B301

        calls: list[int] = []
        restored.warn_once(lambda: calls.append(1))
        assert len(calls) == 1, "a pickle round-trip copy must be a fresh, unwarned instance"

    def test_deepcopy_of_an_already_warned_instance_fires_emit_again(self) -> None:
        original = WarnOncePerInstance()
        original.warn_once(lambda: None)

        restored = copy.deepcopy(original)

        calls: list[int] = []
        restored.warn_once(lambda: calls.append(1))
        assert len(calls) == 1, "a deepcopy must be a fresh, unwarned instance"

    def test_shallow_copy_of_an_already_warned_instance_fires_emit_again(self) -> None:
        original = WarnOncePerInstance()
        original.warn_once(lambda: None)

        restored = copy.copy(original)

        calls: list[int] = []
        restored.warn_once(lambda: calls.append(1))
        assert len(calls) == 1, "a copy.copy must be a fresh, unwarned instance"


class _SinkHost:
    """Mirrors the Extender __getstate__ composition sketch in docs/docs/chapter1/extender.md:
    trial-pickles its own `_sink` attribute and uses a `WarnOncePerInstance` guard to warn once
    if the sink must be dropped. `warnings` is test-only bookkeeping, excluded from pickled state."""

    def __init__(self, sink: Any, warnings: list[str]) -> None:
        self._sink = sink
        self._drop_guard = WarnOncePerInstance()
        self._warnings = warnings

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        del state["_warnings"]
        reason = pickle_failure_reason(self._sink) if self._sink is not None else None
        if reason is not None:
            self._drop_guard.warn_once(lambda: self._warnings.append(reason))
            state["_sink"] = None
        return state


class TestComposedDropAndWarnPattern:
    def test_a_restored_copy_decides_independently_whether_to_warn(self) -> None:
        warnings: list[str] = []
        original = _SinkHost(threading.Lock(), warnings)

        restored = pickle.loads(pickle.dumps(original))  # nosec B301
        assert restored._sink is None, "an unpicklable sink must be dropped on the restored copy"
        assert warnings == ["TypeError"], "the drop must be warned about exactly once"

        pickle.dumps(original)  # nosec B301 -- pickling the same original a second time
        assert warnings == ["TypeError"], "the original's guard already fired; no repeat warning"

        restored._warnings = warnings
        restored._sink = threading.Lock()
        pickle.loads(pickle.dumps(restored))  # nosec B301
        assert warnings == ["TypeError", "TypeError"], (
            "the restored copy's own guard must warn independently of the original's fired state"
        )
