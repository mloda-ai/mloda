"""Pins pickle_failure_reason, is_picklable, and WarnOncePerInstance: a thread-safe,
double-checked-locking once-guard whose pickled/copied instances are always fresh and unwarned,
across every pickle protocol.
"""

from __future__ import annotations

import copy
import itertools
import pickle  # nosec B403
import threading
from dataclasses import dataclass

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


class TestWarnOncePerInstanceThreadSafeUnderConcurrency:
    def test_only_one_emit_fires_across_sixteen_concurrent_threads(self) -> None:
        guard = WarnOncePerInstance()
        counter = itertools.count()
        fired: list[int] = []
        fired_lock = threading.Lock()
        thread_count = 16
        barrier = threading.Barrier(thread_count)

        def emit() -> None:
            value = next(counter)
            with fired_lock:
                fired.append(value)

        def worker() -> None:
            barrier.wait()
            guard.warn_once(emit)

        threads = [threading.Thread(target=worker) for _ in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(fired) == 1, f"emit must fire exactly once across concurrent callers, fired: {len(fired)} times"


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
