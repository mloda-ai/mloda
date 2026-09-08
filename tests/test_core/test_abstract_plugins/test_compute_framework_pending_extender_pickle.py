"""Pins ComputeFramework.__setstate__: a _pending_extender_payload attached before pickling must
materialize into function_extender exactly once, only on the actual unpickle, and be cleared to
None afterward. Holding (not unpickling) an instance with a pending payload must never touch it.
"""

from __future__ import annotations

import pickle  # nosec B403
from typing import Any

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook


class InitDefaultsFramework(ComputeFramework):
    pass


class _MaterializationCountingExtender(Extender):
    """Counts real unpickles of itself via __setstate__, which __init__ never triggers. Guards
    against re-counting on a further round trip of an already-materialized instance, the same
    idempotent pattern a real extender uses for a lazily-built handle (see extender.md)."""

    materializations = 0

    def __init__(self) -> None:
        self._materialized = False

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self._materialized:
            return
        self._materialized = True
        type(self).materializations += 1


class TestPendingExtenderPayloadDefaultsAndConstruction:
    def test_default_construction_has_no_pending_payload(self) -> None:
        fw = InitDefaultsFramework()
        assert hasattr(fw, "_pending_extender_payload") and fw._pending_extender_payload is None

    def test_holding_an_instance_with_a_pending_payload_never_materializes_it(self) -> None:
        _MaterializationCountingExtender.materializations = 0
        payload = pickle.dumps({_MaterializationCountingExtender()})

        fw = InitDefaultsFramework()
        fw._pending_extender_payload = payload

        assert _MaterializationCountingExtender.materializations == 0
        assert fw.function_extender == set()


class TestPendingExtenderPayloadMaterializesOnUnpickle:
    def test_unpickling_materializes_function_extender_and_clears_the_pending_payload(self) -> None:
        _MaterializationCountingExtender.materializations = 0
        payload = pickle.dumps({_MaterializationCountingExtender()})

        fw = InitDefaultsFramework()
        fw._pending_extender_payload = payload

        restored = pickle.loads(pickle.dumps(fw))  # nosec B301

        assert restored._pending_extender_payload is None
        assert len(restored.function_extender) == 1
        restored_extender = next(iter(restored.function_extender))
        assert isinstance(restored_extender, _MaterializationCountingExtender)

    def test_materialization_happens_exactly_once_not_on_every_subsequent_round_trip(self) -> None:
        _MaterializationCountingExtender.materializations = 0
        payload = pickle.dumps({_MaterializationCountingExtender()})

        fw = InitDefaultsFramework()
        fw._pending_extender_payload = payload

        restored_once = pickle.loads(pickle.dumps(fw))  # nosec B301
        restored_twice = pickle.loads(pickle.dumps(restored_once))  # nosec B301

        assert _MaterializationCountingExtender.materializations == 1
        assert restored_twice._pending_extender_payload is None
        assert len(restored_twice.function_extender) == 1
