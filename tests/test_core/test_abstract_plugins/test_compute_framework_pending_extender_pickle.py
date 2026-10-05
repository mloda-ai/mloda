"""Pins ComputeFramework.__setstate__: a _pending_extender_payload attached before pickling must
materialize into function_extender exactly once, only on the actual unpickle, and be cleared to
None afterward. Holding (not unpickling) an instance with a pending payload must never touch it.
"""

from __future__ import annotations

import pickle  # nosec B403
from typing import Any

import pytest

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.run_context import RunContext
from mloda.core.abstract_plugins.function_extender import (
    CompositeExtender,
    Extender,
    ExtenderHook,
    build_hook_extenders,
)


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
        ext = _MaterializationCountingExtender()
        payload = pickle.dumps(({ext}, build_hook_extenders({ext})))

        fw = InitDefaultsFramework()
        fw._pending_extender_payload = payload

        assert _MaterializationCountingExtender.materializations == 0
        assert fw.function_extender == set()


class TestPendingExtenderPayloadMaterializesOnUnpickle:
    @pytest.mark.parametrize("sealed", [False, True])
    def test_unpickling_materializes_function_extender_and_clears_the_pending_payload(self, sealed: bool) -> None:
        _MaterializationCountingExtender.materializations = 0
        ext = _MaterializationCountingExtender()
        payload = pickle.dumps(({ext}, build_hook_extenders({ext})))

        fw = InitDefaultsFramework()
        fw._pending_extender_payload = payload
        if sealed:
            fw.run_context = RunContext(run_id="run-1")

        restored = pickle.loads(pickle.dumps(fw))  # nosec B301

        assert restored._pending_extender_payload is None
        assert len(restored.function_extender) == 1
        restored_extender = next(iter(restored.function_extender))
        assert isinstance(restored_extender, _MaterializationCountingExtender)
        if sealed:
            with pytest.raises(AttributeError):
                restored.function_extender = set()
            with pytest.raises(AttributeError):
                restored._hook_extenders = {}

    def test_materialization_happens_exactly_once_not_on_every_subsequent_round_trip(self) -> None:
        _MaterializationCountingExtender.materializations = 0
        ext = _MaterializationCountingExtender()
        payload = pickle.dumps(({ext}, build_hook_extenders({ext})))

        fw = InitDefaultsFramework()
        fw._pending_extender_payload = payload

        restored_once = pickle.loads(pickle.dumps(fw))  # nosec B301
        restored_twice = pickle.loads(pickle.dumps(restored_once))  # nosec B301

        assert _MaterializationCountingExtender.materializations == 1
        assert restored_twice._pending_extender_payload is None
        assert len(restored_twice.function_extender) == 1


class _IdentExtender(Extender):
    """Same-class, equal-priority extender distinguished only by ident."""

    def __init__(self, ident: str) -> None:
        self.ident = ident

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


def _ident_of(extender: Extender) -> str:
    assert isinstance(extender, _IdentExtender)
    return extender.ident


class TestPendingPayloadPreservesParentSelectionOrder:
    @pytest.mark.parametrize("reverse", [False, True])
    def test_worker_restores_the_parents_composite_order(self, reverse: bool) -> None:
        hook = ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE
        exts = [_IdentExtender(name) for name in ("a", "b", "c", "d")]
        ordered = list(reversed(exts)) if reverse else exts
        table = build_hook_extenders(ordered)
        parent_composite = table[hook]
        assert isinstance(parent_composite, CompositeExtender)
        parent_order = [_ident_of(e) for e in parent_composite.extenders]

        fw = InitDefaultsFramework()
        fw._pending_extender_payload = pickle.dumps((set(exts), table))
        restored = pickle.loads(pickle.dumps(fw))  # nosec B301

        composite = restored.get_function_extender(hook)
        assert isinstance(composite, CompositeExtender)
        assert [_ident_of(e) for e in composite.extenders] == parent_order
        assert all(any(m is x for x in restored.function_extender) for m in composite.extenders)
