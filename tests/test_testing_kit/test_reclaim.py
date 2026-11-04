"""Unit tests of the suite-side reclaim of test-local subclasses (track, mark, reclaim_since)."""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any

import pytest

from mloda._testing.source import reclaim
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup


def _registered(name: str) -> bool:
    """Names only, never class objects, which would pin them."""
    return name in {c.__name__ for c in get_all_subclasses(FeatureGroup)}


def _track_dead_class(name: str) -> str:
    """Build and track a class, return only its name so the caller holds no reference."""
    reclaim.track(type(name, (FeatureGroup,), {}))
    return name


@pytest.fixture
def collections(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Generations of every gc.collect call, recorded while still collecting."""
    calls: list[int] = []
    real: Callable[..., int] = gc.collect

    def counting(generation: int = 2) -> int:
        calls.append(generation)
        return real(generation)

    monkeypatch.setattr(gc, "collect", counting)
    return calls


def test_track_returns_the_class_unchanged() -> None:
    cls = type("ReclaimTrackedIdentityFG", (FeatureGroup,), {})
    assert reclaim.track(cls) is cls
    del cls
    gc.collect()


def test_a_dead_cyclic_tracked_class_is_reclaimed_and_not_reported(collections: list[int]) -> None:
    mark = reclaim.mark()
    name = _track_dead_class("ReclaimDeadCyclicFG")
    assert _registered(name), "the probe never registered; the reclaim assertion would prove nothing"

    assert reclaim.reclaim_since(mark) == []
    assert not _registered(name), f"{name} survived the reclaim"
    assert collections, "a live tracked class must trigger a collection"


def test_a_tracked_class_still_held_is_reported_by_name() -> None:
    mark = reclaim.mark()
    held = reclaim.track(type("ReclaimHeldFG", (FeatureGroup,), {}))
    try:
        reported = reclaim.reclaim_since(mark)
    finally:
        del held
        gc.collect()
        gc.collect()
    assert reported == ["ReclaimHeldFG"]
    assert not _registered("ReclaimHeldFG")


def test_nothing_tracked_since_the_mark_runs_no_collection(collections: list[int]) -> None:
    mark = reclaim.mark()
    assert reclaim.reclaim_since(mark) == []
    assert collections == []


def test_a_class_tracked_before_the_mark_costs_no_collection(collections: list[int]) -> None:
    held = reclaim.track(type("ReclaimBeforeMarkFG", (FeatureGroup,), {}))
    try:
        mark = reclaim.mark()
        assert reclaim.reclaim_since(mark) == []
        assert collections == []
    finally:
        del held
        gc.collect()


def test_the_log_is_truncated_back_to_the_mark(collections: list[int]) -> None:
    mark = reclaim.mark()
    held = reclaim.track(type("ReclaimTruncatedFG", (FeatureGroup,), {}))
    try:
        assert reclaim.reclaim_since(mark) == ["ReclaimTruncatedFG"]
        collections.clear()
        assert reclaim.reclaim_since(mark) == [], "the reported class must not be reported again"
        assert collections == [], "a truncated log has nothing left to collect"
    finally:
        del held
        gc.collect()
        gc.collect()


def test_a_nested_mark_leaves_the_outer_entries_alone() -> None:
    outer = reclaim.mark()
    held = reclaim.track(type("ReclaimOuterFG", (FeatureGroup,), {}))
    try:
        inner = reclaim.mark()
        _track_dead_class("ReclaimInnerFG")
        assert reclaim.reclaim_since(inner) == []
        assert reclaim.reclaim_since(outer) == ["ReclaimOuterFG"]
    finally:
        del held
        gc.collect()
        gc.collect()


def test_a_just_built_dead_class_stops_at_generation_zero(collections: list[int]) -> None:
    mark = reclaim.mark()
    _track_dead_class("ReclaimYoungFG")
    assert reclaim.reclaim_since(mark) == []
    assert collections == [0]


def test_survivors_are_names_not_classes() -> None:
    mark = reclaim.mark()
    held: Any = reclaim.track(type("ReclaimNameOnlyFG", (FeatureGroup,), {}))
    try:
        reported = reclaim.reclaim_since(mark)
    finally:
        del held
        gc.collect()
        gc.collect()
    assert all(isinstance(entry, str) for entry in reported)
