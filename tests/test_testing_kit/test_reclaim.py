"""Unit tests of the suite-side reclaim of test-local subclasses (track, mark, reclaim_since)."""

from __future__ import annotations

import gc
from collections.abc import Callable

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


def _release() -> None:
    """Collect after the test's own reference is deleted."""
    gc.collect()
    gc.collect()


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


def test_a_just_built_dead_class_is_reclaimed_at_generation_zero(collections: list[int]) -> None:
    mark = reclaim.mark()
    name = _track_dead_class("ReclaimYoungFG")
    assert _registered(name), "the probe never registered; the reclaim assertion would prove nothing"

    assert reclaim.reclaim_since(mark) == []
    assert collections == [0]
    assert not _registered(name), f"{name} survived the reclaim"


def test_a_tracked_class_still_held_is_reported_by_name() -> None:
    mark = reclaim.mark()
    held = reclaim.track(type("ReclaimHeldFG", (FeatureGroup,), {}))
    try:
        reported = reclaim.reclaim_since(mark)
    finally:
        del held
        _release()
    assert reported == ["ReclaimHeldFG"]
    assert not _registered("ReclaimHeldFG")


def test_nothing_tracked_since_the_mark_costs_no_collection(collections: list[int]) -> None:
    empty_mark = reclaim.mark()
    assert reclaim.reclaim_since(empty_mark) == []
    assert collections == []

    held = reclaim.track(type("ReclaimBeforeMarkFG", (FeatureGroup,), {}))
    try:
        mark = reclaim.mark()
        assert reclaim.reclaim_since(mark) == []
        assert collections == []
    finally:
        del held
        _release()


def test_dead_entries_are_truncated_so_a_later_reclaim_costs_no_collection(collections: list[int]) -> None:
    mark = reclaim.mark()
    held = reclaim.track(type("ReclaimTruncatedFG", (FeatureGroup,), {}))
    assert reclaim.reclaim_since(mark) == ["ReclaimTruncatedFG"]
    del held
    _release()
    collections.clear()

    assert reclaim.reclaim_since(mark) == []
    assert reclaim.reclaim_since(mark) == []
    assert collections == []


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
        _release()


def test_an_inner_reclaim_keeps_a_survivor_visible_to_the_outer_scope() -> None:
    outer = reclaim.mark()
    inner = reclaim.mark()
    held = reclaim.track(type("ReclaimNestedHeldFG", (FeatureGroup,), {}))
    try:
        assert reclaim.reclaim_since(inner) == ["ReclaimNestedHeldFG"]
        assert reclaim.reclaim_since(outer) == ["ReclaimNestedHeldFG"]
    finally:
        del held
        _release()


def test_a_later_scope_collects_an_earlier_survivor_once_released() -> None:
    scope_a = reclaim.mark()
    held = reclaim.track(type("ReclaimLateFG", (FeatureGroup,), {}))
    try:
        assert reclaim.reclaim_since(scope_a) == ["ReclaimLateFG"]
    finally:
        del held

    scope_b = reclaim.mark()
    assert reclaim.reclaim_since(scope_b) == []
    assert not _registered("ReclaimLateFG"), "the earlier survivor was never retried"


def test_a_released_survivor_is_collected_by_a_later_reclaimed_scope() -> None:
    first = reclaim.reclaimed_scope()
    next(first)
    held = reclaim.track(type("ReclaimPinnedFG", (FeatureGroup,), {}))
    with pytest.raises(AssertionError):
        next(first)
    del held

    second = reclaim.reclaimed_scope()
    next(second)
    with pytest.raises(StopIteration):
        next(second)
    assert not _registered("ReclaimPinnedFG"), "the earlier survivor was never retried"
