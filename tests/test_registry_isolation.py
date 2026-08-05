"""Pin the shared FeatureGroup registry-isolation mechanism (#845, part 1).

The mitigation must stay ONE mechanism: ``tests.registry_isolation.reclaim_leaked_feature_groups`` plus
one autouse fixture in ``tests/conftest.py``, so every test module is isolated and no module carries a copy.
"""

from __future__ import annotations

import gc
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup

from tests import registry_isolation_probe
from tests.registry_isolation import reclaim_leaked_feature_groups


TESTS_ROOT = Path(__file__).resolve().parent

PROBE_MODULE = registry_isolation_probe.__name__

SYNTHETIC_MODULE = "tests._registry_isolation_module_owned_probe"
SYNTHETIC_CLASS = "ModuleOwnedRegistryProbe995FeatureGroup"

FIXTURE_NAME = "_no_feature_group_registry_pollution"
# Assembled rather than written out, so this module is never itself a hit of the pattern it scans for.
FIXTURE_DEF = f"def {FIXTURE_NAME}"


def _registered_names() -> set[str]:
    """Names (never class objects, which would pin them) of this module's registered FeatureGroup subclasses."""
    return {c.__name__ for c in get_all_subclasses(FeatureGroup) if c.__module__ == __name__}


def _registered_names_of(module_name: str) -> set[str]:
    """Names (never class objects, which would pin them) of the registered subclasses from one module."""
    return {c.__name__ for c in get_all_subclasses(FeatureGroup) if c.__module__ == module_name}


def _define_throwaway_subclass() -> str:
    """Define a FeatureGroup subclass and return only its name; returning the class would pin it."""

    class ThrowawayRegistryProbe845FeatureGroup(FeatureGroup):
        pass

    return ThrowawayRegistryProbe845FeatureGroup.__name__


def _define_leaked_subclass() -> type[FeatureGroup]:
    """Define a FeatureGroup subclass and return it, so the caller holds a strong reference: a genuine leak."""

    class LeakedRegistryProbe845FeatureGroup(FeatureGroup):
        pass

    return LeakedRegistryProbe845FeatureGroup


class TestReclaimLeakedFeatureGroups:
    """reclaim_leaked_feature_groups(before, module_name) is the one reclaim-and-report mechanism."""

    def test_reclaims_a_throwaway_subclass(self) -> None:
        """A transient subclass is registered, then reclaimed. One test, not two: xdist could split a pair."""
        before = get_all_subclasses(FeatureGroup)
        name = _define_throwaway_subclass()
        assert name in _registered_names(), "the probe never registered; the reclaim assertion would prove nothing"
        assert reclaim_leaked_feature_groups(before, __name__) == []
        assert name not in _registered_names(), f"{name} survived the reclaim"

    def test_reports_a_genuine_leak(self) -> None:
        """A strongly referenced subclass is reported, so the conftest fixture fails loudly instead of hiding it."""
        before = get_all_subclasses(FeatureGroup)
        leaked_cls = _define_leaked_subclass()
        reported = [c.__name__ for c in reclaim_leaked_feature_groups(before, __name__)]
        expected = leaked_cls.__name__
        del leaked_cls  # drop the strong reference before asserting, so a failure leaves nothing behind
        gc.collect()
        gc.collect()
        assert reported == [expected], f"a strongly referenced subclass must be reported, got {reported}"
        assert expected not in _registered_names(), "the deliberate leak must not outlive this test"

    def test_reclaims_a_subclass_owned_by_another_module(self) -> None:
        """A helper-made subclass is collected too; only the RETURN value stays filtered to module_name.

        Generational GC is paused for the window, so the reclaim under test is the only collection that runs.
        """
        gc.disable()
        try:
            before = get_all_subclasses(FeatureGroup)
            name = registry_isolation_probe.define_helper_subclass()
            registered_before = name in _registered_names_of(PROBE_MODULE)
            reported = [c.__name__ for c in reclaim_leaked_feature_groups(before, __name__)]
            registered_after = name in _registered_names_of(PROBE_MODULE)
        finally:
            gc.enable()
            gc.collect()  # never leave the probe behind for the next test on this worker

        assert registered_before, "the probe never registered; the reclaim assertion would prove nothing"
        assert reported == [], f"a class this module does not own must not be reported, got {reported}"
        assert not registered_after, f"{name} survived the reclaim: an unowned class must still be collected"

    def test_no_new_subclasses_reports_nothing(self) -> None:
        """The cheap path: nothing appeared since the snapshot, so nothing is reported and no collection is needed."""
        before = get_all_subclasses(FeatureGroup)
        assert reclaim_leaked_feature_groups(before, __name__) == []


@contextmanager
def _module_bound_subclass() -> Iterator[str]:
    """Register a FeatureGroup subclass bound in a live module, the shape an import produces."""
    module = types.ModuleType(SYNTHETIC_MODULE)
    sys.modules[SYNTHETIC_MODULE] = module
    cls = type(SYNTHETIC_CLASS, (FeatureGroup,), {"__module__": SYNTHETIC_MODULE})
    setattr(module, SYNTHETIC_CLASS, cls)
    del cls
    try:
        yield SYNTHETIC_CLASS
    finally:
        delattr(module, SYNTHETIC_CLASS)
        del sys.modules[SYNTHETIC_MODULE]
        del module
        gc.collect()


@contextmanager
def _counted_collections(counts: list[int]) -> Iterator[None]:
    """Record the generation of every collection that runs, with automatic collections paused."""

    def on_collect(phase: str, info: dict[str, int]) -> None:
        if phase == "start":
            counts.append(info["generation"])

    gc.disable()
    gc.callbacks.append(on_collect)
    try:
        yield
    finally:
        gc.callbacks.remove(on_collect)
        gc.enable()


class TestModuleOwnedSubclassesSkipTheCollection:
    """An imported module's own class is not the transient leak this reclaims, so it must not cost a collection."""

    def test_a_module_bound_subclass_costs_no_collection(self) -> None:
        """The gate reads the binding instead: a full collection per test sat close to the per-test timeout (#995)."""
        before = get_all_subclasses(FeatureGroup)
        collections: list[int] = []
        with _module_bound_subclass() as name:
            assert name in _registered_names_of(SYNTHETIC_MODULE), (
                "the probe never registered; the collection assertion would prove nothing"
            )
            with _counted_collections(collections):
                reported = reclaim_leaked_feature_groups(before, __name__)

        assert reported == [], f"a class this module does not own must not be reported, got {reported}"
        assert collections == [], f"a module-owned class must cost no collection, got generations {collections}"

    def test_a_just_defined_subclass_is_reclaimed_by_the_young_generation(self) -> None:
        """The counterpart: the transient shape the reclaim exists for never escalates past generation 0."""
        before = get_all_subclasses(FeatureGroup)
        collections: list[int] = []
        with _counted_collections(collections):
            name = _define_throwaway_subclass()
            reported = reclaim_leaked_feature_groups(before, __name__)

        assert reported == []
        assert collections == [0], (
            f"{name} was just defined, so generation 0 must reclaim it and the ladder must stop there; "
            f"a full collection scans the whole heap and costs about a second. Got generations {collections}"
        )


class TestIsolationFixtureIsGlobal:
    """The isolation fixture is autouse in the root conftest, so a new test module inherits it for free."""

    def test_fixture_reaches_every_test(self, request: pytest.FixtureRequest) -> None:
        """This module declares no such fixture, so seeing it here proves tests/conftest.py supplies it."""
        assert FIXTURE_NAME in request.fixturenames, (
            f"{FIXTURE_NAME} must be an autouse fixture in tests/conftest.py so every test module is isolated"
        )


def _fixture_definitions() -> dict[Path, int]:
    """Every file under tests/ that defines the fixture, mapped to its definition count.

    Only conftest.py and test_*.py: a fixture is collected from nowhere else.
    """
    found: dict[Path, int] = {}
    for py_file in sorted({*TESTS_ROOT.rglob("conftest.py"), *TESTS_ROOT.rglob("test_*.py")}):
        if "__pycache__" in py_file.parts:
            continue
        count = py_file.read_text(encoding="utf-8").count(FIXTURE_DEF)
        if count:
            found[py_file] = count
    return found


class TestFixtureIsNotCopied:
    """The per-module copies stay deleted; the fixture lives in exactly one place."""

    def test_defined_exactly_once_in_root_conftest(self) -> None:
        """Exactly one definition of the fixture exists in the tests tree, and it is in tests/conftest.py."""
        found = _fixture_definitions()
        locations = {str(p.relative_to(TESTS_ROOT)): n for p, n in found.items()}
        assert sum(found.values()) == 1, f"{FIXTURE_NAME} must be defined exactly once, found {locations}"
        assert list(found) == [TESTS_ROOT / "conftest.py"], (
            f"the single definition must live in tests/conftest.py, found {locations}"
        )
