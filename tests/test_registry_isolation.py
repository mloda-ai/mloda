"""Pin the shared registry-isolation mechanism (#845, part 1) for the roots FeatureGroup, BaseTransformer, Implementation.

The mitigation must stay ONE mechanism: ``reclaim_leaked_feature_groups`` and ``reclaim_leaked_transformers`` (thin
wrappers over one root-agnostic core in ``tests.registry_isolation``) plus one autouse ``tests/conftest.py`` fixture.
"""

from __future__ import annotations

import ast
import gc
import importlib
import sys
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer
from mloda.core.abstract_plugins.components.input_data.read_db_fg import ReadDBFG
from mloda.core.abstract_plugins.components.input_data.source_feature_group import SourceFeatureGroup
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.implementation import Implementation, loaded_implementations
from mloda.provider import SubtypeDeclaration
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.feature_group.experimental.dynamic_feature_group_factory.dynamic_feature_group_factory import (
    DynamicFeatureGroupCreator,
)

from tests import registry_isolation, registry_isolation_probe
from tests.mixins.neutral_csv_source import neutral_csv_group
from tests.mixins.compute_frameworks.tfs_connection_e2e_mixin import root_binding_groups
from tests.helpers.subclass_walks import count_root_walks
from tests.registry_isolation import (
    FEATURE_GROUPS,
    IMPLEMENTATIONS,
    TRANSFORMERS,
    SubclassLog,
    reclaim_leaked_feature_groups,
    reclaim_leaked_subclasses,
    reclaim_leaked_transformers,
)


TESTS_ROOT = Path(__file__).resolve().parent

PROBE_MODULE = registry_isolation_probe.__name__

# Snapshot at import, before any test runs inside a young window, so it is the interpreter default.
DEFAULT_YOUNG_THRESHOLD = gc.get_threshold()[0]

SYNTHETIC_MODULE = "tests._registry_isolation_module_owned_probe"
SYNTHETIC_CLASS = "ModuleOwnedRegistryProbe995FeatureGroup"
OWN_MODULE_CLASS = "OwnModuleBoundRegistryProbe995FeatureGroup"

FIXTURE_NAME = "_no_feature_group_registry_pollution"
# Assembled rather than written out, so this module is never itself a hit of the pattern it scans for.
FIXTURE_DEF = f"def {FIXTURE_NAME}"


def _registered_names_of_root(root: type[Any]) -> set[str]:
    """Names (never class objects, which would pin them) of this module's registered subclasses of `root`."""
    return {c.__name__ for c in get_all_subclasses(root) if c.__module__ == __name__}


def _registered_names() -> set[str]:
    """Names (never class objects, which would pin them) of this module's registered FeatureGroup subclasses."""
    return _registered_names_of_root(FeatureGroup)


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


def _define_overlap_probe() -> str:
    """Define a SourceFeatureGroup subclass and return only its name; returning the class would pin it."""

    class _OverlapProbeFG(SourceFeatureGroup):
        pass

    return _OverlapProbeFG.__name__


def _registered_transformer_names() -> set[str]:
    """Names (never class objects, which would pin them) of this module's registered BaseTransformer subclasses."""
    return {c.__name__ for c in get_all_subclasses(BaseTransformer) if c.__module__ == __name__}


def _define_throwaway_transformer() -> str:
    """Define a BaseTransformer subclass and return only its name; returning the class would pin it."""

    class ThrowawayRegistryProbeTransformer(BaseTransformer):
        pass

    return ThrowawayRegistryProbeTransformer.__name__


def _define_leaked_transformer() -> type[BaseTransformer]:
    """Define a BaseTransformer subclass and return it, so the caller holds a strong reference: a genuine leak."""

    class LeakedRegistryProbeTransformer(BaseTransformer):
        pass

    return LeakedRegistryProbeTransformer


def _define_rejected_implementation() -> None:
    class RiRejectedImpl(Implementation[Any]):
        compute_framework = PythonDictFramework

        @classmethod
        def calculate(cls, *args: Any) -> Any:
            return None


def _define_pre_super_rejected_group() -> None:
    type("RiPreSuperRejectedFG", (ReadDBFG,), {"PROPERTY_MAPPING": ["k"], "__module__": __name__})


def _define_feature_group_rejected_group() -> None:
    class RiRejectedByFeatureGroupFG(FeatureGroup):
        SUBTYPES = SubtypeDeclaration(key="ri_missing_key")


_REJECTED_DEFINITIONS: list[tuple[SubclassLog[Any], type[Any], str, type[Exception], Callable[[], None]]] = [
    (IMPLEMENTATIONS, Implementation, "RiRejectedImpl", TypeError, _define_rejected_implementation),
    (FEATURE_GROUPS, FeatureGroup, "RiPreSuperRejectedFG", TypeError, _define_pre_super_rejected_group),
    (FEATURE_GROUPS, FeatureGroup, "RiRejectedByFeatureGroupFG", ValueError, _define_feature_group_rejected_group),
]


class TestReclaimLeakedFeatureGroups:
    """reclaim_leaked_feature_groups(mark, module_name) is the one reclaim-and-report mechanism."""

    def test_reclaims_a_throwaway_subclass(self) -> None:
        """A transient subclass is registered, then reclaimed. One test, not two: xdist could split a pair."""
        mark = FEATURE_GROUPS.mark()
        name = _define_throwaway_subclass()
        assert name in _registered_names(), "the probe never registered; the reclaim assertion would prove nothing"
        assert reclaim_leaked_feature_groups(mark, __name__) == []
        assert name not in _registered_names(), f"{name} survived the reclaim"

    def test_reports_a_genuine_leak(self) -> None:
        """A strongly referenced subclass is reported, so the conftest fixture fails loudly instead of hiding it."""
        mark = FEATURE_GROUPS.mark()
        leaked_cls = _define_leaked_subclass()
        reported = [c.__name__ for c in reclaim_leaked_feature_groups(mark, __name__)]
        expected = leaked_cls.__name__
        del leaked_cls  # drop the strong reference before asserting, so a failure leaves nothing behind
        gc.collect()  # the leaked class was promoted by the reclaim, so only a full collection frees it
        gc.collect()  # second pass: finalizer-freed cycles can release further classes
        assert reported == [expected], f"a strongly referenced subclass must be reported, got {reported}"
        assert expected not in _registered_names(), "the deliberate leak must not outlive this test"

    def test_reclaims_a_subclass_owned_by_another_module(self) -> None:
        """A helper-made subclass is collected too; only the RETURN value stays filtered to module_name.

        Generational GC is paused for the window, so the reclaim under test is the only collection that runs.
        """
        gc.disable()
        try:
            mark = FEATURE_GROUPS.mark()
            name = registry_isolation_probe.define_helper_subclass()
            registered_before = name in _registered_names_of(PROBE_MODULE)
            reported = [c.__name__ for c in reclaim_leaked_feature_groups(mark, __name__)]
            registered_after = name in _registered_names_of(PROBE_MODULE)
        finally:
            gc.enable()
            gc.collect()  # never leave the probe behind for the next test on this worker

        assert registered_before, "the probe never registered; the reclaim assertion would prove nothing"
        assert reported == [], f"a class this module does not own must not be reported, got {reported}"
        assert not registered_after, f"{name} survived the reclaim: an unowned class must still be collected"

    @pytest.mark.parametrize("builder", ["helper_module", "type_built"])
    def test_reclaims_another_modules_promoted_transient(self, builder: str) -> None:
        """A foreign-module class that was promoted while held is reclaimed, not reported."""
        was_enabled = gc.isenabled()  # restore what was found, so a paused GC stays paused
        gc.disable()
        try:
            mark = FEATURE_GROUPS.mark()
            if builder == "helper_module":
                name = registry_isolation_probe.define_helper_subclass()
            else:
                name = type("PromotedForeignRegistryProbeFeatureGroup", (FeatureGroup,), {"__module__": "abc"}).__name__
            held = [c for c in FEATURE_GROUPS.live_since(mark) if c.__name__ == name]
            assert len(held) == 1, "the probe never registered; the reclaim assertion would prove nothing"
            gc.collect()  # promotes the held class out of the young generation
            del held
            reported = [c.__name__ for c in reclaim_leaked_feature_groups(mark, __name__)]
            survivors = [c.__name__ for c in FEATURE_GROUPS.live_since(mark) if c.__name__ == name]
        finally:
            if was_enabled:
                gc.enable()
            gc.collect()  # never leave the probe behind for the next test on this worker

        assert reported == [], f"a class this module does not own must not be reported, got {reported}"
        assert survivors == [], f"{name} was promoted, then released, and must still be reclaimed: {survivors}"

    def test_no_new_subclasses_reports_nothing(self) -> None:
        """The cheap path: nothing appeared since the snapshot, so nothing is reported and no collection is needed."""
        mark = FEATURE_GROUPS.mark()
        assert reclaim_leaked_feature_groups(mark, __name__) == []

    @pytest.mark.parametrize(("log", "root", "name", "error", "define"), _REJECTED_DEFINITIONS)
    def test_a_rejected_definition_is_still_reclaimed(
        self, log: SubclassLog[Any], root: type[Any], name: str, error: type[Exception], define: Callable[[], None]
    ) -> None:
        mark = log.mark()
        with pytest.raises(error):
            define()

        assert name in _registered_names_of_root(root), (
            "the rejected class never lingered; the reclaim assertion proves nothing"
        )
        assert reclaim_leaked_subclasses(log, mark, __name__) == []
        assert name not in _registered_names_of_root(root), f"{name} survived the reclaim"

    def test_reclaims_a_promoted_type_built_subclass(self) -> None:
        """A type()-built subclass gets __module__ 'abc'; once promoted past generation 0 it is still reclaimed."""
        name = "PromotedTypeBuiltRegistryProbeFeatureGroup"
        mark = FEATURE_GROUPS.mark()
        collections: list[int] = []
        with _counted_collections(collections):
            _build_type_built_subclass_and_promote(name)
            ladder_start = len(collections)
            reported = reclaim_leaked_feature_groups(mark, __name__)
        escalation = collections[ladder_start:]

        assert reported == [], f"a class the test did not keep must not be reported, got {reported}"
        survivors = {c.__name__ for c in FEATURE_GROUPS.live_since(mark)}
        assert name not in survivors, f"{name} survived the reclaim: its __module__ is 'abc', so it was never seen"
        assert collections[0] == 0 and len(escalation) > 1 and escalation[0] == 0, (
            f"the ladder must escalate past generation 0 for a promoted class, got generations {collections}"
        )

    @pytest.mark.parametrize(
        ("build", "expected_module"),
        [
            pytest.param(
                lambda tmp_path: (type("LeakedTypeBuiltRegistryProbeFeatureGroup", (FeatureGroup,), {}),),
                "abc",
                id="type_built_abc",
            ),
            pytest.param(
                lambda tmp_path: (
                    type(
                        "LeakedTypeBuiltRegistryProbeFeatureGroup",
                        (FeatureGroup,),
                        {"__module__": "tests.mixins.registry_isolation_leak_probe"},
                    ),
                ),
                "tests.mixins.registry_isolation_leak_probe",
                id="mixin_module",
            ),
            pytest.param(
                lambda tmp_path: root_binding_groups(PyArrowTable, lambda *args: None)[:2],
                registry_isolation.SHARED_TEST_MODULE_PREFIX,
                id="root_binding_groups",
            ),
            pytest.param(
                lambda tmp_path: (neutral_csv_group(tmp_path / "n.csv"),),
                registry_isolation.SHARED_TEST_MODULE_PREFIX,
                id="neutral_csv_group",
            ),
        ],
    )
    def test_reports_a_strongly_held_subclass(
        self, tmp_path: Path, build: Callable[[Path], tuple[type[FeatureGroup], ...]], expected_module: str
    ) -> None:
        """A held subclass built by a probe or a shared helper is reported."""
        mark = FEATURE_GROUPS.mark()
        held = build(tmp_path)
        modules = [c.__module__ for c in held]
        # a trailing dot means prefix match (shared builders); otherwise the module must match exactly
        assert all(
            m == expected_module or (expected_module.endswith(".") and m.startswith(expected_module)) for m in modules
        ), f"the probe/builder does not take {expected_module}, got {modules}; the test proves nothing"
        expected = sorted(c.__name__ for c in held)
        reported = [c.__name__ for c in reclaim_leaked_feature_groups(mark, __name__)]
        del held  # drop the strong references before asserting, so a failure leaves nothing behind
        gc.collect()
        gc.collect()
        assert reported == expected, f"a strongly held subclass must be reported, got {reported}"
        survivors = {c.__name__ for c in FEATURE_GROUPS.live_since(mark)}
        assert not survivors & set(expected), "the deliberate leak must not outlive this test"


def _build_type_built_subclass_and_promote(name: str) -> None:
    """Build a subclass without __module__, keep it across a generation 0 collection, then drop every reference."""
    cls = type(name, (FeatureGroup,), {})
    assert cls.__module__ == "abc", "the probe no longer takes its module from abc; the test proves nothing"
    gc.collect(0)
    del cls


class TestReclaimLeakedTransformers:
    """reclaim_leaked_transformers(mark, module_name) is the same mechanism over the BaseTransformer root."""

    def test_reclaims_a_throwaway_transformer_subclass(self) -> None:
        """A transient subclass is registered, then reclaimed. One test, not two: xdist could split a pair."""
        mark = TRANSFORMERS.mark()
        name = _define_throwaway_transformer()
        assert name in _registered_transformer_names(), (
            "the probe never registered; the reclaim assertion would prove nothing"
        )
        assert reclaim_leaked_transformers(mark, __name__) == []
        assert name not in _registered_transformer_names(), f"{name} survived the reclaim"

    def test_reports_a_genuine_transformer_leak(self) -> None:
        """A strongly referenced subclass is reported, so the conftest fixture fails loudly instead of hiding it."""
        mark = TRANSFORMERS.mark()
        leaked_cls = _define_leaked_transformer()
        reported = [c.__name__ for c in reclaim_leaked_transformers(mark, __name__)]
        expected = leaked_cls.__name__
        del leaked_cls  # drop the strong reference before asserting, so a failure leaves nothing behind
        gc.collect()  # the leaked class was promoted by the reclaim, so only a full collection frees it
        gc.collect()  # second pass: finalizer-freed cycles can release further classes
        assert reported == [expected], f"a strongly referenced subclass must be reported, got {reported}"
        assert expected not in _registered_transformer_names(), "the deliberate leak must not outlive this test"


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
        gc.collect()  # the module-bound class sat in a cycle with its module; free it for the next test


@contextmanager
def _subclass_bound_in_this_module(base: type[Any] = FeatureGroup) -> Iterator[str]:
    """Bind a FeatureGroup subclass into THIS module, the shape a test that writes to its own module leaves."""
    this_module = sys.modules[__name__]
    cls = type(OWN_MODULE_CLASS, (base,), {"__module__": __name__})
    setattr(this_module, OWN_MODULE_CLASS, cls)
    del cls
    try:
        yield OWN_MODULE_CLASS
    finally:
        delattr(this_module, OWN_MODULE_CLASS)
        gc.collect()  # never leave the bound class behind for the next test on this worker


@contextmanager
def _counted_collections(counts: list[int], pause_automatic: bool = True) -> Iterator[None]:
    """Record the generation of every collection that runs, with automatic collections paused unless asked not to."""

    def on_collect(phase: str, info: dict[str, int]) -> None:
        if phase == "start":
            counts.append(info["generation"])

    was_enabled = gc.isenabled()  # restore what was found, so nesting this cannot re-enable a paused GC
    if pause_automatic:
        gc.disable()
    gc.callbacks.append(on_collect)
    try:
        yield
    finally:
        gc.callbacks.remove(on_collect)
        if pause_automatic and was_enabled:
            gc.enable()


class TestModuleOwnedSubclassesSkipTheCollection:
    """An imported module's own class is not the transient leak this reclaims, so it must not cost a collection."""

    def test_a_module_bound_subclass_costs_no_collection(self) -> None:
        """The gate reads the binding instead: a full collection per test sat close to the per-test timeout (#995)."""
        mark = FEATURE_GROUPS.mark()
        collections: list[int] = []
        with _module_bound_subclass() as name:
            assert name in _registered_names_of(SYNTHETIC_MODULE), (
                "the probe never registered; the collection assertion would prove nothing"
            )
            with _counted_collections(collections):
                reported = reclaim_leaked_feature_groups(mark, __name__)

        assert reported == [], f"a class this module does not own must not be reported, got {reported}"
        assert collections == [], f"a module-owned class must cost no collection, got generations {collections}"

    def test_a_snapshot_and_reclaim_cost_no_class_tree_walk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The log replaces the root walk: snapshot plus reclaim must never call FeatureGroup.__subclasses__."""
        root_walks = count_root_walks(monkeypatch, FeatureGroup)
        mark = FEATURE_GROUPS.mark()
        with _module_bound_subclass():
            reported = reclaim_leaked_feature_groups(mark, __name__)

        assert reported == []
        assert root_walks == [], f"snapshot and reclaim walked the FeatureGroup tree {len(root_walks)} time(s)"

    @pytest.mark.parametrize("base", [FeatureGroup, ReadDBFG])
    def test_a_subclass_bound_in_the_calling_module_is_still_reported(self, base: type[Any]) -> None:
        """Skipping the collection must not skip the report; a doubly logged class is reported exactly once."""
        mark = FEATURE_GROUPS.mark()
        with _subclass_bound_in_this_module(base) as name:
            reported = [cls.__name__ for cls in reclaim_leaked_feature_groups(mark, __name__)]

        assert reported == [name], (
            f"a class this module bound into itself outlives the test and must be reported, got {reported}"
        )

    def test_a_second_log_on_an_overlapping_root_still_logs(self) -> None:
        """Each log hooks independently: a second log under an already hooked root must record new subclasses."""
        second = SubclassLog(SourceFeatureGroup)  # type: ignore[type-abstract]  # abstract root, as for the module logs
        mark = second.mark()
        fg_mark = FEATURE_GROUPS.mark()
        name = _define_overlap_probe()

        assert name in [c.__name__ for c in second.live_since(mark)]
        assert name in [c.__name__ for c in FEATURE_GROUPS.live_since(fg_mark)]
        assert reclaim_leaked_feature_groups(fg_mark, __name__) == []

    def test_a_just_defined_subclass_is_reclaimed_by_the_young_generation(self) -> None:
        """The counterpart: the transient shape the reclaim exists for never escalates past generation 0."""
        mark = FEATURE_GROUPS.mark()
        collections: list[int] = []
        with _counted_collections(collections):
            name = _define_throwaway_subclass()
            reported = reclaim_leaked_feature_groups(mark, __name__)

        assert reported == []
        assert collections == [0], (
            f"{name} was just defined, so generation 0 must reclaim it and the ladder must stop there; "
            f"a full collection scans the whole heap and costs about a second. Got generations {collections}"
        )


def _define_held_group_during_allocations(held_objects: int) -> list[int]:
    """Hold a fresh FeatureGroup subclass across `held_objects` container allocations, release all, count the reclaim."""
    mark = FEATURE_GROUPS.mark()

    class YoungWindowRegistryProbeFeatureGroup(FeatureGroup):
        pass

    held = [[i] for i in range(held_objects)]  # tracked containers: they drive automatic young collections
    del YoungWindowRegistryProbeFeatureGroup
    del held
    collections: list[int] = []
    with _counted_collections(collections, pause_automatic=False):
        assert reclaim_leaked_feature_groups(mark, __name__) == []
    return collections


class TestYoungWindow:
    """young_window() raises the gen-0 threshold so a held test class is not promoted by automatic collections."""

    def test_allocations_past_the_default_threshold_do_not_escalate_the_ladder(self) -> None:
        """Automatic GC stays on; past the default point only the window keeps the class young, so gen 0 suffices."""
        assert registry_isolation.YOUNG_WINDOW_THRESHOLD > 5 * DEFAULT_YOUNG_THRESHOLD, (
            "the window must out-last the allocation count this test uses"
        )
        collections = _define_held_group_during_allocations(5 * DEFAULT_YOUNG_THRESHOLD)
        assert collections == [0], f"the class was promoted and the ladder escalated, got generations {collections}"

    def test_sets_the_window_threshold_and_restores_the_tuple_on_exit(self) -> None:
        original = gc.get_threshold()
        try:
            gc.set_threshold(1234, 7, 3)
            sentinel = gc.get_threshold()  # read back: 3.14 zeroes the third slot
            with registry_isolation.young_window():
                inside = gc.get_threshold()
            after = gc.get_threshold()
        finally:
            gc.set_threshold(*original)
        assert inside == (registry_isolation.YOUNG_WINDOW_THRESHOLD, *sentinel[1:])
        assert after == sentinel
        assert sentinel[:2] == (1234, 7)

    def test_restores_the_tuple_when_the_body_raises(self) -> None:
        original = gc.get_threshold()
        try:
            gc.set_threshold(1234, 7, 3)
            sentinel = gc.get_threshold()  # read back: 3.14 zeroes the third slot
            with pytest.raises(RuntimeError, match="boom"):
                with registry_isolation.young_window():
                    raise RuntimeError("boom")
            after = gc.get_threshold()
        finally:
            gc.set_threshold(*original)
        assert after == sentinel
        assert sentinel[:2] == (1234, 7)


class TestExtraRootsAreReclaimed:
    """snapshot_roots and reclaim_since reclaim ComputeFramework and Extender subclasses (reclaim only, no report)."""

    @pytest.mark.parametrize("root", registry_isolation.EXTRA_RECLAIM_ROOTS, ids=lambda r: r.__name__)
    def test_a_dead_local_subclass_is_gone_without_a_full_collection(self, root: type) -> None:
        snapshot = registry_isolation.snapshot_roots()

        def define() -> str:
            stubs = {"wraps": lambda self: set(), "__call__": lambda self, func, *a, **kw: func(*a, **kw)}
            return type("DeadLocalRegistryProbeExtra", (root,), stubs).__name__

        name = define()
        collections: list[int] = []
        with _counted_collections(collections):
            registry_isolation.reclaim_since(snapshot, __name__)
        alive = [c.__name__ for c in registry_isolation.RECLAIM_LOGS[root].live_since(snapshot[root])]
        assert name not in alive, f"{name} survived the reclaim"
        assert collections == [0], f"a young dead class must cost generation 0 only, got {collections}"

    @pytest.mark.parametrize("root", registry_isolation.EXTRA_RECLAIM_ROOTS, ids=lambda r: r.__name__)
    def test_a_held_local_extra_root_subclass_is_not_reported(self, root: type) -> None:
        snapshot = registry_isolation.snapshot_roots()
        stubs = {"wraps": lambda self: set(), "__call__": lambda self, func, *a, **kw: func(*a, **kw)}
        held = type("HeldLocalRegistryProbeExtra", (root,), stubs)
        try:
            reported = [c.__name__ for c in registry_isolation.reclaim_since(snapshot, __name__)]
        finally:
            del held  # drop the strong reference before asserting, so a failure leaves nothing behind
            gc.collect()  # the held class is promoted by the reclaim, so only a full collection frees it
        assert reported == [], f"extra roots are reclaim-only and must not be reported, got {reported}"

    def test_a_retained_feature_group_is_still_reported(self) -> None:
        snapshot = registry_isolation.snapshot_roots()
        leaked_cls = _define_leaked_subclass()
        try:
            reported = [c.__name__ for c in registry_isolation.reclaim_since(snapshot, __name__)]
            expected = leaked_cls.__name__
        finally:
            del leaked_cls  # drop the strong reference before asserting, so a failure leaves nothing behind
            gc.collect()  # the leaked class is promoted by the reclaim, so only a full collection frees it
        assert reported == [expected], f"a retained FeatureGroup must be reported, got {reported}"


def _define_implementation() -> tuple[type[Implementation[Any]], str]:
    """A local Implementation naming a local pair; returns the class (held by the caller) and its name."""
    from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
        PythonDictFramework,
    )

    class ReleasedImplProbeFeatureGroup(SourceFeatureGroup):
        pass

    class ReleasedImplProbe(Implementation[Any]):
        feature_group = ReleasedImplProbeFeatureGroup
        compute_framework = PythonDictFramework

        @classmethod
        def calculate(cls, serving: Any, data: Any, features: Any, framework: Any) -> Any:
            return None

    return ReleasedImplProbe, ReleasedImplProbe.__name__


class TestReclaimReleasedImplementations:
    """reclaim_released_implementations(mark, module_name) is the factored make_implementation teardown."""

    def test_released_implementation_costs_no_full_collection(self) -> None:
        mark = IMPLEMENTATIONS.mark()
        impl, name = _define_implementation()
        assert impl in loaded_implementations(), "the probe is not loaded; the assertion would prove nothing"
        del impl
        collections: list[int] = []
        with _counted_collections(collections):
            leaked = registry_isolation.reclaim_released_implementations(mark, __name__)
        assert leaked == []
        assert 2 not in collections, f"a released implementation must not need generation 2, got {collections}"
        assert name not in {c.__name__ for c in IMPLEMENTATIONS.live_since(mark)}

    def test_a_retained_loaded_implementation_is_reported(self) -> None:
        mark = IMPLEMENTATIONS.mark()
        impl, name = _define_implementation()
        try:
            reported = [c.__name__ for c in registry_isolation.reclaim_released_implementations(mark, __name__)]
        finally:
            del impl  # drop the strong reference before asserting, so a failure leaves nothing behind
            gc.collect()  # the retained class was promoted by the reclaim, so only a full collection frees it
            gc.collect()  # second pass: the Implementation and its pair release in separate rounds
        assert reported == [name], f"a retained loaded implementation must be reported, got {reported}"
        assert name not in {c.__name__ for c in IMPLEMENTATIONS.live_since(mark)}

    def test_a_loaded_implementation_of_another_module_is_not_reported(self) -> None:
        mark = IMPLEMENTATIONS.mark()
        impl, _ = _define_implementation()
        impl.__module__ = "tests._other_module_probe"
        try:
            reported = [c.__name__ for c in registry_isolation.reclaim_released_implementations(mark, __name__)]
        finally:
            del impl
            gc.collect()  # never leave the retained probe behind for the next test on this worker
            gc.collect()  # second pass: the Implementation and its pair release in separate rounds
        assert reported == [], f"a class this module does not own must not be reported, got {reported}"


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


_ROOTS = (FeatureGroup, BaseTransformer, Implementation)


def _is_own_module(node: ast.expr, module_name: str) -> bool:
    return (isinstance(node, ast.Name) and node.id == "__name__") or (
        isinstance(node, ast.Constant) and node.value == module_name
    )


def _base_kind(node: ast.expr, imports: dict[str, Any], classes: dict[str, ast.ClassDef], seen: frozenset[str]) -> str:
    """Classify a base as 'root', 'other' or 'unknown' without importing tests modules."""
    if not isinstance(node, ast.Name):
        return "unknown"
    if node.id in classes:
        if node.id in seen:
            return "other"
        kinds = [_base_kind(b, imports, classes, seen | {node.id}) for b in classes[node.id].bases]
        return "root" if "root" in kinds else "other"
    if node.id in imports:
        obj = imports[node.id]
        if isinstance(obj, type):
            return "root" if issubclass(obj, _ROOTS) else "other"
    return "unknown"


def _top_level_imports(tree: ast.Module) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.ImportFrom) or stmt.level or not stmt.module:
            continue
        if stmt.module == "tests" or stmt.module.startswith("tests."):
            continue
        module = importlib.import_module(stmt.module)
        for alias in stmt.names:
            if hasattr(module, alias.name):
                found[alias.asname or alias.name] = getattr(module, alias.name)
    return found


def _sets_foreign_module(cls: ast.ClassDef, module_name: str) -> bool:
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign) and stmt.value is not None:
            names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            if "__module__" in names and not _is_own_module(stmt.value, module_name):
                return True
    return False


def _walk_builder_sites(source: str, module_name: str) -> list[tuple[str, int]]:
    tree = ast.parse(source)
    imports = _top_level_imports(tree)
    classes = {s.name: s for s in tree.body if isinstance(s, ast.ClassDef)}
    sites: list[tuple[str, int]] = []

    def visit(node: ast.AST, path: list[str], in_func: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                if in_func and not _sets_foreign_module(child, module_name):
                    kinds = [_base_kind(b, imports, classes, frozenset()) for b in child.bases]
                    if "root" in kinds or "unknown" in kinds:
                        sites.append((".".join(path), child.lineno))
                visit(child, [*path, child.name], in_func)
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                visit(child, [*path, child.name], True)
            else:
                if in_func and isinstance(child, ast.Dict):
                    for key, value in zip(child.keys, child.values, strict=True):
                        if isinstance(key, ast.Constant) and key.value == "__module__":
                            if _is_own_module(value, module_name):
                                sites.append((".".join(path), child.lineno))
                visit(child, path, in_func)

    visit(tree, [], False)
    return sites


def _builder_sites(source: str, module_name: str) -> list[str]:
    return sorted({path for path, _ in _walk_builder_sites(source, module_name)})


def _shared_builder_hits() -> dict[str, str]:
    """Map '<module>:<path>' to 'file:line' for builder sites in shared helper modules outside tests/mixins/."""
    hits: dict[str, str] = {}
    mixins = TESTS_ROOT / "mixins"
    for file in sorted(TESTS_ROOT.rglob("*.py")):
        if file.name.startswith("test_") or file.name == "__init__.py" or "__pycache__" in file.parts:
            continue
        if mixins in file.parents:
            continue
        module_name = ".".join(file.relative_to(TESTS_ROOT.parent).with_suffix("").parts)
        for path, line in sorted(_walk_builder_sites(file.read_text(), module_name), key=lambda s: s[1]):
            hits.setdefault(f"{module_name}:{path}", f"{file.relative_to(TESTS_ROOT)}:{line}")
    return hits


_SHARED_BUILDER_ALLOWLIST: dict[str, str] = {
    "tests.registry_isolation_probe:define_helper_subclass": "deliberate foreign-module probe for the reclaim gate",
    "tests.test_core.test_prepare.link_side_paths_probe:_sum_group": "called only at import, classes bound at module level",
    "tests.conftest:make_implementation.make": "the fixture's teardown reports every leaked Implementation it built",
}

_FG_IMPORT = "from mloda.core.abstract_plugins.feature_group import FeatureGroup\n"
_EXT_IMPORT = "from mloda.core.abstract_plugins.function_extender import Extender\n"

_BUILDER_CASES: list[Any] = [
    pytest.param(
        _FG_IMPORT + "def make():\n    class G(FeatureGroup): pass\n", ["make"], id="nested-from-feature-group"
    ),
    pytest.param(
        _FG_IMPORT + "def make():\n    class G(FeatureGroup):\n        __module__ = __name__\n",
        ["make"],
        id="nested-module-is-dunder-name",
    ),
    pytest.param("def make(base):\n    class G(base): pass\n", ["make"], id="nested-parameter-base"),
    pytest.param(
        _FG_IMPORT + "class Cls:\n    def method(self):\n        class G(FeatureGroup): pass\n",
        ["Cls.method"],
        id="nested-in-method",
    ),
    pytest.param(
        _FG_IMPORT + "class Local(FeatureGroup): pass\ndef make():\n    class G(Local): pass\n",
        ["make"],
        id="base-is-same-source-feature-group-subclass",
    ),
    pytest.param(
        _FG_IMPORT + "def make(name):\n    return type(name, (FeatureGroup,), {'__module__': __name__})\n",
        ["make"],
        id="type-with-dunder-name-module",
    ),
    pytest.param(
        "def make(meta, base, name):\n    return meta(name, (base,), {'__module__': __name__})\n",
        ["make"],
        id="metaclass-call-with-dunder-name-module",
    ),
    pytest.param(
        _FG_IMPORT + "def outer():\n    def inner():\n        class G(FeatureGroup): pass\n",
        ["outer.inner"],
        id="nested-function-chain",
    ),
    pytest.param(
        _FG_IMPORT + "def make(module):\n    class G(FeatureGroup):\n        __module__ = module\n",
        [],
        id="ok-module-is-parameter",
    ),
    pytest.param("def make():\n    class G: pass\n", [], id="ok-no-bases"),
    pytest.param(_EXT_IMPORT + "def make():\n    class E(Extender): pass\n", [], id="ok-extender-base"),
    pytest.param(
        _FG_IMPORT + "def make(name, module):\n    return type(name, (FeatureGroup,), {'__module__': module})\n",
        [],
        id="ok-type-with-parameter-module",
    ),
    pytest.param(
        _FG_IMPORT + "def make(name):\n    return type(name, (FeatureGroup,), {})\n", [], id="ok-type-without-module"
    ),
    pytest.param(_FG_IMPORT + "class G(FeatureGroup): pass\n", [], id="ok-module-level-class"),
]


class TestSharedSubclassBuildersLiveUnderMixins:
    """Shared helpers outside tests/mixins/ must not build subclasses the leak report cannot see."""

    @pytest.mark.parametrize(("source", "expected"), _BUILDER_CASES)
    def test_builder_sites(self, source: str, expected: list[str]) -> None:
        assert _builder_sites(source, "tests.helpers.fake") == expected

    def test_no_shared_builder_outside_mixins(self) -> None:
        hits = _shared_builder_hits()
        found = [f"{key} at {where}" for key, where in sorted(hits.items()) if key not in _SHARED_BUILDER_ALLOWLIST]
        assert not found, (
            "shared helpers build subclasses whose leaks are invisible; move the builder under tests/mixins/ "
            f"or take the caller's module as __module__: {found}"
        )

    def test_allowlist_has_no_stale_entries(self) -> None:
        stale = sorted(set(_SHARED_BUILDER_ALLOWLIST) - set(_shared_builder_hits()))
        assert not stale, f"allowlist entries no longer match a builder site: {stale}"


_CACHE_RESTORE_PROBE = "CacheRestorePinProbeFeatureGroup"
_CACHE_RESTORE_FIXTURE = "restore_dynamic_feature_group_cache"


class TestDynamicFeatureGroupCacheIsRestoredBetweenTests:
    """test_a's leak-check teardown is the deterministic pin; test_b checks the next test sees no leftover.
    test_b passes trivially if xdist runs it on another worker than test_a."""

    def test_a_creates_a_cached_class(self) -> None:
        DynamicFeatureGroupCreator.create({}, class_name=_CACHE_RESTORE_PROBE)
        assert _CACHE_RESTORE_PROBE in DynamicFeatureGroupCreator._created_classes

    def test_b_the_next_test_no_longer_sees_it(self, request: pytest.FixtureRequest) -> None:
        assert _CACHE_RESTORE_PROBE not in DynamicFeatureGroupCreator._created_classes
        assert _CACHE_RESTORE_FIXTURE in request.fixturenames
