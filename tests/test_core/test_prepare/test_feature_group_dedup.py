"""Tests for dedup_feature_group_subclasses helper and PluginCollector.set_allow_redefinition.

These tests cover the new deduplication behavior for FeatureGroup subclasses that
appear multiple times under the same (module, qualname) key. This commonly happens
in Jupyter notebooks (where IPython's Out[N] holds strong refs to old class objects)
and in `importlib.reload` flows. See plan: dedupe redefined FeatureGroup classes
in long-lived namespaces.

Helpers below simulate Jupyter cell rebinding via ``exec`` into ``__main__.__dict__``
and register the source in ``linecache`` so ``inspect.getsource`` succeeds for the
exec'd classes (mirroring how IPython sets up ``<ipython-input-N-...>`` entries).
"""

from __future__ import annotations

import gc
import linecache
import sys
import textwrap
import types
from typing import Any, cast

import pytest

from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api.plugin_docs import get_feature_group_docs, resolve_feature
from mloda.core.api.plugin_info import FeatureGroupInfo, ResolvedFeature
from mloda.core.prepare.accessible_plugins import PreFilterPlugins, dedup_feature_group_subclasses


# Module-level reference store. This simulates IPython's ``_oh``/``Out[N]`` history,
# which keeps strong refs to all cell-evaluated objects. It MUST be module-scoped so
# the references survive across all assertions inside a single test.
#
# IMPORTANT: Tests MUST use a unique qualname (e.g., ``MyFG_TestN``) per test —
# ``_REF_STORE`` is never cleared during the session, so any class registered here
# survives in ``FeatureGroup.__subclasses__()`` and would interfere with later
# tests that share the same qualname. Filter by ``__qualname__`` when asserting.
_REF_STORE: list[Any] = []


def _exec_fg_in_main(class_name: str, body: str, cell_label: str) -> type[FeatureGroup]:
    """Exec a FeatureGroup subclass into ``__main__``, registering source in linecache.

    The exec'd class behaves like a class defined in a Jupyter cell: its ``__module__``
    is ``"__main__"`` and ``inspect.getsource`` succeeds because we register the
    source text in ``linecache`` (mimicking what IPython does).

    The class is also bound as an attribute on ``sys.modules['__main__']`` so that
    ``getattr(sys.modules['__main__'], class_name)`` returns the most recently
    exec'd version, simulating Jupyter's "live in module" state after a cell rerun.
    """
    main_mod = sys.modules["__main__"]
    src = textwrap.dedent(body)
    filename = f"<{cell_label}>"
    src_lines = src.splitlines(keepends=True)
    linecache.cache[filename] = (len(src), None, src_lines, filename)
    code_obj = compile(src, filename, "exec")
    exec(code_obj, main_mod.__dict__)  # nosec B102
    cls = main_mod.__dict__[class_name]
    return cast(type[FeatureGroup], cls)


def _exec_fg_in_module(class_name: str, body: str, cell_label: str, module: types.ModuleType) -> type[FeatureGroup]:
    """Exec a FeatureGroup subclass into a synthetic real module, registering source in linecache.

    Mirrors ``_exec_fg_in_main`` but targets a caller-provided module object (which the
    caller has inserted into ``sys.modules``), simulating an ``importlib.reload`` flow
    in a real (non-``__main__``) module. ``types.ModuleType`` sets ``__name__``
    automatically, and exec'd class definitions pick up ``__module__`` from the
    globals' ``__name__``, so the exec'd class reports the synthetic module as its
    ``__module__``. Source is registered in ``linecache`` under a distinct synthetic
    filename so source hashes resolve via the linecache AST fallback.
    """
    src = textwrap.dedent(body)
    filename = f"<{cell_label}>"
    src_lines = src.splitlines(keepends=True)
    linecache.cache[filename] = (len(src), None, src_lines, filename)
    code_obj = compile(src, filename, "exec")
    exec(code_obj, module.__dict__)  # nosec B102
    cls = module.__dict__[class_name]
    return cast(type[FeatureGroup], cls)


def _make_fg_source(class_name: str, feature_name: str, extra_body: str = "") -> str:
    """Build a FeatureGroup subclass source string for a given class name and feature.

    The body imports FeatureGroup directly from the abstract_plugins module so that
    ``exec`` into ``__main__.__dict__`` does not need any pre-populated names.
    """
    return f"""
from mloda.core.abstract_plugins.feature_group import FeatureGroup as _FG_BASE_

class {class_name}(_FG_BASE_):
    @classmethod
    def feature_names_supported(cls):
        return {{"{feature_name}"}}
{extra_body}
"""


@pytest.fixture(autouse=True)
def _cleanup_main_module_attrs() -> Any:
    """Snapshot ``__main__`` attributes before each test and restore after.

    This guarantees parallel-safety under ``pytest-xdist -n 8`` and prevents
    bleed-over of test-defined classes into other tests sharing the same xdist worker.
    Note: classes referenced from ``_REF_STORE`` still survive in
    ``FeatureGroup.__subclasses__()`` for the lifetime of the test process,
    which is intentional. Tests filter by their unique qualname to avoid
    interference from other tests.
    """
    main_mod = sys.modules["__main__"]
    snapshot = set(main_mod.__dict__.keys())
    yield
    new_keys = set(main_mod.__dict__.keys()) - snapshot
    for k in new_keys:
        main_mod.__dict__.pop(k, None)


def _filter_by_qualname(classes: set[type[FeatureGroup]], qualname: str) -> set[type[FeatureGroup]]:
    """Filter a set of classes to those with a matching ``__qualname__``."""
    return {c for c in classes if c.__qualname__ == qualname}


# ---------------------------------------------------------------------------
# Case 1: Identical content, double-define in __main__, deduped to 1
# ---------------------------------------------------------------------------
def test_identical_content_double_define_dedups_to_one() -> None:
    """Two identical-source FG defs in __main__ should dedup to a single class."""
    qualname = "MyFG_Test1"
    feature_name = "case1_feature_unique_xyz"
    src = _make_fg_source(qualname, feature_name)

    v1 = _exec_fg_in_main(qualname, src, "cell-test1-v1")
    v2 = _exec_fg_in_main(qualname, src, "cell-test1-v2")

    # Hold strong refs to simulate IPython's _oh history keeping both versions alive.
    _REF_STORE.extend([v1, v2])

    all_fgs = get_all_subclasses(FeatureGroup)
    deduped = dedup_feature_group_subclasses(all_fgs)

    matching = _filter_by_qualname(deduped, qualname)
    assert len(matching) == 1, (
        f"Expected exactly one class with qualname={qualname!r} after dedup, "
        f"got {len(matching)}: {sorted(c.__module__ for c in matching)}"
    )
    only = next(iter(matching))
    assert only.__module__ == "__main__"
    assert only.__qualname__ == qualname


# ---------------------------------------------------------------------------
# Case 2: Different content, double-define, raises ValueError
# ---------------------------------------------------------------------------
def test_different_content_double_define_raises_value_error() -> None:
    """Two different-source FG defs (same qualname/module) should raise ValueError.

    The error message must contain both module paths, two truncated source hashes,
    the synthetic cell labels (so users can locate the redef'd cell), and the
    literal string ``set_allow_redefinition`` to guide the user toward the
    override flag.
    """
    qualname = "MyFG_Test2"
    feature_name = "case2_feature_unique_xyz"
    src_v1 = _make_fg_source(qualname, feature_name)
    src_v2 = _make_fg_source(
        qualname,
        feature_name,
        extra_body="    def extra_method(self):\n        return 42\n",
    )

    v1 = _exec_fg_in_main(qualname, src_v1, "cell-test2-v1")
    v2 = _exec_fg_in_main(qualname, src_v2, "cell-test2-v2")
    _REF_STORE.extend([v1, v2])

    all_fgs = get_all_subclasses(FeatureGroup)

    with pytest.raises(ValueError) as exc_info:
        dedup_feature_group_subclasses(all_fgs)

    msg = str(exc_info.value)

    # Module path appears (both versions live in __main__)
    assert "__main__" in msg, f"Expected '__main__' in error, got: {msg}"

    # Synthetic cell labels appear so users can locate the redef'd cell.
    assert "<cell-test2-v1>" in msg, f"Expected '<cell-test2-v1>' in error, got: {msg}"
    assert "<cell-test2-v2>" in msg, f"Expected '<cell-test2-v2>' in error, got: {msg}"

    # Truncated source hashes (8-char hex prefixes are sufficient indicators).
    # Compute expected hashes deterministically.
    from mloda.core.abstract_plugins.components.base_feature_group_version import BaseFeatureGroupVersion

    h1 = BaseFeatureGroupVersion.class_source_hash(v1)[:8]
    h2 = BaseFeatureGroupVersion.class_source_hash(v2)[:8]
    assert h1 in msg, f"Expected hash prefix {h1!r} in error, got: {msg}"
    assert h2 in msg, f"Expected hash prefix {h2!r} in error, got: {msg}"

    # Hint pointing at the override flag.
    assert "set_allow_redefinition" in msg, f"Expected 'set_allow_redefinition' in error, got: {msg}"


# ---------------------------------------------------------------------------
# Case 3: Different content + override flag
# ---------------------------------------------------------------------------
def test_different_content_with_allow_redefinition_returns_live_class() -> None:
    """With allow_redefinition=True, dedup picks the live class (no error)."""
    qualname = "MyFG_Test3"
    feature_name = "case3_feature_unique_xyz"
    src_v1 = _make_fg_source(qualname, feature_name)
    src_v2 = _make_fg_source(
        qualname,
        feature_name,
        extra_body="    def extra_method(self):\n        return 99\n",
    )

    v1 = _exec_fg_in_main(qualname, src_v1, "cell-test3-v1")
    v2 = _exec_fg_in_main(qualname, src_v2, "cell-test3-v2")
    _REF_STORE.extend([v1, v2])

    all_fgs = get_all_subclasses(FeatureGroup)
    deduped = dedup_feature_group_subclasses(all_fgs, allow_redefinition=True)

    matching = _filter_by_qualname(deduped, qualname)
    assert len(matching) == 1, f"Expected exactly one class with qualname={qualname!r}, got {len(matching)}"
    survivor = next(iter(matching))

    # The survivor must be the live-in-__main__ class (i.e. v2).
    live = getattr(sys.modules["__main__"], qualname)
    assert survivor is live, "Expected survivor to be the live class in __main__"
    assert survivor is v2, "Expected the newer (v2) class to survive when allow_redefinition=True"


# ---------------------------------------------------------------------------
# Case 4: Group with live descendants is preserved untouched
# ---------------------------------------------------------------------------
def test_group_with_live_descendants_is_preserved() -> None:
    """If any member of a (module, qualname) group has live descendants, dedup must
    not drop any member of the group — dropping the older member would orphan its
    subclass.
    """
    parent_qualname = "MyFG_Desc1"
    sub_qualname = "MyFG_Desc1_Sub"
    feature_name = "case4_feature_unique_xyz"

    # v1: parent class
    parent_src_v1 = _make_fg_source(parent_qualname, feature_name)
    parent_v1 = _exec_fg_in_main(parent_qualname, parent_src_v1, "cell-test4-parent-v1")

    # Define a Sub class inheriting from v1 of the parent. This MUST be exec'd while
    # parent_qualname in __main__ resolves to parent_v1.
    sub_src = textwrap.dedent(
        f"""
        class {sub_qualname}({parent_qualname}):
            @classmethod
            def feature_names_supported(cls):
                return {{"{feature_name}_sub"}}
        """
    )
    main_mod = sys.modules["__main__"]
    sub_filename = "<cell-test4-sub>"
    sub_lines = sub_src.splitlines(keepends=True)
    linecache.cache[sub_filename] = (len(sub_src), None, sub_lines, sub_filename)
    sub_code = compile(sub_src, sub_filename, "exec")
    exec(sub_code, main_mod.__dict__)  # nosec B102
    sub_cls = main_mod.__dict__[sub_qualname]

    # v2: redefine the parent (rebinds name in __main__ to a new class object).
    parent_src_v2 = _make_fg_source(parent_qualname, feature_name)
    parent_v2 = _exec_fg_in_main(parent_qualname, parent_src_v2, "cell-test4-parent-v2")

    _REF_STORE.extend([parent_v1, sub_cls, parent_v2])

    # Sanity: parent_v1 still has Sub as a live subclass (Python doesn't auto-detach).
    assert sub_cls in parent_v1.__subclasses__(), "Sub class should still chain to parent_v1"

    all_fgs = get_all_subclasses(FeatureGroup)
    deduped = dedup_feature_group_subclasses(all_fgs)

    parent_matches = _filter_by_qualname(deduped, parent_qualname)
    sub_matches = _filter_by_qualname(deduped, sub_qualname)

    assert parent_v1 in parent_matches, "parent_v1 must be preserved (Sub depends on it)"
    assert parent_v2 in parent_matches, "parent_v2 must be preserved (live descendants in group)"
    assert sub_cls in sub_matches, "Sub class itself must survive in the deduped set"


# ---------------------------------------------------------------------------
# Case 5: type(...)-built classes (no source available) are preserved without raising
# ---------------------------------------------------------------------------
def test_type_built_classes_preserved_without_source() -> None:
    """Dynamically built classes (via ``type(...)``) where ``inspect.getsource`` fails
    must be preserved by dedup; no exception, no false dedup.
    """
    qualname = "MyFG_Dyn1"
    feature_name = "case5_feature_unique_xyz"

    def _make_dyn(suffix: str) -> type[FeatureGroup]:
        # __module__ = "__main__" mirrors a Jupyter factory pattern (the realistic
        # source-unavailable shape) and prevents these classes from leaking into
        # downstream tests that iterate FeatureGroup.__subclasses__() and call
        # version() — get_feature_group_docs filters out __main__ classes.
        body: dict[str, Any] = {
            "feature_names_supported": classmethod(lambda cls: {feature_name + suffix}),
            "__module__": "__main__",
        }
        return type(qualname, (FeatureGroup,), body)

    dyn_v1 = _make_dyn("_a")
    dyn_v2 = _make_dyn("_b")
    _REF_STORE.extend([dyn_v1, dyn_v2])

    # Sanity: inspect.getsource should fail for both (no file backing).
    import inspect

    with pytest.raises((OSError, TypeError)):
        inspect.getsource(dyn_v1)

    all_fgs = get_all_subclasses(FeatureGroup)

    # Must not raise.
    deduped = dedup_feature_group_subclasses(all_fgs)

    matches = _filter_by_qualname(deduped, qualname)
    assert dyn_v1 in matches, "dyn_v1 must be preserved (source unavailable -> preserve group)"
    assert dyn_v2 in matches, "dyn_v2 must be preserved (source unavailable -> preserve group)"


# ---------------------------------------------------------------------------
# Mock compute framework for cases 6 and 7
# ---------------------------------------------------------------------------
class MockCFW(ComputeFramework):
    """Mock compute framework that is always available, for end-to-end dedup tests."""

    @staticmethod
    def is_available() -> bool:
        return True


# ---------------------------------------------------------------------------
# Module-level catalog anchor for the get_feature_group_docs graceful-degradation
# tests (cases 13, 19, 20). It lives in this test module (NOT __main__) and has
# real source on disk, so it is always documentable. This guarantees the "rest
# of the catalog" is non-empty even when this file runs in isolation, where no
# other non-__main__ FeatureGroup subclasses are imported.
# ---------------------------------------------------------------------------
class DocsCatalogAnchorFG(FeatureGroup):
    """Anchor feature group guaranteeing a non-empty documentable catalog in this module."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"docs_catalog_anchor_feature_unique_xyz"}


# ---------------------------------------------------------------------------
# Case 6: plugin_collector=None end-to-end via PreFilterPlugins
# ---------------------------------------------------------------------------
def test_pre_filter_plugins_dedups_with_none_collector() -> None:
    """PreFilterPlugins(plugin_collector=None) must run dedup with default
    allow_redefinition=False, so duplicate identical-content classes collapse
    into a single accessible_plugins entry.
    """
    qualname = "MyFG_Test6"
    feature_name = "case6_feature_unique_xyz"
    src = _make_fg_source(qualname, feature_name)

    v1 = _exec_fg_in_main(qualname, src, "cell-test6-v1")
    v2 = _exec_fg_in_main(qualname, src, "cell-test6-v2")
    _REF_STORE.extend([v1, v2])

    pre_filter = PreFilterPlugins(compute_frameworks={MockCFW}, plugin_collector=None)
    accessible = pre_filter.get_accessible_plugins()

    # Filter to our test's qualname only — other tests on the same xdist worker
    # may have leaked subclasses into the registry.
    matching = [fg for fg in accessible.keys() if fg.__qualname__ == qualname]
    assert len(matching) == 1, (
        f"Expected exactly one accessible_plugins entry with qualname={qualname!r}, "
        f"got {len(matching)}: {[(c.__module__, id(c)) for c in matching]}"
    )


# ---------------------------------------------------------------------------
# Case 7: disabled_feature_groups({StaleFG}) order check (filter-then-dedup)
# ---------------------------------------------------------------------------
def test_disabled_feature_groups_filter_runs_before_dedup() -> None:
    """The enable/disable filter (by class identity) must run BEFORE dedup.

    This guarantees that a stale class ref passed in ``disabled_feature_groups({...})``
    matches the right object by identity, regardless of dedup behavior.
    """
    qualname = "MyFG_Order1"
    feature_name = "case7_feature_unique_xyz"
    src = _make_fg_source(qualname, feature_name)

    v1 = _exec_fg_in_main(qualname, src, "cell-test7-v1")
    v2 = _exec_fg_in_main(qualname, src, "cell-test7-v2")
    _REF_STORE.extend([v1, v2])

    plugin_collector = PluginCollector.disabled_feature_groups({v1})

    pre_filter = PreFilterPlugins(compute_frameworks={MockCFW}, plugin_collector=plugin_collector)
    accessible = pre_filter.get_accessible_plugins()

    matching = [fg for fg in accessible.keys() if fg.__qualname__ == qualname]
    assert len(matching) == 1, (
        f"Expected exactly one accessible_plugins entry with qualname={qualname!r}, got {len(matching)}"
    )
    # v1 was disabled by identity BEFORE dedup ran, so the survivor must be v2.
    assert matching[0] is v2, "Expected v2 to survive (v1 was disabled by identity before dedup)"
    assert v1 not in accessible, "v1 (stale, disabled) must not appear in accessible_plugins"


# ---------------------------------------------------------------------------
# Case 8: resolve_feature after Jupyter-style redefinition returns a single FG
# ---------------------------------------------------------------------------
def test_resolve_feature_after_jupyter_redefinition_succeeds() -> None:
    """``resolve_feature`` must dedup before reporting "Multiple FeatureGroups match"
    so that a Jupyter cell rerun (identical content) returns a single resolved class.
    """
    qualname = "MyFG_Test8"
    feature_name = "case8_feature_unique_xyz"
    src = _make_fg_source(qualname, feature_name)

    v1 = _exec_fg_in_main(qualname, src, "cell-test8-v1")
    v2 = _exec_fg_in_main(qualname, src, "cell-test8-v2")
    _REF_STORE.extend([v1, v2])

    result = resolve_feature(feature_name)

    assert result.error is None, f"resolve_feature should succeed after dedup (no error), got error: {result.error!r}"
    assert result.feature_group is not None, "resolve_feature should return a feature_group"
    assert result.feature_group.__qualname__ == qualname, (
        f"resolved feature_group qualname must be {qualname!r}, got {result.feature_group.__qualname__!r}"
    )
    assert result.feature_group.__module__ == "__main__"


# ---------------------------------------------------------------------------
# Case 9: resolve_feature must not raise on redefinition conflict
# ---------------------------------------------------------------------------
def test_resolve_feature_returns_error_for_redef_conflict_does_not_raise() -> None:
    """``resolve_feature`` is a non-throwing debug API: when ``dedup_feature_group_subclasses``
    detects a different-content redefinition conflict, the error must surface in
    ``ResolvedFeature.error`` rather than as an unhandled ``ValueError``.
    """
    qualname = "MyFG_Test9"
    feature_name = "case9_feature_unique_xyz"
    src_v1 = _make_fg_source(qualname, feature_name)
    src_v2 = _make_fg_source(
        qualname,
        feature_name,
        extra_body="    def extra_method(self):\n        return 42\n",
    )

    v1 = _exec_fg_in_main(qualname, src_v1, "cell-test9-v1")
    v2 = _exec_fg_in_main(qualname, src_v2, "cell-test9-v2")
    _REF_STORE.extend([v1, v2])

    result = resolve_feature(feature_name)

    assert isinstance(result, ResolvedFeature), f"Expected ResolvedFeature, got {type(result).__name__}"
    assert result.feature_group is None, (
        f"feature_group must be None when redefinition conflict prevents resolution, got {result.feature_group!r}"
    )
    assert result.error is not None, "error must be populated when resolution cannot proceed"
    assert any(token in result.error for token in (qualname, "redefined", "set_allow_redefinition")), (
        f"error must mention the qualname, 'redefined', or 'set_allow_redefinition'; got: {result.error!r}"
    )

    # candidates must surface the conflicting classes so callers can inspect them programmatically.
    assert v1 in result.candidates, f"v1 must be in candidates on conflict, got: {result.candidates}"
    assert v2 in result.candidates, f"v2 must be in candidates on conflict, got: {result.candidates}"


# ---------------------------------------------------------------------------
# Case 10: class_source_hash must be stable across unrelated cell edits
# ---------------------------------------------------------------------------
def test_class_source_hash_is_stable_across_unrelated_cell_edits() -> None:
    """``class_source_hash`` must be stable when the FG class body is unchanged but
    surrounding (non-class) cell content differs.

    In Jupyter, each cell maps to a single synthetic linecache filename whose entry
    contains the WHOLE cell text. The current ``_linecache_source_for_class``
    fallback hashes the entire cell, so editing an unrelated comment or variable
    in the same cell flips the hash for an unchanged FeatureGroup class. That
    triggers a false "FeatureGroup redefined with different source code" conflict
    and forces users into ``set_allow_redefinition()`` or kernel restart.

    This test asserts the invariant: identical class body => identical hash,
    regardless of unrelated cell content. It will FAIL today because the hashes
    differ, and PASS once the fallback extracts only the class-local source region.
    """
    qualname = "MyFG_BugA1"

    # Define the class body ONCE so v1 and v2 are byte-for-byte identical for the
    # class region. Any drift here would invalidate the test.
    class_body = textwrap.dedent(
        f"""
        class {qualname}(_FG_):
            @classmethod
            def feature_names_supported(cls):
                return {{"bug_a1_feature_unique_xyz"}}
        """
    )

    cell_v1 = (
        "from mloda.core.abstract_plugins.feature_group import FeatureGroup as _FG_\n"
        "\n"
        "# A comment that changes between cell runs\n"
        "unrelated_var = 1\n"
        f"{class_body}"
    )
    cell_v2 = (
        "from mloda.core.abstract_plugins.feature_group import FeatureGroup as _FG_\n"
        "\n"
        "# A different comment\n"
        "unrelated_var = 99\n"
        f"{class_body}"
    )

    v1 = _exec_fg_in_main(qualname, cell_v1, "ipython-input-bug-v1")
    # Hold v1 BEFORE the second exec rebinds __main__.MyFG_BugA1 to a new class.
    _REF_STORE.append(v1)
    v2 = _exec_fg_in_main(qualname, cell_v2, "ipython-input-bug-v2")
    _REF_STORE.append(v2)

    from mloda.core.abstract_plugins.components.base_feature_group_version import BaseFeatureGroupVersion

    h1 = BaseFeatureGroupVersion.class_source_hash(v1)
    h2 = BaseFeatureGroupVersion.class_source_hash(v2)

    assert h1 == h2, (
        "Identical class bodies must produce equal source hashes regardless of "
        f"unrelated cell content (got {h1[:16]}... vs {h2[:16]}...)"
    )


# ---------------------------------------------------------------------------
# Case 11: resolve_feature(plugin_collector=...) accepts the override flag
# ---------------------------------------------------------------------------
def test_resolve_feature_with_allow_redefinition_collector_succeeds() -> None:
    """``resolve_feature`` must accept a ``plugin_collector`` parameter so a user
    facing a different-source redefinition conflict can pass
    ``PluginCollector().set_allow_redefinition()`` and get the live class back,
    matching the troubleshooting doc claim that this collector flows through to
    ``resolve_feature``.

    Today this test fails with
    ``TypeError: resolve_feature() got an unexpected keyword argument 'plugin_collector'``
    because the signature is ``resolve_feature(feature_name: str)``. After the
    green phase adds the parameter, this test must pass: dedup keeps the live
    class when ``allow_redefinition=True``, and the resolved feature_group is v2.
    """
    qualname = "MyFG_Test10"
    feature_name = "case10_feature_unique_xyz"
    src_v1 = _make_fg_source(qualname, feature_name)
    src_v2 = _make_fg_source(
        qualname,
        feature_name,
        extra_body="    def extra_method(self):\n        return 7\n",
    )

    v1 = _exec_fg_in_main(qualname, src_v1, "cell-test10-v1")
    v2 = _exec_fg_in_main(qualname, src_v2, "cell-test10-v2")
    _REF_STORE.extend([v1, v2])

    plugin_collector = PluginCollector().set_allow_redefinition()
    result = resolve_feature(feature_name, plugin_collector=plugin_collector)

    assert result.error is None, (
        f"resolve_feature with allow_redefinition collector must succeed, got error: {result.error!r}"
    )
    assert result.feature_group is not None, "resolve_feature should return a feature_group"

    live = getattr(sys.modules["__main__"], qualname)
    assert result.feature_group is live, "Expected resolved feature_group to be the live class in __main__"
    assert result.feature_group is v2, "Expected v2 (the newer redefinition) to be the resolved class"
    assert result.feature_group.__qualname__ == qualname


# ---------------------------------------------------------------------------
# Case 16: resolve_feature filters unrelated conflicts from candidates
# ---------------------------------------------------------------------------
def test_resolve_feature_filters_unrelated_conflicts_from_candidates() -> None:
    """When dedup raises on a redef whose classes don't match the requested feature,
    candidates must not include those unrelated classes — candidates is documented
    as 'all matching candidates', not 'all conflicts seen during dedup'.
    """
    qualname = "BarFG_Test16"
    unrelated_feature = "test16_unrelated_unique_xyz"
    requested_feature = "test16_requested_unique_xyz"  # not supported by BarFG_Test16

    src_v1 = _make_fg_source(qualname, unrelated_feature)
    src_v2 = _make_fg_source(
        qualname,
        unrelated_feature,
        extra_body="    def extra_method(self):\n        return 16\n",
    )
    v1 = _exec_fg_in_main(qualname, src_v1, "cell-test16-v1")
    v2 = _exec_fg_in_main(qualname, src_v2, "cell-test16-v2")
    _REF_STORE.extend([v1, v2])

    result = resolve_feature(requested_feature)

    # Dedup raises because BarFG_Test16 has a redef conflict.
    assert result.error is not None, "redef conflict must surface as error"

    # Critical: v1/v2 don't match requested_feature, so they must NOT appear in candidates.
    assert v1 not in result.candidates, (
        f"v1 must not appear in candidates (does not match {requested_feature!r}); got: {result.candidates}"
    )
    assert v2 not in result.candidates, (
        f"v2 must not appear in candidates (does not match {requested_feature!r}); got: {result.candidates}"
    )


# ---------------------------------------------------------------------------
# Case 12: get_feature_group_docs runs dedup, does not raise with allow flag
# ---------------------------------------------------------------------------
def test_get_feature_group_docs_with_allow_redefinition_does_not_raise() -> None:
    """``get_feature_group_docs`` must route through dedup so that a Jupyter-style
    different-source redefinition does not surface as an unhandled ``ValueError``
    when the user passes ``PluginCollector().set_allow_redefinition()``.

    Result list will not include the test class (``plugin_docs.py`` filters out
    ``__module__ == "__main__"``); the contract under test is that the dedup code
    path runs to completion without raising.
    """
    qualname = "MyFG_Test12_Docs"
    feature_name = "case12_feature_unique_xyz"
    src_v1 = _make_fg_source(qualname, feature_name)
    src_v2 = _make_fg_source(
        qualname,
        feature_name,
        extra_body="    def extra_method(self):\n        return 12\n",
    )

    v1 = _exec_fg_in_main(qualname, src_v1, "cell-test12-v1")
    v2 = _exec_fg_in_main(qualname, src_v2, "cell-test12-v2")
    _REF_STORE.extend([v1, v2])

    plugin_collector = PluginCollector().set_allow_redefinition()
    result = get_feature_group_docs(plugin_collector=plugin_collector)

    assert isinstance(result, list)
    from mloda.core.api.plugin_info import FeatureGroupInfo

    assert all(isinstance(item, FeatureGroupInfo) for item in result), (
        f"all items must be FeatureGroupInfo, got: {[type(i).__name__ for i in result]}"
    )
    # plugin_docs filters out __module__ == "__main__" — verify the test class did not leak in.
    assert all(item.module != "__main__" for item in result), "no __main__-bound classes should appear in the result"


# ---------------------------------------------------------------------------
# Case 13: get_feature_group_docs degrades gracefully on conflict without flag
# ---------------------------------------------------------------------------
def test_get_feature_group_docs_degrades_gracefully_on_redef_conflict_without_flag() -> None:
    """Without the ``allow_redefinition`` flag, a Jupyter-style different-source
    redefinition must NOT crash ``get_feature_group_docs``. The docs API is a
    read-only introspection entry point and mirrors ``resolve_feature`` (which
    catches the dedup ``ValueError`` instead of propagating it): it must degrade
    gracefully, returning a list of ``FeatureGroupInfo`` and still documenting
    the rest of the catalog despite the conflicting ``__main__`` classes.
    """
    qualname = "MyFG_Test13_Docs"
    feature_name = "case13_feature_unique_xyz"
    src_v1 = _make_fg_source(qualname, feature_name)
    src_v2 = _make_fg_source(
        qualname,
        feature_name,
        extra_body="    def extra_method(self):\n        return 13\n",
    )

    v1 = _exec_fg_in_main(qualname, src_v1, "cell-test13-v1")
    v2 = _exec_fg_in_main(qualname, src_v2, "cell-test13-v2")
    _REF_STORE.extend([v1, v2])

    result = get_feature_group_docs()

    assert isinstance(result, list), f"get_feature_group_docs must not raise on a redef conflict, got {type(result)}"
    assert all(isinstance(item, FeatureGroupInfo) for item in result), (
        f"all items must be FeatureGroupInfo, got: {[type(i).__name__ for i in result]}"
    )
    assert len(result) > 0, "the rest of the catalog must still be documented despite the conflict"


# ---------------------------------------------------------------------------
# Case 17: get_feature_group_docs filter→dedup ordering
# ---------------------------------------------------------------------------
def test_get_feature_group_docs_filter_runs_before_dedup() -> None:
    """The plugin_collector identity filter must run BEFORE dedup so that a stale
    class ref passed via disabled_feature_groups({stale}) eliminates the conflict
    before dedup sees it. Mirrors test_disabled_feature_groups_filter_runs_before_dedup
    (for PreFilterPlugins) at the plugin_docs entry point.
    """
    qualname = "MyFG_Test17_Docs"
    feature_name = "case17_feature_unique_xyz"
    src_v1 = _make_fg_source(qualname, feature_name)
    src_v2 = _make_fg_source(
        qualname,
        feature_name,
        extra_body="    def extra_method(self):\n        return 17\n",
    )
    v1 = _exec_fg_in_main(qualname, src_v1, "cell-test17-v1")
    v2 = _exec_fg_in_main(qualname, src_v2, "cell-test17-v2")
    _REF_STORE.extend([v1, v2])

    plugin_collector = PluginCollector.disabled_feature_groups({v1})

    # With v1 filtered before dedup, only v2 remains — no conflict, must not raise.
    result = get_feature_group_docs(plugin_collector=plugin_collector)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Case 18: resolve_feature filter→dedup ordering
# ---------------------------------------------------------------------------
def test_resolve_feature_filter_runs_before_dedup() -> None:
    """resolve_feature must filter via plugin_collector BEFORE dedup so a stale class
    ref passed via disabled_feature_groups({stale}) eliminates the conflict before
    dedup sees it.
    """
    qualname = "MyFG_Test18"
    feature_name = "case18_feature_unique_xyz"
    src_v1 = _make_fg_source(qualname, feature_name)
    src_v2 = _make_fg_source(
        qualname,
        feature_name,
        extra_body="    def extra_method(self):\n        return 18\n",
    )
    v1 = _exec_fg_in_main(qualname, src_v1, "cell-test18-v1")
    v2 = _exec_fg_in_main(qualname, src_v2, "cell-test18-v2")
    _REF_STORE.extend([v1, v2])

    plugin_collector = PluginCollector.disabled_feature_groups({v1})

    result = resolve_feature(feature_name, plugin_collector=plugin_collector)

    assert result.error is None, (
        f"resolve_feature should succeed when stale class is filtered before dedup, got error: {result.error!r}"
    )
    assert result.feature_group is v2, (
        f"resolved feature_group must be v2 (v1 was disabled), got: {result.feature_group}"
    )


# ---------------------------------------------------------------------------
# Case 14 + 15: linecache AST fallback in class_source_hash
# ---------------------------------------------------------------------------
def test_class_source_hash_fallback_fires_when_inspect_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``inspect.getsource`` raises ``OSError``, ``class_source_hash`` must
    fall back to ``_linecache_source_for_class`` and produce a valid SHA-256
    hash from the AST-extracted class-local source segment.

    Note: under pytest, ``__main__.__file__`` points to the pytest entry point,
    so ``inspect.getsource`` already fails naturally for ``_exec_fg_in_main``
    classes (it tries to read pytest's binary and cannot find the class
    definition). Forcing the failure with ``monkeypatch`` here makes the test
    robust against future ``inspect.getsource`` changes that might start
    succeeding for these classes.
    """
    import inspect

    from mloda.core.abstract_plugins.components.base_feature_group_version import BaseFeatureGroupVersion

    qualname = "MyFG_FallbackFires"
    feature_name = "fallback_fires_feature_unique_xyz"
    src = _make_fg_source(qualname, feature_name)

    cls = _exec_fg_in_main(qualname, src, "cell-fallback-fires-v1")
    _REF_STORE.append(cls)

    def failing_getsource(obj: Any) -> str:
        raise OSError("forced failure for fallback test")

    monkeypatch.setattr(inspect, "getsource", failing_getsource)

    h = BaseFeatureGroupVersion.class_source_hash(cls)

    assert isinstance(h, str)
    assert len(h) == 64, f"SHA-256 hash should be 64 hex chars, got {len(h)}: {h!r}"
    assert all(c in "0123456789abcdef" for c in h), f"Hash should be all-hex, got {h!r}"


def test_class_source_hash_fallback_extracts_class_local_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the fallback fires, it must extract ONLY the class-local source
    region from the cell, so identical class bodies produce identical hashes
    even when surrounding (non-class) cell content differs.

    This is the intent that drove the fallback (commit b6af301 "hash only
    class-local source in linecache fallback"). Existing happy-path test
    ``test_class_source_hash_is_stable_across_unrelated_cell_edits`` may
    pass via ``inspect.getsource`` directly; this test forces the fallback
    so the invariant is verified for that branch specifically.
    """
    import inspect

    from mloda.core.abstract_plugins.components.base_feature_group_version import BaseFeatureGroupVersion

    qualname = "MyFG_FallbackStable"

    class_body = textwrap.dedent(
        f"""
        class {qualname}(_FG_):
            @classmethod
            def feature_names_supported(cls):
                return {{"fallback_stable_feature_unique_xyz"}}
        """
    )

    cell_v1 = (
        f"from mloda.core.abstract_plugins.feature_group import FeatureGroup as _FG_\n\nunrelated_var = 1\n{class_body}"
    )
    cell_v2 = (
        "from mloda.core.abstract_plugins.feature_group import FeatureGroup as _FG_\n"
        "\n"
        "unrelated_var = 99\n"
        f"{class_body}"
    )

    v1 = _exec_fg_in_main(qualname, cell_v1, "ipython-input-fallback-stable-v1")
    _REF_STORE.append(v1)
    v2 = _exec_fg_in_main(qualname, cell_v2, "ipython-input-fallback-stable-v2")
    _REF_STORE.append(v2)

    def failing_getsource(obj: Any) -> str:
        raise OSError("forced failure for fallback test")

    monkeypatch.setattr(inspect, "getsource", failing_getsource)

    h1 = BaseFeatureGroupVersion.class_source_hash(v1)
    h2 = BaseFeatureGroupVersion.class_source_hash(v2)

    assert h1 == h2, (
        "Fallback must extract only the class-local source segment so "
        f"identical class bodies hash identically; got {h1[:16]}... vs {h2[:16]}..."
    )


# ---------------------------------------------------------------------------
# Case 19: get_feature_group_docs annotates unintrospectable classes
# ---------------------------------------------------------------------------
def test_get_feature_group_docs_annotates_unintrospectable_class_with_unavailable_version() -> None:
    """A FeatureGroup subclass built at runtime via ``type(...)`` with a fake
    ``__module__`` has no importable source file, so ``fg_class.version()`` raises
    ``TypeError`` (from ``inspect.getsource`` inside
    ``BaseFeatureGroupVersion.class_source_hash``). ``get_feature_group_docs`` must
    degrade gracefully instead of crashing the whole catalog call: the
    unintrospectable class is annotated with ``version == "unavailable"`` (not
    skipped), and all other classes are still documented.

    NOTE: this class is held only via a LOCAL reference (enough to keep it alive in
    ``FeatureGroup.__subclasses__()`` during the call). It is deliberately NOT
    appended to ``_REF_STORE``: that would permanently leak a non-``__main__``
    class into every later ``get_feature_group_docs()`` result on the same xdist
    worker. After the assertions the local ref is deleted and ``gc.collect()`` runs
    so the class is unregistered from ``__subclasses__``.
    """
    fake_module = "some_fake_module_name_xyz_19"
    fake_cls = cast(
        type[FeatureGroup],
        type("MyFG_Test19_FakeModule", (FeatureGroup,), {"__module__": fake_module}),
    )

    result = get_feature_group_docs()

    assert isinstance(result, list), (
        f"get_feature_group_docs must not raise on an unintrospectable class, got {type(result)}"
    )
    assert all(isinstance(item, FeatureGroupInfo) for item in result), (
        f"all items must be FeatureGroupInfo, got: {[type(i).__name__ for i in result]}"
    )

    fake_entries = [item for item in result if item.module == fake_module]
    assert len(fake_entries) == 1, (
        f"the unintrospectable class must be documented (annotate, not skip); "
        f"expected exactly one entry for module {fake_module!r}, got: {fake_entries}"
    )
    assert fake_entries[0].name == "MyFG_Test19_FakeModule"
    assert fake_entries[0].version == "unavailable", (
        f"version must be 'unavailable' when source introspection fails, got: {fake_entries[0].version!r}"
    )

    other_entries = [item for item in result if item.module != fake_module]
    assert len(other_entries) > 0, "the rest of the catalog must still be documented alongside the annotated class"

    # Cleanup: drop the only strong ref and collect so the fake class vanishes
    # from FeatureGroup.__subclasses__() and cannot leak into later tests.
    del fake_cls
    gc.collect()


# ---------------------------------------------------------------------------
# Case 20: graceful redef handling must not leak __main__ classes into output
# ---------------------------------------------------------------------------
def test_get_feature_group_docs_redef_conflict_does_not_leak_main_classes() -> None:
    """Graceful degradation on a redefinition conflict must not change the output
    contract of ``get_feature_group_docs``: classes living in ``__main__``
    (including the conflicting redefinitions themselves) stay filtered from the
    documented catalog. Without any collector, the call returns a list and no
    entry has ``module == "__main__"``.
    """
    qualname = "MyFG_Test20_Docs"
    feature_name = "case20_feature_unique_xyz"
    src_v1 = _make_fg_source(qualname, feature_name)
    src_v2 = _make_fg_source(
        qualname,
        feature_name,
        extra_body="    def extra_method(self):\n        return 20\n",
    )

    v1 = _exec_fg_in_main(qualname, src_v1, "cell-test20-v1")
    v2 = _exec_fg_in_main(qualname, src_v2, "cell-test20-v2")
    _REF_STORE.extend([v1, v2])

    result = get_feature_group_docs()

    assert isinstance(result, list), f"get_feature_group_docs must not raise on a redef conflict, got {type(result)}"
    assert all(item.module != "__main__" for item in result), (
        "conflicting __main__ classes must not leak into the documented catalog"
    )


# ---------------------------------------------------------------------------
# Case 21: redef conflict in a REAL module collapses to the live class
# ---------------------------------------------------------------------------
def test_get_feature_group_docs_reload_conflict_collapses_to_live_class() -> None:
    """A redefinition conflict in a real (non-``__main__``) module (the
    ``importlib.reload`` flow) must NOT surface both the stale and the live
    version of the class in the docs output. ``get_feature_group_docs`` is a
    "current state" catalog: on a redefinition conflict it must collapse each
    conflicting group to the most recently defined live class (the behavior of
    dedup with ``allow_redefinition=True``), so exactly ONE entry per
    conflicting class appears, and it is the live version.

    Today's graceful-degradation path unions ALL conflicting classes back into
    the result, so this test fails with TWO entries for the synthetic module
    (stale v1 plus live v2) instead of one.

    Cleanup runs BEFORE the assertions: a red failure must not leave the
    conflicting classes live in a real module in ``sys.modules``, or every
    later dedup on the same xdist worker would see the conflict.
    """
    module_name = "fake_reload_mod_case21"
    qualname = "MyFG_Case21_Reload"
    feature_name = "case21_feature_unique_xyz"

    mod = types.ModuleType(module_name)
    sys.modules[module_name] = mod

    src_v1 = _make_fg_source(qualname, feature_name)
    src_v2 = _make_fg_source(
        qualname,
        feature_name,
        extra_body="    def extra_method(self):\n        return 21\n",
    )

    # Local refs only (NOT _REF_STORE): these non-__main__ classes must vanish
    # from FeatureGroup.__subclasses__() at the end of this test.
    v1 = _exec_fg_in_module(qualname, src_v1, "reload-case21-v1", mod)
    v2 = _exec_fg_in_module(qualname, src_v2, "reload-case21-v2", mod)

    # Sanity: this is a genuine conflict scenario; the latest version is live
    # in the module (exec rebinding) and the source hashes differ.
    assert getattr(mod, qualname) is v2, "v2 must be the live class bound in the synthetic module"
    v1_version = v1.version()
    v2_version = v2.version()
    assert v1_version != v2_version, "the two versions must have differing source hashes for a real conflict"

    result = get_feature_group_docs()

    # Snapshot plain data, then clean up so the assertions below cannot leak
    # the live-module conflict into later tests on this worker.
    entry_versions = [item.version for item in result if item.module == module_name]

    sys.modules.pop(module_name, None)
    del mod
    del v1
    del v2
    gc.collect()

    assert len(entry_versions) == 1, (
        f"a redefinition conflict must collapse to the live class: expected exactly one "
        f"docs entry for module {module_name!r}, got {len(entry_versions)}: {entry_versions}"
    )
    assert entry_versions[0] == v2_version, (
        f"the single documented entry must be the live (most recent) version; "
        f"expected {v2_version!r}, got {entry_versions[0]!r}"
    )
    assert entry_versions[0] != v1_version, "the stale (pre-reload) version must not be the documented entry"
