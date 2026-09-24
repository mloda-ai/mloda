import ast
import gc
import importlib
import importlib.metadata
import inspect
import os
import shutil
import sys
import types
import uuid
import warnings
import weakref
from pathlib import Path
from typing import Any, Callable, Iterator, cast

import pytest

import mloda.core.version as version_module
from mloda.core.abstract_plugins.components.base_feature_group_version import (
    SOURCE_INTROSPECTION_ERRORS,
    _implementation_hash_cache,
)
from mloda.core.abstract_plugins.components.code_closure import closure_parts, dependency_entry
from mloda.core.api.plugin_docs import get_feature_group_docs
from mloda.provider import BaseFeatureGroupVersion, FeatureGroup, FeatureSet, ThirdPartyVersionMode
from mloda_plugins.feature_group.experimental.data_quality.missing_value.python_dict import (
    PythonDictMissingValueFeatureGroup,
)
from mloda_plugins.feature_group.experimental.dynamic_feature_group_factory.dynamic_feature_group_factory import (
    DynamicFeatureGroupCreator,
)
from tests.test_core.test_abstract_plugins.test_abstract_feature_group import BaseTestFeatureGroup1
from tests.test_core.test_prepare.test_feature_group_dedup import _exec_fg_in_module


class TestBaseFeatureGroupVersion:
    def test_version_composite(self, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
        # Patch importlib.metadata.version to a known value and reset the
        # mloda_version memo so the patched value is actually recomputed even
        # when an earlier test in the same worker already warmed the cache.
        # monkeypatch teardown restores both the version function and the
        # pre-test cache value, so no "1.2.3" poisoning leaks to other tests.
        monkeypatch.setattr(importlib.metadata, "version", lambda pkg: "1.2.3")
        monkeypatch.setattr(version_module, "_mloda_version_cache", None)

        # implementation_hash's INCLUDE-mode dependency walk also resolves through
        # importlib.metadata.version (via dependency_entry, itself memoized), so the
        # patched "1.2.3" must be cleared before and after this call, and the class's
        # own implementation-hash cache entry dropped, or a later test computing a real
        # dependency's version (or this class's hash) would observe the poisoned value.
        dependency_entry.cache_clear()

        def _cleanup() -> None:
            dependency_entry.cache_clear()
            _implementation_hash_cache.pop(BaseTestFeatureGroup1, None)

        request.addfinalizer(_cleanup)

        composite = BaseTestFeatureGroup1.version()
        # Expected format: "1.2.3-{module_name}-{hash}"
        expected_prefix = f"1.2.3-{BaseTestFeatureGroup1.__module__}-"
        assert composite.startswith(expected_prefix), (
            f"Composite version should start with '{expected_prefix}', got '{composite}'"
        )

        # Split the composite string into parts.
        parts = composite.split("-")
        assert len(parts) == 3, "Composite version should have three parts separated by '-'"

        # Check that the hash part is 64 hex characters (SHA-256 produces 64 hex digits).
        hash_val = parts[2]
        assert len(hash_val) == 64, "Hash length should be 64 characters"
        # Verify that the hash is valid hexadecimal.
        int(hash_val, 16)

    def test_invalid_target_class_for_hash(self) -> None:
        # Calling BaseFeatureGroupVersion.class_source_hash with a class not inheriting from FeatureGroup should raise a ValueError.
        with pytest.raises(ValueError):
            BaseFeatureGroupVersion.class_source_hash(str)


def _install_counting_getsource(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Wraps inspect.getsource with a call recorder.

    The module under test does ``import inspect`` and resolves ``inspect.getsource`` at call
    time, so patching the shared ``inspect`` module attribute intercepts its source reads.
    """
    calls: list[object] = []
    real_getsource = inspect.getsource

    def counting_getsource(obj: Any) -> str:
        calls.append(obj)
        return real_getsource(obj)

    monkeypatch.setattr(inspect, "getsource", counting_getsource)
    return calls


def _define_same_name_class_variant_a() -> type[FeatureGroup]:
    class RedefinedCacheProbeFeatureGroup(FeatureGroup):
        """Variant A of a redefined class, body deliberately distinct."""

    return RedefinedCacheProbeFeatureGroup


def _define_same_name_class_variant_b() -> type[FeatureGroup]:
    class RedefinedCacheProbeFeatureGroup(FeatureGroup):
        """Variant B of a redefined class, body deliberately different from variant A."""

    return RedefinedCacheProbeFeatureGroup


class TestClassSourceHashCaching:
    """Caching contract for BaseFeatureGroupVersion.class_source_hash.

    Within one process, the source of a given class OBJECT is read at most
    once; later calls return the cached hash, or re-raise the cached failure.
    Different class objects are cached independently, including a redefined
    class with the same name.
    """

    def test_class_source_hash_is_cached_per_class(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _install_counting_getsource(monkeypatch)

        class CachedProbeFeatureGroup(FeatureGroup):
            """Local feature group used to observe how often its source is read."""

        first = BaseFeatureGroupVersion.class_source_hash(CachedProbeFeatureGroup)
        second = BaseFeatureGroupVersion.class_source_hash(CachedProbeFeatureGroup)

        assert first == second
        reads = [obj for obj in calls if obj is CachedProbeFeatureGroup]
        assert len(reads) <= 1, (
            f"Source of CachedProbeFeatureGroup was read {len(reads)} times across two "
            "class_source_hash calls; the second call must be served from the per-class cache."
        )

    def test_class_source_hash_distinct_classes_hashed_independently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _install_counting_getsource(monkeypatch)

        class IndependentProbeFeatureGroupOne(FeatureGroup):
            """First independent probe class with its own body."""

        class IndependentProbeFeatureGroupTwo(FeatureGroup):
            """Second independent probe class with a deliberately different body."""

        hash_one = BaseFeatureGroupVersion.class_source_hash(IndependentProbeFeatureGroupOne)
        hash_two = BaseFeatureGroupVersion.class_source_hash(IndependentProbeFeatureGroupTwo)

        assert hash_one != hash_two
        reads_one = [obj for obj in calls if obj is IndependentProbeFeatureGroupOne]
        reads_two = [obj for obj in calls if obj is IndependentProbeFeatureGroupTwo]
        assert len(reads_one) == 1, "First class must trigger exactly one source read"
        assert len(reads_two) == 1, "Second class must trigger its own source read"

    def test_class_source_hash_redefined_class_gets_fresh_hash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _install_counting_getsource(monkeypatch)

        first_class = _define_same_name_class_variant_a()
        second_class = _define_same_name_class_variant_b()
        assert first_class.__name__ == second_class.__name__
        assert first_class is not second_class

        hash_first = BaseFeatureGroupVersion.class_source_hash(first_class)
        hash_first_again = BaseFeatureGroupVersion.class_source_hash(first_class)
        assert hash_first == hash_first_again
        first_reads = [obj for obj in calls if obj is first_class]
        assert len(first_reads) == 1, (
            f"Source of the first class object was read {len(first_reads)} times across two calls; "
            "the second call must hit the cache."
        )

        hash_second = BaseFeatureGroupVersion.class_source_hash(second_class)
        assert hash_second != hash_first, (
            "A new class object with the same name must be hashed fresh, not served from the old object's cache entry."
        )
        second_reads = [obj for obj in calls if obj is second_class]
        assert len(second_reads) == 1, "The redefined class object must trigger its own source read"

    @pytest.mark.parametrize(
        ("class_name", "module", "expected_error"),
        [
            ("FailedLookupRealFileProbeFG", __name__, OSError),
            ("FailedLookupFakeModuleProbeFG", "fake_module_for_failed_source_hash_probe", TypeError),
        ],
        ids=["real_file", "fake_module"],
    )
    @pytest.mark.parametrize(
        "hash_fn",
        [BaseFeatureGroupVersion.class_source_hash, BaseFeatureGroupVersion.implementation_hash],
        ids=["class_source_hash", "implementation_hash"],
    )
    def test_failed_lookup_is_cached_per_class(
        self,
        monkeypatch: pytest.MonkeyPatch,
        hash_fn: Callable[[type[Any]], str],
        class_name: str,
        module: str,
        expected_error: type[Exception],
    ) -> None:
        calls = _install_counting_getsource(monkeypatch)
        cls = cast(type[FeatureGroup], type(class_name, (FeatureGroup,), {"__module__": module}))

        with pytest.raises(expected_error) as first:
            hash_fn(cls)
        with pytest.raises(expected_error) as second:
            hash_fn(cls)

        # Clean up before asserting: a failed assertion's traceback would otherwise leak the class.
        first_error = (type(first.value), str(first.value))
        second_error = (type(second.value), str(second.value))
        reads = len([obj for obj in calls if obj is cls])
        ref = weakref.ref(cls)
        calls.clear()
        del first, second, cls
        gc.collect()

        assert second_error == first_error
        assert reads == 1, f"A failed lookup must be served from the cache, but getsource ran {reads} times"
        assert ref() is None, "A class whose lookup failed must stay garbage-collectable"


class TestMlodaVersionMemoization:
    """Memoization contract for BaseFeatureGroupVersion.mloda_version.

    Within one process, the underlying importlib.metadata lookup runs at most
    once; all subsequent mloda_version calls return the memoized string. Docs
    enumeration calls mloda_version once per FeatureGroup subclass per
    enumeration, and the repeated metadata parsing is a hot path.
    """

    def test_mloda_version_metadata_lookup_is_memoized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Earlier tests in the same process may already have populated the
        # memo; reset the module-level cache so this test observes the first
        # lookup. raising=False creates the attribute if it does not exist yet.
        monkeypatch.setattr(version_module, "_mloda_version_cache", None, raising=False)

        calls: list[str] = []
        real_metadata_version = importlib.metadata.version

        def counting_metadata_version(distribution_name: str) -> str:
            calls.append(distribution_name)
            return real_metadata_version(distribution_name)

        # The module under test does ``import importlib.metadata`` and
        # resolves ``importlib.metadata.version`` at call time, so patching
        # the shared module attribute intercepts its metadata lookups.
        monkeypatch.setattr(importlib.metadata, "version", counting_metadata_version)

        first = BaseFeatureGroupVersion.mloda_version()
        second = BaseFeatureGroupVersion.mloda_version()
        third = BaseFeatureGroupVersion.mloda_version()

        assert isinstance(first, str)
        assert first != ""
        assert first == second == third
        assert len(calls) == 1, (
            f"importlib.metadata lookup ran {len(calls)} times across three mloda_version calls; "
            "it must run at most once per process, with later calls served from the memo."
        )


# --------------------------------------------------------------------------
# implementation_hash / version() closure-walk fixtures.
#
# A fixture package "pkg" is written to tmp_path under a unique top-level
# package name per test (the pinned test uses a fixed name), then imported.
# To observe an edit's effect, the package is purged from sys.modules, one
# file is rewritten with a bumped mtime, and the leaf class is re-imported.
# --------------------------------------------------------------------------

DECO_SRC = """\
def deco(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
"""

DEEP_SRC = """\
DEEP_CONST = 1


def deep_helper(x: int) -> int:
    return x + DEEP_CONST
"""

MIXIN_SRC = """\
class Mixin:
    def mixin_method(self) -> str:
        return "mixin"
"""

MOD_SRC = """\
def f(x: int) -> int:
    return x + 1
"""

DISPATCH_SRC = """\
def dispatched(x: int) -> bool:
    return x < 100
"""

HELPERS_SRC_TMPL = """\
from {top}.deep import deep_helper

MODULE_CONST = 10
LIMIT = 5


def helper(x: int) -> int:
    return deep_helper(x) + MODULE_CONST


def unrelated_helper() -> str:
    return "unrelated function in helpers module"
"""

HELPERS_REORDERED_TMPL = """\
from {top}.deep import deep_helper

MODULE_CONST = 10
LIMIT = 5


def unrelated_helper() -> str:
    return "unrelated function in helpers module"


def helper(x: int) -> int:
    return deep_helper(x) + MODULE_CONST
"""

BASE_SRC_TMPL = """\
import {top}.mod
from {top} import dispatch as dispatch_module
from {top}.deco import deco
from {top}.helpers import LIMIT, helper
from mloda.provider import FeatureGroup


DIRECT_CONST = 7


class Base(FeatureGroup):
    \"\"\"Base docstring.\"\"\"

    # a comment in base
    CLASS_ATTR = 1

    class Nested:
        def nested_method(self) -> str:
            return "nested"

    def annotated(self, x: int) -> None:
        pass

    def _same_module_helper(self, x: int) -> int:
        return x + 1

    @deco
    def calculate_feature(self, x: int) -> int:
        # comment in calculate_feature
        value = self._same_module_helper(x)
        value = helper(value)
        value = value + LIMIT
        value = value + DIRECT_CONST
        dispatched = getattr(dispatch_module, "dispatched")
        if dispatched(value):
            value = value
        value = {top}.mod.f(value)
        return value
"""

SUB_SRC_TMPL = """\
from {top}.base import Base


class Leaf(Base):
    \"\"\"Leaf docstring.\"\"\"

    # a comment in leaf


def unrelated_in_sub() -> str:
    return "unrelated function in sub module"
"""


def _base_files(top: str) -> dict[str, str]:
    return {
        "deco.py": DECO_SRC,
        "deep.py": DEEP_SRC,
        "mixin.py": MIXIN_SRC,
        "mod.py": MOD_SRC,
        "dispatch.py": DISPATCH_SRC,
        "helpers.py": HELPERS_SRC_TMPL.format(top=top),
        "base.py": BASE_SRC_TMPL.format(top=top),
        "sub.py": SUB_SRC_TMPL.format(top=top),
    }


def _apply_case(files: dict[str, str], file_name: str, old: str, new: str) -> dict[str, str]:
    updated = dict(files)
    text = updated[file_name]
    assert old in text, f"case anchor text not found in {file_name!r}: {old!r}"
    updated[file_name] = text.replace(old, new, 1)
    return updated


def _write_package(tmp_path: Path, top: str, files: dict[str, str]) -> None:
    pkg_dir = tmp_path / top
    pkg_dir.mkdir(parents=True, exist_ok=True)
    init_file = pkg_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")
    for rel_path, source in files.items():
        full = pkg_dir / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(source)
    importlib.invalidate_caches()


def _rewrite_package(tmp_path: Path, top: str, files: dict[str, str]) -> None:
    pkg_dir = tmp_path / top
    for rel_path, source in files.items():
        full = pkg_dir / rel_path
        full.write_text(source)
        current_ns = full.stat().st_mtime_ns
        bumped_ns = current_ns + 2_000_000_000
        os.utime(full, ns=(bumped_ns, bumped_ns))
    importlib.invalidate_caches()


def _purge_package(top: str) -> None:
    for name in list(sys.modules):
        if name == top or name.startswith(top + "."):
            del sys.modules[name]


def _unique_top(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class _FixturePkgHelper:
    """Writes/rewrites/imports a fixture package under tmp_path; purges and gc-checks on teardown."""

    def __init__(self, tmp_path: Path) -> None:
        self._tmp_path = tmp_path
        self._tops: list[str] = []
        self._tracked: list["weakref.ReferenceType[Any]"] = []

    def write_base(self, top: str | None = None) -> str:
        resolved_top = top or _unique_top("fgv")
        _write_package(self._tmp_path, resolved_top, _base_files(resolved_top))
        if resolved_top not in self._tops:
            self._tops.append(resolved_top)
        return resolved_top

    def write(self, top: str, files: dict[str, str]) -> None:
        _write_package(self._tmp_path, top, files)
        if top not in self._tops:
            self._tops.append(top)

    def rewrite(self, top: str, files: dict[str, str]) -> None:
        _purge_package(top)
        _rewrite_package(self._tmp_path, top, files)

    def import_leaf(self, top: str, module: str = "sub", attr: str = "Leaf") -> type[Any]:
        mod = importlib.import_module(f"{top}.{module}")
        return cast(type[Any], getattr(mod, attr))

    def track(self, cls: type[Any]) -> None:
        self._tracked.append(weakref.ref(cls))

    def purge(self, top: str) -> None:
        _purge_package(top)

    def teardown(self) -> None:
        for top in self._tops:
            _purge_package(top)
        gc.collect()
        alive = [ref for ref in self._tracked if ref() is not None]
        assert not alive, f"fixture classes leaked past teardown: {alive}"


@pytest.fixture
def fixture_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_FixturePkgHelper]:
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    helper = _FixturePkgHelper(tmp_path)
    yield helper
    helper.teardown()


# case: (case_id, file_name, old_template, new_template); templates are formatted with top=<top>.
_MUST_CHANGE_CASES: list[tuple[str, str, str, str]] = [
    ("same_module_helper_body", "base.py", "return x + 1", "return x + 2"),
    ("base_class_method_body", "base.py", "return value", "return value if value else value"),
    ("class_attribute", "base.py", "CLASS_ATTR = 1", "CLASS_ATTR = 2"),
    ("module_constant", "base.py", "DIRECT_CONST = 7", "DIRECT_CONST = 8"),
    ("imported_constant", "helpers.py", "LIMIT = 5", "LIMIT = 6"),
    (
        "imported_helper_body",
        "helpers.py",
        "return deep_helper(x) + MODULE_CONST",
        "return deep_helper(x) + MODULE_CONST + 1",
    ),
    ("two_hop_helper_body", "deep.py", "return x + DEEP_CONST", "return x + DEEP_CONST + 1"),
    ("constant_used_by_helper", "helpers.py", "MODULE_CONST = 10", "MODULE_CONST = 11"),
    ("getattr_dispatched_function_body", "dispatch.py", "return x < 100", "return not (x >= 100)"),
    ("single_operator", "dispatch.py", "return x < 100", "return x <= 100"),
    (
        "decorator_body",
        "deco.py",
        "        return func(*args, **kwargs)",
        "        result = func(*args, **kwargs)\n        return result",
    ),
    (
        "base_list_add_mixin",
        "sub.py",
        "from {top}.base import Base\n\n\nclass Leaf(Base):",
        "from {top}.base import Base\nfrom {top}.mixin import Mixin\n\n\nclass Leaf(Base, Mixin):",
    ),
    ("annotation_only", "base.py", "def annotated(self, x: int) -> None:", "def annotated(self, x: float) -> None:"),
    ("nested_class_body", "base.py", 'return "nested"', 'return "nested-changed"'),
    ("dotted_chain_pkg_mod_f", "mod.py", "return x + 1", "return x + 2"),
]

_MUST_NOT_CHANGE_CASES: list[tuple[str, str, str, str]] = [
    ("base_docstring", "base.py", '"""Base docstring."""', '"""Changed base docstring."""'),
    ("leaf_docstring", "sub.py", '"""Leaf docstring."""', '"""Changed leaf docstring."""'),
    ("base_comment", "base.py", "# a comment in base", "# different comment in base"),
    ("leaf_comment", "sub.py", "# a comment in leaf", "# different comment in leaf"),
    (
        "blank_lines_above_leaf",
        "sub.py",
        "from {top}.base import Base\n\n\nclass Leaf(Base):",
        "from {top}.base import Base\n\n\n\n\nclass Leaf(Base):",
    ),
    (
        "unrelated_in_leaf_module",
        "sub.py",
        '"unrelated function in sub module"',
        '"changed unrelated function in sub module"',
    ),
    (
        "unrelated_in_referenced_module",
        "helpers.py",
        '"unrelated function in helpers module"',
        '"changed unrelated function in helpers module"',
    ),
]


class TestImplementationHashEditMatrix:
    """implementation_hash reacts to reachable-code edits, and only to those."""

    @pytest.mark.parametrize(
        ("case_id", "file_name", "old_tmpl", "new_tmpl"), _MUST_CHANGE_CASES, ids=[c[0] for c in _MUST_CHANGE_CASES]
    )
    def test_edit_changes_hash(
        self,
        fixture_pkg: _FixturePkgHelper,
        case_id: str,
        file_name: str,
        old_tmpl: str,
        new_tmpl: str,
    ) -> None:
        top = fixture_pkg.write_base()
        leaf_before = fixture_pkg.import_leaf(top)
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        edited = _apply_case(_base_files(top), file_name, old_tmpl.format(top=top), new_tmpl.format(top=top))
        fixture_pkg.rewrite(top, {file_name: edited[file_name]})
        leaf_after = fixture_pkg.import_leaf(top)
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after != before, f"case {case_id!r} edits reachable code and must change implementation_hash"

    @pytest.mark.parametrize(
        ("case_id", "file_name", "old_tmpl", "new_tmpl"),
        _MUST_NOT_CHANGE_CASES,
        ids=[c[0] for c in _MUST_NOT_CHANGE_CASES],
    )
    def test_edit_does_not_change_hash(
        self,
        fixture_pkg: _FixturePkgHelper,
        case_id: str,
        file_name: str,
        old_tmpl: str,
        new_tmpl: str,
    ) -> None:
        top = fixture_pkg.write_base()
        leaf_before = fixture_pkg.import_leaf(top)
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        edited = _apply_case(_base_files(top), file_name, old_tmpl.format(top=top), new_tmpl.format(top=top))
        fixture_pkg.rewrite(top, {file_name: edited[file_name]})
        leaf_after = fixture_pkg.import_leaf(top)
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after == before, f"case {case_id!r} must not affect implementation_hash"

    def test_reordering_definitions_in_referenced_module_does_not_change_hash(
        self, fixture_pkg: _FixturePkgHelper
    ) -> None:
        top = fixture_pkg.write_base()
        leaf_before = fixture_pkg.import_leaf(top)
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        fixture_pkg.rewrite(top, {"helpers.py": HELPERS_REORDERED_TMPL.format(top=top)})
        leaf_after = fixture_pkg.import_leaf(top)
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after == before, "reordering unrelated definitions in a referenced module must not affect the hash"

    def test_implementation_hash_is_64_hex_chars(self, fixture_pkg: _FixturePkgHelper) -> None:
        top = fixture_pkg.write_base()
        leaf = fixture_pkg.import_leaf(top)
        digest = BaseFeatureGroupVersion.implementation_hash(leaf)
        assert len(digest) == 64
        int(digest, 16)


class TestVersionUsesImplementationHash:
    """FeatureGroup.version() format stays '{mloda_version}-{module}-{hash}', hash now from implementation_hash."""

    def test_version_ends_with_implementation_hash(self, fixture_pkg: _FixturePkgHelper) -> None:
        top = fixture_pkg.write_base()
        leaf = fixture_pkg.import_leaf(top)
        digest = BaseFeatureGroupVersion.implementation_hash(leaf)
        assert leaf.version().endswith(f"-{digest}")

    def test_version_changes_when_base_class_helper_changes(self, fixture_pkg: _FixturePkgHelper) -> None:
        """The original bug: base-class and helper edits used to leave version() unchanged."""
        top = fixture_pkg.write_base()
        leaf_before = fixture_pkg.import_leaf(top)
        before = leaf_before.version()

        edited = _apply_case(_base_files(top), "base.py", "return x + 1", "return x + 2")
        fixture_pkg.rewrite(top, {"base.py": edited["base.py"]})
        leaf_after = fixture_pkg.import_leaf(top)
        after = leaf_after.version()

        assert after != before, "editing a base-class helper method must change version()"


# --------------------------------------------------------------------------
# Pinned implementation_hash: fixed fixture package, fixed name, fixed content.
# EXPECTED is pinned to prove the hash is identical across supported Pythons;
# it must only change together with a deliberate canonical-form change.
# --------------------------------------------------------------------------

_PIN_TOP = "mloda_pin_fixture_pkg_fixed"

PIN_HELPER_SRC = '''\
"""Pin helper module docstring."""

PIN_MODULE_CONST = 3.5


def pin_helper(x: int) -> str:
    return f"{x:03d}"
'''

PIN_BASE_SRC = f'''\
"""Pin base module docstring."""

from mloda.provider import FeatureGroup

from {_PIN_TOP}.pin_helper import PIN_MODULE_CONST, pin_helper

NON_ASCII = "café☕"
BYTES_CONST = b"\\x00\\x01\\xff"


def pin_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


class PinLeaf(FeatureGroup):
    """Pin leaf docstring."""

    @pin_decorator
    def calculate_feature(self, x: int) -> str:
        return pin_helper(x) + NON_ASCII + str(BYTES_CONST) + str(PIN_MODULE_CONST)
'''


class TestPinnedImplementationHash:
    def test_pinned_hash_matches_recorded_value(self, fixture_pkg: _FixturePkgHelper) -> None:
        fixture_pkg.write(_PIN_TOP, {"pin_helper.py": PIN_HELPER_SRC, "pin_base.py": PIN_BASE_SRC})
        leaf = fixture_pkg.import_leaf(_PIN_TOP, "pin_base", "PinLeaf")

        digest = BaseFeatureGroupVersion.implementation_hash(leaf)

        # Pinned to prove the digest is identical across supported Pythons; it
        # must only change together with a deliberate canonical-form change.
        EXPECTED = "4fcce064f264dad26a182c4eaa441c6899f71911dcb521de60f2e5c5a36cef31"
        assert digest == EXPECTED


class TestImplementationHashCaching:
    """Caching contract for BaseFeatureGroupVersion.implementation_hash: mirrors class_source_hash's."""

    def _install_counting_parse(self, monkeypatch: pytest.MonkeyPatch) -> list[object]:
        """Patches the shared ``ast`` module's ``parse``; ``code_closure`` does ``import ast`` and
        resolves ``ast.parse`` at call time, so this intercepts its module-source parsing."""
        calls: list[object] = []
        real_parse = ast.parse

        def counting_parse(*args: Any, **kwargs: Any) -> ast.AST:
            calls.append((args, kwargs))
            return cast(ast.AST, real_parse(*args, **kwargs))

        monkeypatch.setattr(ast, "parse", counting_parse)
        return calls

    def test_implementation_hash_is_cached_per_class(
        self, fixture_pkg: _FixturePkgHelper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        top = fixture_pkg.write_base()
        leaf = fixture_pkg.import_leaf(top)

        first = BaseFeatureGroupVersion.implementation_hash(leaf)
        calls = self._install_counting_parse(monkeypatch)
        second = BaseFeatureGroupVersion.implementation_hash(leaf)

        assert first == second
        assert calls == [], "the second call must be served from the per-class cache, reparsing nothing"

    def _install_counting_build_index(self, monkeypatch: pytest.MonkeyPatch) -> list[object]:
        """Patches code_closure._build_module_index with a call recorder.

        Counts full module-index builds, not every ``ast.parse`` call: a lazy
        implementation may reparse a small definition span on first reach without
        rebuilding the whole module's index, which is an implementation detail,
        not a caching-contract violation.
        """
        from mloda.core.abstract_plugins.components import code_closure

        calls: list[object] = []
        real_build = code_closure._build_module_index

        def counting_build(module: types.ModuleType) -> Any:
            calls.append(module)
            return real_build(module)

        monkeypatch.setattr(code_closure, "_build_module_index", counting_build)
        return calls

    def test_shared_module_is_indexed_once_across_two_leaf_classes(
        self, fixture_pkg: _FixturePkgHelper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        top = fixture_pkg.write_base()
        files = _base_files(top)
        files["sub2.py"] = f'from {top}.base import Base\n\n\nclass LeafTwo(Base):\n    """Second leaf."""\n'
        fixture_pkg.write(top, files)
        leaf_one = fixture_pkg.import_leaf(top, "sub", "Leaf")
        leaf_two = fixture_pkg.import_leaf(top, "sub2", "LeafTwo")

        calls = self._install_counting_build_index(monkeypatch)
        BaseFeatureGroupVersion.implementation_hash(leaf_one)
        first_module_count = len(calls)
        calls.clear()
        BaseFeatureGroupVersion.implementation_hash(leaf_two)
        second_module_count = len(calls)

        assert second_module_count <= 1, (
            f"computing leaf_two's hash rebuilt the module index for {second_module_count} modules (first, "
            f"cold call built {first_module_count}); base.py, helpers.py, deep.py, dispatch.py, deco.py and "
            "mod.py are shared with leaf_one and must come from the per-module index cache, only sub2.py's "
            "own module is new"
        )

    def test_implementation_hash_success_does_not_keep_class_alive(self, fixture_pkg: _FixturePkgHelper) -> None:
        top = fixture_pkg.write_base()
        leaf = fixture_pkg.import_leaf(top)
        BaseFeatureGroupVersion.implementation_hash(leaf)

        ref = weakref.ref(leaf)
        del leaf
        fixture_pkg.purge(top)
        gc.collect()

        assert ref() is None, "implementation_hash must not pin a successfully hashed class alive via its cache"


def _two_hundred_functions_ast_retention_module_src() -> str:
    """200 tiny functions; a caller reaches only 3 of them (used_0..used_2)."""
    used = [f"def used_{i}(x: int) -> int:\n    return x + {i}\n\n\n" for i in range(3)]
    unrelated = [f"def unrelated_{i}(x: int) -> int:\n    return x + {i}\n\n\n" for i in range(197)]
    return "".join(used) + "".join(unrelated)


class TestImplementationHashDoesNotRetainAst:
    """implementation_hash must not keep parsed ast.AST nodes alive once it returns.

    The plan forbids retaining a module's full parsed tree: modules are indexed once, keeping
    only source text plus per-definition line spans, re-parsing just a definition's span on
    first reach. A naive lazy implementation that instead caches the whole tree to defer
    canonicalization would keep every ast.AST node in the fixture module alive.
    """

    def test_ast_nodes_do_not_survive_hashing(self, fixture_pkg: _FixturePkgHelper) -> None:
        top = _unique_top("astretain")
        fixture_pkg.write(
            top,
            {
                "many.py": _two_hundred_functions_ast_retention_module_src(),
                "sub.py": (
                    f"from {top}.many import used_0, used_1, used_2\n"
                    "from mloda.provider import FeatureGroup\n\n\n"
                    "class Leaf(FeatureGroup):\n"
                    "    def calculate_feature(self, x: int) -> int:\n"
                    "        return used_0(x) + used_1(x) + used_2(x)\n"
                ),
            },
        )
        leaf = fixture_pkg.import_leaf(top, "sub", "Leaf")

        gc.collect()
        before = sum(1 for obj in gc.get_objects() if isinstance(obj, ast.AST))

        BaseFeatureGroupVersion.implementation_hash(leaf)

        gc.collect()
        after = sum(1 for obj in gc.get_objects() if isinstance(obj, ast.AST))

        growth = after - before
        assert growth < 50, (
            f"implementation_hash must not retain parsed AST nodes past its own call; "
            f"{growth} ast.AST objects survived (before={before}, after={after})"
        )


def _make_synthetic_module(label: str) -> types.ModuleType:
    """A fresh file-less module registered in sys.modules, for _exec_fg_in_module."""
    module = types.ModuleType(f"synthetic_impl_hash_{label}")
    sys.modules[module.__name__] = module
    return module


def _define_local_probe_variant_a() -> type[FeatureGroup]:
    class LocalBodyProbeFeatureGroup(FeatureGroup):
        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return 1

    return LocalBodyProbeFeatureGroup


def _define_local_probe_variant_b() -> type[FeatureGroup]:
    class LocalBodyProbeFeatureGroup(FeatureGroup):
        @classmethod
        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
            return 2

    return LocalBodyProbeFeatureGroup


class TestExistingFallbacksStayGreen:
    """Behaviors that predate this cycle: version() must keep distinguishing these class shapes."""

    def test_exec_defined_class_in_fileless_module_versions_differ(self) -> None:
        body_a = """
        from mloda.provider import FeatureGroup

        class ExecProbeFG(FeatureGroup):
            def calculate_feature(self) -> int:
                return 1
        """
        body_b = """
        from mloda.provider import FeatureGroup

        class ExecProbeFG(FeatureGroup):
            def calculate_feature(self) -> int:
                return 2
        """
        cls_a = _exec_fg_in_module(
            "ExecProbeFG", body_a, "impl-hash-exec-a", _make_synthetic_module("impl-hash-exec-a")
        )
        cls_b = _exec_fg_in_module(
            "ExecProbeFG", body_b, "impl-hash-exec-b", _make_synthetic_module("impl-hash-exec-b")
        )

        version_a = cls_a.version()
        version_b = cls_b.version()

        sys.modules.pop(cls_a.__module__, None)
        sys.modules.pop(cls_b.__module__, None)

        assert version_a != version_b

    def test_locals_class_version_changes_with_body(self) -> None:
        cls_a = _define_local_probe_variant_a()
        cls_b = _define_local_probe_variant_b()
        assert cls_a.__name__ == cls_b.__name__
        assert cls_a is not cls_b
        assert cls_a.version() != cls_b.version()


# --------------------------------------------------------------------------
# Dependency mode: ThirdPartyVersionMode and code_closure.dependency_entry.
#
# A fake distribution ("fakedep_<uuid>") is written under tmp_path as both an
# importable package and a matching *.dist-info/METADATA, so
# importlib.metadata.version resolves it like a real installed dependency.
# --------------------------------------------------------------------------


def _write_fake_dist(tmp_path: Path, name: str, version: str, init_src: str) -> None:
    pkg_dir = tmp_path / name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text(init_src)
    dist_info = tmp_path / f"{name}-{version}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    importlib.invalidate_caches()


def _rename_fake_dist_info(tmp_path: Path, name: str, old_version: str, new_version: str) -> None:
    old_dir = tmp_path / f"{name}-{old_version}.dist-info"
    new_dir = tmp_path / f"{name}-{new_version}.dist-info"
    old_dir.rename(new_dir)
    (new_dir / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {new_version}\n")
    importlib.invalidate_caches()


class _FakeDistHelper:
    """Writes/rewrites a fake third-party distribution (package + dist-info) under tmp_path."""

    def __init__(self, tmp_path: Path) -> None:
        self._tmp_path = tmp_path
        self.names: list[str] = []

    def make(self, version: str, init_src: str, name: str | None = None) -> str:
        fakedep_name = name or _unique_top("fakedep")
        _write_fake_dist(self._tmp_path, fakedep_name, version, init_src)
        self.names.append(fakedep_name)
        return fakedep_name

    def bump_version(self, name: str, old_version: str, new_version: str) -> None:
        _purge_package(name)
        _rename_fake_dist_info(self._tmp_path, name, old_version, new_version)

    def rewrite_body(self, name: str, new_src: str) -> None:
        _purge_package(name)
        init_file = self._tmp_path / name / "__init__.py"
        init_file.write_text(new_src)
        current_ns = init_file.stat().st_mtime_ns
        bumped_ns = current_ns + 2_000_000_000
        os.utime(init_file, ns=(bumped_ns, bumped_ns))
        importlib.invalidate_caches()

    def remove_package(self, name: str) -> None:
        _purge_package(name)
        shutil.rmtree(self._tmp_path / name)
        importlib.invalidate_caches()


@pytest.fixture
def fakedep(tmp_path: Path) -> Iterator[_FakeDistHelper]:
    helper = _FakeDistHelper(tmp_path)
    yield helper
    dependency_entry.cache_clear()
    for name in helper.names:
        _purge_package(name)


FAKEDEP_INIT_V1 = """\
def helper() -> str:
    return "helper-body-v1"


class FakeBase:
    def fake_method(self) -> str:
        return "fake-body-v1"
"""

FAKEDEP_INIT_EDITED = """\
def helper() -> str:
    return "helper-body-edited"


class FakeBase:
    def fake_method(self) -> str:
        return "fake-body-edited"
"""


def _fakedep_plain_import_base(fakedep_name: str) -> str:
    return f"""\
import {fakedep_name}

from mloda.provider import FeatureGroup


class Base(FeatureGroup):
    def calculate_feature(self, x: int) -> str:
        return {fakedep_name}.helper()
"""


def _fakedep_from_import_base(fakedep_name: str) -> str:
    return f"""\
from {fakedep_name} import helper

from mloda.provider import FeatureGroup


class Base(FeatureGroup):
    def calculate_feature(self, x: int) -> str:
        return helper()
"""


_DEP_LEAF_SRC_TMPL = """\
from {top}.base import Base


class Leaf(Base):
    pass
"""

_DEP_EXCLUDE_LEAF_SRC_TMPL = """\
from {top}.base import Base
from mloda.provider import ThirdPartyVersionMode


class Leaf(Base):
    @classmethod
    def version_third_party_mode(cls):
        return ThirdPartyVersionMode.EXCLUDE
"""

_TRY_IMPORT_BASE_TMPL = """\
try:
    import {fakedep_name}
except ImportError:
    {fakedep_name} = None

from mloda.provider import FeatureGroup


class Base(FeatureGroup):
    def calculate_feature(self, x: int) -> str:
        if {fakedep_name} is None:
            return "no-dep"
        return {fakedep_name}.helper()
"""

_MRO_LEAF_SRC_TMPL = """\
import {fakedep_name}

from mloda.provider import FeatureGroup


class Leaf({fakedep_name}.FakeBase, FeatureGroup):
    def calculate_feature(self, x: int) -> str:
        return "leaf"
"""


class TestThirdPartyVersionMode:
    """FeatureGroup.version_third_party_mode: default hook and subclass override."""

    def test_default_mode_is_include(self) -> None:
        assert FeatureGroup.version_third_party_mode() == ThirdPartyVersionMode.INCLUDE

    def test_subclass_override_is_honoured(self) -> None:
        class OverriddenModeFeatureGroup(FeatureGroup):
            @classmethod
            def version_third_party_mode(cls) -> Any:
                return ThirdPartyVersionMode.EXCLUDE

        assert OverriddenModeFeatureGroup.version_third_party_mode() == ThirdPartyVersionMode.EXCLUDE


class TestDependencyModeHash:
    """implementation_hash reacts to a third-party dependency's version only in INCLUDE mode."""

    @pytest.mark.parametrize(
        "base_src_fn", [_fakedep_plain_import_base, _fakedep_from_import_base], ids=["plain_import", "from_import"]
    )
    def test_include_mode_hash_differs_between_dependency_versions(
        self, fixture_pkg: _FixturePkgHelper, fakedep: _FakeDistHelper, base_src_fn: Callable[[str], str]
    ) -> None:
        name = fakedep.make("1.0", FAKEDEP_INIT_V1)
        top = _unique_top("depinc")
        fixture_pkg.write(top, {"base.py": base_src_fn(name), "sub.py": _DEP_LEAF_SRC_TMPL.format(top=top)})
        dependency_entry.cache_clear()
        leaf_before = fixture_pkg.import_leaf(top)
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        fixture_pkg.purge(top)
        fakedep.bump_version(name, "1.0", "2.0")
        dependency_entry.cache_clear()
        leaf_after = fixture_pkg.import_leaf(top)
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after != before, "INCLUDE mode must reflect a dependency's version bump"

    @pytest.mark.parametrize(
        "base_src_fn", [_fakedep_plain_import_base, _fakedep_from_import_base], ids=["plain_import", "from_import"]
    )
    def test_exclude_mode_hash_identical_between_dependency_versions(
        self, fixture_pkg: _FixturePkgHelper, fakedep: _FakeDistHelper, base_src_fn: Callable[[str], str]
    ) -> None:
        name = fakedep.make("1.0", FAKEDEP_INIT_V1)
        top = _unique_top("depexc")
        fixture_pkg.write(top, {"base.py": base_src_fn(name), "sub.py": _DEP_EXCLUDE_LEAF_SRC_TMPL.format(top=top)})
        dependency_entry.cache_clear()
        leaf_before = fixture_pkg.import_leaf(top)
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        fixture_pkg.purge(top)
        fakedep.bump_version(name, "1.0", "2.0")
        dependency_entry.cache_clear()
        leaf_after = fixture_pkg.import_leaf(top)
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after == before, "EXCLUDE mode must not react to a dependency's version bump"

    @pytest.mark.parametrize("leaf_tmpl", [_DEP_LEAF_SRC_TMPL, _DEP_EXCLUDE_LEAF_SRC_TMPL], ids=["include", "exclude"])
    def test_dependency_body_edit_never_changes_hash(
        self, fixture_pkg: _FixturePkgHelper, fakedep: _FakeDistHelper, leaf_tmpl: str
    ) -> None:
        name = fakedep.make("1.0", FAKEDEP_INIT_V1)
        top = _unique_top("depbody")
        fixture_pkg.write(top, {"base.py": _fakedep_plain_import_base(name), "sub.py": leaf_tmpl.format(top=top)})
        dependency_entry.cache_clear()
        leaf_before = fixture_pkg.import_leaf(top)
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        fakedep.rewrite_body(name, FAKEDEP_INIT_EDITED)
        fixture_pkg.purge(top)
        dependency_entry.cache_clear()
        leaf_after = fixture_pkg.import_leaf(top)
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after == before, "third-party code is never walked, in either mode"


class TestTryExceptOptionalDependency:
    """A try/import/except ImportError module hashes identically installed or not, in EXCLUDE mode."""

    def test_exclude_mode_hash_identical_with_and_without_dependency_installed(
        self, fixture_pkg: _FixturePkgHelper, fakedep: _FakeDistHelper
    ) -> None:
        name = fakedep.make("1.0", FAKEDEP_INIT_V1)
        top = _unique_top("depopt")
        fixture_pkg.write(
            top,
            {
                "base.py": _TRY_IMPORT_BASE_TMPL.format(fakedep_name=name),
                "sub.py": _DEP_EXCLUDE_LEAF_SRC_TMPL.format(top=top),
            },
        )
        dependency_entry.cache_clear()
        leaf_installed = fixture_pkg.import_leaf(top)
        with_dep = BaseFeatureGroupVersion.implementation_hash(leaf_installed)

        fixture_pkg.purge(top)
        fakedep.remove_package(name)
        dependency_entry.cache_clear()
        leaf_missing = fixture_pkg.import_leaf(top)
        without_dep = BaseFeatureGroupVersion.implementation_hash(leaf_missing)

        assert with_dep == without_dep


class TestIgnoredFacadeReexportFallsThroughToLiveValue:
    """An ignored import source (core surface / stdlib / out-of-scope mloda_plugins) is not a dependency:

    when it re-exports a first-party function, the walk must fall through to the live value so the
    function's own defining module decides reachability, instead of stopping at the facade import.
    """

    def test_real_case_row_count_reexported_via_mloda_user_python_dict_is_reachable(self) -> None:
        parts = closure_parts(PythonDictMissingValueFeatureGroup, False)
        prefix = "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils:row_count:"
        assert any(part.startswith(prefix) for part in parts), (
            "row_count, re-exported through the ignored mloda.user.python_dict facade, must be reachable"
        )

    def test_fixture_facade_reexport_edit_changes_hash(
        self, fixture_pkg: _FixturePkgHelper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        top = _unique_top("facade")
        fixture_pkg.write(top, {"helpers.py": "def helper(x: int) -> int:\n    return x + 1\n"})
        facade_name = f"mloda.user._facade_{uuid.uuid4().hex[:10]}"
        fixture_pkg.write(
            top,
            {
                "sub.py": (
                    f"from {facade_name} import helper\n"
                    "from mloda.provider import FeatureGroup\n\n\n"
                    "class Leaf(FeatureGroup):\n"
                    "    def calculate_feature(self, x: int) -> int:\n"
                    "        return helper(x)\n"
                )
            },
        )
        helpers_mod = importlib.import_module(f"{top}.helpers")
        facade_mod = types.ModuleType(facade_name)
        facade_mod.helper = helpers_mod.helper  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, facade_name, facade_mod)

        leaf_before = fixture_pkg.import_leaf(top)
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        fixture_pkg.rewrite(top, {"helpers.py": "def helper(x: int) -> int:\n    return x + 2\n"})
        helpers_mod_after = importlib.import_module(f"{top}.helpers")
        facade_mod.helper = helpers_mod_after.helper  # type: ignore[attr-defined]
        leaf_after = fixture_pkg.import_leaf(top)
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after != before, "editing the facade-reexported helper's body must change implementation_hash"


class TestThirdPartyBaseClassNotRoot:
    """A third-party base class in the MRO is a dependency, not a walked root."""

    def test_third_party_base_body_edit_does_not_change_hash(
        self, fixture_pkg: _FixturePkgHelper, fakedep: _FakeDistHelper
    ) -> None:
        name = fakedep.make("1.0", FAKEDEP_INIT_V1)
        top = _unique_top("depmrobody")
        fixture_pkg.write(top, {"leaf.py": _MRO_LEAF_SRC_TMPL.format(fakedep_name=name)})
        dependency_entry.cache_clear()
        leaf_before = fixture_pkg.import_leaf(top, "leaf", "Leaf")
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        fakedep.rewrite_body(name, FAKEDEP_INIT_EDITED)
        fixture_pkg.purge(top)
        dependency_entry.cache_clear()
        leaf_after = fixture_pkg.import_leaf(top, "leaf", "Leaf")
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after == before, "a third-party MRO base's body is never walked"

    def test_include_mode_hash_changes_with_third_party_base_dependency_version(
        self, fixture_pkg: _FixturePkgHelper, fakedep: _FakeDistHelper
    ) -> None:
        name = fakedep.make("1.0", FAKEDEP_INIT_V1)
        top = _unique_top("depmroversion")
        fixture_pkg.write(top, {"leaf.py": _MRO_LEAF_SRC_TMPL.format(fakedep_name=name)})
        dependency_entry.cache_clear()
        leaf_before = fixture_pkg.import_leaf(top, "leaf", "Leaf")
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        fixture_pkg.purge(top)
        fakedep.bump_version(name, "1.0", "2.0")
        dependency_entry.cache_clear()
        leaf_after = fixture_pkg.import_leaf(top, "leaf", "Leaf")
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after != before, "a third-party MRO base still contributes a dependency entry in INCLUDE mode"


class TestDependencyEntry:
    """Unit contract of code_closure.dependency_entry, independent of the closure walk."""

    def test_stdlib_module_returns_none(self) -> None:
        dependency_entry.cache_clear()
        assert dependency_entry("json") is None

    def test_builtins_returns_none(self) -> None:
        dependency_entry.cache_clear()
        assert dependency_entry("builtins") is None

    def test_core_surface_module_returns_none(self) -> None:
        dependency_entry.cache_clear()
        assert dependency_entry("mloda.core.abstract_plugins.feature_group") is None

    def test_mloda_plugins_module_returns_none(self) -> None:
        dependency_entry.cache_clear()
        assert dependency_entry("mloda_plugins.feature_group") is None

    def test_fake_dist_returns_dep_entry_with_version(
        self, fixture_pkg: _FixturePkgHelper, fakedep: _FakeDistHelper
    ) -> None:
        name = fakedep.make("1.0", FAKEDEP_INIT_V1)
        importlib.import_module(name)
        dependency_entry.cache_clear()
        assert dependency_entry(name) == f"dep:{name}==1.0"

    def test_installed_looking_package_without_metadata_or_dunder_version_returns_bare_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setattr(sys, "dont_write_bytecode", True)
        name = _unique_top("baredep")
        _write_package(tmp_path, name, {})
        importlib.import_module(name)
        dependency_entry.cache_clear()
        assert dependency_entry(name) == f"dep:{name}"
        _purge_package(name)

    def test_package_with_dunder_version_and_no_metadata_returns_that_version(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setattr(sys, "dont_write_bytecode", True)
        name = _unique_top("verdep")
        _write_package(tmp_path, name, {"__init__.py": '__version__ = "9.9"\n'})
        importlib.import_module(name)
        dependency_entry.cache_clear()
        assert dependency_entry(name) == f"dep:{name}==9.9"
        _purge_package(name)


# --------------------------------------------------------------------------
# Cycle D: fallbacks and the error contract (deep chains, unwrap loops,
# pathological helper modules, loaders without get_source, <locals> classes
# reached as values, lazy canonicalization, namespace-package dependencies).
# --------------------------------------------------------------------------


def _chain_module_src(leaf_extra: str = "") -> str:
    """600 helpers h0..h599, each calling the next; h599 is the base case."""
    parts = [f"def h{i}(x: int) -> int:\n    return h{i + 1}(x)\n\n\n" for i in range(599)]
    parts.append(f"def h599(x: int) -> int:\n    return x{leaf_extra}\n")
    return "".join(parts)


_CHAIN_LEAF_SRC = """\
from {top}.chain import h0
from mloda.provider import FeatureGroup


class Leaf(FeatureGroup):
    def calculate_feature(self, x: int) -> int:
        return h0(x)
"""


class TestDeepHelperChainDoesNotRecurse:
    """A 600-deep same-module helper chain must not raise RecursionError."""

    def test_deep_chain_hashes_without_recursion_error(self, fixture_pkg: _FixturePkgHelper) -> None:
        top = _unique_top("chain")
        fixture_pkg.write(top, {"chain.py": _chain_module_src(), "sub.py": _CHAIN_LEAF_SRC.format(top=top)})
        leaf = fixture_pkg.import_leaf(top, "sub", "Leaf")

        digest = BaseFeatureGroupVersion.implementation_hash(leaf)

        assert len(digest) == 64
        int(digest, 16)

    def test_deep_chain_hash_changes_when_leaf_helper_body_changes(self, fixture_pkg: _FixturePkgHelper) -> None:
        top = _unique_top("chain")
        fixture_pkg.write(top, {"chain.py": _chain_module_src(), "sub.py": _CHAIN_LEAF_SRC.format(top=top)})
        leaf_before = fixture_pkg.import_leaf(top, "sub", "Leaf")
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        fixture_pkg.rewrite(top, {"chain.py": _chain_module_src(" + 1")})
        leaf_after = fixture_pkg.import_leaf(top, "sub", "Leaf")
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after != before


_UNWRAP_WEIRD_SRC = """\
import unittest.mock

MOCK_OBJ = unittest.mock.MagicMock()


class InfiniteAttr:
    def __getattr__(self, name: str) -> "InfiniteAttr":
        return InfiniteAttr()


INFINITE_OBJ = InfiniteAttr()
"""

_UNWRAP_LEAF_SRC = """\
from {top}.weird import MOCK_OBJ, INFINITE_OBJ
from mloda.provider import FeatureGroup


class Leaf(FeatureGroup):
    def calculate_feature(self, x: int) -> object:
        return MOCK_OBJ, INFINITE_OBJ
"""


class TestUnwrapLoopTerminates:
    """A value whose ``__wrapped__`` chain never ends must not hang implementation_hash.

    A hang here has no clean assertable failure; the suite's 10s pytest-timeout is the
    backstop. This test's own work is tiny, so a pass means the walk terminated promptly.
    """

    def test_wrapped_unwrap_loop_returns_promptly(self, fixture_pkg: _FixturePkgHelper) -> None:
        top = _unique_top("unwrap")
        fixture_pkg.write(top, {"weird.py": _UNWRAP_WEIRD_SRC, "sub.py": _UNWRAP_LEAF_SRC.format(top=top)})
        leaf = fixture_pkg.import_leaf(top, "sub", "Leaf")

        digest = BaseFeatureGroupVersion.implementation_hash(leaf)

        assert len(digest) == 64


class TestPathologicalHelperModule:
    """A reachable helper module whose live objects work but whose source no longer parses."""

    def test_unparseable_helper_source_degrades_without_raising(
        self, fixture_pkg: _FixturePkgHelper, tmp_path: Path
    ) -> None:
        top = _unique_top("patho")
        fixture_pkg.write(
            top,
            {
                "helper.py": "def helper(x: int) -> int:\n    return x + 1\n",
                "sub.py": (
                    f"from {top}.helper import helper\n"
                    "from mloda.provider import FeatureGroup\n\n\n"
                    "class Leaf(FeatureGroup):\n"
                    "    def calculate_feature(self, x: int) -> int:\n"
                    "        return helper(x)\n"
                ),
            },
        )
        fixture_pkg.import_leaf(top, "sub", "Leaf")
        assert f"{top}.helper" in sys.modules

        # Purge only the leaf module; the helper module object (and its live
        # `helper` function) stays in sys.modules, but its file on disk stops parsing.
        sys.modules.pop(f"{top}.sub", None)
        helper_file = tmp_path / top / "helper.py"
        helper_file.write_text("this is not valid python !!! ???\n")
        bumped_ns = helper_file.stat().st_mtime_ns + 2_000_000_000
        os.utime(helper_file, ns=(bumped_ns, bumped_ns))
        importlib.invalidate_caches()

        leaf = fixture_pkg.import_leaf(top, "sub", "Leaf")

        digest = BaseFeatureGroupVersion.implementation_hash(leaf)

        assert len(digest) == 64

    def test_raising_module_getattr_is_never_invoked(self, fixture_pkg: _FixturePkgHelper) -> None:
        top = _unique_top("patho")
        fixture_pkg.write(
            top,
            {
                "helpers2.py": (
                    "CALLS = []\n\n\n"
                    "def __getattr__(name: str) -> object:\n"
                    "    CALLS.append(name)\n"
                    "    raise RuntimeError(f'should not be called: {name}')\n"
                ),
                "sub2.py": (
                    f"from {top} import helpers2\n"
                    "from mloda.provider import FeatureGroup\n\n\n"
                    "class Leaf(FeatureGroup):\n"
                    "    def calculate_feature(self, x: int) -> object:\n"
                    "        return helpers2.missing_name\n"
                ),
            },
        )
        leaf = fixture_pkg.import_leaf(top, "sub2", "Leaf")

        digest = BaseFeatureGroupVersion.implementation_hash(leaf)

        assert len(digest) == 64
        helpers2 = sys.modules[f"{top}.helpers2"]
        assert helpers2.CALLS == [], "module __getattr__ must never run during the closure walk"


class TestDocsDoNotRaiseOnPathologicalModule:
    """get_feature_group_docs must degrade gracefully, not raise, for a pathological helper module."""

    def test_docs_return_real_version_for_leaf_with_unparseable_helper(
        self, fixture_pkg: _FixturePkgHelper, tmp_path: Path
    ) -> None:
        top = _unique_top("pathodocs")
        fixture_pkg.write(
            top,
            {
                "helper.py": "def helper(x: int) -> int:\n    return x + 1\n",
                "sub.py": (
                    f"from {top}.helper import helper\n"
                    "from mloda.provider import FeatureGroup\n\n\n"
                    "class DocsPathologicalLeaf(FeatureGroup):\n"
                    "    def calculate_feature(self, x: int) -> int:\n"
                    "        return helper(x)\n"
                ),
            },
        )
        fixture_pkg.import_leaf(top, "sub", "DocsPathologicalLeaf")

        sys.modules.pop(f"{top}.sub", None)
        helper_file = tmp_path / top / "helper.py"
        helper_file.write_text("this is not valid python !!! ???\n")
        bumped_ns = helper_file.stat().st_mtime_ns + 2_000_000_000
        os.utime(helper_file, ns=(bumped_ns, bumped_ns))
        importlib.invalidate_caches()
        fixture_pkg.import_leaf(top, "sub", "DocsPathologicalLeaf")

        docs = get_feature_group_docs(name="DocsPathologicalLeaf")

        assert len(docs) == 1
        assert docs[0].version != "unavailable"
        assert docs[0].version.startswith(f"{BaseFeatureGroupVersion.mloda_version()}-")


class TestNoSyntaxWarningOnParse:
    """Parsing a helper module with an invalid escape sequence must not emit SyntaxWarning."""

    def test_invalid_escape_sequence_emits_no_syntax_warning(self, fixture_pkg: _FixturePkgHelper) -> None:
        top = _unique_top("escseq")
        fixture_pkg.write(
            top,
            {
                "escapes.py": 'PATTERN = "\\d+"\n\n\ndef escaped_helper(x: str) -> bool:\n    return x == PATTERN\n',
                "sub.py": (
                    f"from {top}.escapes import escaped_helper\n"
                    "from mloda.provider import FeatureGroup\n\n\n"
                    "class Leaf(FeatureGroup):\n"
                    "    def calculate_feature(self, x: str) -> bool:\n"
                    "        return escaped_helper(x)\n"
                ),
            },
        )
        leaf = fixture_pkg.import_leaf(top, "sub", "Leaf")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            BaseFeatureGroupVersion.implementation_hash(leaf)

        syntax_warnings = [w for w in caught if issubclass(w.category, SyntaxWarning)]
        assert not syntax_warnings, f"unexpected SyntaxWarning(s): {[str(w.message) for w in syntax_warnings]}"


class TestLoaderWithoutGetSource:
    """A module loaded by a loader without get_source (e.g. pytest's assertion rewriter) still indexes."""

    def test_module_without_get_source_loader_indexes_from_file_bytes(
        self, fixture_pkg: _FixturePkgHelper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        top = _unique_top("noloader")
        files = {
            "helper.py": "def helper(x: int) -> int:\n    return x + 1\n",
            "sub.py": (
                f"from {top}.helper import helper\n"
                "from mloda.provider import FeatureGroup\n\n\n"
                "class Leaf(FeatureGroup):\n"
                "    def calculate_feature(self, x: int) -> int:\n"
                "        return helper(x)\n"
            ),
        }
        fixture_pkg.write(top, files)
        leaf_normal = fixture_pkg.import_leaf(top, "sub", "Leaf")
        normal_hash = BaseFeatureGroupVersion.implementation_hash(leaf_normal)

        fixture_pkg.purge(top)
        leaf_patched = fixture_pkg.import_leaf(top, "sub", "Leaf")
        helper_module = sys.modules[f"{top}.helper"]

        class _NoGetSourceLoader:
            """Stand-in for a loader without get_source, e.g. pytest's assertion-rewrite hook."""

        monkeypatch.setattr(helper_module, "__loader__", _NoGetSourceLoader(), raising=False)
        calls = _install_counting_getsource(monkeypatch)

        patched_hash = BaseFeatureGroupVersion.implementation_hash(leaf_patched)

        assert patched_hash == normal_hash
        helper_calls = [obj for obj in calls if getattr(obj, "__module__", None) == helper_module.__name__]
        assert helper_calls == [], (
            f"inspect.getsource must not run for a module indexed from file bytes; got {helper_calls}"
        )


def _locals_a_module_src(body: str) -> str:
    return (
        "from mloda.provider import FeatureGroup\n\n\n"
        "def make():\n"
        "    class Helper(FeatureGroup):\n"
        f'        def value(self) -> str:\n            return "{body}"\n\n'
        "    return Helper\n"
    )


def _locals_b_module_src(body: str) -> str:
    return (
        "from mloda.provider import FeatureGroup\n\n\n"
        "class Maker:\n"
        "    def build(self):\n"
        "        class Helper(FeatureGroup):\n"
        f'            def value(self) -> str:\n                return "{body}"\n\n'
        "        return Helper\n"
    )


def _locals_c_module_src(body: str) -> str:
    return (
        "import contextlib\n\n"
        "from mloda.provider import FeatureGroup\n\n\n"
        "def make():\n"
        "    with contextlib.nullcontext():\n"
        "        class Helper(FeatureGroup):\n"
        f'            def value(self) -> str:\n                return "{body}"\n\n'
        "    return Helper\n"
    )


class TestLocalsClassReachedThroughIndex:
    """A <locals> class hashed as the ROOT (how every test-local FeatureGroup is versioned).

    Each fixture module keeps the returned class only in a local variable inside the test:
    the leaf's own module never exposes it through a module-level binding, so the walk
    cannot take the enclosing-function shortcut that a module-level ``HelperCls = make()``
    binding would offer; the root itself must resolve through the module index.
    """

    def test_module_level_function_local_class_as_root(
        self, fixture_pkg: _FixturePkgHelper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        top = _unique_top("localsa")
        fixture_pkg.write(top, {"local_a.py": _locals_a_module_src("a")})
        module_before = importlib.import_module(f"{top}.local_a")
        cls_before = module_before.make()

        calls = _install_counting_getsource(monkeypatch)
        before = BaseFeatureGroupVersion.implementation_hash(cls_before)
        assert calls == [], f"a root <locals> class must hash through the index, not inspect.getsource; got {calls}"
        assert cls_before.version().endswith(f"-{before}")

        fixture_pkg.rewrite(top, {"local_a.py": _locals_a_module_src("a-changed")})
        module_after = importlib.import_module(f"{top}.local_a")
        cls_after = module_after.make()
        after = BaseFeatureGroupVersion.implementation_hash(cls_after)

        assert after != before, "editing the body of a module-level-function-returned root class must change the hash"

    def test_method_returned_local_class_as_root(
        self, fixture_pkg: _FixturePkgHelper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        top = _unique_top("localsb")
        fixture_pkg.write(top, {"local_b.py": _locals_b_module_src("b")})
        module_before = importlib.import_module(f"{top}.local_b")
        cls_before = module_before.Maker().build()
        assert cls_before.__qualname__ == "Maker.build.<locals>.Helper"

        calls = _install_counting_getsource(monkeypatch)
        before = BaseFeatureGroupVersion.implementation_hash(cls_before)
        assert calls == [], f"a root <locals> class must hash through the index, not inspect.getsource; got {calls}"
        assert cls_before.version().endswith(f"-{before}")

        fixture_pkg.rewrite(top, {"local_b.py": _locals_b_module_src("b-changed")})
        module_after = importlib.import_module(f"{top}.local_b")
        cls_after = module_after.Maker().build()
        after = BaseFeatureGroupVersion.implementation_hash(cls_after)

        assert after != before, "editing the body of a method-returned root class must change the hash"

    def test_with_block_local_class_as_root(
        self, fixture_pkg: _FixturePkgHelper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        top = _unique_top("localsc")
        fixture_pkg.write(top, {"local_c.py": _locals_c_module_src("c")})
        module_before = importlib.import_module(f"{top}.local_c")
        cls_before = module_before.make()

        calls = _install_counting_getsource(monkeypatch)
        before = BaseFeatureGroupVersion.implementation_hash(cls_before)
        assert calls == [], f"a root <locals> class must hash through the index, not inspect.getsource; got {calls}"
        assert cls_before.version().endswith(f"-{before}")

        fixture_pkg.rewrite(top, {"local_c.py": _locals_c_module_src("c-changed")})
        module_after = importlib.import_module(f"{top}.local_c")
        cls_after = module_after.make()
        after = BaseFeatureGroupVersion.implementation_hash(cls_after)

        assert after != before, "editing the body of a with-block-defined root class must change the hash"


_LOCALS_D_MODULE_SRC = """\
from mloda.provider import FeatureGroup


def make(flag: bool):
    if flag:
        class Helper(FeatureGroup):
            def value(self) -> str:
                return "true-branch"
    else:
        class Helper(FeatureGroup):
            def value(self) -> str:
                return "false-branch"
    return Helper
"""


class TestAmbiguousLocalsClassFallsBack:
    """Two same-named <locals> classes in an if/else must not crash the walk when hashed as root."""

    def test_duplicate_local_class_name_still_returns_a_hash_as_root(self, fixture_pkg: _FixturePkgHelper) -> None:
        top = _unique_top("localsd")
        fixture_pkg.write(top, {"local_d.py": _LOCALS_D_MODULE_SRC})
        module = importlib.import_module(f"{top}.local_d")
        cls = module.make(True)

        digest = BaseFeatureGroupVersion.implementation_hash(cls)

        assert len(digest) == 64
        int(digest, 16)
        assert cls.version().endswith(f"-{digest}")


def _many_functions_module_src() -> str:
    unrelated = [f"def unrelated_{i}(x: int) -> int:\n    return x + {i}\n\n\n" for i in range(50)]
    return "".join(unrelated) + "def used(x: int) -> int:\n    return x + 1\n"


class TestLazyCanonicalization:
    """Only reached definitions get canonicalized; a 50-function module is mostly untouched."""

    def test_unrelated_definitions_are_not_canonicalized(
        self, fixture_pkg: _FixturePkgHelper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        top = _unique_top("lazy")
        fixture_pkg.write(
            top,
            {
                "many.py": _many_functions_module_src(),
                "sub.py": (
                    f"from {top}.many import used\n"
                    "from mloda.provider import FeatureGroup\n\n\n"
                    "class Leaf(FeatureGroup):\n"
                    "    def calculate_feature(self, x: int) -> int:\n"
                    "        return used(x)\n"
                ),
            },
        )
        leaf = fixture_pkg.import_leaf(top, "sub", "Leaf")

        from mloda.core.abstract_plugins.components import code_closure

        dump_count = 0
        real_dump = code_closure.canonical_dump

        def counting_dump(node: Any) -> str:
            nonlocal dump_count
            dump_count += 1
            return real_dump(node)

        monkeypatch.setattr(code_closure, "canonical_dump", counting_dump)
        BaseFeatureGroupVersion.implementation_hash(leaf)

        assert dump_count < 10, f"expected lazy canonicalization, got {dump_count} dumps for a 50-function module"


class TestNamespacePackageDependency:
    """dependency_entry for a namespace top uses the first regular (non-namespace) package on the path."""

    def test_namespace_top_with_dunder_version_returns_versioned_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setattr(sys, "dont_write_bytecode", True)
        ns_name = _unique_top("nsdep")
        sub_dir = tmp_path / ns_name / "sub"
        sub_dir.mkdir(parents=True)
        (sub_dir / "__init__.py").write_text('__version__ = "3.1"\n')
        importlib.invalidate_caches()

        importlib.import_module(f"{ns_name}.sub")
        dependency_entry.cache_clear()

        assert dependency_entry(f"{ns_name}.sub.anything") == f"dep:{ns_name}.sub==3.1"

        dependency_entry.cache_clear()
        _purge_package(ns_name)

    def test_namespace_top_without_dunder_version_returns_bare_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setattr(sys, "dont_write_bytecode", True)
        ns_name = _unique_top("nsdep")
        sub_dir = tmp_path / ns_name / "sub"
        sub_dir.mkdir(parents=True)
        (sub_dir / "__init__.py").write_text("")
        importlib.invalidate_caches()

        importlib.import_module(f"{ns_name}.sub")
        dependency_entry.cache_clear()

        assert dependency_entry(f"{ns_name}.sub.anything") == f"dep:{ns_name}.sub"

        dependency_entry.cache_clear()
        _purge_package(ns_name)


# --------------------------------------------------------------------------
# Review-fix round (accepted findings F1-F10): each test fails today against
# the pre-fix implementation for the reason noted on the test/class.
# --------------------------------------------------------------------------

_PROXY_HELPERS_SRC = """\
CALLS = []


class LazyProxy:
    def __init__(self, factory):
        object.__setattr__(self, "_factory", factory)

    @property
    def __class__(self):
        CALLS.append(1)
        raise RuntimeError("proxy __class__ must not be read by the closure walk")


def _connect():
    raise RuntimeError("no database configured in this process")


DB = LazyProxy(_connect)
"""

_PROXY_SUB_TMPL = """\
from {top}.helpers import DB
from mloda.provider import FeatureGroup


class ProxyLeafFG(FeatureGroup):
    def calculate_feature(self, x: int) -> object:
        return DB
"""


class TestLazyProxyClassPropertyIsNeverRead:
    """F1: classifying live values via isinstance() reads a proxy's __class__, which can raise."""

    def test_implementation_hash_succeeds_without_invoking_class_property(self, fixture_pkg: _FixturePkgHelper) -> None:
        top = _unique_top("proxy")
        fixture_pkg.write(top, {"helpers.py": _PROXY_HELPERS_SRC, "sub.py": _PROXY_SUB_TMPL.format(top=top)})
        leaf = fixture_pkg.import_leaf(top, "sub", "ProxyLeafFG")

        digest = BaseFeatureGroupVersion.implementation_hash(leaf)

        assert len(digest) == 64
        helpers = sys.modules[f"{top}.helpers"]
        assert helpers.CALLS == [], "the proxy's __class__ property must never be invoked by the closure walk"

    def test_get_feature_group_docs_does_not_raise_for_leaf_with_lazy_proxy_value(
        self, fixture_pkg: _FixturePkgHelper
    ) -> None:
        top = _unique_top("proxydocs")
        class_name = f"ProxyDocsLeafFG_{uuid.uuid4().hex[:10]}"
        sub_src = _PROXY_SUB_TMPL.format(top=top).replace("ProxyLeafFG", class_name)
        fixture_pkg.write(top, {"helpers.py": _PROXY_HELPERS_SRC, "sub.py": sub_src})
        fixture_pkg.import_leaf(top, "sub", class_name)

        docs = get_feature_group_docs(name=class_name)

        assert len(docs) == 1
        assert docs[0].version != "unavailable", (
            "an exception reading a value's __class__ must be contained, not escape get_feature_group_docs"
        )


def _dyn_version_calc_a(cls: Any, data: Any, features: Any) -> Any:
    return {"x": 1}


def _dyn_version_calc_b(cls: Any, data: Any, features: Any) -> Any:
    return {"x": 2}


class _DynamicVersionProbeUserBase(FeatureGroup):
    """Base class handed to DynamicFeatureGroupCreator.create in the F2 tests below."""


class TestDynamicallyCreatedClassModuleIsStdlib:
    """F2: a leaf whose own module is stdlib/builtins (e.g. ABCMeta's type() gives __module__=='abc')
    must raise, not silently hash to an empty closure shared by every such class."""

    def test_stdlib_module_leaf_raises_on_implementation_hash(self) -> None:
        class_name = f"DynVersionProbe_{uuid.uuid4().hex[:10]}"
        cls = DynamicFeatureGroupCreator.create(
            {"calculate_feature": _dyn_version_calc_a},
            class_name=class_name,
            feature_group_cls=_DynamicVersionProbeUserBase,
        )
        assert cls.__module__ in sys.stdlib_module_names or cls.__module__ == "builtins"

        with pytest.raises(SOURCE_INTROSPECTION_ERRORS):
            BaseFeatureGroupVersion.implementation_hash(cls)

    def test_two_such_classes_with_different_bodies_both_raise(self) -> None:
        suffix = uuid.uuid4().hex[:10]
        cls_a = DynamicFeatureGroupCreator.create(
            {"calculate_feature": _dyn_version_calc_a},
            class_name=f"DynVersionA_{suffix}",
            feature_group_cls=_DynamicVersionProbeUserBase,
        )
        cls_b = DynamicFeatureGroupCreator.create(
            {"calculate_feature": _dyn_version_calc_b},
            class_name=f"DynVersionB_{suffix}",
            feature_group_cls=_DynamicVersionProbeUserBase,
        )

        with pytest.raises(SOURCE_INTROSPECTION_ERRORS):
            BaseFeatureGroupVersion.implementation_hash(cls_a)
        with pytest.raises(SOURCE_INTROSPECTION_ERRORS):
            BaseFeatureGroupVersion.implementation_hash(cls_b)

    def test_docs_report_unavailable_version(self) -> None:
        class_name = f"DynVersionDocsProbe_{uuid.uuid4().hex[:10]}"
        DynamicFeatureGroupCreator.create(
            {"calculate_feature": _dyn_version_calc_a},
            class_name=class_name,
            feature_group_cls=_DynamicVersionProbeUserBase,
        )

        docs = get_feature_group_docs(name=class_name)

        assert len(docs) == 1
        assert docs[0].version == "unavailable"


_COND_ROOT_SUB_SRC = """\
import sys

from mloda.provider import FeatureGroup

if sys.version_info < (3, 0):
    class Leaf(FeatureGroup):
        def calculate_feature(self, x: int) -> int:
            return 1
else:
    class Leaf(FeatureGroup):
        def calculate_feature(self, x: int) -> int:
            return 2
"""


class TestConditionalRootPicksTheLiveCandidate:
    """F3: several same-qualname ClassDef candidates (if/else) must hash the LIVE one, not the first in source order."""

    def test_editing_the_live_else_branch_changes_the_hash(self, fixture_pkg: _FixturePkgHelper) -> None:
        top = _unique_top("condlive")
        fixture_pkg.write(top, {"sub.py": _COND_ROOT_SUB_SRC})
        leaf_before = fixture_pkg.import_leaf(top)
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        fixture_pkg.rewrite(top, {"sub.py": _COND_ROOT_SUB_SRC.replace("return 2", "return 3")})
        leaf_after = fixture_pkg.import_leaf(top)
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after != before, "editing the live (else-branch) Leaf's method body must change the hash"

    def test_editing_only_the_dead_if_branch_does_not_change_the_hash(self, fixture_pkg: _FixturePkgHelper) -> None:
        top = _unique_top("conddead")
        fixture_pkg.write(top, {"sub.py": _COND_ROOT_SUB_SRC})
        leaf_before = fixture_pkg.import_leaf(top)
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        fixture_pkg.rewrite(top, {"sub.py": _COND_ROOT_SUB_SRC.replace("return 1", "return 11")})
        leaf_after = fixture_pkg.import_leaf(top)
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after == before, "editing only the dead (unreached) if-branch must not change the hash"


class TestTypeCheckingOnlyDependencyImportIsVersionIndependent:
    """F4: dependency_entry must not depend on whether the top module happens to already be in sys.modules."""

    def test_hash_identical_whether_type_checking_only_dependency_is_imported_or_not(
        self, fixture_pkg: _FixturePkgHelper, fakedep: _FakeDistHelper
    ) -> None:
        name = fakedep.make("1.0", "class Table:\n    pass\n")
        top = _unique_top("tcdep")
        sub_src = (
            "from __future__ import annotations\n\n"
            "from typing import TYPE_CHECKING\n\n"
            "from mloda.provider import FeatureGroup\n\n"
            "if TYPE_CHECKING:\n"
            f"    import {name}\n\n\n"
            "class Leaf(FeatureGroup):\n"
            f"    def calculate_feature(self, x: {name}.Table) -> int:\n"
            "        return 1\n"
        )
        fixture_pkg.write(top, {"sub.py": sub_src})

        dependency_entry.cache_clear()
        _purge_package(name)
        leaf_without = fixture_pkg.import_leaf(top)
        without_hash = BaseFeatureGroupVersion.implementation_hash(leaf_without)

        fixture_pkg.purge(top)
        dependency_entry.cache_clear()
        importlib.import_module(name)
        leaf_with = fixture_pkg.import_leaf(top)
        with_hash = BaseFeatureGroupVersion.implementation_hash(leaf_with)

        assert with_hash == without_hash, (
            "INCLUDE-mode hash must not depend on whether the TYPE_CHECKING-only dependency is already imported"
        )


_TRY_IMPORT_FIRST_PARTY_FALLBACK_TMPL = """\
try:
    from {fakedep_name} import helper
except ImportError:
    from {top}.slow import helper

from mloda.provider import FeatureGroup


class Leaf(FeatureGroup):
    def calculate_feature(self, x: int) -> int:
        return helper(x)
"""


class TestTryExceptOptionalDependencyFallsThroughToFirstPartyFallback:
    """F5: when the accelerated import is unavailable, the live first-party fallback must still be walked."""

    def test_first_party_fallback_body_edit_changes_hash_when_dependency_missing(
        self, fixture_pkg: _FixturePkgHelper
    ) -> None:
        top = _unique_top("optfallback")
        fakedep_name = _unique_top("notinstalledfastlib")
        fixture_pkg.write(
            top,
            {
                "slow.py": "def helper(x: int) -> int:\n    return x + 1\n",
                "sub.py": _TRY_IMPORT_FIRST_PARTY_FALLBACK_TMPL.format(fakedep_name=fakedep_name, top=top),
            },
        )
        dependency_entry.cache_clear()
        leaf_before = fixture_pkg.import_leaf(top)
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        fixture_pkg.rewrite(top, {"slow.py": "def helper(x: int) -> int:\n    return x + 2\n"})
        dependency_entry.cache_clear()
        leaf_after = fixture_pkg.import_leaf(top)
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after != before, "editing the live first-party fallback's body must change implementation_hash"


_DISPATCH_INIT_SRC = "from .impl import double\n"
_DISPATCH_IMPL_SRC = "def double(x: int) -> int:\n    return x * 2\n"
_DISPATCH_IMPL_SRC_EDITED = "def double(x: int) -> int:\n    return x * 3\n"
_DISPATCH_GETATTR_LEAF_TMPL = """\
from {top} import dispatch
from mloda.provider import FeatureGroup


class Leaf(FeatureGroup):
    def calculate_feature(self, x: int, name: str) -> int:
        return getattr(dispatch, name)(x)
"""


class TestModuleReexportedNameReachableThroughGetattr:
    """F7: a module used whole must also resolve names it re-exports through its own imports."""

    def test_reexported_function_body_edit_changes_hash(self, fixture_pkg: _FixturePkgHelper) -> None:
        top = _unique_top("reexp")
        fixture_pkg.write(
            top,
            {
                "dispatch/__init__.py": _DISPATCH_INIT_SRC,
                "dispatch/impl.py": _DISPATCH_IMPL_SRC,
                "sub.py": _DISPATCH_GETATTR_LEAF_TMPL.format(top=top),
            },
        )
        leaf_before = fixture_pkg.import_leaf(top)
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        fixture_pkg.rewrite(top, {"dispatch/impl.py": _DISPATCH_IMPL_SRC_EDITED})
        leaf_after = fixture_pkg.import_leaf(top)
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after != before, (
            "editing impl.double's body (re-exported through dispatch/__init__.py) must change the hash"
        )


_FORMFEED_HELPER_SRC = "\x0c\nFACTOR = 1\n\n\ndef helper(x: int) -> int:\n    return x + 1\n"
_FORMFEED_HELPER_SRC_EDITED = "\x0c\nFACTOR = 1\n\n\ndef helper(x: int) -> int:\n    return x + 2\n"
_U2028_HELPER_SRC = "# note:   pasted text\n\n\ndef helper(x: int) -> int:\n    return x + 1\n"
_U2028_HELPER_SRC_EDITED = "# note:   pasted text\n\n\ndef helper(x: int) -> int:\n    return x + 2\n"
_SIMPLE_HELPER_LEAF_TMPL = """\
from {top}.helpers import helper
from mloda.provider import FeatureGroup


class Leaf(FeatureGroup):
    def calculate_feature(self, x: int) -> int:
        return helper(x)
"""


class TestLineBoundaryCharactersDoNotShiftDefinitionSpans:
    """F9: str.splitlines() also splits on \\x0c and U+2028, which are not Python line boundaries."""

    def test_form_feed_before_helper_def_edit_changes_hash(self, fixture_pkg: _FixturePkgHelper) -> None:
        top = _unique_top("formfeed")
        fixture_pkg.write(top, {"helpers.py": _FORMFEED_HELPER_SRC, "sub.py": _SIMPLE_HELPER_LEAF_TMPL.format(top=top)})
        leaf_before = fixture_pkg.import_leaf(top)
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        fixture_pkg.rewrite(top, {"helpers.py": _FORMFEED_HELPER_SRC_EDITED})
        leaf_after = fixture_pkg.import_leaf(top)
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after != before, "a form feed before helper's def must not shift the def's recorded span"

    def test_u2028_in_comment_above_helper_edit_changes_hash(self, fixture_pkg: _FixturePkgHelper) -> None:
        top = _unique_top("u2028")
        fixture_pkg.write(top, {"helpers.py": _U2028_HELPER_SRC, "sub.py": _SIMPLE_HELPER_LEAF_TMPL.format(top=top)})
        leaf_before = fixture_pkg.import_leaf(top)
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        fixture_pkg.rewrite(top, {"helpers.py": _U2028_HELPER_SRC_EDITED})
        leaf_after = fixture_pkg.import_leaf(top)
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after != before, "a U+2028 inside a comment above helper's def must not shift the def's recorded span"


def _latin1_coding_cookie_helper_bytes(b_value: int) -> bytes:
    """Line 2 has 5 latin-1 bytes (0xC3 0xA9) that a naive utf-8 decode misreads as 5 utf-8 'e-acute'
    characters instead of 10 latin-1 characters, drifting the column offset of ``B`` on that line."""
    return (
        "# -*- coding: latin-1 -*-\n"
        f'NAME = "\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9"; B = {b_value}\n'
        "\n\n"
        "def helper(x: int) -> int:\n"
        "    return x + B\n"
    ).encode("latin-1")


_LATIN1_SUB_TMPL = """\
from {top}.helpers import helper
from mloda.provider import FeatureGroup


class Leaf(FeatureGroup):
    def calculate_feature(self, x: int) -> int:
        return helper(x)
"""


class TestLatin1CodingCookieHashing:
    """F9: the bytes path decodes with a hardcoded utf-8, ignoring a module's own coding cookie."""

    def test_hashing_a_latin1_cookied_module_is_deterministic(
        self, fixture_pkg: _FixturePkgHelper, tmp_path: Path
    ) -> None:
        top = _unique_top("latin1det")
        fixture_pkg.write(top, {"sub.py": _LATIN1_SUB_TMPL.format(top=top)})
        (tmp_path / top / "helpers.py").write_bytes(_latin1_coding_cookie_helper_bytes(2))
        importlib.invalidate_caches()
        leaf = fixture_pkg.import_leaf(top)

        first = BaseFeatureGroupVersion.implementation_hash(leaf)
        second = BaseFeatureGroupVersion.implementation_hash(leaf)

        assert first == second

    def test_editing_b_after_the_drifted_latin1_prefix_changes_the_hash(
        self, fixture_pkg: _FixturePkgHelper, tmp_path: Path
    ) -> None:
        top = _unique_top("latin1edit")
        fixture_pkg.write(top, {"sub.py": _LATIN1_SUB_TMPL.format(top=top)})
        helper_file = tmp_path / top / "helpers.py"
        helper_file.write_bytes(_latin1_coding_cookie_helper_bytes(2))
        importlib.invalidate_caches()
        leaf_before = fixture_pkg.import_leaf(top)
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        fixture_pkg.purge(top)
        helper_file.write_bytes(_latin1_coding_cookie_helper_bytes(3))
        bumped_ns = helper_file.stat().st_mtime_ns + 2_000_000_000
        os.utime(helper_file, ns=(bumped_ns, bumped_ns))
        importlib.invalidate_caches()
        leaf_after = fixture_pkg.import_leaf(top)
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after != before, (
            "editing B's value after a latin-1-decoded multi-byte prefix on the same line must change the hash"
        )


_SEMICOLON_JOINED_HELPER_SRC = "A = 1; B = 2\n"
_SEMICOLON_JOINED_HELPER_SRC_EDITED = "A = 1; B = 3\n"
_SEMICOLON_JOINED_LEAF_TMPL = """\
from {top}.helpers import B
from mloda.provider import FeatureGroup


class Leaf(FeatureGroup):
    def calculate_feature(self, x: int) -> int:
        return x + B
"""


class TestSemicolonJoinedConstantUsesItsOwnSpan:
    """F10: span re-parse must pick the candidate's own node by column offset, not the span's first node."""

    def test_editing_only_b_value_changes_the_hash(self, fixture_pkg: _FixturePkgHelper) -> None:
        top = _unique_top("semi")
        fixture_pkg.write(
            top, {"helpers.py": _SEMICOLON_JOINED_HELPER_SRC, "sub.py": _SEMICOLON_JOINED_LEAF_TMPL.format(top=top)}
        )
        leaf_before = fixture_pkg.import_leaf(top)
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        fixture_pkg.rewrite(top, {"helpers.py": _SEMICOLON_JOINED_HELPER_SRC_EDITED})
        leaf_after = fixture_pkg.import_leaf(top)
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after != before, "editing only B's value in a semicolon-joined line must change the hash"


class _RaisingGetSourceLoader:
    """A loader whose get_source raises with a message that embeds a live object's default repr
    (a memory address), to probe whether a contained per-task error's message is run-independent."""

    def get_source(self, name: str) -> str:
        raise RuntimeError(f"boom {object()!r}")


class TestContainedErrorMessageDeterminism:
    """run() contains a raising task as a ``nosrc:<tag>:<message>`` part; that message must not carry
    anything run-specific (e.g. a default object repr's memory address), or the hash becomes non-deterministic."""

    def test_hash_is_identical_across_two_fresh_runs_of_identical_source(
        self, fixture_pkg: _FixturePkgHelper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        top = _unique_top("contained")
        fixture_pkg.write(
            top,
            {
                "helpers.py": "def helper(x: int) -> int:\n    return x + 1\n",
                "sub.py": (
                    f"from {top}.helpers import helper\n"
                    "from mloda.provider import FeatureGroup\n\n\n"
                    "class Leaf(FeatureGroup):\n"
                    "    def calculate_feature(self, x: int) -> int:\n"
                    "        return helper(x)\n"
                ),
            },
        )

        def run_once() -> str:
            fixture_pkg.purge(top)
            leaf = fixture_pkg.import_leaf(top)
            helpers_module = sys.modules[f"{top}.helpers"]
            monkeypatch.setattr(helpers_module, "__loader__", _RaisingGetSourceLoader(), raising=False)
            return BaseFeatureGroupVersion.implementation_hash(leaf)

        first = run_once()
        second = run_once()

        assert first == second, "a contained per-task error's nosrc part must not embed a run-specific value"


_NONE_RETURNING_HELPER_TMPL = """\
def compute():
    {extra}return None


MODE = compute()
"""

_NONE_VALUE_LEAF_SRC = """\
from {top}.helpers import MODE
from mloda.provider import FeatureGroup


class Leaf(FeatureGroup):
    def calculate_feature(self, x: int) -> object:
        return MODE
"""


class TestNoneShortCircuitOnlyAppliesToTheOptionalDependencySentinel:
    """The walk must not silently ignore every first-party value that happens to be None: only the
    ``try: import x / except ImportError: x = None`` sentinel is meant to add nothing to the hash."""

    def test_editing_the_body_of_a_function_that_still_returns_none_changes_the_hash(
        self, fixture_pkg: _FixturePkgHelper
    ) -> None:
        top = _unique_top("nonebody")
        fixture_pkg.write(
            top,
            {
                "helpers.py": _NONE_RETURNING_HELPER_TMPL.format(extra=""),
                "sub.py": _NONE_VALUE_LEAF_SRC.format(top=top),
            },
        )
        leaf_before = fixture_pkg.import_leaf(top)
        before = BaseFeatureGroupVersion.implementation_hash(leaf_before)

        fixture_pkg.rewrite(top, {"helpers.py": _NONE_RETURNING_HELPER_TMPL.format(extra="y = 1\n    ")})
        leaf_after = fixture_pkg.import_leaf(top)
        after = BaseFeatureGroupVersion.implementation_hash(leaf_after)

        assert after != before, (
            "compute() still returns None, but its body changed; MODE = compute() must still be walked"
        )
