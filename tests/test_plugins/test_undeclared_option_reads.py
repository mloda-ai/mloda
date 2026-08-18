"""Static sweep: every Options key a shipped ``feature_group`` plugin reads must be a DECLARED key.

A plugin that reads ``options.get("foo")`` (directly, through a class constant, through a
``DefaultOptionKeys`` member, or through a local alias of one) is promising its users a ``foo`` option.
If ``foo`` is in no declaration (issue os-004 / mloda#775), that promise is undocumented and
unvalidated. This sweep resolves each read to its key and flags any key that no shipped plugin
declares, accepting only a narrow allowlist of genuinely engine-internal and genuinely dynamic reads.

The check spans TWO declaration surfaces and is deliberately GLOBAL across both: a key passes when
SOME shipped FeatureGroup declares it in ``PROPERTY_MAPPING`` or SOME shipped reader declares it in
``READER_OPTIONS``. Readers need their own surface because their keys are consumed at reader-selection
time, before the framework materializes any ``PROPERTY_MAPPING`` default. Per-group owning-surface
attribution is out of scope for this sweep.

Scope notes:

* ``SCAN_ROOT`` is the whole ``mloda_plugins/feature_group`` tree, ``input_data`` readers included;
  the allowlist paths below are written relative to it.
* ``reference_time`` is a ``DefaultOptionKeys`` member yet is NOT treated as framework-reserved here.
  The time-based groups expose it as a user-overridable column option, so os-004 reclassifies it as a
  key those groups must DECLARE. The remaining ``DefaultOptionKeys`` values stay framework-reserved.

Subclass-leak policy: this module NEEDS a leaked test-tree ``BaseInputData`` subclass to stay reachable
through ``__subclasses__()`` mid-test, because that is what proves the module-prefix filter in
``reader_declared_union`` really excludes it; the probe is built inside a helper and reclaimed by the
module teardown gc pass, and it is never a final reader, so it cannot reach reader selection.
"""

from __future__ import annotations

import ast
import gc
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar, NamedTuple

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
from mloda.provider import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.forecasting.base import ForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.encoding.base import EncodingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.pipeline.base import SklearnPipelineFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.scaling.base import ScalingFeatureGroup
from mloda_plugins.feature_group.experimental.dynamic_feature_group_factory.dynamic_feature_group_factory import (
    DynamicFeatureGroupCreator,
)
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup
from mloda_plugins.feature_group.input_data.read_context_files import ConcatenatedFileContent
from mloda_plugins.feature_group.input_data.read_document import ReadDocument
from mloda_plugins.feature_group.input_data.read_files.markdown_document_reader import MarkdownDocumentReader

# Anchor the scan root to the repo layout via __file__, not the cwd: a cwd-relative root makes the
# rglob loops empty and the sweep pass vacuously. This file sits two parents below the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOT = _REPO_ROOT / "mloda_plugins" / "feature_group"
assert SCAN_ROOT.exists(), f"scan root not found; check the parents index for the repo root: {SCAN_ROOT}"

# Accessor calls whose result is an Options-like object: cls.get_singular_option_from_options(...).get(...),
# get_options(...).get(...) and cls.options_with_defaults(...).get(...). Anything else (arbitrary methods
# containing "option") is not an Options read. options_with_defaults is load-bearing here: a group that
# resolves its declared defaults reads every key through it, so leaving it out blinds the scan to those
# reads entirely (that is how ConcatenatedFileContent's disallowed_files/file_type reads went missing).
_OPTIONS_ACCESSORS = frozenset({"get_singular_option_from_options", "get_options", "options_with_defaults"})
_OPTIONS_RECEIVER_NAMES = frozenset({"options", "option"})
_READ_METHODS = frozenset({"get", "get_options_key", "reader_option"})
# Reads whose receiver is never options-like, so they are recognized by method name alone: get_options_key
# is a FeatureSet method (receiver features/self) and reader_option is a BaseInputData classmethod
# (receiver cls) taking the key FIRST and the Options second. Both names are distinctive enough to carry
# the match on their own, and both take the key as their first positional argument.
_METHOD_NAME_ONLY_READS = frozenset({"get_options_key", "reader_option"})

# DefaultOptionKeys.<member> resolves to the member's string value. Built from the enum so it cannot drift.
_DEFAULT_OPTION_KEY_MEMBERS: dict[str, str] = {member.name: member.value for member in DefaultOptionKeys}

# Framework-reserved keys any plugin may read without declaring. reference_time is excluded on purpose:
# os-004 makes it a declared per-group option, so a read of it must resolve through declared_union instead.
FRAMEWORK_KEYS: frozenset[str] = frozenset(_DEFAULT_OPTION_KEY_MEMBERS.values()) - {
    DefaultOptionKeys.reference_time.value
}

# Literal keys that are engine-internal signals rather than user-facing options, scoped to their owning
# source file (relative to SCAN_ROOT): a read of the same literal from any other file is still flagged.
ALLOWED_LITERAL_KEYS: dict[str, tuple[str, str, str]] = {
    "initial_requested_data": (
        "experimental/source_input_feature.py",
        "SourceInputFeatureComposite",
        "engine-internal request-scoping signal set as a Feature attribute; not a user-facing option",
    ),
}

# Reads whose key is computed at runtime, keyed by (path relative to SCAN_ROOT, enclosing function).
READ_DOCUMENT_DYNAMIC_READ = ("input_data/read_document.py", "load_data")
ALLOWED_DYNAMIC_READS: dict[tuple[str, str], tuple[str, str]] = {
    ("experimental/forecasting/forecasting_artifact.py", "custom_loader"): (
        "ForecastingArtifact",
        "artifact is keyed by the runtime feature name, not a fixed option key",
    ),
    READ_DOCUMENT_DYNAMIC_READ: (
        "ReadDocument",
        "reader selection keys the option by the reader's own class name (data_access_name), so the key "
        "is dynamic by construction and no fixed declared key can cover it",
    ),
}

# Keys that ONLY the reader surface declares: nothing about them is in any PROPERTY_MAPPING, so they are
# the probe for whether the union really reaches READER_OPTIONS. "BaseInputData" is the reserved key the
# framework itself writes; the other two are read inside reader matching.
READER_ONLY_KEYS: frozenset[str] = frozenset({"BaseInputData", "data_access_handle", "document_suffixes"})

# Individual source files whose reads are asserted key-by-key below, so a scanner blind spot cannot
# silently drop them again. Relative to SCAN_ROOT.
READ_CONTEXT_FILES_REL = "input_data/read_context_files.py"
READ_FILE_REL = "input_data/read_file.py"
READ_DOCUMENT_REL = "input_data/read_document.py"

# Declared by the throwaway reader below, never by shipped code.
_PROBE_READER_KEY = "uor_leaked_test_tree_probe_key"


@pytest.fixture(scope="module", autouse=True)
def _load_all_plugins() -> Iterator[None]:
    """Load every shipped plugin so the declared unions enumerate the full plugin set.

    The teardown gc pass reclaims the throwaway reader this module builds: a test-local
    ``BaseInputData`` subclass sits in reference cycles, so it would otherwise linger in
    ``BaseInputData.__subclasses__()`` where other tests enumerate readers.
    """
    PluginLoader().all()
    yield
    gc.collect()
    gc.collect()


@pytest.fixture(autouse=True)
def _cleanup_dynamic_feature_groups() -> Iterator[None]:
    """This module only imports ``ConcatenatedFileContent``, but a sibling test file in the same xdist
    worker can register its join class first; pop it here too so this module never carries the leak
    forward regardless of run order.
    """
    yield
    DynamicFeatureGroupCreator._created_classes.pop(ConcatenatedFileContent.join_feature_name, None)


def _make_leaked_reader_probe() -> type[BaseInputData]:
    """A throwaway reader declaring one distinctive key, built here so it stays collectable.

    It deliberately does not override ``load_data`` and declares no ``_final_reader_requires``, so
    ``is_final_reader()`` is False and reader discovery never collects it.
    """

    class UorLeakedTestTreeReaderProbe(BaseInputData):
        READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
            _PROBE_READER_KEY: PropertySpec("Test-tree only; must never reach the declared union.", default=None),
        }

    return UorLeakedTestTreeReaderProbe


class _Read(NamedTuple):
    """One resolved Options read: key is None when the key is computed dynamically."""

    key: str | None
    lineno: int
    cls_name: str | None
    func_name: str | None


def _is_options_source(node: ast.expr) -> bool:
    """True for a direct Options source: an options/option name, an ``.options`` attribute, or an accessor call."""
    if isinstance(node, ast.Name):
        return node.id in _OPTIONS_RECEIVER_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr == "options"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr in _OPTIONS_ACCESSORS
    return False


def _receiver_is_options(node: ast.expr, func_assignments: dict[str, ast.expr]) -> bool:
    """True when ``node`` is an Options object, including a local variable single-assigned from one."""
    if isinstance(node, ast.Name) and node.id not in _OPTIONS_RECEIVER_NAMES:
        rhs = func_assignments.get(node.id)
        return rhs is not None and _is_options_source(rhs)
    return _is_options_source(node)


def _single(values: set[str] | None) -> str | None:
    """The one value in a set, or None when absent or ambiguous."""
    if values is not None and len(values) == 1:
        return next(iter(values))
    return None


def _collect_class_constants(root: Path) -> tuple[dict[tuple[str, str], str], dict[str, set[str]]]:
    """Map every class-body ``NAME = "literal"`` to (class, name) and to a loose name-to-values index."""
    by_class: dict[tuple[str, str], str] = {}
    by_name: dict[str, set[str]] = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
                    continue
                target = stmt.targets[0]
                value = stmt.value
                if isinstance(target, ast.Name) and isinstance(value, ast.Constant) and isinstance(value.value, str):
                    by_class[(node.name, target.id)] = value.value
                    by_name.setdefault(target.id, set()).add(value.value)
    return by_class, by_name


def _func_assignments(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, ast.expr]:
    """Top-level ``name = expr`` statements assigned exactly once in a function body."""
    counts: dict[str, int] = {}
    values: dict[str, ast.expr] = {}
    for stmt in func_node.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            counts[name] = counts.get(name, 0) + 1
            values[name] = stmt.value
    return {name: expr for name, expr in values.items() if counts[name] == 1}


def _resolve_key(
    node: ast.expr,
    cls_name: str | None,
    by_class: dict[tuple[str, str], str],
    by_name: dict[str, set[str]],
    func_assignments: dict[str, ast.expr],
) -> str | None:
    """Resolve a key expression to its string, or None when it is computed dynamically."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.Attribute):
        base = node.value
        if not isinstance(base, ast.Name):
            return None
        if base.id in ("cls", "self"):
            if cls_name is not None:
                hit = by_class.get((cls_name, node.attr))
                if hit is not None:
                    return hit
            return _single(by_name.get(node.attr))
        if base.id == "DefaultOptionKeys":
            return _DEFAULT_OPTION_KEY_MEMBERS.get(node.attr)
        hit = by_class.get((base.id, node.attr))
        return hit if hit is not None else _single(by_name.get(node.attr))
    if isinstance(node, ast.Name):
        expr = func_assignments.get(node.id)
        if expr is not None:
            return _resolve_key(expr, cls_name, by_class, by_name, func_assignments)
    return None


def _key_arg(call: ast.Call) -> ast.expr | None:
    """The key argument of a read call: the first positional, else a ``key=`` keyword, else None."""
    if call.args:
        return call.args[0]
    for kw in call.keywords:
        if kw.arg == "key":
            return kw.value
    return None


def _key_node(node: ast.AST, func_assignments: dict[str, ast.expr]) -> ast.expr | None:
    """The key expression of an Options read (``.get``/``.get_options_key``, subscript, or ``in``/``not in``), else None."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        method = node.func.attr
        if method in _READ_METHODS and (
            method in _METHOD_NAME_ONLY_READS or _receiver_is_options(node.func.value, func_assignments)
        ):
            return _key_arg(node)
    if isinstance(node, ast.Subscript) and _receiver_is_options(node.value, func_assignments):
        return node.slice
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)):
        if _receiver_is_options(node.comparators[0], func_assignments):
            return node.left
    return None


def _reads_in_tree(
    tree: ast.Module,
    by_class: dict[tuple[str, str], str],
    by_name: dict[str, set[str]],
) -> list[_Read]:
    """Collect every Options read in a module, threading the enclosing class and function for context."""
    out: list[_Read] = []

    def visit(
        node: ast.AST,
        cls_name: str | None,
        func_node: ast.FunctionDef | ast.AsyncFunctionDef | None,
        func_assignments: dict[str, ast.expr],
    ) -> None:
        key_node = _key_node(node, func_assignments)
        if key_node is not None:
            assert isinstance(node, ast.expr)
            key = _resolve_key(key_node, cls_name, by_class, by_name, func_assignments)
            func_name = func_node.name if func_node is not None else None
            out.append(_Read(key, node.lineno, cls_name, func_name))
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, child.name, func_node, func_assignments)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, cls_name, child, _func_assignments(child))
            else:
                visit(child, cls_name, func_node, func_assignments)

    visit(tree, None, None, {})
    return out


def _count_reads(root: Path) -> int:
    """Total Options reads found under ``root``; a nonzero floor guards against a vacuous scan."""
    by_class, by_name = _collect_class_constants(root)
    total = 0
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        total += len(_reads_in_tree(tree, by_class, by_name))
    return total


def _resolved_keys_in(rel: str) -> set[str]:
    """Every statically resolved option key the scanner sees in one file under SCAN_ROOT."""
    path = SCAN_ROOT / rel
    by_class, by_name = _collect_class_constants(SCAN_ROOT)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {read.key for read in _reads_in_tree(tree, by_class, by_name) if read.key is not None}


def _reader_option_call_keys(rel: str) -> set[str]:
    """Literal keys read through ``cls.reader_option(key, options)`` in one file under SCAN_ROOT.

    Independent of the scanner on purpose: it pins the ACCESSOR the reader uses, while
    ``_resolved_keys_in`` pins that the scanner still resolves the key behind it.
    """
    path = SCAN_ROOT / rel
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "reader_option" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            keys.add(first.value)
    return keys


def _dynamic_allowed(rel: str, func_name: str | None, dynamic_allow: dict[tuple[str, str], tuple[str, str]]) -> bool:
    """A dynamic read is documented when its (path, function) exactly matches an allowlist entry."""
    for (path_rel, allowed_func), _ in dynamic_allow.items():
        if allowed_func == func_name and rel == path_rel:
            return True
    return False


def find_violations(
    root: Path,
    declared: frozenset[str],
    framework_keys: frozenset[str],
    literal_allow: dict[str, tuple[str, str, str]],
    dynamic_allow: dict[tuple[str, str], tuple[str, str]],
) -> list[str]:
    """Every Options read under ``root`` whose key is neither declared, framework-reserved, nor allowlisted."""
    by_class, by_name = _collect_class_constants(root)
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for read in _reads_in_tree(tree, by_class, by_name):
            plugin = read.cls_name if read.cls_name is not None else "<module>"
            if read.key is not None:
                if read.key in declared or read.key in framework_keys:
                    continue
                allow_entry = literal_allow.get(read.key)
                if allow_entry is not None and allow_entry[0] == rel:
                    continue
                violations.append(f"{rel}:{read.lineno}: undeclared option read of key {read.key!r} (plugin {plugin})")
            elif not _dynamic_allowed(rel, read.func_name, dynamic_allow):
                violations.append(
                    f"{rel}:{read.lineno}: undocumented dynamic option read (plugin {plugin}, function {read.func_name})"
                )
    return violations


def property_mapping_declared_union() -> frozenset[str]:
    """Every PROPERTY_MAPPING key declared by any shipped (mloda_plugins) FeatureGroup subclass.

    Restricting to ``mloda_plugins`` modules drops throwaway test subclasses that other modules leak into
    ``FeatureGroup.__subclasses__()`` under xdist; those could only WIDEN the set and mask a real read.
    """
    keys: set[str] = set()
    for cls in get_all_subclasses(FeatureGroup):
        if cls.__module__.startswith("mloda_plugins"):
            keys |= set(cls.declared_option_keys())
    return frozenset(keys)


def reader_declared_union() -> frozenset[str]:
    """Every READER_OPTIONS key declared by ``BaseInputData`` or a shipped reader subclass.

    Filtered by module prefix for the same reason as the PROPERTY_MAPPING side, with one extra prefix:
    ``BaseInputData`` itself lives under ``mloda.core`` and is what declares the reserved
    ``"BaseInputData"`` key, so the filter admits ``mloda.core`` while still excluding ``tests.*``.
    """
    readers: list[type[BaseInputData]] = [BaseInputData, *get_all_subclasses(BaseInputData)]
    keys: set[str] = set()
    for cls in readers:
        if cls.__module__.startswith(("mloda_plugins", "mloda.core")):
            keys |= set(cls.declared_reader_option_keys())
    return frozenset(keys)


def declared_union() -> frozenset[str]:
    """Both declaration surfaces: a key passes when EITHER a FeatureGroup or a reader declares it."""
    return property_mapping_declared_union() | reader_declared_union()


def test_no_undeclared_static_option_reads() -> None:
    """No shipped feature_group plugin reads an Options key that nothing declares (outside the narrow allowlist)."""
    assert SCAN_ROOT.exists(), SCAN_ROOT
    # Vacuity floor, NOT a target: its only job is to prove the rglob loops really walked the tree
    # instead of finding nothing. Keep it around half the measured total (55 today) so that removing a
    # read site stays a legitimate change; pinning it to the exact count made deleting reads fail here.
    assert _count_reads(SCAN_ROOT) >= 28, "scan found too few reads; SCAN_ROOT is likely misconfigured"
    violations = find_violations(
        SCAN_ROOT,
        declared_union(),
        FRAMEWORK_KEYS,
        ALLOWED_LITERAL_KEYS,
        ALLOWED_DYNAMIC_READS,
    )
    assert violations == [], (
        "Undeclared option reads (declare the key in the plugin's PROPERTY_MAPPING or READER_OPTIONS):\n"
        + "\n".join(violations)
    )


def test_reference_time_declared_by_time_groups() -> None:
    """The time-based groups declare the reference_time column option they read."""
    assert "reference_time" in TimeWindowFeatureGroup.declared_option_keys()
    assert "reference_time" in ForecastingFeatureGroup.declared_option_keys()


def test_artifact_storage_path_declared_by_sklearn_groups() -> None:
    """The sklearn groups declare the artifact_storage_path option their artifact reads."""
    assert "artifact_storage_path" in EncodingFeatureGroup.declared_option_keys()
    assert "artifact_storage_path" in ScalingFeatureGroup.declared_option_keys()
    assert "artifact_storage_path" in SklearnPipelineFeatureGroup.declared_option_keys()


def test_allowlist_paths_are_relative_to_the_scan_root() -> None:
    """Every allowlist path resolves under SCAN_ROOT; a stale path silently stops matching and flags nothing."""
    for path_rel, _, _ in ALLOWED_LITERAL_KEYS.values():
        assert (SCAN_ROOT / path_rel).exists(), path_rel
    for path_rel, _ in ALLOWED_DYNAMIC_READS:
        assert (SCAN_ROOT / path_rel).exists(), path_rel


def test_literal_allowlist_is_load_bearing() -> None:
    """Drop the literal allowlist and its one key is flagged, so the entry (and its path) really matches."""
    violations = find_violations(SCAN_ROOT, declared_union(), FRAMEWORK_KEYS, {}, ALLOWED_DYNAMIC_READS)

    assert any("'initial_requested_data'" in v for v in violations), violations


@pytest.mark.parametrize("key", sorted(READER_ONLY_KEYS))
def test_declared_union_recognizes_reader_declarations(key: str) -> None:
    """The union covers the READER_OPTIONS surface, not just PROPERTY_MAPPING."""
    assert key in reader_declared_union()
    assert key in declared_union()


@pytest.mark.parametrize("key", sorted(READER_ONLY_KEYS))
def test_reader_keys_are_attributable_to_the_reader_surface_alone(key: str) -> None:
    """No PROPERTY_MAPPING and no framework-reserved key covers these, so only READER_OPTIONS carries them."""
    assert key not in property_mapping_declared_union()
    assert key not in FRAMEWORK_KEYS


def test_reader_declarations_are_load_bearing_for_the_tree_wide_scan() -> None:
    """Exclude the reader surface from the union and the readers' own match-time reads are flagged."""
    violations = find_violations(
        SCAN_ROOT,
        property_mapping_declared_union(),
        FRAMEWORK_KEYS,
        ALLOWED_LITERAL_KEYS,
        ALLOWED_DYNAMIC_READS,
    )

    assert any("'document_suffixes'" in v for v in violations), violations
    assert any("'data_access_handle'" in v for v in violations), violations
    assert all(v.startswith("input_data/") for v in violations), violations


def test_reader_union_ignores_readers_declared_in_the_test_tree() -> None:
    """A leaked test-tree reader is visible through __subclasses__() yet cannot widen the union."""
    probe = _make_leaked_reader_probe()

    assert _PROBE_READER_KEY in probe.declared_reader_option_keys()
    reachable = {key for cls in get_all_subclasses(BaseInputData) for key in cls.declared_reader_option_keys()}
    assert _PROBE_READER_KEY in reachable, "probe is unreachable, so the module-prefix filter is untested here"
    assert _PROBE_READER_KEY not in reader_declared_union()
    assert _PROBE_READER_KEY not in declared_union()


def test_read_document_dynamic_allowlist_entry_is_load_bearing() -> None:
    """Without its allowlist entry, the reader-selection read in ReadDocument.load_data is flagged."""
    without_entry = {key: value for key, value in ALLOWED_DYNAMIC_READS.items() if key != READ_DOCUMENT_DYNAMIC_READ}
    assert len(without_entry) == len(ALLOWED_DYNAMIC_READS) - 1

    violations = find_violations(SCAN_ROOT, declared_union(), FRAMEWORK_KEYS, ALLOWED_LITERAL_KEYS, without_entry)

    assert len(violations) == 1, violations
    assert violations[0].startswith("input_data/read_document.py:"), violations[0]
    assert "undocumented dynamic option read" in violations[0], violations[0]
    assert "function load_data" in violations[0], violations[0]


def test_read_document_selection_key_is_the_reader_class_name() -> None:
    """The allowlisted read is keyed by the reader's own class name, so no fixed declared key can cover it."""
    assert ALLOWED_DYNAMIC_READS[READ_DOCUMENT_DYNAMIC_READ][0] == ReadDocument.__name__
    assert MarkdownDocumentReader.data_access_name() == MarkdownDocumentReader.__name__
    assert MarkdownDocumentReader.__name__ not in declared_union()


class TestEveryKnownReadSiteStaysVisible:
    """Per-key pins on the three reader/group files whose reads a scanner blind spot could drop."""

    @pytest.mark.parametrize("key", ["disallowed_files", "file_type"])
    def test_read_context_files_default_backed_reads_are_resolved(self, key: str) -> None:
        """The two keys read through ``self.options_with_defaults(options)`` are seen by the scanner.

        These are exactly the reads that vanished when the read site moved behind the accessor: the
        receiver stopped being a name the scanner recognized as options-like.
        """
        assert key in _resolved_keys_in(READ_CONTEXT_FILES_REL)

    @pytest.mark.parametrize("key", ["file_paths", "target_folder", "document_reader_class"])
    def test_read_context_files_direct_reads_are_resolved(self, key: str) -> None:
        """Control: the three keys still read straight off ``options`` are seen too."""
        assert key in _resolved_keys_in(READ_CONTEXT_FILES_REL)

    def test_read_context_files_resolves_all_five_declared_keys(self) -> None:
        """The scanner's view of the file covers the whole declared inventory, with nothing extra."""
        assert _resolved_keys_in(READ_CONTEXT_FILES_REL) == ConcatenatedFileContent.declared_option_keys()

    @pytest.mark.parametrize("rel", [READ_FILE_REL, READ_DOCUMENT_REL])
    def test_reader_match_time_reads_are_resolved(self, rel: str) -> None:
        """Both readers' match-time keys stay visible whichever accessor resolves them."""
        resolved = _resolved_keys_in(rel)

        assert "document_suffixes" in resolved
        assert "data_access_handle" in resolved

    @pytest.mark.parametrize("rel", [READ_FILE_REL, READ_DOCUMENT_REL])
    def test_document_suffixes_is_read_through_the_reader_option_accessor(self, rel: str) -> None:
        """``document_suffixes`` resolves its declared default through the presence-honouring accessor.

        RED until the ``reader_option(key, options)`` accessor lands and both readers call it: today
        they use ``options.get(...) or cls.reader_option_default(...)``, which silently replaces an
        explicit empty value with the declared default.
        """
        assert "document_suffixes" in _reader_option_call_keys(rel)


def test_scanner_resolves_a_read_behind_options_with_defaults(tmp_path: Path) -> None:
    """A read behind ``x = obj.options_with_defaults(options)`` is resolved, not silently ignored."""
    source = (
        "class Foo:\n"
        "    def f(self, options):\n"
        "        effective = self.options_with_defaults(options)\n"
        "        return effective.get('undeclared_x')\n"
    )
    (tmp_path / "mod.py").write_text(source)

    violations = find_violations(tmp_path, frozenset(), frozenset(), {}, {})

    assert any("undeclared_x" in v for v in violations), violations
    assert find_violations(tmp_path, frozenset({"undeclared_x"}), frozenset(), {}, {}) == []


def test_scanner_resolves_a_direct_read_off_options_with_defaults(tmp_path: Path) -> None:
    """The un-aliased ``self.options_with_defaults(options).get(key)`` chain resolves too."""
    source = (
        "class Foo:\n"
        "    def f(self, options):\n"
        "        return self.options_with_defaults(options).get('undeclared_x')\n"
    )
    (tmp_path / "mod.py").write_text(source)

    violations = find_violations(tmp_path, frozenset(), frozenset(), {}, {})

    assert any("undeclared_x" in v for v in violations), violations


def test_scanner_flags_reader_option(tmp_path: Path) -> None:
    """``cls.reader_option('k', options)`` is recognized by method name, with the KEY as first argument."""
    source = (
        "class Foo:\n"
        "    @classmethod\n"
        "    def f(cls, options):\n"
        "        return cls.reader_option('undeclared_x', options)\n"
    )
    (tmp_path / "mod.py").write_text(source)

    violations = find_violations(tmp_path, frozenset(), frozenset(), {}, {})

    assert any("undeclared_x" in v for v in violations), violations
    assert find_violations(tmp_path, frozenset({"undeclared_x"}), frozenset(), {}, {}) == []


def test_scanner_reads_the_first_positional_argument_of_reader_option(tmp_path: Path) -> None:
    """The key is the FIRST argument: the Options in second position is never mistaken for the key."""
    source = (
        "class Foo:\n"
        "    @classmethod\n"
        "    def f(cls, options):\n"
        "        return cls.reader_option('undeclared_x', options)\n"
    )
    (tmp_path / "mod.py").write_text(source)

    violations = find_violations(tmp_path, frozenset(), frozenset(), {}, {})

    assert len(violations) == 1, violations
    assert "'undeclared_x'" in violations[0], violations[0]
    assert "dynamic" not in violations[0], violations[0]


def test_scanner_ignores_reader_option_default(tmp_path: Path) -> None:
    """``reader_option_default(key)`` consults no Options, so it is not an Options read.

    The recognition is an exact method-name match, so the longer sibling name does not match.
    """
    source = "class Foo:\n    @classmethod\n    def f(cls):\n        return cls.reader_option_default('undeclared_x')\n"
    (tmp_path / "mod.py").write_text(source)

    assert find_violations(tmp_path, frozenset(), frozenset(), {}, {}) == []


def test_scanner_flags_new_undeclared_literal_read(tmp_path: Path) -> None:
    """A brand-new literal read of an undeclared key is reported."""
    (tmp_path / "mod.py").write_text("def f(options):\n    return options.get('totally_new_key')\n")
    violations = find_violations(tmp_path, frozenset(), frozenset(), {}, {})
    assert any("totally_new_key" in v for v in violations), violations


def test_scanner_resolves_declared_class_constant(tmp_path: Path) -> None:
    """``options.get(cls.CONST)`` resolves through the class constant: declared passes, undeclared flags."""
    source = (
        "class Foo:\n"
        "    BAR = 'foo'\n"
        "\n"
        "    @classmethod\n"
        "    def f(cls, options):\n"
        "        return options.get(cls.BAR)\n"
    )
    (tmp_path / "mod.py").write_text(source)
    assert find_violations(tmp_path, frozenset({"foo"}), frozenset(), {}, {}) == []
    assert any("foo" in v for v in find_violations(tmp_path, frozenset(), frozenset(), {}, {}))


def test_scanner_resolves_default_option_keys_enum(tmp_path: Path) -> None:
    """``options.get(DefaultOptionKeys.reference_time)`` resolves to the enum member's value."""
    (tmp_path / "mod.py").write_text("def f(options):\n    return options.get(DefaultOptionKeys.reference_time)\n")
    assert find_violations(tmp_path, frozenset({"reference_time"}), frozenset(), {}, {}) == []


def test_scanner_resolves_local_alias(tmp_path: Path) -> None:
    """A local alias of a resolvable key is followed: declared passes, undeclared flags."""
    source = "def f(options):\n    k = DefaultOptionKeys.reference_time\n    return options.get(k)\n"
    (tmp_path / "mod.py").write_text(source)
    assert find_violations(tmp_path, frozenset({"reference_time"}), frozenset(), {}, {}) == []
    assert any("reference_time" in v for v in find_violations(tmp_path, frozenset(), frozenset(), {}, {}))


def test_scanner_flags_undocumented_dynamic_read(tmp_path: Path) -> None:
    """A computed key flags unless its (path, function) is documented in the dynamic allowlist."""
    (tmp_path / "mod.py").write_text("def foo(options, x):\n    return options.get(str(x))\n")
    assert any("dynamic" in v for v in find_violations(tmp_path, frozenset(), frozenset(), {}, {}))
    allow: dict[tuple[str, str], tuple[str, str]] = {("mod.py", "foo"): ("X", "runtime key")}
    assert find_violations(tmp_path, frozenset(), frozenset(), {}, allow) == []


def test_scanner_flags_aliased_options_receiver(tmp_path: Path) -> None:
    """A read through a local alias of ``feature.options`` is flagged, not silently ignored."""
    source = "def f(feature):\n    opts = feature.options\n    return opts.get('undeclared_x')\n"
    (tmp_path / "mod.py").write_text(source)
    violations = find_violations(tmp_path, frozenset(), frozenset(), {}, {})
    assert any("undeclared_x" in v for v in violations), violations


def test_scanner_flags_get_options_key(tmp_path: Path) -> None:
    """``features.get_options_key(...)`` is recognized by method name even off a non-options receiver."""
    (tmp_path / "mod.py").write_text("def f(features):\n    return features.get_options_key('undeclared_x')\n")
    violations = find_violations(tmp_path, frozenset(), frozenset(), {}, {})
    assert any("undeclared_x" in v for v in violations), violations


def test_scanner_flags_not_in_membership(tmp_path: Path) -> None:
    """``"k" not in options`` is scanned like ``"k" in options``."""
    source = "def f(options):\n    if 'undeclared_x' not in options:\n        return None\n    return options\n"
    (tmp_path / "mod.py").write_text(source)
    violations = find_violations(tmp_path, frozenset(), frozenset(), {}, {})
    assert any("undeclared_x" in v for v in violations), violations


def test_scanner_flags_keyword_arg_key(tmp_path: Path) -> None:
    """A key passed as the ``key=`` keyword is resolved like a positional key."""
    (tmp_path / "mod.py").write_text("def f(options):\n    return options.get(key='undeclared_x')\n")
    violations = find_violations(tmp_path, frozenset(), frozenset(), {}, {})
    assert any("undeclared_x" in v for v in violations), violations


def test_scanner_scopes_literal_allowlist_to_owner(tmp_path: Path) -> None:
    """A literal-allowlisted key passes only in its owning file; the same read elsewhere is flagged."""
    allow: dict[str, tuple[str, str, str]] = {"scoped_key": ("owner.py", "Owner", "engine-internal")}
    (tmp_path / "owner.py").write_text("def f(options):\n    return options.get('scoped_key')\n")
    (tmp_path / "other.py").write_text("def g(options):\n    return options.get('scoped_key')\n")
    violations = find_violations(tmp_path, frozenset(), frozenset(), allow, {})
    assert any(v.startswith("other.py") for v in violations), violations
    assert not any(v.startswith("owner.py") for v in violations), violations
