"""Static sweep: every Options key a feature_group plugin reads must be a declared PROPERTY_MAPPING key.

A feature group that reads ``options.get("foo")`` (directly, through a class constant, through a
``DefaultOptionKeys`` member, or through a local alias of one) is promising its users a ``foo`` option.
If ``foo`` is in no ``PROPERTY_MAPPING`` (issue os-004 / mloda#775), that promise is undocumented and
unvalidated. This sweep resolves each read to its key and flags any key that no shipped FeatureGroup
declares, accepting only a narrow allowlist of genuinely engine-internal and genuinely dynamic reads.

The check is deliberately GLOBAL: a key passes when SOME shipped feature group declares it, not the one
that reads it. That is the intended model for shared keys like ``reference_time``; per-group owning-surface
attribution is out of scope for this sweep.

Scope notes:

* ``SCAN_ROOT`` is the whole ``mloda_plugins/feature_group`` tree (issue os-013 widened it from the
  experimental subtree), so the ``input_data`` readers are swept too; allowlist paths are relative to it.
* ``reference_time`` is a ``DefaultOptionKeys`` member yet is NOT treated as framework-reserved here.
  The time-based groups expose it as a user-overridable column option, so os-004 reclassifies it as a
  key those groups must DECLARE. The remaining ``DefaultOptionKeys`` values stay framework-reserved.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

import pytest

from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
from mloda.provider import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.forecasting.base import ForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.encoding.base import EncodingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.pipeline.base import SklearnPipelineFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.scaling.base import ScalingFeatureGroup
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup
from mloda_plugins.feature_group.input_data.read_context_files import ConcatenatedFileContent
from mloda_plugins.feature_group.input_data.read_db_feature import ReadDBFeature
from mloda_plugins.feature_group.input_data.read_document_feature import ReadDocumentFeature
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature

# Anchor the scan root to the repo layout via __file__, not the cwd: a cwd-relative root makes the
# rglob loops empty and the sweep pass vacuously. This file sits two parents below the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOT = _REPO_ROOT / "mloda_plugins" / "feature_group"
assert SCAN_ROOT.exists(), f"scan root not found; check the parents index for the repo root: {SCAN_ROOT}"

# Accessor calls whose result is an Options-like object: cls.get_singular_option_from_options(...).get(...)
# and get_options(...).get(...). Anything else (arbitrary methods containing "option") is not an Options read.
_OPTIONS_ACCESSORS = frozenset({"get_singular_option_from_options", "get_options"})
_OPTIONS_RECEIVER_NAMES = frozenset({"options", "option"})
_READ_METHODS = frozenset({"get", "get_options_key"})
# get_options_key is a FeatureSet method: its receiver (features/self) is never options-like, so it is
# recognized by method name alone. The name is distinctive enough to carry the match on its own.
_METHOD_NAME_ONLY_READS = frozenset({"get_options_key"})

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
    "BaseInputData": (
        "input_data/read_dbs/sqlite.py",
        "SQLITEReader",
        "engine-internal (ReaderClass, data_access) tuple recorded by BaseInputData matching and "
        "consumed by init_reader; not a user-facing option",
    ),
}

# Reads whose key is computed at runtime, keyed by (path relative to SCAN_ROOT, enclosing function).
ALLOWED_DYNAMIC_READS: dict[tuple[str, str], tuple[str, str]] = {
    ("experimental/forecasting/forecasting_artifact.py", "custom_loader"): (
        "ForecastingArtifact",
        "artifact is keyed by the runtime feature name, not a fixed option key",
    ),
    ("input_data/read_document.py", "load_data"): (
        "ReadDocument",
        "reader-selection contract: the options key is the reader's class name, computed at runtime",
    ),
}


@pytest.fixture(scope="module", autouse=True)
def _load_all_plugins() -> None:
    """Load every shipped plugin so declared_union enumerates the full FeatureGroup set."""
    PluginLoader().all()


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


def declared_union() -> frozenset[str]:
    """Every PROPERTY_MAPPING key declared by any shipped (mloda_plugins) FeatureGroup subclass.

    Restricting to ``mloda_plugins`` modules drops throwaway test subclasses that other modules leak into
    ``FeatureGroup.__subclasses__()`` under xdist; those could only WIDEN the set and mask a real read.
    """
    keys: set[str] = set()
    for cls in get_all_subclasses(FeatureGroup):
        if cls.__module__.startswith("mloda_plugins"):
            keys |= set(cls.declared_option_keys())
    return frozenset(keys)


def test_no_undeclared_static_option_reads() -> None:
    """No feature_group plugin reads an Options key that no FeatureGroup declares (outside the narrow allowlist)."""
    assert SCAN_ROOT.exists(), SCAN_ROOT
    assert _count_reads(SCAN_ROOT) >= 40, "scan found too few reads; SCAN_ROOT is likely misconfigured"
    violations = find_violations(
        SCAN_ROOT,
        declared_union(),
        FRAMEWORK_KEYS,
        ALLOWED_LITERAL_KEYS,
        ALLOWED_DYNAMIC_READS,
    )
    assert violations == [], (
        "Undeclared option reads (declare the key in the plugin's PROPERTY_MAPPING):\n" + "\n".join(violations)
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


def test_reader_groups_declare_document_suffixes() -> None:
    """The file and document readers declare the document_suffixes option they read."""
    assert "document_suffixes" in ReadFileFeature.declared_option_keys()
    assert "document_suffixes" in ReadDocumentFeature.declared_option_keys()


def test_reader_groups_declare_data_access_handle() -> None:
    """The reader groups declare the data_access_handle option their readers consume."""
    assert "data_access_handle" in ReadFileFeature.declared_option_keys()
    assert "data_access_handle" in ReadDocumentFeature.declared_option_keys()
    assert "data_access_handle" in ReadDBFeature.declared_option_keys()


def test_concatenated_file_content_declares_its_keys() -> None:
    """ConcatenatedFileContent declares every option key it reads."""
    declared = ConcatenatedFileContent.declared_option_keys()
    for key in ("disallowed_files", "file_paths", "target_folder", "file_type", "document_reader_class"):
        assert key in declared, key


def test_scanner_flags_new_undeclared_literal_read(tmp_path: Path) -> None:
    """A brand-new literal read of an undeclared key is reported."""
    (tmp_path / "mod.py").write_text("def f(options):\n    return options.get('totally_new_key')\n")
    violations = find_violations(tmp_path, frozenset(), frozenset(), {}, {})
    assert any("totally_new_key" in v for v in violations), violations


def test_scanner_reports_plugin_key_and_location(tmp_path: Path) -> None:
    """A violation names the reading class, the key, and the file:line of the read."""
    source = "class Foo:\n    @classmethod\n    def f(cls, options):\n        return options.get('totally_new_key')\n"
    (tmp_path / "mod.py").write_text(source)
    violations = find_violations(tmp_path, frozenset(), frozenset(), {}, {})
    assert any("plugin Foo" in v and "totally_new_key" in v and v.startswith("mod.py:4") for v in violations), (
        violations
    )


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
