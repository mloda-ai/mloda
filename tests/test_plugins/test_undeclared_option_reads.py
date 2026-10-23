"""Static sweep: every Options key a shipped ``feature_group`` plugin reads must be a DECLARED key.

A plugin that reads ``options.get("foo")`` (directly, through a class constant, through a
``DefaultOptionKeys`` member, or through a local alias of one) is promising its users a ``foo`` option.
If ``foo`` is in no declaration (issue os-004 / mloda#775), that promise is undocumented and
unvalidated. This sweep resolves each read to its key and flags any key that no shipped plugin
declares, accepting only a narrow allowlist of genuinely engine-internal and genuinely dynamic reads.

The check is deliberately GLOBAL: a key passes when SOME shipped FeatureGroup declares it in
``PROPERTY_MAPPING``. Per-group attribution is out of scope for this sweep.

Scope notes:

* ``SCAN_ROOTS`` are the ``mloda_plugins/feature_group`` tree and the core ``input_data`` format group
  bases; allowlist paths are relative to the repo root, constants are collected over both roots.
* ``reference_time`` is a ``DefaultOptionKeys`` member yet is NOT treated as framework-reserved here.
  The time-based groups expose it as a user-overridable column option, so os-004 reclassifies it as a
  key those groups must DECLARE. The remaining ``DefaultOptionKeys`` values stay framework-reserved.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Sequence
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

# Anchor the scan root to the repo layout via __file__, not the cwd: a cwd-relative root makes the
# rglob loops empty and the sweep pass vacuously. This file sits two parents below the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOT = _REPO_ROOT / "mloda_plugins" / "feature_group"
assert SCAN_ROOT.exists(), f"scan root not found; check the parents index for the repo root: {SCAN_ROOT}"
CORE_SCAN_ROOT = _REPO_ROOT / "mloda" / "core" / "abstract_plugins" / "components" / "input_data"
assert CORE_SCAN_ROOT.exists(), f"core scan root not found: {CORE_SCAN_ROOT}"
SCAN_ROOTS = (SCAN_ROOT, CORE_SCAN_ROOT)

# Accessor calls whose result is an Options-like object: cls.get_singular_option_from_options(...).get(...),
# get_options(...).get(...) and cls.options_with_defaults(...).get(...). Anything else (arbitrary methods
# containing "option") is not an Options read. options_with_defaults is load-bearing here: a group that
# resolves its declared defaults reads every key through it, so leaving it out blinds the scan to those
# reads entirely.
_OPTIONS_ACCESSORS = frozenset({"get_singular_option_from_options", "get_options", "options_with_defaults"})
_OPTIONS_RECEIVER_NAMES = frozenset({"options", "option"})
_READ_METHODS = frozenset({"get", "get_options_key", "read_option"})
# get_options_key (FeatureSet method) and read_option (FeatureGroup classmethod) have non-options receivers,
# so they match by method name alone; both take the key as the first positional argument or `key=`.
_METHOD_NAME_ONLY_READS = frozenset({"get_options_key", "read_option"})

# DefaultOptionKeys.<member> resolves to the member's string value. Built from the enum so it cannot drift.
_DEFAULT_OPTION_KEY_MEMBERS: dict[str, str] = {member.name: member.value for member in DefaultOptionKeys}

# Framework-reserved keys any plugin may read without declaring. reference_time is excluded on purpose:
# os-004 makes it a declared per-group option, so a read of it must resolve through declared_union instead.
FRAMEWORK_KEYS: frozenset[str] = frozenset(_DEFAULT_OPTION_KEY_MEMBERS.values()) - {
    DefaultOptionKeys.reference_time.value
}

# Literal keys that are engine-internal signals rather than user-facing options, scoped to their owning
# source files (relative to the repo root): a read of the same literal from any other file is still flagged.
_FORMAT_BASES = "mloda/core/abstract_plugins/components/input_data/"
ALLOWED_LITERAL_KEYS: dict[str, tuple[tuple[str, ...], str, str]] = {
    "initial_requested_data": (
        ("mloda_plugins/feature_group/experimental/source_input_feature.py",),
        "SourceInputFeatureComposite",
        "engine-internal request-scoping signal set as a Feature attribute; not a user-facing option",
    ),
    "query_text": (
        (_FORMAT_BASES + "format_feature_group.py", _FORMAT_BASES + "read_db_fg.py"),
        "ReadDBFG",
        "declared by ReadDBFG subclasses that serve query text; the bases read it to route or decline",
    ),
}

# Reads whose key is computed at runtime, keyed by (path relative to the repo root, enclosing function).
ALLOWED_DYNAMIC_READS: dict[tuple[str, str], tuple[str, str]] = {
    (_FORMAT_BASES + "file_source_fg.py", "find_sources"): (
        "FileSourceFG",
        "key is one of the group's pointer keys, resolved at runtime",
    ),
    (_FORMAT_BASES + "file_source_fg.py", "_pointer_key"): (
        "FileSourceFG",
        "key is one of the group's pointer keys, resolved at runtime",
    ),
    (_FORMAT_BASES + "file_source_fg.py", "_pointer_value"): (
        "FileSourceFG",
        "key is one of the group's pointer keys, resolved at runtime",
    ),
    (_FORMAT_BASES + "format_feature_group.py", "is_pointed"): (
        "FormatFeatureGroup",
        "key is one of the group's pointer keys, resolved at runtime",
    ),
    (_FORMAT_BASES + "format_feature_group.py", "_matches_by_default_rules"): (
        "FormatFeatureGroup",
        "key is one of the group's pointer keys, resolved at runtime",
    ),
    (_FORMAT_BASES + "read_db_fg.py", "_matches_by_default_rules"): (
        "ReadDBFG",
        "key is one of the group's pointer keys, or a claim route's required key, resolved at runtime",
    ),
    (_FORMAT_BASES + "read_db_fg.py", "_candidates"): (
        "ReadDBFG",
        "key is one of the group's pointer keys, or a claim route's required key, resolved at runtime",
    ),
    (_FORMAT_BASES + "api/api_input_data.py", "matches"): (
        "ApiInputData",
        "key is the subclass's data access name",
    ),
    ("mloda_plugins/feature_group/experimental/forecasting/forecasting_artifact.py", "custom_loader"): (
        "ForecastingArtifact",
        "artifact is keyed by the runtime feature name, not a fixed option key",
    ),
}


@pytest.fixture(scope="module", autouse=True)
def _load_all_plugins() -> Iterator[None]:
    """Load every shipped plugin so the declared union enumerates the full plugin set."""
    PluginLoader().all()
    yield


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


def _module_name(path: Path, base: Path) -> str:
    """Dotted module name of ``path`` relative to ``base``; ``pkg/__init__.py`` maps to ``pkg``."""
    parts = list(path.relative_to(base).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _collect_module_constants(constant_roots: Sequence[Path], base: Path) -> dict[str, dict[str, str]]:
    """Map each module's dotted name to its top-level ``NAME = "literal"`` constants."""
    out: dict[str, dict[str, str]] = {}
    for root in constant_roots:
        for path in sorted(root.rglob("*.py")):
            consts: dict[str, str] = {}
            for stmt in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    value = stmt.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        consts[stmt.targets[0].id] = value.value
            out[_module_name(path, base)] = consts
    return out


def _file_constants(tree: ast.Module, module: str, module_consts: dict[str, dict[str, str]]) -> dict[str, str]:
    """Constants a bare name can resolve to in one file: its own plus absolute ``from X import NAME [as ALIAS]``."""
    out = dict(module_consts.get(module, {}))
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and stmt.module is not None:
            source = module_consts.get(stmt.module, {})
            for alias in stmt.names:
                if alias.name in source:
                    out[alias.asname or alias.name] = source[alias.name]
    return out


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
    file_consts: dict[str, str],
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
            return _resolve_key(expr, cls_name, by_class, by_name, func_assignments, file_consts)
        return file_consts.get(node.id)
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
    file_consts: dict[str, str],
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
            key = _resolve_key(key_node, cls_name, by_class, by_name, func_assignments, file_consts)
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


def _iter_reads(root: Path, base: Path | None, constant_roots: Sequence[Path] | None) -> Iterator[tuple[str, _Read]]:
    """Every Options read under ``root`` with its path relative to ``base`` (default ``root``)."""
    base = base if base is not None else root
    roots = constant_roots if constant_roots is not None else (root,)
    by_class, by_name = _collect_class_constants(root)
    module_consts = _collect_module_constants(roots, base)
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        file_consts = _file_constants(tree, _module_name(path, base), module_consts)
        for read in _reads_in_tree(tree, by_class, by_name, file_consts):
            yield path.relative_to(base).as_posix(), read


def _count_reads(root: Path, *, base: Path | None = None, constant_roots: Sequence[Path] | None = None) -> int:
    """Total Options reads found under ``root``; a nonzero floor guards against a vacuous scan."""
    return sum(1 for _ in _iter_reads(root, base, constant_roots))


def _resolved_read_keys(
    root: Path, *, base: Path | None = None, constant_roots: Sequence[Path] | None = None
) -> set[str]:
    """The non-None resolved keys of all reads under ``root``."""
    return {read.key for _, read in _iter_reads(root, base, constant_roots) if read.key is not None}


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
    literal_allow: dict[str, tuple[tuple[str, ...], str, str]],
    dynamic_allow: dict[tuple[str, str], tuple[str, str]],
    *,
    base: Path | None = None,
    constant_roots: Sequence[Path] | None = None,
) -> list[str]:
    """Every Options read under ``root`` whose key is neither declared, framework-reserved, nor allowlisted."""
    violations: list[str] = []
    for rel, read in _iter_reads(root, base, constant_roots):
        plugin = read.cls_name if read.cls_name is not None else "<module>"
        if read.key is not None:
            if read.key in declared or read.key in framework_keys:
                continue
            allow_entry = literal_allow.get(read.key)
            if allow_entry is not None and rel in allow_entry[0]:
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


def test_no_undeclared_static_option_reads() -> None:
    """No shipped feature_group plugin reads an Options key that nothing declares (outside the narrow allowlist)."""
    # Vacuity floors, NOT targets: their only job is to prove the rglob loops really walked each tree
    # instead of finding nothing. Each is about half the measured count (30 plugins, 18 core today) so that
    # removing a read site stays a legitimate change; pinning the exact count made deleting reads fail here.
    floors = {SCAN_ROOT: 15, CORE_SCAN_ROOT: 9}
    violations: list[str] = []
    for root in SCAN_ROOTS:
        assert root.exists(), root
        count = _count_reads(root, base=_REPO_ROOT, constant_roots=SCAN_ROOTS)
        assert count >= floors[root], f"scan found too few reads under {root}; the root is likely misconfigured"
        violations += find_violations(
            root,
            property_mapping_declared_union(),
            FRAMEWORK_KEYS,
            ALLOWED_LITERAL_KEYS,
            ALLOWED_DYNAMIC_READS,
            base=_REPO_ROOT,
            constant_roots=SCAN_ROOTS,
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


def test_allowlist_paths_are_relative_to_the_scan_root() -> None:
    """Every allowlist path resolves under _REPO_ROOT; a stale path silently stops matching and flags nothing."""
    for paths_rel, _, _ in ALLOWED_LITERAL_KEYS.values():
        for path_rel in paths_rel:
            assert (_REPO_ROOT / path_rel).exists(), path_rel
    for path_rel, _ in ALLOWED_DYNAMIC_READS:
        assert (_REPO_ROOT / path_rel).exists(), path_rel


def test_literal_allowlist_is_load_bearing() -> None:
    """Drop the literal allowlist and its one key is flagged, so the entry (and its path) really matches."""
    violations: list[str] = []
    for root in SCAN_ROOTS:
        violations += find_violations(
            root,
            property_mapping_declared_union(),
            FRAMEWORK_KEYS,
            {},
            ALLOWED_DYNAMIC_READS,
            base=_REPO_ROOT,
            constant_roots=SCAN_ROOTS,
        )

    assert any("'initial_requested_data'" in v for v in violations), violations


def test_the_format_groups_declare_the_match_time_keys() -> None:
    """The handle and document_suffixes keys now come from the shipped format groups' PROPERTY_MAPPING."""
    assert {"data_access_handle", "document_suffixes"} <= property_mapping_declared_union()


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


def test_scanner_flags_read_option(tmp_path: Path) -> None:
    """``cls.read_option(KEY, options)`` is recognized by method name, with literal and class-constant keys."""
    source = (
        "class P:\n"
        "    BAR = 'foo'\n"
        "    @classmethod\n"
        "    def a(cls, options):\n"
        "        return cls.read_option('undeclared_x', options)\n"
        "    @classmethod\n"
        "    def b(cls, options):\n"
        "        return cls.read_option(cls.BAR, options)\n"
        "    @classmethod\n"
        "    def c(cls, options):\n"
        "        return cls.read_option(key='undeclared_y', options=options)\n"
    )
    (tmp_path / "mod.py").write_text(source)
    violations = find_violations(tmp_path, frozenset(), frozenset(), {}, {})
    assert any("undeclared_x" in v for v in violations), violations
    assert any("'foo'" in v for v in violations), violations
    assert any("undeclared_y" in v for v in violations), violations
    declared = frozenset({"undeclared_x", "foo", "undeclared_y"})
    assert find_violations(tmp_path, declared, frozenset(), {}, {}) == []


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
    allow: dict[str, tuple[tuple[str, ...], str, str]] = {"scoped_key": (("owner.py",), "Owner", "engine-internal")}
    (tmp_path / "owner.py").write_text("def f(options):\n    return options.get('scoped_key')\n")
    (tmp_path / "other.py").write_text("def g(options):\n    return options.get('scoped_key')\n")
    violations = find_violations(tmp_path, frozenset(), frozenset(), allow, {})
    assert any(v.startswith("other.py") for v in violations), violations
    assert not any(v.startswith("owner.py") for v in violations), violations


def test_scanner_allows_a_literal_key_in_every_owning_path(tmp_path: Path) -> None:
    """A literal allowlist entry with two owning paths passes in both files and still flags a third."""
    allow: dict[str, tuple[tuple[str, ...], str, str]] = {"shared_key": (("one.py", "two.py"), "Owner", "internal")}
    for name in ("one.py", "two.py", "three.py"):
        (tmp_path / name).write_text("def f(options):\n    return options.get('shared_key')\n")
    violations = find_violations(tmp_path, frozenset(), frozenset(), allow, {})
    assert [v.split(":")[0] for v in violations] == ["three.py"], violations


def test_scanner_resolves_a_module_level_constant(tmp_path: Path) -> None:
    """A module-level ``NAME = 'literal'`` resolves for a bare NAME read in the same file."""
    (tmp_path / "mod.py").write_text("KEY = 'module_key'\n\n\ndef f(options):\n    return options.get(KEY)\n")
    assert find_violations(tmp_path, frozenset({"module_key"}), frozenset(), {}, {}) == []
    violations = find_violations(tmp_path, frozenset(), frozenset(), {}, {})
    assert any("'module_key'" in v and "dynamic" not in v for v in violations), violations


def test_scanner_resolves_a_constant_imported_by_module_path(tmp_path: Path) -> None:
    """``from consts import KEY`` resolves through consts.py under the constant roots."""
    (tmp_path / "consts.py").write_text("KEY = 'imported_key'\n")
    (tmp_path / "user.py").write_text("from consts import KEY\n\n\ndef f(options):\n    return options.get(KEY)\n")
    assert find_violations(tmp_path, frozenset({"imported_key"}), frozenset(), {}, {}, base=tmp_path) == []
    violations = find_violations(tmp_path, frozenset(), frozenset(), {}, {}, base=tmp_path)
    assert any("'imported_key'" in v for v in violations), violations


def test_scanner_resolves_an_aliased_import(tmp_path: Path) -> None:
    """``from consts import KEY as OTHER`` resolves the alias to the constant's value."""
    (tmp_path / "consts.py").write_text("KEY = 'aliased_key'\n")
    (tmp_path / "user.py").write_text(
        "from consts import KEY as OTHER\n\n\ndef f(options):\n    return options.get(OTHER)\n"
    )
    assert find_violations(tmp_path, frozenset({"aliased_key"}), frozenset(), {}, {}, base=tmp_path) == []
    violations = find_violations(tmp_path, frozenset(), frozenset(), {}, {}, base=tmp_path)
    assert any("'aliased_key'" in v for v in violations), violations


def test_scanner_keeps_an_import_from_an_unknown_module_dynamic(tmp_path: Path) -> None:
    """No name-only fallback: an import from a module outside the constant roots stays a dynamic read."""
    (tmp_path / "other_consts.py").write_text("KEY = 'wrong_module_key'\n")
    (tmp_path / "user.py").write_text("from elsewhere import KEY\n\n\ndef f(options):\n    return options.get(KEY)\n")
    violations = find_violations(tmp_path, frozenset(), frozenset(), {}, {}, base=tmp_path)
    assert any(v.startswith("user.py") and "dynamic" in v for v in violations), violations


def test_scanner_resolves_constants_across_constant_roots(tmp_path: Path) -> None:
    """``constant_roots`` lets a file under one root resolve a constant defined under another."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "consts.py").write_text("KEY = 'cross_root_key'\n")
    (tmp_path / "a" / "user.py").write_text(
        "from b.consts import KEY\n\n\ndef f(options):\n    return options.get(KEY)\n"
    )
    roots = (tmp_path / "a", tmp_path / "b")
    declared = frozenset({"cross_root_key"})
    assert find_violations(tmp_path / "a", declared, frozenset(), {}, {}, base=tmp_path, constant_roots=roots) == []
    violations = find_violations(tmp_path / "a", frozenset(), frozenset(), {}, {}, base=tmp_path, constant_roots=roots)
    assert any(v.startswith("a/user.py") and "'cross_root_key'" in v for v in violations), violations


def test_core_format_constant_reads_resolve() -> None:
    """The core format bases' module-level constant reads resolve to their keys, not to dynamic reads."""
    keys = _resolved_read_keys(CORE_SCAN_ROOT, base=_REPO_ROOT, constant_roots=SCAN_ROOTS)
    assert {"data_access_handle", "query_text", "document_suffixes"} <= keys
