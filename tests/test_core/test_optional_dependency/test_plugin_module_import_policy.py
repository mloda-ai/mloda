"""Static AST policy check: a module under mloda_plugins/** may only import, at module level, a
third-party root that its own pyproject extra (directly or through IMPLIED) declares. An unguarded
import of an undeclared library makes the plugin loader silently skip the module when it is absent.
It also checks that each require() literal names an extra, and that MODULE_EXTRA maps every backend-reaching module."""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Collection, Iterator
from pathlib import Path
from typing import NamedTuple

import pytest

import mloda

from tests.test_core.test_abstract_plugins.test_plugin_registry.test_compute_framework_exports import (
    _package_directory,
)
from tests.test_core.test_optional_dependency.test_no_optional_backend_imports_in_core import (
    _dynamic_import_aliases,
    _dynamic_import_root,
    _is_type_checking_test,
)
from tests.test_core.test_optional_pyarrow.test_pyproject_optional_extras import _load_optional_deps

_BACKENDS = "mloda_plugins.compute_framework.base_implementations"
_EXPERIMENTAL = "mloda_plugins.feature_group.experimental"
_INPUT_DATA = "mloda_plugins.feature_group.input_data"

# Module -> home extra (the future per-backend package). Only EAGER_MODULES import it unconditionally.
MODULE_EXTRA: dict[str, str] = {
    f"{_BACKENDS}.duckdb.duckdb_filter_engine": "duckdb",
    f"{_BACKENDS}.duckdb.duckdb_framework": "duckdb",
    f"{_BACKENDS}.duckdb.duckdb_mask_engine": "duckdb",
    f"{_BACKENDS}.duckdb.duckdb_merge_engine": "duckdb",
    f"{_BACKENDS}.duckdb.duckdb_pyarrow_transformer": "duckdb",
    f"{_BACKENDS}.duckdb.duckdb_relation": "duckdb",
    f"{_BACKENDS}.iceberg.iceberg_filter_engine": "iceberg",
    f"{_BACKENDS}.iceberg.iceberg_framework": "iceberg",
    f"{_BACKENDS}.iceberg.iceberg_pyarrow_transformer": "iceberg",
    f"{_BACKENDS}.pandas.dataframe": "pandas",
    f"{_BACKENDS}.pandas.pandas_filter_engine": "pandas",
    f"{_BACKENDS}.pandas.pandas_mask_engine": "pandas",
    f"{_BACKENDS}.pandas.pandas_merge_engine": "pandas",
    f"{_BACKENDS}.pandas.pandas_pyarrow_transformer": "pandas",
    f"{_BACKENDS}.pandas.pandas_type_semantics": "pandas",
    f"{_BACKENDS}.polars.dataframe": "polars",
    f"{_BACKENDS}.polars.lazy_dataframe": "polars",
    f"{_BACKENDS}.polars.polars_expr_mask_engine": "polars",
    f"{_BACKENDS}.polars.polars_filter_engine": "polars",
    f"{_BACKENDS}.polars.polars_lazy_merge_engine": "polars",
    f"{_BACKENDS}.polars.polars_lazy_pyarrow_transformer": "polars",
    f"{_BACKENDS}.polars.polars_mask_engine": "polars",
    f"{_BACKENDS}.polars.polars_merge_engine": "polars",
    f"{_BACKENDS}.polars.polars_pyarrow_transformer": "polars",
    f"{_BACKENDS}.polars.polars_type_semantics": "polars",
    f"{_BACKENDS}.pyarrow.pyarrow_file_source_transformer": "pyarrow",
    f"{_BACKENDS}.pyarrow.pyarrow_filter_engine": "pyarrow",
    f"{_BACKENDS}.pyarrow.pyarrow_mask_engine": "pyarrow",
    f"{_BACKENDS}.pyarrow.pyarrow_merge_engine": "pyarrow",
    f"{_BACKENDS}.pyarrow.pyarrow_type_semantics": "pyarrow",
    f"{_BACKENDS}.pyarrow.pyarrow_value_set": "pyarrow",
    f"{_BACKENDS}.pyarrow.table": "pyarrow",
    f"{_BACKENDS}.python_dict.python_dict_pyarrow_transformer": "pyarrow",
    f"{_BACKENDS}.spark.spark_filter_engine": "spark",
    f"{_BACKENDS}.spark.spark_framework": "spark",
    f"{_BACKENDS}.spark.spark_mask_engine": "spark",
    f"{_BACKENDS}.spark.spark_merge_engine": "spark",
    f"{_BACKENDS}.spark.spark_pyarrow_transformer": "spark",
    f"{_BACKENDS}.spark.spark_type_semantics": "spark",
    f"{_BACKENDS}.sql.sql_base_filter_engine": "pyarrow",
    f"{_BACKENDS}.sql.sql_base_mask_engine": "pyarrow",
    f"{_BACKENDS}.sql.sql_base_merge_engine": "pyarrow",
    f"{_BACKENDS}.sql.sql_base_pyarrow_transformer": "pyarrow",
    f"{_BACKENDS}.sql.sql_base_relation": "pyarrow",
    f"{_BACKENDS}.sql.sql_type_semantics": "pyarrow",
    f"{_BACKENDS}.sql.sql_utils": "pyarrow",
    f"{_BACKENDS}.sql.sql_window": "pyarrow",
    f"{_BACKENDS}.sqlite.sqlite_filter_engine": "sqlite",
    f"{_BACKENDS}.sqlite.sqlite_framework": "sqlite",
    f"{_BACKENDS}.sqlite.sqlite_mask_engine": "sqlite",
    f"{_BACKENDS}.sqlite.sqlite_merge_engine": "sqlite",
    f"{_BACKENDS}.sqlite.sqlite_pyarrow_transformer": "sqlite",
    f"{_BACKENDS}.sqlite.sqlite_relation": "sqlite",
    f"{_BACKENDS}.sqlite.sqlite_value_sample": "sqlite",
    f"{_EXPERIMENTAL}.aggregated_feature_group.pandas": "pandas",
    f"{_EXPERIMENTAL}.aggregated_feature_group.polars_lazy": "polars",
    f"{_EXPERIMENTAL}.aggregated_feature_group.pyarrow": "pyarrow",
    f"{_EXPERIMENTAL}.clustering.pandas": "pandas",
    f"{_EXPERIMENTAL}.data_quality.missing_value.pandas": "pandas",
    f"{_EXPERIMENTAL}.data_quality.missing_value.pyarrow": "pyarrow",
    f"{_EXPERIMENTAL}.dimensionality_reduction.pandas": "pandas",
    f"{_EXPERIMENTAL}.environment.installed_packages_feature_group": "pandas",
    f"{_EXPERIMENTAL}.environment.list_directory_feature_group": "pandas",
    f"{_EXPERIMENTAL}.forecasting.pandas": "pandas",
    f"{_EXPERIMENTAL}.geo_distance.pandas": "pandas",
    f"{_EXPERIMENTAL}.node_centrality.pandas": "pandas",
    f"{_EXPERIMENTAL}.sklearn.encoding.base": "sklearn",
    f"{_EXPERIMENTAL}.sklearn.encoding.pandas": "sklearn",
    f"{_EXPERIMENTAL}.sklearn.pipeline.base": "sklearn",
    f"{_EXPERIMENTAL}.sklearn.pipeline.pandas": "sklearn",
    f"{_EXPERIMENTAL}.sklearn.scaling.base": "sklearn",
    f"{_EXPERIMENTAL}.sklearn.scaling.pandas": "sklearn",
    f"{_EXPERIMENTAL}.sklearn.sklearn_artifact": "sklearn",
    f"{_EXPERIMENTAL}.text_cleaning.pandas": "pandas",
    f"{_EXPERIMENTAL}.text_cleaning.python_dict": "text_cleaning",
    f"{_EXPERIMENTAL}.time_window.pandas": "pandas",
    f"{_EXPERIMENTAL}.time_window.pyarrow": "pyarrow",
    f"{_INPUT_DATA}.read_context_files": "pandas",
    f"{_INPUT_DATA}.read_dbs.sqlite": "sqlite",
    f"{_INPUT_DATA}.read_files.feather": "pyarrow",
    f"{_INPUT_DATA}.read_files.json": "pyarrow",
    f"{_INPUT_DATA}.read_files.orc": "pyarrow",
    f"{_INPUT_DATA}.read_files.parquet": "pyarrow",
    f"{_INPUT_DATA}.read_files.yaml_document_reader": "yaml",
}

# Extras a module reaches only lazily, guarded or through an import edge, on top of its home extra.
ALSO_NEEDS: dict[str, tuple[str, ...]] = {
    f"{_BACKENDS}.pandas.pandas_pyarrow_transformer": ("pyarrow",),
    f"{_BACKENDS}.polars.polars_lazy_pyarrow_transformer": ("pyarrow",),
    f"{_BACKENDS}.polars.polars_pyarrow_transformer": ("pyarrow",),
    f"{_EXPERIMENTAL}.aggregated_feature_group.pyarrow": ("pandas",),
    f"{_EXPERIMENTAL}.clustering.pandas": ("sklearn",),
    f"{_EXPERIMENTAL}.data_quality.missing_value.pyarrow": ("pandas",),
    f"{_EXPERIMENTAL}.dimensionality_reduction.pandas": ("sklearn",),
    f"{_EXPERIMENTAL}.forecasting.pandas": ("sklearn",),
    f"{_EXPERIMENTAL}.sklearn.encoding.pandas": ("pandas",),
    f"{_EXPERIMENTAL}.sklearn.pipeline.pandas": ("pandas",),
    f"{_EXPERIMENTAL}.sklearn.scaling.pandas": ("pandas",),
    f"{_EXPERIMENTAL}.text_cleaning.pandas": ("text_cleaning",),
    f"{_EXPERIMENTAL}.time_window.pyarrow": ("pandas",),
}

# The only modules allowed to import their home extra's roots unconditionally.
EAGER_MODULES: frozenset[str] = frozenset(
    {
        "mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow",
        "mloda_plugins.feature_group.experimental.data_quality.missing_value.pyarrow",
        "mloda_plugins.feature_group.experimental.time_window.pyarrow",
        "mloda_plugins.feature_group.experimental.geo_distance.pandas",
        "mloda_plugins.feature_group.experimental.time_window.pandas",
        "mloda_plugins.feature_group.input_data.read_dbs.sqlite",
    }
)

# Modules asserted to reach no third-party root.
BACKEND_NEUTRAL: frozenset[str] = frozenset(
    {
        "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_file_source_transformer",
        "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_filter_engine",
        "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework",
        "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_mask_engine",
        "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_merge_engine",
        "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_type_semantics",
        "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils",
        "mloda_plugins.feature_group.experimental.aggregated_feature_group.base",
        "mloda_plugins.feature_group.experimental.clustering.base",
        "mloda_plugins.feature_group.experimental.data_quality.missing_value.base",
        "mloda_plugins.feature_group.experimental.data_quality.missing_value.python_dict",
        "mloda_plugins.feature_group.experimental.dimensionality_reduction.base",
        "mloda_plugins.feature_group.experimental.forecasting.base",
        "mloda_plugins.feature_group.experimental.geo_distance.base",
        "mloda_plugins.feature_group.experimental.node_centrality.base",
        "mloda_plugins.feature_group.experimental.text_cleaning.base",
        "mloda_plugins.feature_group.experimental.time_window.base",
    }
)

BACKEND_EXTRAS: frozenset[str] = frozenset(
    {"pyarrow", "pandas", "polars", "numpy", "yaml", "duckdb", "sqlite", "iceberg", "spark", "sklearn", "text_cleaning"}
)

# Import roots a library hard-depends on, still covered by the extra that ships the library itself.
IMPLIED: dict[str, frozenset[str]] = {
    "pandas": frozenset({"numpy"}),
    "sklearn": frozenset({"numpy", "scipy"}),
}

# Distribution name -> top-level import it provides, for names that don't match "lowercased, '-' to '_'".
DISTRIBUTION_IMPORT_ROOT: dict[str, str] = {"pyyaml": "yaml", "scikit-learn": "sklearn"}

_REQUIREMENT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

_STDLIB: frozenset[str] = sys.stdlib_module_names

_MLODA_PLUGINS_DIR: Path = _package_directory("mloda_plugins")

_FIRST_PARTY_ROOT_DIRS: dict[str, list[Path]] = {
    "mloda": [Path(entry) for entry in mloda.__path__ if Path(entry).is_dir()],
    "mloda_plugins": [_MLODA_PLUGINS_DIR],
}

# Other mloda.* core is deliberately not followed: it is backend-neutral, never a plugin backend.
_MAP_FOLLOW_PREFIXES: tuple[str, ...] = ("mloda_plugins.", "mloda.user.")


def _strip_requirement(requirement: str) -> str:
    without_marker = requirement.split(";", 1)[0]
    without_extras = without_marker.split("[", 1)[0]
    match = _REQUIREMENT_NAME_RE.match(without_extras.strip())
    return match.group(0) if match else without_extras.strip()


def _import_root_for_distribution(distribution: str) -> str:
    mapped = DISTRIBUTION_IMPORT_ROOT.get(distribution.lower())
    if mapped is not None:
        return mapped
    return distribution.lower().replace("-", "_")


def _extra_import_roots(extra: str) -> frozenset[str]:
    roots: set[str] = set()
    for requirement in _load_optional_deps()[extra]:
        distribution = _strip_requirement(requirement)
        if distribution.lower() == "mloda":
            continue
        roots.add(_import_root_for_distribution(distribution))
    return frozenset(roots)


def allowed_roots(extra: str) -> frozenset[str]:
    base = _extra_import_roots(extra)
    implied: set[str] = set()
    for root in base:
        implied |= IMPLIED.get(root, frozenset())
    return base | implied


class ImportTarget(NamedTuple):
    candidate: str
    module: str

    @property
    def root(self) -> str:
        return self.candidate.partition(".")[0]


def _resolve_relative(module: str | None, level: int, package: str) -> str:
    if level == 0:
        assert module is not None
        return module
    bits = package.rsplit(".", level - 1)
    assert len(bits) >= level, f"relative import escapes its package: level {level} in package '{package}'"
    base = bits[0]
    return f"{base}.{module}" if module else base


def _is_import_failure_handler(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    candidates = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    catches = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
    for candidate in candidates:
        if isinstance(candidate, ast.Name) and candidate.id in catches:
            return True
        if isinstance(candidate, ast.Attribute) and candidate.attr in catches:
            return True
    return False


def _try_catches_import_failure(node: ast.Try) -> bool:
    return any(_is_import_failure_handler(handler) for handler in node.handlers)


def _unconditional_statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """A `try` body is skipped only if a handler catches an import failure; handlers, else, and finally always run."""
    for node in body:
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.If):
            if _is_type_checking_test(node.test):
                yield from _unconditional_statements(node.orelse)
            else:
                yield from _unconditional_statements(node.body)
                yield from _unconditional_statements(node.orelse)
            continue
        if isinstance(node, ast.Try):
            if not _try_catches_import_failure(node):
                yield from _unconditional_statements(node.body)
            for handler in node.handlers:
                yield from _unconditional_statements(handler.body)
            yield from _unconditional_statements(node.orelse)
            yield from _unconditional_statements(node.finalbody)
            continue
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            yield from _unconditional_statements(node.body)
            yield from _unconditional_statements(node.orelse)
            continue
        if isinstance(node, (ast.With, ast.AsyncWith)):
            yield from _unconditional_statements(node.body)
            continue
        if isinstance(node, ast.ClassDef):
            yield from _unconditional_statements(node.body)
            continue


def _calls_excluding_lambdas(value: ast.expr) -> Iterator[ast.Call]:
    stack: list[ast.AST] = [value]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.Call):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def _require_call_target(call: ast.Call) -> str | None:
    func = call.func
    is_require = (isinstance(func, ast.Name) and func.id == "require") or (
        isinstance(func, ast.Attribute) and func.attr == "require"
    )
    if not is_require or not call.args:
        return None
    first = call.args[0]
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        return None
    return first.value


def _statement_import_targets(node: ast.Import | ast.ImportFrom, package: str) -> Iterator[ImportTarget]:
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield ImportTarget(alias.name, alias.name)
        return
    if node.module == "__future__" and node.level == 0:
        return
    module = _resolve_relative(node.module, node.level, package)
    for alias in node.names:
        yield ImportTarget(f"{module}.{alias.name}", module)


def _call_import_target(call: ast.Call, aliases: set[str]) -> ImportTarget | None:
    name = _dynamic_import_root(call, aliases)
    if name is None:
        name = _require_call_target(call)
    return ImportTarget(name, name) if name is not None else None


def unconditional_import_targets(tree: ast.Module, package: str) -> frozenset[ImportTarget]:
    aliases = _dynamic_import_aliases(tree)
    targets: set[ImportTarget] = set()

    for node in _unconditional_statements(tree.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            targets.update(_statement_import_targets(node, package))
        elif isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None:
                continue
            for call in _calls_excluding_lambdas(value):
                target = _call_import_target(call, aliases)
                if target is not None:
                    targets.add(target)

    return frozenset(targets)


def reachable_import_targets(tree: ast.Module, package: str) -> frozenset[ImportTarget]:
    aliases = _dynamic_import_aliases(tree)
    targets: set[ImportTarget] = set()

    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.If) and _is_type_checking_test(node.test):
            stack.extend(node.orelse)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            targets.update(_statement_import_targets(node, package))
        elif isinstance(node, ast.Call):
            target = _call_import_target(node, aliases)
            if target is not None:
                targets.add(target)
        stack.extend(ast.iter_child_nodes(node))

    return frozenset(targets)


def _package_init(dirs: list[Path]) -> Path | None:
    for base in dirs:
        init_file = base / "__init__.py"
        if init_file.is_file():
            return init_file
    return None


def _first_party_file(dotted: str) -> Path | None:
    parts = dotted.split(".")
    dirs = _FIRST_PARTY_ROOT_DIRS.get(parts[0])
    if dirs is None:
        return None
    if len(parts) == 1:
        return _package_init(dirs)

    for part in parts[1:-1]:
        dirs = [candidate for base in dirs if (candidate := base / part).is_dir()]
        if not dirs:
            return None

    last = parts[-1]
    for base in dirs:
        module_file = base / f"{last}.py"
        if module_file.is_file():
            return module_file
    for base in dirs:
        init_file = base / last / "__init__.py"
        if init_file.is_file():
            return init_file
    return None


def _resolve_first_party_target(target: ImportTarget) -> str:
    """A first-party import resolving to no file on disk must fail loudly, never return None."""
    if _first_party_file(target.candidate) is not None:
        return target.candidate
    if target.module != target.candidate and _first_party_file(target.module) is not None:
        return target.module
    raise AssertionError(f"cannot resolve first-party import '{target.candidate}' to a file on disk")


def _module_source(dotted: str) -> tuple[str, str]:
    path = _first_party_file(dotted)
    assert path is not None, f"first-party module '{dotted}' has no file on disk"
    package = dotted if path.name == "__init__.py" else dotted.rpartition(".")[0]
    return package, path.read_text(encoding="utf-8")


def _parent_packages(dotted: str) -> frozenset[str]:
    """Ancestor `__init__.py` files run before `dotted` itself does; a namespace package contributes nothing."""
    parts = dotted.split(".")
    parents: set[str] = set()
    for end in range(1, len(parts)):
        candidate = ".".join(parts[:end])
        if _first_party_file(candidate) is not None:
            parents.add(candidate)
    return frozenset(parents)


class _DirectEdges(NamedTuple):
    third_party_roots: frozenset[str]
    first_party_modules: frozenset[str]


_DIRECT_EDGES_CACHE: dict[str, _DirectEdges] = {}
_EFFECTIVE_ROOTS_CACHE: dict[str, frozenset[str]] = {}
_REACHABLE_OWN_ROOTS_CACHE: dict[str, frozenset[str]] = {}


def _direct_edges(module: str) -> _DirectEdges:
    """Parsed once and memoized; safe even inside an import cycle since this never recurses into another module."""
    cached = _DIRECT_EDGES_CACHE.get(module)
    if cached is not None:
        return cached

    package, source = _module_source(module)
    tree = ast.parse(source, filename=module)
    targets = unconditional_import_targets(tree, package)

    third_party: set[str] = set()
    first_party: set[str] = set(_parent_packages(module))
    for target in targets:
        root = target.root
        if root == "__future__" or root in _STDLIB:
            continue
        if root in _FIRST_PARTY_ROOT_DIRS:
            first_party.add(_resolve_first_party_target(target))
        else:
            third_party.add(root)

    result = _DirectEdges(frozenset(third_party), frozenset(first_party))
    _DIRECT_EDGES_CACHE[module] = result
    return result


def effective_roots(module: str) -> frozenset[str]:
    """Caches only once the full reachable set is walked, so a cycle-truncated aggregate is never cached."""
    cached = _EFFECTIVE_ROOTS_CACHE.get(module)
    if cached is not None:
        return cached

    visited: set[str] = set()
    pending = [module]
    roots: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        edges = _direct_edges(current)
        roots |= edges.third_party_roots
        pending.extend(edges.first_party_modules - visited)

    result = frozenset(roots)
    _EFFECTIVE_ROOTS_CACHE[module] = result
    return result


def _own_reachable_roots(module: str) -> frozenset[str]:
    cached = _REACHABLE_OWN_ROOTS_CACHE.get(module)
    if cached is not None:
        return cached

    package, source = _module_source(module)
    tree = ast.parse(source, filename=module)
    roots = frozenset(
        target.root
        for target in reachable_import_targets(tree, package)
        if target.root != "__future__" and target.root not in _STDLIB and target.root not in _FIRST_PARTY_ROOT_DIRS
    )
    _REACHABLE_OWN_ROOTS_CACHE[module] = roots
    return roots


# Follows import-time first-party edges only; a sibling imported inside a function is not a dependency edge.
def reachable_roots(module: str) -> frozenset[str]:
    visited: set[str] = set()
    pending = [module]
    roots: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        roots |= _own_reachable_roots(current)
        for candidate in _direct_edges(current).first_party_modules:
            if candidate.startswith(_MAP_FOLLOW_PREFIXES) and candidate not in visited:
                pending.append(candidate)
    return frozenset(roots)


def _dotted_plugin_module(path: Path) -> str:
    relative = path.relative_to(_MLODA_PLUGINS_DIR).with_suffix("")
    return ".".join(("mloda_plugins", *relative.parts))


_PLUGIN_MODULE_NAMES: list[str] = sorted(
    _dotted_plugin_module(path) for path in _MLODA_PLUGINS_DIR.rglob("*.py") if path.name != "__init__.py"
)

_HINT = (
    "Use mloda.core.optional_dependency.require() at the point of use or guard the import, "
    "or list the module in EAGER_MODULES with a MODULE_EXTRA row."
)


def test_every_plugin_module_imports_only_what_its_extra_declares() -> None:
    assert _PLUGIN_MODULE_NAMES, "could not locate any module under mloda_plugins/**"

    violations: list[str] = []
    for module in _PLUGIN_MODULE_NAMES:
        allowed = allowed_roots(MODULE_EXTRA[module]) if module in EAGER_MODULES else frozenset()
        offending = effective_roots(module) - allowed
        if offending:
            violations.append(f"{module}: {sorted(offending)}")

    assert violations == [], (
        "The following plugin modules import a third-party root their extra does not declare:\n"
        + "\n".join(violations)
        + "\n\n"
        + _HINT
    )


def test_module_extra_table_has_no_stale_rows() -> None:
    optional = _load_optional_deps()
    for module, extra in MODULE_EXTRA.items():
        assert module in _PLUGIN_MODULE_NAMES, f"MODULE_EXTRA row '{module}' does not exist on disk"
        if module in EAGER_MODULES:
            assert effective_roots(module), f"MODULE_EXTRA row '{module}' has no unguarded third-party imports"
        assert reachable_roots(module), f"MODULE_EXTRA row '{module}' reaches no third-party root"
        assert extra in optional, f"MODULE_EXTRA row '{module}' names extra '{extra}', absent from pyproject.toml"
        assert extra in BACKEND_EXTRAS, f"MODULE_EXTRA row '{module}' names extra '{extra}', not a backend extra"

    for module, extras in ALSO_NEEDS.items():
        assert module in MODULE_EXTRA, f"ALSO_NEEDS key '{module}' has no MODULE_EXTRA row"
        for extra in extras:
            assert extra in optional, f"ALSO_NEEDS entry '{module}' names extra '{extra}', absent from pyproject.toml"
            assert extra in BACKEND_EXTRAS, f"ALSO_NEEDS entry '{module}' names extra '{extra}', not a backend extra"
            covered_without = allowed_roots(MODULE_EXTRA[module])
            for other in extras:
                if other != extra:
                    covered_without = covered_without | allowed_roots(other)
            assert reachable_roots(module) - covered_without, (
                f"ALSO_NEEDS entry '{module}' extra '{extra}' is redundant: no reachable root needs it"
            )

    orphans = EAGER_MODULES - MODULE_EXTRA.keys()
    assert not orphans, f"EAGER_MODULES names modules without a MODULE_EXTRA row: {sorted(orphans)}"


def test_every_plugin_module_with_a_backend_root_is_mapped(_isolated_effective_roots_cache: None) -> None:
    violations: list[str] = []
    for module in _PLUGIN_MODULE_NAMES:
        roots = reachable_roots(module)
        if not roots:
            continue
        home = MODULE_EXTRA.get(module)
        if home is None:
            violations.append(f"{module}: no MODULE_EXTRA row, reaches {sorted(roots)}")
            continue
        covered = allowed_roots(home)
        for extra in ALSO_NEEDS.get(module, ()):
            covered = covered | allowed_roots(extra)
        uncovered = roots - covered
        if uncovered:
            violations.append(f"{module}: home extra '{home}' and ALSO_NEEDS do not cover {sorted(uncovered)}")

    assert violations == [], (
        "Plugin modules reaching an unmapped third-party root:\n"
        + "\n".join(violations)
        + "\n\nAdd a MODULE_EXTRA row, or an ALSO_NEEDS entry for the extra that declares the root."
    )


def test_backend_neutral_modules_reach_no_third_party_root(_isolated_effective_roots_cache: None) -> None:
    for module in sorted(BACKEND_NEUTRAL):
        assert module in _PLUGIN_MODULE_NAMES, f"BACKEND_NEUTRAL entry '{module}' does not exist on disk"
        assert reachable_roots(module) == frozenset(), f"BACKEND_NEUTRAL entry '{module}' reaches a third-party root"


_UNCONDITIONAL_IMPORT_CASES: list[tuple[str, str, frozenset[str]]] = [
    ("plain_import", "import numpy as np\n", frozenset({"numpy"})),
    ("from_import", "from pyarrow import compute\n", frozenset({"pyarrow"})),
    ("submodule_import", "import duckdb.experimental\n", frozenset({"duckdb"})),
    (
        "guarded_by_try_except",
        "try:\n    import numpy as np\nexcept ImportError:\n    np = None\n",
        frozenset(),
    ),
    (
        "guarded_by_type_checking",
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    import pyarrow as pa\n",
        frozenset({"typing"}),
    ),
    (
        "guarded_by_function",
        "def f():\n    import numpy as np\n    return np\n",
        frozenset(),
    ),
    (
        "dynamic_import_module_body",
        "import importlib\n\npa = importlib.import_module('pyarrow')\n",
        frozenset({"importlib", "pyarrow"}),
    ),
    (
        "with_block_import_reported",
        "with open('f') as fh:\n    import numpy\n",
        frozenset({"numpy"}),
    ),
    (
        "plain_if_import_reported",
        "if condition:\n    import numpy\n",
        frozenset({"numpy"}),
    ),
    (
        "class_body_import_reported",
        "class Foo:\n    import numpy\n",
        frozenset({"numpy"}),
    ),
    (
        "try_except_value_error_reports_import",
        "try:\n    import numpy\nexcept ValueError:\n    numpy = None\n",
        frozenset({"numpy"}),
    ),
    (
        "try_except_tuple_with_import_error_hides_import",
        "try:\n    import numpy\nexcept (ImportError, AttributeError):\n    numpy = None\n",
        frozenset(),
    ),
    (
        "try_except_exception_hides_import",
        "try:\n    import numpy\nexcept Exception:\n    numpy = None\n",
        frozenset(),
    ),
    (
        "except_import_error_handler_fallback_import_reported",
        "try:\n    import numpy\nexcept ImportError:\n    import numpy_fallback as numpy\n",
        frozenset({"numpy_fallback"}),
    ),
    (
        "type_checking_else_branch_import_reported",
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    import pyarrow as pa\nelse:\n    import numpy\n",
        frozenset({"typing", "numpy"}),
    ),
    (
        "require_call_module_level_reported",
        "require('numpy', 'x')\n",
        frozenset({"numpy"}),
    ),
    (
        "require_call_inside_function_not_reported",
        "def f():\n    require('numpy', 'x')\n",
        frozenset(),
    ),
]

_UNCONDITIONAL_IMPORT_IDS = [case[0] for case in _UNCONDITIONAL_IMPORT_CASES]


@pytest.mark.parametrize(
    ("source", "expected_roots"), [case[1:] for case in _UNCONDITIONAL_IMPORT_CASES], ids=_UNCONDITIONAL_IMPORT_IDS
)
def test_unconditional_import_targets_reports_only_direct_module_body_imports(
    source: str, expected_roots: frozenset[str]
) -> None:
    tree = ast.parse(source)
    targets = unconditional_import_targets(tree, package="synthetic")
    assert {target.root for target in targets} == expected_roots


def test_regression_pyarrow_module_reports_numpy_violation() -> None:
    tree = ast.parse("import numpy as np\nimport pyarrow as pa\n")
    targets = unconditional_import_targets(tree, package="synthetic")
    reported_roots = {target.root for target in targets}

    violation = reported_roots - allowed_roots("pyarrow")

    assert violation == {"numpy"}


_REACHABLE_IMPORT_CASES: list[tuple[str, str, frozenset[str]]] = [
    ("function_body_import_counted", "def f():\n    import numpy\n", frozenset({"numpy"})),
    (
        "try_except_import_error_body_counted",
        "try:\n    import numpy\nexcept ImportError:\n    numpy = None\n",
        frozenset({"numpy"}),
    ),
    (
        "type_checking_body_excluded",
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    import pyarrow as pa\n",
        frozenset({"typing"}),
    ),
    (
        "type_checking_else_branch_counted",
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    import pyarrow as pa\nelse:\n    import numpy\n",
        frozenset({"typing", "numpy"}),
    ),
    (
        "require_inside_method_counted",
        "class Foo:\n    def method(self):\n        require('numpy', 'x')\n",
        frozenset({"numpy"}),
    ),
    (
        "return_require_counted",
        "def f():\n    return require('numpy', 'x')\n",
        frozenset({"numpy"}),
    ),
    (
        "require_inside_lambda_counted",
        "getter = lambda: require('numpy', 'x')\n",
        frozenset({"numpy"}),
    ),
    (
        "loaded_call_not_counted",
        "def f():\n    return loaded('numpy')\n",
        frozenset(),
    ),
    (
        "future_import_skipped",
        "from __future__ import annotations\n",
        frozenset(),
    ),
    (
        "dynamic_import_inside_function_counted",
        "import importlib\n\ndef f():\n    return importlib.import_module('pyarrow')\n",
        frozenset({"importlib", "pyarrow"}),
    ),
]

_REACHABLE_IMPORT_IDS = [case[0] for case in _REACHABLE_IMPORT_CASES]


@pytest.mark.parametrize(
    ("source", "expected_roots"), [case[1:] for case in _REACHABLE_IMPORT_CASES], ids=_REACHABLE_IMPORT_IDS
)
def test_reachable_import_targets_reports_guarded_and_lazy_imports_but_not_type_checking_bodies(
    source: str, expected_roots: frozenset[str]
) -> None:
    tree = ast.parse(source)
    targets = reachable_import_targets(tree, package="synthetic")
    assert {target.root for target in targets} == expected_roots


@pytest.fixture
def _isolated_effective_roots_cache() -> Iterator[None]:
    _DIRECT_EDGES_CACHE.clear()
    _EFFECTIVE_ROOTS_CACHE.clear()
    _REACHABLE_OWN_ROOTS_CACHE.clear()
    yield
    _DIRECT_EDGES_CACHE.clear()
    _EFFECTIVE_ROOTS_CACHE.clear()
    _REACHABLE_OWN_ROOTS_CACHE.clear()


_CYCLE_MODULES = ("cycletest_root.a", "cycletest_root.b", "cycletest_root.c")


@pytest.mark.parametrize("first_asked", _CYCLE_MODULES)
def test_effective_roots_survives_a_first_party_import_cycle_regardless_of_entry_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_asked: str,
    _isolated_effective_roots_cache: None,
) -> None:
    """A imports B imports C imports A, and A also imports numpy: every member must report numpy regardless of order."""
    (tmp_path / "a.py").write_text("import numpy\nimport cycletest_root.b\n")
    (tmp_path / "b.py").write_text("import cycletest_root.c\n")
    (tmp_path / "c.py").write_text("import cycletest_root.a\n")
    monkeypatch.setitem(_FIRST_PARTY_ROOT_DIRS, "cycletest_root", [tmp_path])

    assert effective_roots(first_asked) == frozenset({"numpy"})
    for module in _CYCLE_MODULES:
        assert effective_roots(module) == frozenset({"numpy"}), module


def test_effective_roots_includes_a_parent_packages_own_unguarded_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolated_effective_roots_cache: None,
) -> None:
    package_dir = tmp_path / "parentpkgtest"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("import numpy\n")
    (package_dir / "child.py").write_text("")
    monkeypatch.setitem(_FIRST_PARTY_ROOT_DIRS, "ptest", [tmp_path])

    assert effective_roots("ptest.parentpkgtest.child") == frozenset({"numpy"})


class RequireSite(NamedTuple):
    path: Path
    lineno: int
    literal: str

    @property
    def root(self) -> str:
        return self.literal.partition(".")[0]


# Only a first positional string literal is scanned; keyword and f-string forms are not.
def require_sites(root: Path) -> list[RequireSite]:
    sites: list[RequireSite] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found = [
            RequireSite(path, node.lineno, literal)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and (literal := _require_call_target(node)) is not None
        ]
        sites.extend(sorted(found, key=lambda site: site.lineno))
    return sites


_UNDECLARED_MESSAGE = (
    "{path}:{lineno}: require('{literal}') hints 'pip install mloda[{root}]' but pyproject.toml has no '{root}' extra"
)


def undeclared_require_messages(sites: list[RequireSite], extras: Collection[str]) -> list[str]:
    return [
        _UNDECLARED_MESSAGE.format(path=site.path, lineno=site.lineno, literal=site.literal, root=site.root)
        for site in sites
        if site.root not in extras
    ]


_SYNTHETIC_EXTRAS = frozenset({"pyarrow", "numpy"})

_REQUIRE_SITE_CASES: list[tuple[str, str, list[tuple[int, str]]]] = [
    ("undeclared_at_module_level", "import os\n\nrequire('somelib', 'x')\n", [(3, "somelib")]),
    ("declared_root_reports_nothing", "require('numpy', 'x')\n", []),
    ("declared_submodule_reduces_to_root", "require('pyarrow.flight', 'x')\n", []),
    ("undeclared_submodule_reduces_to_root", "require('somelib.sub', 'x')\n", [(1, "somelib.sub")]),
    ("undeclared_inside_function_body", "def f():\n    require('somelib', 'x')\n", [(2, "somelib")]),
]


@pytest.mark.parametrize(
    ("source", "expected"), [case[1:] for case in _REQUIRE_SITE_CASES], ids=[case[0] for case in _REQUIRE_SITE_CASES]
)
def test_undeclared_require_messages_reports_only_roots_missing_from_extras(
    tmp_path: Path, source: str, expected: list[tuple[int, str]]
) -> None:
    module = tmp_path / "mod.py"
    module.write_text(source)

    messages = undeclared_require_messages(require_sites(tmp_path), _SYNTHETIC_EXTRAS)

    assert len(messages) == len(expected)
    for message, (lineno, literal) in zip(messages, expected):
        root = literal.partition(".")[0]
        assert message.startswith(f"{module}:{lineno}: ")
        assert f"mloda[{root}]" in message
        assert f"no '{root}' extra" in message


_REQUIRE_EXTRA_HINT = (
    "Add the extra under [project.optional-dependencies], or teach require() a distribution-to-extra mapping."
)


def test_every_require_call_names_an_existing_extra() -> None:
    extras = _load_optional_deps()
    messages: list[str] = []
    for package in ("mloda", "mloda_plugins"):
        directories = list(dict.fromkeys(directory.resolve() for directory in _FIRST_PARTY_ROOT_DIRS[package]))
        assert directories, f"no {package} package directory"
        sites = [site for directory in directories for site in require_sites(directory)]
        assert sites, f"no require() call under {package}/"
        messages.extend(undeclared_require_messages(sites, extras))

    assert messages == [], (
        "require() names an extra pyproject.toml does not declare:\n"
        + "\n".join(messages)
        + "\n\n"
        + _REQUIRE_EXTRA_HINT
    )


def test_reachable_roots_follows_a_module_level_import_and_counts_the_callee_guarded_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolated_effective_roots_cache: None,
) -> None:
    (tmp_path / "a.py").write_text("import reachtest_root.b\n")
    (tmp_path / "b.py").write_text("def load():\n    import numpy\n    return numpy\n")
    monkeypatch.setitem(_FIRST_PARTY_ROOT_DIRS, "reachtest_root", [tmp_path])
    monkeypatch.setattr(sys.modules[__name__], "_MAP_FOLLOW_PREFIXES", ("reachtest_root.",))

    assert reachable_roots("reachtest_root.a") == frozenset({"numpy"})


def test_reachable_roots_does_not_follow_a_module_imported_only_inside_a_function(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolated_effective_roots_cache: None,
) -> None:
    (tmp_path / "a.py").write_text("def load():\n    import reachtest_root.b\n")
    (tmp_path / "b.py").write_text("import numpy\n")
    monkeypatch.setitem(_FIRST_PARTY_ROOT_DIRS, "reachtest_root", [tmp_path])
    monkeypatch.setattr(sys.modules[__name__], "_MAP_FOLLOW_PREFIXES", ("reachtest_root.",))

    assert reachable_roots("reachtest_root.a") == frozenset()


def test_reachable_roots_does_not_follow_a_module_outside_the_followed_prefixes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _isolated_effective_roots_cache: None,
) -> None:
    followed_dir = tmp_path / "followed"
    other_dir = tmp_path / "other"
    followed_dir.mkdir()
    other_dir.mkdir()
    (followed_dir / "a.py").write_text("import reachtest_root.c\nimport reachother_root.b\n")
    (followed_dir / "c.py").write_text("import pyarrow\n")
    (other_dir / "b.py").write_text("import pandas\n")
    monkeypatch.setitem(_FIRST_PARTY_ROOT_DIRS, "reachtest_root", [followed_dir])
    monkeypatch.setitem(_FIRST_PARTY_ROOT_DIRS, "reachother_root", [other_dir])
    monkeypatch.setattr(sys.modules[__name__], "_MAP_FOLLOW_PREFIXES", ("reachtest_root.",))

    assert reachable_roots("reachtest_root.a") == frozenset({"pyarrow"})
