"""Static AST policy check: a module under mloda_plugins/** may only import, at module level, a
third-party root that its own pyproject extra (directly or through IMPLIED) declares. An unguarded
import of an undeclared library makes the plugin loader silently skip the module when it is absent."""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterator
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

# A row exists only when a module unconditionally reaches a third party; the extra is the future per-backend package.
MODULE_EXTRA: dict[str, str] = {
    "mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow": "pyarrow",
    "mloda_plugins.feature_group.experimental.data_quality.missing_value.pyarrow": "pyarrow",
    "mloda_plugins.feature_group.experimental.time_window.pyarrow": "pyarrow",
    "mloda_plugins.feature_group.experimental.geo_distance.pandas": "pandas",
    "mloda_plugins.feature_group.experimental.time_window.pandas": "pandas",
    "mloda_plugins.feature_group.input_data.read_dbs.sqlite": "sqlite",
}

# Import roots a library hard-depends on, still covered by the extra that ships the library itself.
IMPLIED: dict[str, frozenset[str]] = {"pandas": frozenset({"numpy"})}

# Distribution name -> top-level import it provides, for names that don't match "lowercased, '-' to '_'".
DISTRIBUTION_IMPORT_ROOT: dict[str, str] = {"pyyaml": "yaml", "scikit-learn": "sklearn"}

_REQUIREMENT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

_STDLIB: frozenset[str] = sys.stdlib_module_names

_MLODA_PLUGINS_DIR: Path = _package_directory("mloda_plugins")

_FIRST_PARTY_ROOT_DIRS: dict[str, list[Path]] = {
    "mloda": [Path(entry) for entry in mloda.__path__ if Path(entry).is_dir()],
    "mloda_plugins": [_MLODA_PLUGINS_DIR],
}


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


def unconditional_import_targets(tree: ast.Module, package: str) -> frozenset[ImportTarget]:
    aliases = _dynamic_import_aliases(tree)
    targets: set[ImportTarget] = set()

    for node in _unconditional_statements(tree.body):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.add(ImportTarget(alias.name, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__" and node.level == 0:
                continue
            module = _resolve_relative(node.module, node.level, package)
            for alias in node.names:
                targets.add(ImportTarget(f"{module}.{alias.name}", module))
        elif isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None:
                continue
            for call in _calls_excluding_lambdas(value):
                root = _dynamic_import_root(call, aliases)
                if root is not None:
                    targets.add(ImportTarget(root, root))
                    continue
                required = _require_call_target(call)
                if required is not None:
                    targets.add(ImportTarget(required, required))

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


def _dotted_plugin_module(path: Path) -> str:
    relative = path.relative_to(_MLODA_PLUGINS_DIR).with_suffix("")
    return ".".join(("mloda_plugins", *relative.parts))


_PLUGIN_MODULE_NAMES: list[str] = sorted(
    _dotted_plugin_module(path) for path in _MLODA_PLUGINS_DIR.rglob("*.py") if path.name != "__init__.py"
)

_HINT = (
    "Fix by reaching the library at the point of use with mloda.core.optional_dependency.require() "
    "(or guard the import), or add a MODULE_EXTRA row whose extra declares it."
)


def test_every_plugin_module_imports_only_what_its_extra_declares() -> None:
    assert _PLUGIN_MODULE_NAMES, "could not locate any module under mloda_plugins/**"

    violations: list[str] = []
    for module in _PLUGIN_MODULE_NAMES:
        allowed = allowed_roots(MODULE_EXTRA[module]) if module in MODULE_EXTRA else frozenset()
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
        assert effective_roots(module), f"MODULE_EXTRA row '{module}' has no unguarded third-party imports"
        assert extra in optional, f"MODULE_EXTRA row '{module}' names extra '{extra}', absent from pyproject.toml"


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


@pytest.fixture
def _isolated_effective_roots_cache() -> Iterator[None]:
    _DIRECT_EDGES_CACHE.clear()
    _EFFECTIVE_ROOTS_CACHE.clear()
    yield
    _DIRECT_EDGES_CACHE.clear()
    _EFFECTIVE_ROOTS_CACHE.clear()


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
