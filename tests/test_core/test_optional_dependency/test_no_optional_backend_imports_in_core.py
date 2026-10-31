"""Rot guard: no module under mloda/ may import an optional backend library at runtime.

Optional backends are reached through mloda.core.optional_dependency (require/loaded), which is the single
place where the sys.modules gating and the "install the extra" message are implemented. A hand-rolled
`try: import pandas` in core silently duplicates that logic, so it is a violation.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pytest

import mloda

OPTIONAL_BACKEND_ROOTS = frozenset({"pyarrow", "pandas", "polars", "duckdb", "pyspark", "pyiceberg"})

# mloda is a namespace package, so __file__ is None: its directories come from __path__.
CORE_ROOTS = [Path(entry) for entry in mloda.__path__ if Path(entry).is_dir()]

# Callees that reach a module by name at runtime. importlib.import_module is matched on the attribute too.
DYNAMIC_IMPORT_NAMES = frozenset({"import_module", "__import__"})

_HINT = (
    "Import optional backends through mloda.core.optional_dependency instead: "
    "require(module, reason) at the point of use, loaded(module) on hot paths."
)


def _is_type_checking_test(test: ast.expr) -> bool:
    """Matches `if TYPE_CHECKING:` and `if typing.TYPE_CHECKING:`."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


def _type_checking_imports(nodes: Iterable[ast.AST]) -> set[int]:
    """Node ids of imports inside the body of an `if TYPE_CHECKING:` block. Annotations only, never executed."""
    guarded: set[int] = set()
    for node in nodes:
        if not isinstance(node, ast.If) or not _is_type_checking_test(node.test):
            continue
        for statement in node.body:
            for child in ast.walk(statement):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    guarded.add(id(child))
    return guarded


def _imported_roots(node: ast.Import | ast.ImportFrom) -> set[str]:
    """Root packages an import statement pulls in. Relative imports never leave mloda."""
    if isinstance(node, ast.Import):
        return {alias.name.partition(".")[0] for alias in node.names}
    if node.level or not node.module:
        return set()
    return {node.module.partition(".")[0]}


def _dynamic_import_aliases(nodes: Iterable[ast.AST]) -> set[str]:
    """Bare names that reach a module by name: the builtins plus any `from importlib import import_module as X`."""
    aliases = set(DYNAMIC_IMPORT_NAMES)
    for node in nodes:
        if not isinstance(node, ast.ImportFrom) or node.module != "importlib":
            continue
        for alias in node.names:
            if alias.name == "import_module":
                aliases.add(alias.asname or alias.name)
    return aliases


def _is_dynamic_import(node: ast.Call, aliases: set[str]) -> bool:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr == "import_module"
    if isinstance(node.func, ast.Name):
        return node.func.id in aliases
    return False


def _dynamic_import_root(node: ast.Call, aliases: set[str]) -> str | None:
    """Root package of a dynamic import of a string literal. A computed name cannot be judged statically."""
    if not _is_dynamic_import(node, aliases) or not node.args:
        return None
    first = node.args[0]
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        return None
    return first.value.partition(".")[0]


def _violations(root: Path) -> list[str]:
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        nodes = list(ast.walk(tree))
        guarded = _type_checking_imports(nodes)
        aliases = _dynamic_import_aliases(nodes)
        for node in nodes:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if id(node) in guarded:
                    continue
                for backend in sorted(_imported_roots(node) & OPTIONAL_BACKEND_ROOTS):
                    found.append(f"{path}:{node.lineno}: imports {backend}")
            elif isinstance(node, ast.Call):
                dynamic = _dynamic_import_root(node, aliases)
                if dynamic in OPTIONAL_BACKEND_ROOTS:
                    found.append(f"{path}:{node.lineno}: imports {dynamic}")
    return found


def _backends(violations: list[str]) -> list[str]:
    """The backend names a violation list reports, sorted."""
    return sorted(violation.rsplit("imports ", 1)[1] for violation in violations)


# Whole-tree AST scan: 0.9s serial, up to 6.9s under -n 3 on a loaded host, and has exceeded 10s on a loaded CI runner.
@pytest.mark.timeout(30)
def test_core_does_not_import_optional_backends() -> None:
    """Every module under mloda/ must be importable without any optional backend installed."""
    assert CORE_ROOTS, "could not locate the mloda package directory"

    violations: list[str] = []
    for root in CORE_ROOTS:
        violations.extend(_violations(root))

    assert violations == [], "mloda core imports optional backends directly:\n" + "\n".join(violations) + "\n" + _HINT


def test_guard_flags_a_hand_rolled_import(tmp_path: Path) -> None:
    """The guard itself must catch a re-introduced backend import, in any of its import forms."""
    (tmp_path / "hand_rolled.py").write_text(
        "def f():\n"
        "    try:\n"
        "        import pandas\n"
        "    except ImportError:\n"
        "        raise ImportError('nope')\n"
        "    return pandas\n"
    )
    (tmp_path / "from_form.py").write_text("from pyarrow import flight\n")
    (tmp_path / "submodule_form.py").write_text("import duckdb.experimental\n")

    assert _backends(_violations(tmp_path)) == ["duckdb", "pandas", "pyarrow"]


def test_guard_flags_a_hand_rolled_importlib_call(tmp_path: Path) -> None:
    """The rot this guard exists to prevent: importlib is now the sanctioned way to reach a backend.

    A hand-rolled importlib.import_module + try/except in core duplicates require() exactly, and an
    ast.Import-only walk never sees it.
    """
    (tmp_path / "dynamic.py").write_text(
        "import importlib\n"
        "\n"
        "def f():\n"
        "    try:\n"
        "        return importlib.import_module('pandas')\n"
        "    except ImportError:\n"
        "        raise ImportError('pandas is required')\n"
    )

    assert _backends(_violations(tmp_path)) == ["pandas"]


def test_guard_flags_a_dunder_import_call(tmp_path: Path) -> None:
    (tmp_path / "dunder.py").write_text("def f():\n    return __import__('polars')\n")

    assert _backends(_violations(tmp_path)) == ["polars"]


def test_guard_flags_an_aliased_import_module_call(tmp_path: Path) -> None:
    """Renaming the callee does not change what it reaches."""
    (tmp_path / "aliased.py").write_text(
        "from importlib import import_module as _im\n\n\ndef f():\n    return _im('pyarrow')\n"
    )

    assert _backends(_violations(tmp_path)) == ["pyarrow"]


def test_guard_flags_a_dynamic_submodule_literal(tmp_path: Path) -> None:
    """The extra is named by the root package, so a dynamic 'pyarrow.flight' counts as pyarrow."""
    (tmp_path / "sub.py").write_text("from importlib import import_module\n\nx = import_module('pyarrow.flight')\n")

    assert _backends(_violations(tmp_path)) == ["pyarrow"]


def test_guard_ignores_a_dynamic_import_of_a_non_literal(tmp_path: Path) -> None:
    """A computed module name cannot be judged statically, so it is not flagged and must not crash the guard.

    This is the shape of mloda/core/optional_dependency.py itself, which is why that file needs no exemption.
    """
    (tmp_path / "variable.py").write_text(
        "import importlib\n"
        "\n"
        "\n"
        "def require(module: str):\n"
        "    return importlib.import_module(module)\n"
        "\n"
        "\n"
        "def no_args():\n"
        "    return importlib.import_module()\n"
    )

    assert _violations(tmp_path) == []


def test_guard_ignores_a_dynamic_import_of_a_non_backend(tmp_path: Path) -> None:
    (tmp_path / "stdlib.py").write_text(
        "import importlib\n\nx = importlib.import_module('json')\ny = __import__('os')\n"
    )

    assert _violations(tmp_path) == []


def test_guard_allows_type_checking_imports(tmp_path: Path) -> None:
    """Annotation-only imports do not execute, so they are not violations."""
    (tmp_path / "annotations_only.py").write_text(
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    import pyarrow as pa\n"
    )

    assert _violations(tmp_path) == []


def test_guard_parses_each_file_once_and_walks_each_tree_at_most_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "annotations_only.py").write_text(
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    import pyarrow as pa\n"
    )
    (tmp_path / "aliased.py").write_text(
        "from importlib import import_module as _im\n\n\ndef f():\n    return _im('pandas')\n"
    )
    real_parse = ast.parse
    real_walk = ast.walk
    parses: list[str] = []
    tree_walks: list[ast.AST] = []

    def counting_parse(source: str, *args: Any, **kwargs: Any) -> ast.Module:
        parses.append(str(kwargs.get("filename", "")))
        tree: ast.Module = real_parse(source, *args, **kwargs)
        return tree

    def counting_walk(node: ast.AST) -> Iterator[ast.AST]:
        if isinstance(node, ast.Module):
            tree_walks.append(node)
        return real_walk(node)

    with monkeypatch.context() as patch:
        patch.setattr(ast, "parse", counting_parse)
        patch.setattr(ast, "walk", counting_walk)
        violations = _violations(tmp_path)

    assert len(parses) == 2, f"expected one parse per file, got {len(parses)}"
    assert len(tree_walks) <= 4, f"expected at most 2 whole-tree walks per file (4), got {len(tree_walks)}"
    assert _backends(violations) == ["pandas"]


def test_guard_allows_non_backend_imports(tmp_path: Path) -> None:
    (tmp_path / "plain.py").write_text("import json\nfrom pathlib import Path\nfrom . import sibling\n")

    assert _violations(tmp_path) == []
