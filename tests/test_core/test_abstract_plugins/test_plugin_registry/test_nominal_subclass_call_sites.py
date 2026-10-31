"""Bare ``issubclass`` calls under ``mloda/`` and ``mloda_plugins/`` are limited to a justified allowlist.

``issubclass`` against an mloda ABC walks ABCMeta's descendants and grows its negative cache on every failed
check against a base with many subclasses; ``is_nominal_subclass`` skips that walk. Blind spots: ``isinstance``
against ABC classes, ``builtins.issubclass`` or aliased calls, and code outside these two packages.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import pytest

pytestmark = pytest.mark.timeout(30)

# Anchored via __file__, not the cwd: a cwd-relative root makes every lookup empty and the sweep vacuous.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ROOTS = (_REPO_ROOT / "mloda", _REPO_ROOT / "mloda_plugins")
for _root in _ROOTS:
    assert _root.is_dir(), f"root not found; check the parents index for the repo root: {_root}"

# Site (repo-relative path, enclosing function qualname) -> why the right-hand side is not an mloda class.
ALLOWED: dict[tuple[str, str], str] = {
    ("mloda/core/runtime/flight/flight_server.py", "FlightServer.__new__"): "pyarrow FlightServerBase",
    ("mloda/core/abstract_plugins/components/code_closure.py", "_Walker._handle_def"): "the builtin type",
    ("mloda/core/abstract_plugins/components/code_closure.py", "_Walker._handle_value"): "the builtin type",
    (
        "mloda/core/abstract_plugins/compute_framework.py",
        "ComputeFramework._takes_as_is",
    ): "external data type from expected_data_framework()",
}


def _collect(node: ast.AST, module: str, scope: list[str], out: set[tuple[str, str]]) -> None:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _collect(child, module, [*scope, child.name], out)
            continue
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "issubclass":
            out.add((module, ".".join(scope) or "<module>"))
        _collect(child, module, scope, out)


@lru_cache(maxsize=1)
def _sweep() -> tuple[int, frozenset[tuple[str, str]]]:
    found: set[tuple[str, str]] = set()
    count = 0
    for root in _ROOTS:
        for path in sorted(root.rglob("*.py")):
            module = path.relative_to(_REPO_ROOT).as_posix()
            count += 1
            _collect(ast.parse(path.read_text(encoding="utf-8"), filename=module), module, [], found)
    return count, frozenset(found)


def test_the_sweep_does_not_collapse() -> None:
    assert _sweep()[0] >= 150


def test_bare_issubclass_calls_match_the_allowlist() -> None:
    found = _sweep()[1]
    unexpected = sorted(found - ALLOWED.keys())
    stale = sorted(ALLOWED.keys() - found)

    message = ""
    if unexpected:
        message += (
            "Unexpected bare issubclass sites:\n"
            + "\n".join(f"  {m} {q}" for m, q in unexpected)
            + "\nUse is_nominal_subclass for mloda classes, or add a justified allowlist entry.\n"
        )
    if stale:
        message += "Stale allowlist entries:\n" + "\n".join(f"  {m} {q}" for m, q in stale) + "\n"
    assert not message, message
