"""Bare ``issubclass`` calls under ``mloda/`` and ``mloda_plugins/`` are limited to a justified allowlist.

``issubclass`` against an mloda ABC walks ABCMeta's descendants and grows its negative cache on every failed
check against a base with many subclasses; ``is_nominal_subclass`` skips that walk. Blind spots: ``isinstance``
against ABC classes, ``builtins.issubclass`` or aliased calls, and code outside these two packages.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

# Anchored via __file__, not the cwd: a cwd-relative root makes every lookup empty and the sweep vacuous.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ROOTS = (_REPO_ROOT / "mloda", _REPO_ROOT / "mloda_plugins")
assert all(r.is_dir() for r in _ROOTS), f"roots not found; check the parents index for the repo root: {_ROOTS}"

Site = tuple[str, str, str]

# Site (repo-relative path, enclosing qualname, unparsed second argument) -> why it is not an mloda class.
ALLOWED: dict[Site, str] = {
    ("mloda/core/runtime/flight/flight_server.py", "FlightServer.__new__", "_flight().FlightServerBase"): (
        "pyarrow FlightServerBase"
    ),
    ("mloda/core/abstract_plugins/components/code_closure.py", "_Walker._handle_def", "type"): "the builtin type",
    ("mloda/core/abstract_plugins/components/code_closure.py", "_Walker._handle_value", "type"): "the builtin type",
    ("mloda/core/abstract_plugins/compute_framework.py", "ComputeFramework._takes_as_is", "expected"): (
        "external data type from expected_data_framework()"
    ),
}


def _collect(node: ast.AST, module: str, scope: list[str], out: set[Site]) -> None:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _collect(child, module, [*scope, child.name], out)
            continue
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "issubclass":
            rhs = ast.unparse(child.args[1]) if len(child.args) > 1 else "<missing>"
            out.add((module, ".".join(scope) or "<module>", rhs))
        _collect(child, module, scope, out)


@lru_cache(maxsize=1)
def _sweep() -> tuple[int, frozenset[Site]]:
    found: set[Site] = set()
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
            + "\n".join(f"  {m} {q} issubclass(..., {r})" for m, q, r in unexpected)
            + "\nUse is_nominal_subclass for mloda classes, or add a justified allowlist entry.\n"
        )
    if stale:
        message += (
            "Stale allowlist entries:\n" + "\n".join(f"  {m} {q} issubclass(..., {r})" for m, q, r in stale) + "\n"
        )
    assert not message, message
