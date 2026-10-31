"""Bare ``issubclass`` and plugin-base ``isinstance`` calls in ``mloda/`` and ``mloda_plugins/`` must be allowlisted.

``issubclass`` against an mloda ABC walks ABCMeta's descendants and grows its negative cache on every failed
check against a base with many subclasses; ``is_nominal_subclass`` skips that walk. A failed ``isinstance`` walks
once per instance type until the next ABC invalidation, so it is swept only against the plugin root bases, whose trees
are large. Blind spots: ``builtins.``-qualified or aliased calls, intermediate bases such as ``SourceFeatureGroup``,
and code outside these two packages.
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
_PLUGIN_BASE_NAMES = {"FeatureGroup", "ComputeFramework", "Extender", "Implementation"}

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

# isinstance sites against a plugin base, same key shape and rationale as ALLOWED.
_FROM_CFW = "from_cfw is a ComputeFramework, UUID or None, and the ComputeFramework tree is small"
ALLOWED_ISINSTANCE: dict[Site, str] = {
    ("mloda/core/core/step/feature_group_step.py", "FeatureGroupStep._source_data", "ComputeFramework"): _FROM_CFW,
    ("mloda/core/core/step/join_step.py", "JoinStep.execute", "ComputeFramework"): _FROM_CFW,
    (
        "mloda/core/core/step/transform_frame_work_step.py",
        "TransformFrameworkStep.execute",
        "ComputeFramework",
    ): _FROM_CFW,
}


def _references_plugin_base(node: ast.AST) -> bool:
    return any(
        (isinstance(n, ast.Name) and n.id in _PLUGIN_BASE_NAMES)
        or (isinstance(n, ast.Attribute) and n.attr in _PLUGIN_BASE_NAMES)
        for n in ast.walk(node)
    )


def _collect(node: ast.AST, module: str, scope: list[str], out: dict[str, set[Site]]) -> None:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _collect(child, module, [*scope, child.name], out)
            continue
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id in out:
            rhs = ast.unparse(child.args[1]) if len(child.args) > 1 else "<missing>"
            if child.func.id == "issubclass" or (len(child.args) > 1 and _references_plugin_base(child.args[1])):
                out[child.func.id].add((module, ".".join(scope) or "<module>", rhs))
        _collect(child, module, scope, out)


@lru_cache(maxsize=1)
def _sweep() -> tuple[int, dict[str, frozenset[Site]]]:
    found: dict[str, set[Site]] = {"issubclass": set(), "isinstance": set()}
    count = 0
    for root in _ROOTS:
        for path in sorted(root.rglob("*.py")):
            module = path.relative_to(_REPO_ROOT).as_posix()
            count += 1
            _collect(ast.parse(path.read_text(encoding="utf-8"), filename=module), module, [], found)
    return count, {name: frozenset(sites) for name, sites in found.items()}


def test_the_sweep_does_not_collapse() -> None:
    assert _sweep()[0] >= 150


def test_bare_issubclass_calls_match_the_allowlist() -> None:
    found = _sweep()[1]["issubclass"]
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


def test_isinstance_calls_against_plugin_bases_match_the_allowlist() -> None:
    found = _sweep()[1]["isinstance"]
    unexpected = sorted(found - ALLOWED_ISINSTANCE.keys())
    stale = sorted(ALLOWED_ISINSTANCE.keys() - found)

    message = ""
    if unexpected:
        message += (
            "Unexpected isinstance sites against plugin bases:\n"
            + "\n".join(f"  {m} {q} isinstance(..., {r})" for m, q, r in unexpected)
            + "\nUse is_nominal_subclass(type(x), Base), or add a justified allowlist entry.\n"
        )
    if stale:
        message += (
            "Stale isinstance allowlist entries:\n"
            + "\n".join(f"  {m} {q} isinstance(..., {r})" for m, q, r in stale)
            + "\n"
        )
    assert not message, message
