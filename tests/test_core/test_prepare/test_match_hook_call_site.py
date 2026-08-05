"""Issue #991: one call site for the match hook, so a fix can no longer land on one seam only.

Both seams called ``match_feature_group_criteria`` behind their own ``try`` and drifted apart twice, so this
sweep parses every module under ``mloda/`` and requires the only hook call to sit in the shared helper, with
no exception handler left at either seam. Blind spots: a ``getattr``-reached matcher, and ``mloda_plugins``.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import pytest

# Anchored via __file__, not the cwd: a cwd-relative root makes every lookup empty and the sweep vacuous.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CORE_ROOT = _REPO_ROOT / "mloda"
assert _CORE_ROOT.is_dir(), f"core root not found; check the parents index for the repo root: {_CORE_ROOT}"

HOOK = "match_feature_group_criteria"

# The one home of the call, the marked-abort re-raise and the bool() coercion.
HELPER_MODULE = "mloda/core/abstract_plugins/components/match_hook.py"
HELPER_FUNCTION = "call_match_hook"

# The two seams that keep only their own recording and rollback around that call.
FILTER_SEAM = ("mloda/core/filter/global_filter.py", "criteria")
RESOLUTION_SEAM = ("mloda/core/prepare/identify_feature_group.py", "_filter_feature_group_by_criteria")
SEAMS = (FILTER_SEAM, RESOLUTION_SEAM)


class CallSite(NamedTuple):
    """One call of the match hook, with the function that makes it."""

    module: str
    lineno: int
    function: str

    def location(self) -> str:
        return f"{self.module}:{self.lineno} {self.function}()"


def _is_super_delegation(node: ast.Call) -> bool:
    """``super().<hook>(...)`` reaches a base implementation from an override; it is no caller site."""
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    value = func.value
    return isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "super"


def _is_hook_call(node: ast.Call) -> bool:
    """A call of the hook by name: ``<anything>.<hook>(...)`` or a bare ``<hook>(...)``.

    A ``def <hook>`` is a definition node, never a call, so definitions are out by construction.
    """
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr == HOOK
    return isinstance(func, ast.Name) and func.id == HOOK


def _collect_calls(node: ast.AST, module: str, functions: list[str], out: list[CallSite]) -> None:
    """Walk ``node``, recording each hook call with the function that encloses it."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _collect_calls(child, module, [*functions, child.name], out)
            continue
        if isinstance(child, ast.Call) and _is_hook_call(child) and not _is_super_delegation(child):
            out.append(CallSite(module, child.lineno, functions[-1] if functions else "<module>"))
        _collect_calls(child, module, functions, out)


def classify_calls(source: str, module: str) -> list[CallSite]:
    """Every call of the match hook in ``source``, definitions and ``super()`` delegations excluded."""
    out: list[CallSite] = []
    _collect_calls(ast.parse(source, filename=module), module, [], out)
    return sorted(out)


class _Sweep(NamedTuple):
    """The modules under ``mloda/`` the sweep parsed, and every hook call found in them."""

    modules: tuple[str, ...]
    sites: tuple[CallSite, ...]


@lru_cache(maxsize=1)
def sweep() -> _Sweep:
    """Parse every module under ``mloda/`` and collect the hook calls."""
    modules: list[str] = []
    sites: list[CallSite] = []
    for path in sorted(_CORE_ROOT.rglob("*.py")):
        module = path.relative_to(_REPO_ROOT).as_posix()
        modules.append(module)
        sites.extend(classify_calls(path.read_text(encoding="utf-8"), module))
    return _Sweep(tuple(modules), tuple(sorted(sites)))


def _definition(module: str, function: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """The single definition of ``function`` in ``module``; a missing or duplicated one fails here."""
    tree = ast.parse((_REPO_ROOT / module).read_text(encoding="utf-8"), filename=module)
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function
    ]
    assert len(found) == 1, f"{module} holds {len(found)} definitions of {function}; this pin names exactly one"
    return found[0]


def test_the_helper_module_is_scanned_and_holds_the_call() -> None:
    """Non-vacuity: a helper that moved or was never written must fail here, not make the sweep find nothing."""
    scanned = sweep().modules

    assert HELPER_MODULE in scanned, (
        f"{HELPER_MODULE} is not among the {len(scanned)} modules swept under mloda/. Create the helper that "
        f"owns the hook call, or update HELPER_MODULE to where it now lives."
    )
    functions = [site.function for site in sweep().sites if site.module == HELPER_MODULE]
    assert functions == [HELPER_FUNCTION], (
        f"{HELPER_MODULE} must call {HOOK} exactly once, inside {HELPER_FUNCTION}(); found {functions}"
    )


def test_the_helper_is_the_only_match_hook_call_site() -> None:
    strays = [site for site in sweep().sites if site.module != HELPER_MODULE]

    assert strays == [], (
        f"These call {HOOK} outside the shared helper:\n"
        + "\n".join(f"  {site.location()}" for site in strays)
        + f"\n\nRoute the call through {HELPER_FUNCTION}() in {HELPER_MODULE}, which owns the call, the "
        "marked-abort re-raise and the bool() coercion, and keep only the recording and the rollback at the "
        "seam. A second call site means every fix has to land twice, and drift when it does not."
    )


@pytest.mark.parametrize(("module", "function"), SEAMS)
def test_neither_seam_holds_an_exception_handler_of_its_own(module: str, function: str) -> None:
    """A seam that catches again re-implements the containment policy the helper owns."""
    handlers = [node.lineno for node in ast.walk(_definition(module, function)) if isinstance(node, ast.ExceptHandler)]

    assert handlers == [], (
        f"{module}::{function}() catches at line(s) {handlers}. The call, the marked-abort re-raise and the "
        f"bool() coercion belong to {HELPER_FUNCTION}(); read its outcome instead and keep only this seam's "
        "own recording and rollback here."
    )


def test_the_scanner_reports_a_plain_hook_call() -> None:
    source = (
        "def criteria(self, feature_group, filter, data_access_collection=None):\n"
        "    return feature_group.match_feature_group_criteria(filter.name, filter.options, None)\n"
    )

    sites = classify_calls(source, "snippet.py")

    assert [(site.lineno, site.function) for site in sites] == [(2, "criteria")]


def test_the_scanner_ignores_the_hook_definition() -> None:
    """A ``def`` is where the hook lives, never a place that calls it."""
    source = (
        "class ProbeGroup:\n"
        "    @classmethod\n"
        "    def match_feature_group_criteria(cls, feature_name, options, data_access_collection=None):\n"
        "        return True\n"
    )

    assert classify_calls(source, "snippet.py") == []


def test_the_scanner_ignores_a_super_delegation() -> None:
    """An override reaching its base implementation keeps the default matcher; it opens no second seam."""
    source = (
        "class ProbeGroup(Base):\n"
        "    @classmethod\n"
        "    def match_feature_group_criteria(cls, feature_name, options, data_access_collection=None):\n"
        "        return super().match_feature_group_criteria(feature_name, options, data_access_collection)\n"
    )

    assert classify_calls(source, "snippet.py") == []


def _splice_hook_call(source: str, function: str) -> tuple[str, int]:
    """Insert a hook call at the top of ``function``'s body; in memory only, never written back to disk."""
    targets = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function
    ]
    assert len(targets) == 1, f"{function} is not a single definition in this module: {len(targets)} found"
    first = targets[0].body[0]
    pad = " " * first.col_offset
    lines = source.splitlines()
    at = first.lineno - 1
    spliced = f"{pad}feature_group.{HOOK}(filter.filter_feature.name, filter.filter_feature.options, None)"
    return "\n".join([*lines[:at], spliced, *lines[at:]]) + "\n", first.lineno


def test_the_sweep_flags_a_hook_call_spliced_into_the_real_seam() -> None:
    """End to end on real source, mutated in memory only: a call that comes back at a seam is reported."""
    module, function = FILTER_SEAM
    source = (_REPO_ROOT / module).read_text(encoding="utf-8")

    mutated, lineno = _splice_hook_call(source, function)
    spliced = [site for site in classify_calls(mutated, module) if site.lineno == lineno]

    assert [site.function for site in spliced] == [function], (
        f"a {HOOK} call spliced into {function}() itself was not flagged: {spliced}"
    )
