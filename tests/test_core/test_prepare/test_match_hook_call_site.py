"""Issue #991: one call site for the match hook, so a fix can no longer land on one seam only.

Both seams called ``match_feature_group_criteria`` behind their own ``try`` and drifted apart twice, so this
sweep parses every module under ``mloda/`` and requires the only hook call to sit in the shared helper, with
nothing at either seam that catches, suppresses or otherwise drops what the helper re-raised. A reference to
the hook counts as a site too, since a bound method is called a line later. Blind spots: a matcher reached by
string (``getattr``), and ``mloda_plugins``.
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

# The context var whose per-candidate window the resolution seam opens, and resets in its finally.
CONTEXT_VAR = "MATCH_REJECTION_REASONS"

SUPPRESS = "suppress"


class CallSite(NamedTuple):
    """One place the match hook is reached, with the function that reaches it."""

    module: str
    lineno: int
    function: str

    def location(self) -> str:
        return f"{self.module}:{self.lineno} {self.function}()"


def _is_super_attribute(node: ast.Attribute) -> bool:
    """``super().<hook>`` reaches a base implementation from an override; it opens no seam of its own."""
    value = node.value
    return isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "super"


def _is_super_delegation(node: ast.Call) -> bool:
    """``super().<hook>(...)``: the call form of the same delegation."""
    func = node.func
    return isinstance(func, ast.Attribute) and _is_super_attribute(func)


def _is_hook_call(node: ast.Call) -> bool:
    """A call of the hook by name: ``<anything>.<hook>(...)`` or a bare ``<hook>(...)``.

    A ``def <hook>`` is a definition node, never a call, so definitions are out by construction.
    """
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr == HOOK
    return isinstance(func, ast.Name) and func.id == HOOK


def _is_hook_reference(node: ast.Attribute) -> bool:
    """``<anything>.<hook>`` read as a value: bound here, called a line later, invisible to a call-only scan."""
    return node.attr == HOOK and isinstance(node.ctx, ast.Load) and not _is_super_attribute(node)


def _collect_calls(node: ast.AST, module: str, functions: list[str], out: list[CallSite]) -> None:
    """Walk ``node``, recording each hook call and each hook reference with the function that encloses it."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _collect_calls(child, module, [*functions, child.name], out)
            continue
        if isinstance(child, ast.Call) and _is_hook_call(child):
            if not _is_super_delegation(child):
                out.append(CallSite(module, child.lineno, functions[-1] if functions else "<module>"))
            # Walk the func node's own children only: the node itself names the hook this call already reported.
            _collect_calls(child.func, module, functions, out)
            for argument in [*child.args, *(keyword.value for keyword in child.keywords)]:
                _collect_calls(argument, module, functions, out)
            continue
        if isinstance(child, ast.Attribute) and _is_hook_reference(child):
            out.append(CallSite(module, child.lineno, functions[-1] if functions else "<module>"))
        _collect_calls(child, module, functions, out)


def classify_calls(source: str, module: str) -> list[CallSite]:
    """Every call or reference of the match hook in ``source``, definitions and ``super()`` delegations excluded."""
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


def _definition_in(source: str, module: str, function: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """The single definition of ``function`` in ``source``; a missing or duplicated one fails here."""
    tree = ast.parse(source, filename=module)
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function
    ]
    assert len(found) == 1, f"{module} holds {len(found)} definitions of {function}; this pin names exactly one"
    return found[0]


def _definition(module: str, function: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """The single definition of ``function`` in ``module``, read from disk."""
    return _definition_in((_REPO_ROOT / module).read_text(encoding="utf-8"), module, function)


def _is_suppress(expr: ast.expr) -> bool:
    """``suppress(...)`` or ``contextlib.suppress(...)``. Restated here, not imported: no test may import a test."""
    if not isinstance(expr, ast.Call):
        return False
    func = expr.func
    if isinstance(func, ast.Name):
        return func.id == SUPPRESS
    return isinstance(func, ast.Attribute) and func.attr == SUPPRESS


def _finally_escapes(finalbody: list[ast.stmt]) -> bool:
    """A return, break or continue leaving ``finally`` discards whatever exception is in flight."""
    escapes = (ast.Return, ast.Break, ast.Continue)
    return any(isinstance(node, escapes) for statement in finalbody for node in ast.walk(statement))


def silent_drops(node: ast.AST) -> list[int]:
    """Line of every place inside ``node`` that drops an exception with no ``except`` clause to show for it."""
    drops: list[int] = []
    for child in ast.walk(node):
        if isinstance(child, (ast.With, ast.AsyncWith)) and any(
            _is_suppress(item.context_expr) for item in child.items
        ):
            drops.append(child.lineno)
        elif isinstance(child, ast.Try) and _finally_escapes(child.finalbody):
            # ast exposes no position for the finally keyword, so the try line carries the decision.
            drops.append(child.lineno)
    return sorted(drops)


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


def test_the_sweep_does_not_collapse() -> None:
    """The helper canary is satisfied by a single scanned module, so pin the size of the walk too."""
    scanned = sweep().modules

    # Vacuity floor, not a target: 140 modules today, so deleting one stays a legitimate change.
    assert len(scanned) >= 120, (
        f"the sweep parsed only {len(scanned)} modules under mloda/; it stopped covering the tree, and both "
        "call-site pins would pass while reading almost nothing"
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


@pytest.mark.parametrize(("module", "function"), SEAMS)
def test_neither_seam_drops_an_exception_without_an_except(module: str, function: str) -> None:
    """``except`` is not the only way to swallow: a suppress block and a returning finally do it with none."""
    drops = silent_drops(_definition(module, function))

    assert drops == [], (
        f"{module}::{function}() drops an exception at line(s) {drops} with no except clause to show for it. A "
        f"suppress block or a finally that returns swallows the marked abort {HELPER_FUNCTION}() deliberately "
        "re-raised, and the handler pin beside this one sees none of it."
    )


def test_the_resolution_seam_keeps_only_the_context_var_reset_in_its_finally() -> None:
    """``outcome`` is bound only where nothing raised, so every read of it belongs inside the same try."""
    module, function = RESOLUTION_SEAM
    definition = _definition(module, function)
    blocks = [node for node in ast.walk(definition) if isinstance(node, ast.Try) and node.finalbody]

    assert len(blocks) == 1, f"{module}::{function}() holds {len(blocks)} try/finally blocks; this pin names one"
    statements = [ast.unparse(statement) for statement in blocks[0].finalbody]
    assert len(statements) == 1 and statements[0].startswith(f"{CONTEXT_VAR}.reset("), (
        f"the finally of {function}() must hold the window reset alone, got: {statements}. Harvesting there "
        "keeps the harvest on a path the try body never reached."
    )
    assert isinstance(definition.body[-1], ast.Try), (
        f"every path out of {function}() must leave through that try. A read of the outcome after it runs on a "
        "path no handler binds, so the first except clause added to the try turns it into an UnboundLocalError."
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


def test_the_scanner_ignores_a_bound_super_delegation() -> None:
    """Binding the base implementation first is the same delegation, one line apart."""
    source = (
        "class ProbeGroup(Base):\n"
        "    @classmethod\n"
        "    def match_feature_group_criteria(cls, feature_name, options, data_access_collection=None):\n"
        "        inherited = super().match_feature_group_criteria\n"
        "        return inherited(feature_name, options, data_access_collection)\n"
    )

    assert classify_calls(source, "snippet.py") == []


def test_the_scanner_reports_a_hook_bound_for_a_later_call() -> None:
    """A seam that binds the hook and calls the name is a seam; a call-only scan reads it as nothing at all."""
    source = (
        "def criteria(self, feature_group, filter, data_access_collection=None):\n"
        "    hook = feature_group.match_feature_group_criteria\n"
        "    return hook(filter.name, filter.options, None)\n"
    )

    sites = classify_calls(source, "snippet.py")

    assert [(site.lineno, site.function) for site in sites] == [(2, "criteria")]


def test_the_scanner_counts_a_call_once_and_a_reference_beside_it() -> None:
    """The attribute inside a call the scanner already reported must not come back as a second site."""
    source = (
        "def criteria(self, feature_group, filter, data_access_collection=None):\n"
        "    if filter is None:\n"
        "        return feature_group.match_feature_group_criteria(filter.name, filter.options, None)\n"
        "    hook = feature_group.match_feature_group_criteria\n"
        "    return hook(filter.name, filter.options, None)\n"
    )

    sites = classify_calls(source, "snippet.py")

    assert [(site.lineno, site.function) for site in sites] == [(3, "criteria"), (4, "criteria")]


def _splice(source: str, function: str, block: list[str]) -> tuple[str, int]:
    """Insert ``block`` at the top of ``function``'s body; in memory only, never written back to disk."""
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
    return "\n".join([*lines[:at], *(f"{pad}{line}" for line in block), *lines[at:]]) + "\n", first.lineno


def _splice_hook_call(source: str, function: str) -> tuple[str, int]:
    """Insert a hook call at the top of ``function``'s body."""
    call = f"feature_group.{HOOK}(filter.filter_feature.name, filter.filter_feature.options, None)"
    return _splice(source, function, [call])


def test_the_sweep_flags_a_hook_call_spliced_into_the_real_seam() -> None:
    """End to end on real source, mutated in memory only: a call that comes back at a seam is reported."""
    module, function = FILTER_SEAM
    source = (_REPO_ROOT / module).read_text(encoding="utf-8")

    mutated, lineno = _splice_hook_call(source, function)
    spliced = [site for site in classify_calls(mutated, module) if site.lineno == lineno]

    assert [site.function for site in spliced] == [function], (
        f"a {HOOK} call spliced into {function}() itself was not flagged: {spliced}"
    )


def test_the_sweep_flags_a_hook_reference_spliced_into_the_real_seam() -> None:
    """The same end to end for the bound form, which no call node names."""
    module, function = FILTER_SEAM
    source = (_REPO_ROOT / module).read_text(encoding="utf-8")

    mutated, lineno = _splice(source, function, [f"hook = feature_group.{HOOK}"])
    spliced = [site for site in classify_calls(mutated, module) if site.lineno == lineno]

    assert [site.function for site in spliced] == [function], (
        f"a {HOOK} reference spliced into {function}() itself was not flagged: {spliced}"
    )


def test_the_drop_scanner_flags_a_suppress_block() -> None:
    """``suppress`` discards the exception exactly as a bare except does, and shows no except clause."""
    source = (
        "def criteria(self, feature_group, filter, data_access_collection=None):\n"
        "    with contextlib.suppress(Exception):\n"
        "        return call_match_hook(feature_group, filter.name, filter.options, None).matched\n"
        "    return False\n"
    )

    assert silent_drops(_definition_in(source, "snippet.py", "criteria")) == [2]


def test_the_drop_scanner_flags_a_bare_suppress_import() -> None:
    """``from contextlib import suppress`` spells the same drop without the module name."""
    source = (
        "def criteria(self, feature_group, filter, data_access_collection=None):\n"
        "    with suppress(Exception):\n"
        "        return call_match_hook(feature_group, filter.name, filter.options, None).matched\n"
        "    return False\n"
    )

    assert silent_drops(_definition_in(source, "snippet.py", "criteria")) == [2]


def test_the_drop_scanner_flags_a_finally_that_returns() -> None:
    """A return in finally discards the in-flight exception, marker and all."""
    source = (
        "def criteria(self, feature_group, filter, data_access_collection=None):\n"
        "    try:\n"
        "        return call_match_hook(feature_group, filter.name, filter.options, None).matched\n"
        "    finally:\n"
        "        return False\n"
    )

    # Anchored at the try line: ast.Try carries no lineno for the finally keyword.
    assert silent_drops(_definition_in(source, "snippet.py", "criteria")) == [2]


def test_the_drop_scanner_passes_a_finally_that_only_cleans_up() -> None:
    """The control: the resolution seam's own finally resets a context var and drops nothing."""
    source = (
        "def criteria(self, feature_group, filter, data_access_collection=None):\n"
        "    token = MATCH_REJECTION_REASONS.set({})\n"
        "    try:\n"
        "        return call_match_hook(feature_group, filter.name, filter.options, None).matched\n"
        "    finally:\n"
        "        MATCH_REJECTION_REASONS.reset(token)\n"
    )

    assert silent_drops(_definition_in(source, "snippet.py", "criteria")) == []


@pytest.mark.parametrize(("module", "function"), SEAMS)
def test_the_drop_pin_flags_a_suppress_spliced_into_a_real_seam(module: str, function: str) -> None:
    """End to end on real source, mutated in memory only: a suppress at either seam is reported."""
    source = (_REPO_ROOT / module).read_text(encoding="utf-8")

    mutated, lineno = _splice(source, function, ["with contextlib.suppress(Exception):", "    pass"])

    assert silent_drops(_definition_in(mutated, module, function)) == [lineno], (
        f"a suppress block spliced into {function}() itself was not flagged"
    )
