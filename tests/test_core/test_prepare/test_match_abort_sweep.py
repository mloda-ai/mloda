"""Static sweep: every raise a match hook can reach must state whether it escalates or stays contained.

``IdentifyFeatureGroupClass._filter_feature_group_by_criteria`` contains any raise out of
``match_feature_group_criteria`` as a non-match for that candidate, unless the exception was marked with
``escalate_match_abort``. The marked set is hand-maintained, so a NEW unmarked raise on the match path is
silently downgraded to "this candidate does not match" and a rival plugin wins the feature. This sweep walks
the static call graph from the match seams through the declared match-path modules and requires every raise
it reaches to carry a decision: either the escalation call, or a ``# Contained: <reason>`` comment anchored to
that individual raise.

What it deliberately does NOT cover:

* Plugin code under ``mloda_plugins``. Their match hooks are user-shaped code; containment is the point.
* Dynamic dispatch. The graph resolves calls by NAME inside the declared modules only, so a hook reached
  purely through a runtime-built callable is invisible here. ``check_required_when`` is seeded for exactly
  that reason: it runs inside the matcher wrapper ``install_required_when_guard`` installs, and no static
  call edge reaches it from the seams.
* Whether a contained raise is the RIGHT call. The comment records the author's decision; this sweep only
  proves a decision was made and written down next to the raise.

Anchoring policy: the reason must sit on the raise line or in the contiguous comment block directly above the
raise. A reason merely present somewhere in the enclosing function is not accepted, otherwise a second raise
added later would inherit the first one's reason and the gate would rot.
"""

from __future__ import annotations

import ast
import io
import tokenize
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Literal, NamedTuple

# Anchor the scan to the repo layout via __file__, not the cwd: a cwd-relative root makes every lookup below
# empty and the sweep pass vacuously. This file sits three parents below the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CORE_ROOT = _REPO_ROOT / "mloda"
assert _CORE_ROOT.is_dir(), f"core root not found; check the parents index for the repo root: {_CORE_ROOT}"

# Modules the match path runs through, repo-relative. The graph never leaves this set, so a module missing
# here is a blind spot; test_every_declared_module_is_reachable keeps stale entries out.
MATCH_PATH_MODULES: dict[str, str] = {
    "mloda/core/prepare/identify_feature_group.py": "the containment seam itself",
    "mloda/core/abstract_plugins/feature_group.py": "the default matcher every group inherits",
    "mloda/core/abstract_plugins/components/feature_chainer/feature_chain_parser_mixin.py": (
        "the chain-parser matcher, the most common override"
    ),
    "mloda/core/abstract_plugins/components/feature_chainer/feature_chain_parser.py": (
        "name parsing and property resolution run inside that matcher"
    ),
    "mloda/core/abstract_plugins/components/feature_chainer/feature_chain_author_guards.py": (
        "the required_when guard wrapped around a group's matcher"
    ),
    "mloda/core/abstract_plugins/components/feature_chainer/property_spec.py": (
        "PropertySpec resolution feeds the matcher's option reads"
    ),
    "mloda/core/abstract_plugins/components/input_data/base_input_data.py": (
        "reader matching and file pinning, reached from the input-data matchers"
    ),
    "mloda/core/abstract_plugins/components/input_data/api/api_input_data.py": "the api reader's match hook",
    "mloda/core/abstract_plugins/components/input_data/creator/data_creator.py": "the data-creator match hook",
    "mloda/core/abstract_plugins/components/match_data/match_data.py": "reader selection during matching",
}

for _module in MATCH_PATH_MODULES:
    assert (_REPO_ROOT / _module).exists(), f"declared match-path module not found: {_module}"

# Entry points of the walk. The first two are the seam; the last two are reached from code the graph cannot
# see, so they are seeded by hand.
SEEDS: dict[str, str] = {
    "match_feature_group_criteria": "the match hook every candidate is asked",
    "_filter_feature_group_by_criteria": "the seam that contains a raising hook",
    "_resolve_pinned_file": "framework helper invoked from reader match hooks that live in mloda_plugins",
    "check_required_when": "runs inside the matcher wrapper install_required_when_guard installs",
}

# Raising functions OUTSIDE the declared modules that the match path calls. Each needs a decision too, but it
# is recorded here instead of at the raise, because these functions serve callers far away from matching.
RAISING_HELPERS_OUTSIDE_THE_PATH: dict[tuple[str, str], str] = {
    ("mloda/core/abstract_plugins/components/options.py", "get_in_features"): (
        "contained: a malformed in_features declaration is that candidate's own defect"
    ),
    ("mloda/core/abstract_plugins/plugin_loader/plugin_loader.py", "load_group"): (
        "contained: reader auto-load runs during matching, and a broken plugin group must not abort a run "
        "another candidate can serve"
    ),
    ("mloda/core/abstract_plugins/plugin_loader/plugin_loader.py", "all"): (
        "contained: reader auto-load runs during matching, and a broken plugin group must not abort a run "
        "another candidate can serve"
    ),
}

_ESCALATION = "escalate_match_abort"
_CONTAINED_TAG = "Contained:"
_MARKED_TAG = "Marked:"

# Attribute calls whose name is a public method of a builtin container are never a framework function, so
# `"".join(...)` and friends must not pull an unrelated same-named function into the graph.
_BUILTIN_CONTAINER_METHODS: frozenset[str] = frozenset(
    name
    for container in (str, list, dict, set, tuple, frozenset, int, bytes)
    for name in dir(container)
    if not name.startswith("_")
)

Kind = Literal["marked", "contained", "mismarked", "unannotated"]


class RaiseSite(NamedTuple):
    """One ``raise <exc>`` statement and the escalation decision written at it."""

    module: str
    lineno: int
    function: str
    kind: Kind
    reason: str | None

    def location(self) -> str:
        return f"{self.module}:{self.lineno} {self.function}()"


class ExternalCall(NamedTuple):
    """A call from the match path into a raising function defined outside the declared modules."""

    module: str
    function: str
    caller: str


class _Definition(NamedTuple):
    module: str
    node: ast.FunctionDef | ast.AsyncFunctionDef


class _Sweep(NamedTuple):
    sites: tuple[RaiseSite, ...]
    reachable: frozenset[tuple[str, str]]
    external: tuple[ExternalCall, ...]


def _called_names(node: ast.AST) -> set[str]:
    """Every name called inside ``node``: ``f(...)`` by id, ``x.f(...)`` by attribute."""
    names: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute) and func.attr not in _BUILTIN_CONTAINER_METHODS:
            names.add(func.attr)
    return names


def _escalates(node: ast.AST) -> bool:
    """True when ``escalate_match_abort`` is called anywhere inside ``node``."""
    return _ESCALATION in _called_names(node)


def _raises_an_exception(node: ast.AST) -> bool:
    """True when ``node`` contains a ``raise <exc>``; a bare re-raise does not count."""
    return any(isinstance(sub, ast.Raise) and sub.exc is not None for sub in ast.walk(node))


def _own_line_and_trailing_comments(source: str) -> tuple[dict[int, str], dict[int, str]]:
    """Comment text per line, split into own-line comments and comments trailing code.

    tokenize, not a raw ``#`` scan: a ``#`` inside a string literal is not a comment token.
    """
    lines = source.splitlines()
    own_line: dict[int, str] = {}
    trailing: dict[int, str] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        row, col = token.start
        text = token.string.lstrip("#").strip()
        if lines[row - 1][:col].strip():
            trailing[row] = text
        else:
            own_line[row] = text
    return own_line, trailing


def _tagged_reason(text: str, tag: str) -> str | None:
    """The reason behind ``tag`` in one comment, or None when the tag is absent or the reason is empty."""
    if not text.startswith(tag):
        return None
    reason = text[len(tag) :].strip()
    return reason or None


def _anchored_reason(
    own_line: dict[int, str],
    trailing: dict[int, str],
    lineno: int,
    tag: str,
) -> str | None:
    """The tagged reason for the raise at ``lineno``: trailing on that line, or in the block directly above.

    The block ends at the first line that is not an own-line comment, so a blank or code line separates a
    raise from a reason written for something else.
    """
    reason = _tagged_reason(trailing.get(lineno, ""), tag)
    if reason is not None:
        return reason
    row = lineno - 1
    while row in own_line:
        reason = _tagged_reason(own_line[row], tag)
        if reason is not None:
            return reason
        row -= 1
    return None


def _collect_raises(
    node: ast.AST,
    functions: list[str],
    handler: ast.ExceptHandler | None,
    out: list[tuple[ast.Raise, list[str], ast.ExceptHandler | None]],
) -> None:
    """Walk ``node``, recording each raise with its enclosing function chain and except handler."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # A nested def leaves the handler's scope: its raises are not that handler's re-raise.
            _collect_raises(child, [*functions, child.name], None, out)
        elif isinstance(child, ast.ExceptHandler):
            _collect_raises(child, functions, child, out)
        else:
            if isinstance(child, ast.Raise):
                out.append((child, functions, handler))
            _collect_raises(child, functions, handler, out)


def classify_raises(source: str, module: str, functions: frozenset[str] | None = None) -> list[RaiseSite]:
    """Classify every ``raise <exc>`` in ``source``, restricted to the named functions when given.

    A pure function over text so the classifier can be exercised on snippets, independent of the tree.
    ``functions`` filters by name: a raise counts when any function enclosing it is named in the filter.
    """
    tree = ast.parse(source, filename=module)
    own_line, trailing = _own_line_and_trailing_comments(source)
    found: list[tuple[ast.Raise, list[str], ast.ExceptHandler | None]] = []
    _collect_raises(tree, [], None, found)

    sites: list[RaiseSite] = []
    for raise_node, chain, handler in found:
        if not chain:
            continue
        if functions is not None and not any(name in functions for name in chain):
            continue
        if raise_node.exc is None:
            # A bare re-raise carries no decision of its own: it sits under the escalate_match_abort line.
            continue
        marked = _escalates(raise_node.exc) or (handler is not None and any(_escalates(s) for s in handler.body))
        claimed = _anchored_reason(own_line, trailing, raise_node.lineno, _MARKED_TAG)
        contained = _anchored_reason(own_line, trailing, raise_node.lineno, _CONTAINED_TAG)
        kind: Kind
        if marked:
            kind, reason = "marked", claimed
        elif claimed is not None:
            kind, reason = "mismarked", claimed
        elif contained is not None:
            kind, reason = "contained", contained
        else:
            kind, reason = "unannotated", None
        sites.append(RaiseSite(module, raise_node.lineno, chain[-1], kind, reason))
    return sorted(sites)


def _index_functions(root: Path) -> dict[str, list[_Definition]]:
    """Every function defined under ``root``, indexed by name; one name can hold several definitions."""
    index: dict[str, list[_Definition]] = {}
    for path in sorted(root.rglob("*.py")):
        module = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=module)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                index.setdefault(node.name, []).append(_Definition(module, node))
    return index


@lru_cache(maxsize=1)
def sweep() -> _Sweep:
    """Walk the match path from the seeds and classify every raise it reaches.

    Indexing ``mloda/`` (not ``mloda_plugins/``) is what lets the walk notice calls that leave the declared
    modules; the walk itself never follows them.
    """
    core_index = _index_functions(_CORE_ROOT)
    declared_index: dict[str, list[_Definition]] = {}
    for name, definitions in core_index.items():
        inside = [d for d in definitions if d.module in MATCH_PATH_MODULES]
        if inside:
            declared_index[name] = inside

    queue: deque[_Definition] = deque()
    # A definition is identified by (module, line): two same-named methods in one module stay distinct.
    visited: set[tuple[str, int]] = set()
    for seed in SEEDS:
        for definition in declared_index.get(seed, []):
            if (definition.module, definition.node.lineno) not in visited:
                visited.add((definition.module, definition.node.lineno))
                queue.append(definition)

    reachable: set[tuple[str, str]] = set()
    external: set[ExternalCall] = set()
    while queue:
        definition = queue.popleft()
        reachable.add((definition.module, definition.node.name))
        caller = f"{definition.module}::{definition.node.name}"
        for called in _called_names(definition.node):
            for target in declared_index.get(called, []):
                key = (target.module, target.node.lineno)
                if key not in visited:
                    visited.add(key)
                    queue.append(target)
            for target in core_index.get(called, []):
                if target.module not in MATCH_PATH_MODULES and _raises_an_exception(target.node):
                    external.add(ExternalCall(target.module, called, caller))

    sites: list[RaiseSite] = []
    for module in sorted({module for module, _ in reachable}):
        names = frozenset(name for reached_module, name in reachable if reached_module == module)
        source = (_REPO_ROOT / module).read_text(encoding="utf-8")
        sites.extend(classify_raises(source, module, names))

    return _Sweep(tuple(sorted(sites)), frozenset(reachable), tuple(sorted(external)))


def _of_kind(kind: Kind) -> list[RaiseSite]:
    return [site for site in sweep().sites if site.kind == kind]


def test_every_reachable_raise_is_marked_or_contained() -> None:
    """No raise the match path can reach is left without an escalation decision."""
    unannotated = _of_kind("unannotated")

    assert unannotated == [], (
        "Raise sites reachable from the match seams with no escalation decision:\n"
        + "\n".join(f"  {site.location()}" for site in unannotated)
        + "\n\nFor each one, decide and record it at the raise: wrap the exception in "
        f"{_ESCALATION}(...) when the run must fail instead of silently losing the candidate, or write "
        f"'# {_CONTAINED_TAG} <reason>' directly above the raise when the seam may read it as a non-match."
    )


def test_no_marked_comment_without_an_escalation() -> None:
    """No raise claims to be marked while nothing at the site actually escalates."""
    mismarked = _of_kind("mismarked")

    assert mismarked == [], (
        f"Raise sites carrying a '# {_MARKED_TAG}' comment that no {_ESCALATION} call backs:\n"
        + "\n".join(f"  {site.location()}: {site.reason}" for site in mismarked)
    )


def test_calls_into_raising_helpers_outside_the_path_are_declared() -> None:
    """Every raising helper the match path calls outside the declared modules is declared with a reason."""
    undeclared = [
        call for call in sweep().external if (call.module, call.function) not in RAISING_HELPERS_OUTSIDE_THE_PATH
    ]

    assert undeclared == [], (
        "The match path calls raising functions outside the declared modules:\n"
        + "\n".join(f"  {call.module}::{call.function} from {call.caller}" for call in undeclared)
        + "\n\nEither declare the (module, function) pair in RAISING_HELPERS_OUTSIDE_THE_PATH with the reason "
        "its raises may stay contained, or add the module to MATCH_PATH_MODULES so its raises are swept."
    )


def test_declared_helpers_outside_the_path_are_still_called() -> None:
    """Every declared helper is still reached, so stale entries cannot accumulate."""
    called = {(call.module, call.function) for call in sweep().external}
    stale = sorted(entry for entry in RAISING_HELPERS_OUTSIDE_THE_PATH if entry not in called)

    assert stale == [], f"RAISING_HELPERS_OUTSIDE_THE_PATH entries the match path no longer calls: {stale}"


def test_every_declared_module_is_reachable() -> None:
    """Each declared module contributes at least one reachable function."""
    reached = {module for module, _ in sweep().reachable}
    unreached = sorted(set(MATCH_PATH_MODULES) - reached)

    assert unreached == [], (
        f"MATCH_PATH_MODULES entries no seed reaches: {unreached}. Drop the entry, or add the seed that "
        "reaches it if the call edge is dynamic."
    )


def test_known_escalations_are_enumerated() -> None:
    """Canary: the escalations that exist today are all seen, so a silently empty sweep cannot pass."""
    base_input_data = "mloda/core/abstract_plugins/components/input_data/base_input_data.py"
    match_data = "mloda/core/abstract_plugins/components/match_data/match_data.py"
    mixin = "mloda/core/abstract_plugins/components/feature_chainer/feature_chain_parser_mixin.py"
    expected = {
        (base_input_data, "add_base_input_data_to_options"),
        (base_input_data, "_resolve_pinned_file"),
        (match_data, "add_base_input_data_to_options"),
        (mixin, "_validate_forwarded_name_mismatch"),
    }

    marked = {(site.module, site.function) for site in _of_kind("marked")}

    assert expected <= marked, f"known escalations missing from the sweep: {sorted(expected - marked)}"
    # Vacuity floor, not a target: 18 sites today, so removing one stays a legitimate change.
    assert len(sweep().sites) >= 15, f"sweep enumerated only {len(sweep().sites)} raise sites; it is not walking"


def test_classifier_flags_an_unannotated_raise() -> None:
    """A raise with no decision at it is reported, whatever the enclosing function says elsewhere."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        '    """# Contained: a docstring mention is not a decision at the raise."""\n'
        "    if feature is None:\n"
        "        raise ValueError('no feature')\n"
        "    return True\n"
    )

    sites = classify_raises(source, "snippet.py", frozenset({"match_feature_group_criteria"}))

    assert [site.kind for site in sites] == ["unannotated"]
    assert sites[0].function == "match_feature_group_criteria"


def test_classifier_accepts_a_contained_comment_and_an_escalation() -> None:
    """Both ways of recording a decision are accepted, and neither is reported as unannotated."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    if feature is None:\n"
        "        # Contained: a missing feature is this candidate's own defect.\n"
        "        raise ValueError('no feature')\n"
        "    if options is None:\n"
        "        raise escalate_match_abort(ValueError('no options'))\n"
        "    return True\n"
    )

    sites = classify_raises(source, "snippet.py", frozenset({"match_feature_group_criteria"}))

    assert [site.kind for site in sites] == ["contained", "marked"]
    assert sites[0].reason == "a missing feature is this candidate's own defect."


def test_classifier_flags_a_marked_comment_without_an_escalation() -> None:
    """A comment that claims escalation without one is a lie the sweep must catch, not accept as annotated."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    # Marked: claims to escalate, nothing here does.\n"
        "    raise ValueError('no feature')\n"
    )

    sites = classify_raises(source, "snippet.py", frozenset({"match_feature_group_criteria"}))

    assert [site.kind for site in sites] == ["mismarked"]
