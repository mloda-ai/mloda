"""Static sweep: every raise a match hook can reach must state whether it escalates or stays contained.

``IdentifyFeatureGroupClass._filter_feature_group_by_criteria`` contains a raise out of
``match_feature_group_criteria`` as a non-match unless ``escalate_match_abort`` marked it, so a NEW unmarked
raise on the match path silently loses the feature to a rival plugin. This sweep walks the static call graph
from the match seams through the declared match-path modules and requires a decision at each individual
raise: the escalation call, or a ``# Contained: <reason>`` comment on the raise line or in the comment block
directly above it. A reason elsewhere in the enclosing function is not accepted; a later raise would inherit
it and the gate would rot.

A sibling sweep walks the same reachable set for ``except`` handlers: a mark only survives if every handler
between the raise and the seam re-raises it, so each handler must re-raise on ``is_match_abort`` or carry a
``# Swallows: <reason>`` comment.

Not covered: plugin code under ``mloda_plugins``; dynamic dispatch, which is why SEEDS is hand-written;
decorators.
"""

from __future__ import annotations

import ast
import io
import tokenize
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Literal, NamedTuple

# Anchored via __file__, not the cwd: a cwd-relative root makes every lookup empty and the sweep vacuous.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CORE_ROOT = _REPO_ROOT / "mloda"
assert _CORE_ROOT.is_dir(), f"core root not found; check the parents index for the repo root: {_CORE_ROOT}"

# Modules the match path runs through, repo-relative. The graph never leaves this set, so a module missing
# here is a blind spot; test_every_declared_module_is_reachable keeps stale entries out.
MATCH_PATH_MODULES: dict[str, str] = {
    "mloda/core/prepare/identify_feature_group.py": "the containment seam itself",
    "mloda/core/abstract_plugins/feature_group.py": "the default matcher every group inherits",
    "mloda/core/abstract_plugins/components/feature_chainer/feature_chain_parser_mixin.py": "the chain-parser matcher",
    "mloda/core/abstract_plugins/components/feature_chainer/feature_chain_parser.py": "the parsing that matcher runs",
    "mloda/core/abstract_plugins/components/feature_chainer/feature_chain_author_guards.py": "the required_when guard",
    "mloda/core/abstract_plugins/components/feature_chainer/property_spec.py": "the matcher reads the spec sentinel",
    "mloda/core/abstract_plugins/components/input_data/base_input_data.py": "reader matching and file pinning",
    "mloda/core/abstract_plugins/components/input_data/api/api_input_data.py": "the api reader's match hook",
    "mloda/core/abstract_plugins/components/input_data/creator/data_creator.py": "the data-creator match hook",
    "mloda/core/abstract_plugins/components/match_data/match_data.py": "reader selection during matching",
}

for _module in MATCH_PATH_MODULES:
    assert (_REPO_ROOT / _module).exists(), f"declared match-path module not found: {_module}"

# Entry points. The first two are the seam; the rest are reached from code the graph cannot see.
SEEDS: dict[str, str] = {
    "match_feature_group_criteria": "the match hook every candidate is asked",
    "_filter_feature_group_by_criteria": "the seam that contains a raising hook",
    "_resolve_pinned_file": "called from reader match hooks in mloda_plugins",
    "check_required_when": "runs inside the wrapper install_required_when_guard installs",
    "guarded": "the installed closure IS a guarded class's matcher; the setattr is no call edge",
}

# Raising functions OUTSIDE the declared modules that the match path calls; the decision is recorded here
# instead of at the raise. Resolution is by name, so an entry can be a name COLLISION, not a call edge.
_CANDIDATE_OWN_DECLARATION = "contained: validates the requesting feature's own declaration during matching"
_READER_AUTO_LOAD = "contained: reader auto-load during matching; a broken plugin group must not abort the run"
_DECIDED_ABOVE_BY_READER_SELECTION = (
    "decided above by the marked raise in both add_base_input_data_to_options callers; this write only "
    "ever reaches an absent key"
)

RAISING_HELPERS_OUTSIDE_THE_PATH: dict[tuple[str, str], str] = {
    ("mloda/core/abstract_plugins/components/options.py", "__init__"): _CANDIDATE_OWN_DECLARATION,
    ("mloda/core/abstract_plugins/components/options.py", "add_to_group"): _DECIDED_ABOVE_BY_READER_SELECTION,
    ("mloda/core/abstract_plugins/components/options.py", "get_in_features"): _CANDIDATE_OWN_DECLARATION,
    ("mloda/core/abstract_plugins/components/feature.py", "__init__"): _CANDIDATE_OWN_DECLARATION,
    ("mloda/core/abstract_plugins/plugin_loader/plugin_loader.py", "__init__"): _READER_AUTO_LOAD,
    ("mloda/core/abstract_plugins/plugin_loader/plugin_loader.py", "load_group"): _READER_AUTO_LOAD,
    ("mloda/core/abstract_plugins/plugin_loader/plugin_loader.py", "all"): _READER_AUTO_LOAD,
    ("mloda/core/abstract_plugins/components/link.py", "matches"): (
        "name collision: the match path calls the input-data hooks named matches, not Link.matches"
    ),
    ("mloda/core/abstract_plugins/components/utils.py", "get_all_subclasses"): (
        "real edge, collided verdict: it raises nothing itself; the set.add / set.update names do"
    ),
    ("mloda/core/prepare/resolve_links.py", "update"): "name collision: dict.update, not the link resolver's",
    ("mloda/core/runtime/run.py", "join"): "name collision: str.join, not the runner's join",
}

# Swallowing functions OUTSIDE the declared modules that the match path calls; a handler there is decided here.
SWALLOWING_HELPERS_OUTSIDE_THE_PATH: dict[tuple[str, str], str] = {
    ("mloda/core/abstract_plugins/components/utils.py", "safe_field"): (
        "it degrades one field in a rendering path, so swallowing a marked exception is its contract"
    ),
    ("mloda/core/abstract_plugins/components/utils.py", "escalate_match_abort"): (
        "the guard around the marker write; failing to mark must not replace the exception being marked"
    ),
}

_ESCALATION = "escalate_match_abort"
_ABORT_CHECK = "is_match_abort"
_CONTAINED_TAG = "Contained:"
_MARKED_TAG = "Marked:"
_SWALLOW_TAG = "Swallows:"

# Constructing a class runs these, so a call on a class name is an edge into them.
_CONSTRUCTORS: frozenset[str] = frozenset({"__init__", "__post_init__"})

Kind = Literal["marked", "contained", "mismarked", "unannotated"]

HandlerKind = Literal["escalating", "swallowing", "misannotated", "unannotated"]


class RaiseSite(NamedTuple):
    """One ``raise <exc>`` statement and the escalation decision written at it."""

    module: str
    lineno: int
    function: str
    kind: Kind
    reason: str | None

    def location(self) -> str:
        return f"{self.module}:{self.lineno} {self.function}()"


class HandlerSite(NamedTuple):
    """One ``except`` clause and the containment decision written at it."""

    module: str
    lineno: int
    function: str
    kind: HandlerKind
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
    handlers: tuple[HandlerSite, ...]
    swallowing_external: tuple[ExternalCall, ...]


def _scan(node: ast.AST) -> tuple[frozenset[str], bool]:
    """The names ``node`` calls and whether it raises, in one walk (the index scans all of ``mloda/``)."""
    names: set[str] = set()
    raises = False
    stack: list[ast.AST] = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, ast.Call):
            func = current.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
        elif isinstance(current, ast.Raise) and current.exc is not None:
            raises = True
        for field, value in ast.iter_fields(current):
            if field == "decorator_list":
                # A decorator runs at definition time, not when the match path calls the function.
                continue
            for child in value if isinstance(value, list) else [value]:
                if isinstance(child, ast.AST):
                    stack.append(child)
    return frozenset(names), raises


def _escalates(node: ast.AST) -> bool:
    return _ESCALATION in _scan(node)[0]


def _handler_escalates(handler: ast.ExceptHandler) -> bool:
    """A raise as a direct child re-raises unconditionally; otherwise the abort check needs a raise beside it."""
    if any(isinstance(statement, ast.Raise) for statement in handler.body):
        return True
    calls: set[str] = set()
    raises = False
    for statement in handler.body:
        calls |= _scan(statement)[0]
        raises = raises or any(isinstance(node, ast.Raise) for node in ast.walk(statement))
    return raises and _ABORT_CHECK in calls


def _swallows(node: ast.AST) -> bool:
    """Does this definition hold a handler that neither re-raises nor checks the marker."""
    return any(isinstance(child, ast.ExceptHandler) and not _handler_escalates(child) for child in ast.walk(node))


def _own_line_and_trailing_comments(source: str) -> tuple[dict[int, str], dict[int, str]]:
    """Comment text per line, split into own-line and trailing; tokenize, so a ``#`` in a string is not one."""
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
    """The tagged reason for the raise at ``lineno``: trailing on that line, or in the block directly above."""
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


def _collect_raises(node: ast.AST, functions: list[str], out: list[tuple[ast.Raise, list[str]]]) -> None:
    """Walk ``node``, recording each raise with its enclosing function chain."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _collect_raises(child, [*functions, child.name], out)
        else:
            if isinstance(child, ast.Raise):
                out.append((child, functions))
            _collect_raises(child, functions, out)


def classify_raises(source: str, module: str, functions: frozenset[str] | None = None) -> list[RaiseSite]:
    """Classify every ``raise <exc>`` in ``source``, restricted to raises inside the named functions."""
    tree = ast.parse(source, filename=module)
    own_line, trailing = _own_line_and_trailing_comments(source)
    found: list[tuple[ast.Raise, list[str]]] = []
    _collect_raises(tree, [], found)

    sites: list[RaiseSite] = []
    for raise_node, chain in found:
        if not chain:
            continue
        if functions is not None and not any(name in functions for name in chain):
            continue
        if raise_node.exc is None:
            # A bare re-raise carries no decision of its own: it sits under the escalate_match_abort line.
            continue
        # Only the raised expression counts: crediting the enclosing handler would bless later raises too.
        marked = _escalates(raise_node.exc)
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


def _collect_handlers(node: ast.AST, functions: list[str], out: list[tuple[ast.ExceptHandler, list[str]]]) -> None:
    """Walk ``node``, recording each except clause with its enclosing function chain."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _collect_handlers(child, [*functions, child.name], out)
        else:
            if isinstance(child, ast.ExceptHandler):
                out.append((child, functions))
            _collect_handlers(child, functions, out)


def classify_handlers(source: str, module: str, functions: frozenset[str] | None = None) -> list[HandlerSite]:
    """Classify every ``except`` clause in ``source``, restricted to handlers inside the named functions."""
    tree = ast.parse(source, filename=module)
    own_line, trailing = _own_line_and_trailing_comments(source)
    found: list[tuple[ast.ExceptHandler, list[str]]] = []
    _collect_handlers(tree, [], found)

    sites: list[HandlerSite] = []
    for handler, chain in found:
        if not chain:
            continue
        if functions is not None and not any(name in functions for name in chain):
            continue
        escalating = _handler_escalates(handler)
        # Anchored at the except line only: a reason elsewhere in the function would bless the next handler too.
        declared = _anchored_reason(own_line, trailing, handler.lineno, _SWALLOW_TAG)
        kind: HandlerKind
        if escalating:
            kind, reason = ("misannotated", declared) if declared is not None else ("escalating", None)
        elif declared is not None:
            kind, reason = "swallowing", declared
        else:
            kind, reason = "unannotated", None
        sites.append(HandlerSite(module, handler.lineno, chain[-1], kind, reason))
    return sorted(sites)


class _Index(NamedTuple):
    """Every definition under ``mloda/``, plus the names whose call can raise."""

    functions: dict[str, list[_Definition]]
    constructors: dict[str, list[_Definition]]
    raising: frozenset[str]


def _raising_names(calls_per_name: dict[str, frozenset[str]], direct: frozenset[str]) -> frozenset[str]:
    """Names that raise directly or through anything they call: one non-raising wrapper defeats depth-1."""
    callers: dict[str, set[str]] = {}
    for name, called in calls_per_name.items():
        for callee in called:
            callers.setdefault(callee, set()).add(name)

    raising = set(direct)
    queue = deque(sorted(raising))
    while queue:
        for caller in callers.get(queue.popleft(), ()):
            if caller not in raising:
                raising.add(caller)
                queue.append(caller)
    return frozenset(raising)


@lru_cache(maxsize=1)
def _index() -> _Index:
    """Index ``mloda/`` by name: functions, the constructors a class name runs, and the names that can raise."""
    functions: dict[str, list[_Definition]] = {}
    constructors: dict[str, list[_Definition]] = {}
    calls: dict[str, set[str]] = {}
    direct: set[str] = set()
    for path in sorted(_CORE_ROOT.rglob("*.py")):
        module = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=module)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                called, raises = _scan(node)
                functions.setdefault(node.name, []).append(_Definition(module, node))
                calls.setdefault(node.name, set()).update(called)
                if raises:
                    direct.add(node.name)
            elif isinstance(node, ast.ClassDef):
                # A call on a CLASS name is an edge into what construction runs, which no call node names.
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in _CONSTRUCTORS:
                        constructors.setdefault(node.name, []).append(_Definition(module, child))
    frozen = {name: frozenset(called) for name, called in calls.items()}
    return _Index(functions, constructors, _raising_names(frozen, frozenset(direct)))


@lru_cache(maxsize=1)
def sweep() -> _Sweep:
    """Walk the match path from the seeds and classify every raise it reaches."""
    # The index covers all of mloda/, so a call leaving the declared modules is seen, though never followed.
    index = _index()

    def targets_of(called: str) -> list[_Definition]:
        return index.functions.get(called, []) + index.constructors.get(called, [])

    queue: deque[_Definition] = deque()
    # A definition is identified by (module, line): two same-named methods in one module stay distinct.
    visited: set[tuple[str, int]] = set()
    for seed in SEEDS:
        for definition in targets_of(seed):
            if definition.module in MATCH_PATH_MODULES and (definition.module, definition.node.lineno) not in visited:
                visited.add((definition.module, definition.node.lineno))
                queue.append(definition)

    reachable: set[tuple[str, str]] = set()
    external: set[ExternalCall] = set()
    swallowing: set[ExternalCall] = set()
    while queue:
        definition = queue.popleft()
        reachable.add((definition.module, definition.node.name))
        caller = f"{definition.module}::{definition.node.name}"
        for called in _scan(definition.node)[0]:
            for target in targets_of(called):
                if target.module in MATCH_PATH_MODULES:
                    key = (target.module, target.node.lineno)
                    if key not in visited:
                        visited.add(key)
                        queue.append(target)
                    continue
                if target.node.name in index.raising:
                    external.add(ExternalCall(target.module, target.node.name, caller))
                if _swallows(target.node):
                    swallowing.add(ExternalCall(target.module, target.node.name, caller))

    sites: list[RaiseSite] = []
    handlers: list[HandlerSite] = []
    for module in sorted({module for module, _ in reachable}):
        names = frozenset(name for reached_module, name in reachable if reached_module == module)
        source = (_REPO_ROOT / module).read_text(encoding="utf-8")
        sites.extend(classify_raises(source, module, names))
        handlers.extend(classify_handlers(source, module, names))

    return _Sweep(
        tuple(sorted(sites)),
        frozenset(reachable),
        tuple(sorted(external)),
        tuple(sorted(handlers)),
        tuple(sorted(swallowing)),
    )


def _of_kind(kind: Kind) -> list[RaiseSite]:
    return [site for site in sweep().sites if site.kind == kind]


def test_every_reachable_raise_is_marked_or_contained() -> None:
    unannotated = _of_kind("unannotated")

    assert unannotated == [], (
        "Raise sites reachable from the match seams with no escalation decision:\n"
        + "\n".join(f"  {site.location()}" for site in unannotated)
        + f"\n\nRecord the decision at the raise: wrap the exception in {_ESCALATION}(...), or write "
        f"'# {_CONTAINED_TAG} <reason>' directly above it."
    )


def test_no_marked_comment_without_an_escalation() -> None:
    mismarked = _of_kind("mismarked")

    assert mismarked == [], (
        f"Raise sites carrying a '# {_MARKED_TAG}' comment that no {_ESCALATION} call backs:\n"
        + "\n".join(f"  {site.location()}: {site.reason}" for site in mismarked)
    )


def test_calls_into_raising_helpers_outside_the_path_are_declared() -> None:
    undeclared = [
        call for call in sweep().external if (call.module, call.function) not in RAISING_HELPERS_OUTSIDE_THE_PATH
    ]

    assert undeclared == [], (
        "The match path calls raising functions outside the declared modules:\n"
        + "\n".join(f"  {call.module}::{call.function} from {call.caller}" for call in undeclared)
        + "\n\nDeclare the (module, function) pair in RAISING_HELPERS_OUTSIDE_THE_PATH with a reason, or add "
        "the module to MATCH_PATH_MODULES so its raises are swept."
    )


def test_declared_helpers_outside_the_path_are_still_called() -> None:
    """Stale entries cannot accumulate: every declared helper must still be reached."""
    called = {(call.module, call.function) for call in sweep().external}
    stale = sorted(entry for entry in RAISING_HELPERS_OUTSIDE_THE_PATH if entry not in called)

    assert stale == [], f"RAISING_HELPERS_OUTSIDE_THE_PATH entries the match path no longer calls: {stale}"


def test_every_seed_resolves_inside_a_declared_module() -> None:
    """A seed that no longer names a definition seeds nothing, and the walk would shrink in silence."""
    index = _index()
    unresolved = sorted(
        seed
        for seed in SEEDS
        if not any(
            definition.module in MATCH_PATH_MODULES
            for definition in index.functions.get(seed, []) + index.constructors.get(seed, [])
        )
    )

    assert unresolved == [], (
        f"SEEDS entries that name no definition in a declared module: {unresolved}. Rename the seed with the "
        "definition it points at."
    )


def test_every_declared_module_is_reachable() -> None:
    reached = {module for module, _ in sweep().reachable}
    unreached = sorted(set(MATCH_PATH_MODULES) - reached)

    assert unreached == [], (
        f"MATCH_PATH_MODULES entries no seed reaches: {unreached}. Drop the entry, or add the seed that reaches it."
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


def test_classifier_flags_an_unannotated_raise_inside_an_escalating_handler() -> None:
    """An escalation elsewhere in the handler annotates nothing: only the raised expression counts."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        escalate_match_abort(exc)\n"
        "        if options is None:\n"
        "            raise ValueError('brand new')\n"
        "        raise\n"
    )

    sites = classify_raises(source, "snippet.py", frozenset({"match_feature_group_criteria"}))

    assert [site.kind for site in sites] == ["unannotated"]
    assert sites[0].lineno == 7


def test_classifier_flags_an_unannotated_raise() -> None:
    """A '# Contained:' mention inside a docstring is not a decision at the raise."""
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
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    # Marked: claims to escalate, nothing here does.\n"
        "    raise ValueError('no feature')\n"
    )

    sites = classify_raises(source, "snippet.py", frozenset({"match_feature_group_criteria"}))

    assert [site.kind for site in sites] == ["mismarked"]


# The handler sweep: same reachable set, one decision per except clause.


def _handlers_of_kind(kind: HandlerKind) -> list[HandlerSite]:
    return [site for site in sweep().handlers if site.kind == kind]


def _handlers(source: str) -> list[HandlerSite]:
    return classify_handlers(source, "snippet.py", frozenset({"match_feature_group_criteria"}))


def test_every_reachable_handler_escalates_or_declares_a_swallow() -> None:
    unannotated = _handlers_of_kind("unannotated")

    assert unannotated == [], (
        "Handlers reachable from the match seams with no containment decision:\n"
        + "\n".join(f"  {site.location()}" for site in unannotated)
        + "\n\nA handler that swallows a marked raise undoes the escalation. Record the decision at the "
        f"handler: re-raise when is_match_abort(exc) holds, or write '# {_SWALLOW_TAG} <reason>' on the except line."
    )


def test_no_swallows_comment_on_a_reraising_handler() -> None:
    misannotated = _handlers_of_kind("misannotated")

    assert misannotated == [], f"Handlers carrying a '# {_SWALLOW_TAG}' comment that re-raise anyway:\n" + "\n".join(
        f"  {site.location()}: {site.reason}" for site in misannotated
    )


def test_calls_into_swallowing_helpers_outside_the_path_are_declared() -> None:
    undeclared = [
        call
        for call in sweep().swallowing_external
        if (call.module, call.function) not in SWALLOWING_HELPERS_OUTSIDE_THE_PATH
    ]

    assert undeclared == [], (
        "The match path calls functions outside the declared modules that swallow exceptions:\n"
        + "\n".join(f"  {call.module}::{call.function} from {call.caller}" for call in undeclared)
        + "\n\nDeclare the (module, function) pair in SWALLOWING_HELPERS_OUTSIDE_THE_PATH with a reason, or add "
        "the module to MATCH_PATH_MODULES so its handlers are swept."
    )


def test_declared_swallowing_helpers_outside_the_path_are_still_called() -> None:
    """Stale entries cannot accumulate: every declared swallowing helper must still be reached."""
    called = {(call.module, call.function) for call in sweep().swallowing_external}
    stale = sorted(entry for entry in SWALLOWING_HELPERS_OUTSIDE_THE_PATH if entry not in called)

    assert stale == [], f"SWALLOWING_HELPERS_OUTSIDE_THE_PATH entries the match path no longer calls: {stale}"


def test_known_escalating_handlers_are_enumerated() -> None:
    """Canary: the handlers that re-raise today are all seen, so a silently empty handler sweep cannot pass."""
    seam = "mloda/core/prepare/identify_feature_group.py"
    feature_group = "mloda/core/abstract_plugins/feature_group.py"
    mixin = "mloda/core/abstract_plugins/components/feature_chainer/feature_chain_parser_mixin.py"
    guards = "mloda/core/abstract_plugins/components/feature_chainer/feature_chain_author_guards.py"
    expected = {
        (seam, "_filter_feature_group_by_criteria"),
        (feature_group, "is_root"),
        (mixin, "match_parser_criteria"),
        (guards, "check_required_when"),
    }

    escalating = {(site.module, site.function) for site in _handlers_of_kind("escalating")}

    assert expected <= escalating, f"known escalating handlers missing from the sweep: {sorted(expected - escalating)}"
    # Vacuity floor, not a target: 15 handlers today, so removing one stays a legitimate change.
    assert len(sweep().handlers) >= 12, f"sweep enumerated only {len(sweep().handlers)} handlers; it is not walking"


def test_handler_classifier_flags_a_blanket_handler() -> None:
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except Exception:\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["unannotated"]
    assert sites[0].lineno == 4


def test_handler_classifier_accepts_a_conditional_reraise() -> None:
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["escalating"]


def test_handler_classifier_accepts_an_escalation_and_a_bare_reraise() -> None:
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except Exception as exc:\n"
        "        escalate_match_abort(exc)\n"
        "        raise\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["escalating"]


def test_handler_classifier_accepts_a_trailing_swallows_comment() -> None:
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError:  # Swallows: a malformed name is this candidate's own defect.\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["swallowing"]
    assert sites[0].reason == "a malformed name is this candidate's own defect."


def test_handler_classifier_accepts_an_own_line_swallows_comment() -> None:
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    # Swallows: a missing file is a non-match for this reader only.\n"
        "    except OSError:\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["swallowing"]
    assert sites[0].reason == "a missing file is a non-match for this reader only."


def test_handler_classifier_requires_a_raise_beside_the_abort_check() -> None:
    """Reading is_match_abort without re-raising escalates nothing."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        if is_match_abort(exc):\n"
        "            record(exc)\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["unannotated"]


def test_handler_classifier_does_not_inherit_a_swallows_reason() -> None:
    """A reason on an earlier handler annotates nothing: the next handler needs its own."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except OSError:  # Swallows: a missing file is a non-match for this reader only.\n"
        "        return False\n"
        "    except ValueError:\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["swallowing", "unannotated"]


def test_handler_classifier_flags_a_swallows_comment_on_a_reraise() -> None:
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:  # Swallows: stale, this handler re-raises.\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["misannotated"]
    assert sites[0].reason == "stale, this handler re-raises."
