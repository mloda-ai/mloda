"""Static sweep: every raise a match hook can reach must state whether it escalates or stays contained.

``IdentifyFeatureGroupClass._filter_feature_group_by_criteria`` contains a raise out of
``match_feature_group_criteria`` as a non-match unless ``escalate_match_abort`` marked it, so a NEW unmarked
raise on the match path silently loses the feature to a rival plugin. This sweep walks the static call graph
from the match seams through the declared match-path modules and requires a decision at each individual
raise: the escalation call, or a ``# Contained: <reason>`` comment on the raise line or in the comment block
directly above it at the raise's own column. A reason elsewhere in the enclosing function is not accepted; a
later raise would inherit it and the gate would rot.

A sibling sweep walks the same reachable set for every place an exception can be dropped: an ``except``
clause, a ``contextlib.suppress`` block, and a ``finally`` that returns. A mark only survives if each of them
re-raises the caught object itself, so each must re-raise on ``is_match_abort`` or carry a
``# Swallows: <reason>`` comment.

Both judge one site alone, which misses the clauses of one ``try`` shadowing each other: Python runs the
first matching handler, so an earlier one whose type meets a later clause's, in either inheritance
direction, wins over it and the abort check below never runs. A third pass compares the clauses on each
``try``. Types resolve by name through the builtins and the class definitions under ``mloda/``, never by
importing the module under test; a type neither resolves is reported instead of passed.

A fourth pass reads the other pairing on one ``try``: a ``finally`` that discards what is in flight, by leaving
through a return, break or continue or by raising a replacement, undoes every escalation it encloses: the
re-raises of its own clauses, and the marked raises in the arms those clauses never catch.

Not covered: plugin code under ``mloda_plugins``; dynamic dispatch, which is why SEEDS is hand-written;
decorators; ``except*`` groups, which neither pass over a ``try`` reads, as their clauses split an exception
group between them instead of racing for it, so the third pass's first-match reasoning does not hold there
and the fourth drops them with it; a raise in the body of a nested try that has handlers, assumed caught there, and
one inside a ``suppress`` block naming ``Exception`` or ``BaseException``, assumed dropped there.
"""

from __future__ import annotations

import ast
import builtins
import io
import sys
import tokenize
from collections import deque
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Literal, NamedTuple, TypeGuard, cast

# Anchored via __file__, not the cwd: a cwd-relative root makes every lookup empty and the sweep vacuous.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CORE_ROOT = _REPO_ROOT / "mloda"
assert _CORE_ROOT.is_dir(), f"core root not found; check the parents index for the repo root: {_CORE_ROOT}"

# Modules the match path runs through, repo-relative. The graph never leaves this set, so a module missing
# here is a blind spot; test_every_declared_module_is_reachable keeps stale entries out.
MATCH_PATH_MODULES: dict[str, str] = {
    "mloda/core/abstract_plugins/components/match_hook.py": "the helper that owns the call and the containment",
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
_MATCHES_COLLISION = "name collision: the match path calls the input-data hooks named matches, not Link.matches"
_UPDATE_COLLISION = "name collision: dict.update, not the link resolver's"
_JOIN_COLLISION = "name collision: str.join, not the runner's join"

RAISING_HELPERS_OUTSIDE_THE_PATH: dict[tuple[str, str], str] = {
    ("mloda/core/abstract_plugins/components/options.py", "__init__"): _CANDIDATE_OWN_DECLARATION,
    ("mloda/core/abstract_plugins/components/options.py", "add_to_group"): _DECIDED_ABOVE_BY_READER_SELECTION,
    ("mloda/core/abstract_plugins/components/options.py", "get_in_features"): _CANDIDATE_OWN_DECLARATION,
    ("mloda/core/abstract_plugins/components/feature.py", "__init__"): _CANDIDATE_OWN_DECLARATION,
    ("mloda/core/abstract_plugins/plugin_loader/plugin_loader.py", "__init__"): _READER_AUTO_LOAD,
    ("mloda/core/abstract_plugins/plugin_loader/plugin_loader.py", "load_group"): _READER_AUTO_LOAD,
    ("mloda/core/abstract_plugins/plugin_loader/plugin_loader.py", "all"): _READER_AUTO_LOAD,
    ("mloda/core/abstract_plugins/components/link.py", "matches"): _MATCHES_COLLISION,
    ("mloda/core/abstract_plugins/components/utils.py", "get_all_subclasses"): (
        "real edge, collided verdict: it raises nothing itself; the set.add / set.update names do"
    ),
    ("mloda/core/prepare/resolve_links.py", "update"): _UPDATE_COLLISION,
    ("mloda/core/runtime/run.py", "join"): _JOIN_COLLISION,
}

# Swallowing functions OUTSIDE the declared modules that the match path calls; the containment there is decided
# here. The closure is transitive and resolves by name, so an entry can be a name COLLISION, not a call edge.
SWALLOWING_HELPERS_OUTSIDE_THE_PATH: dict[tuple[str, str], str] = {
    ("mloda/core/abstract_plugins/components/utils.py", "safe_field"): (
        "it degrades one field in a rendering path, so swallowing a marked exception is its contract"
    ),
    ("mloda/core/abstract_plugins/components/utils.py", "escalate_match_abort"): (
        "the guard around the marker write; failing to mark must not replace the exception being marked"
    ),
    ("mloda/core/abstract_plugins/components/utils.py", "contained_raise_reason"): (
        "it swallows only through safe_field, whose degrade-one-field contract is declared above"
    ),
    ("mloda/core/abstract_plugins/components/utils.py", "get_all_subclasses"): (
        "real edge, collided verdict: it swallows nothing itself; the set.add / set.update names do"
    ),
    ("mloda/core/abstract_plugins/components/options.py", "__init__"): _CANDIDATE_OWN_DECLARATION,
    ("mloda/core/abstract_plugins/components/feature.py", "__init__"): _CANDIDATE_OWN_DECLARATION,
    ("mloda/core/abstract_plugins/plugin_loader/plugin_loader.py", "__init__"): _READER_AUTO_LOAD,
    ("mloda/core/abstract_plugins/plugin_loader/plugin_loader.py", "load_group"): _READER_AUTO_LOAD,
    ("mloda/core/abstract_plugins/plugin_loader/plugin_loader.py", "all"): _READER_AUTO_LOAD,
    ("mloda/core/abstract_plugins/components/link.py", "matches"): _MATCHES_COLLISION,
    ("mloda/core/prepare/resolve_links.py", "update"): _UPDATE_COLLISION,
    ("mloda/core/runtime/run.py", "join"): _JOIN_COLLISION,
}

_ESCALATION = "escalate_match_abort"
_ABORT_CHECK = "is_match_abort"
_SUPPRESS = "suppress"
# The types a suppress must name before a raise inside it can be assumed dropped there.
_BLANKET_TYPES: frozenset[str] = frozenset({"Exception", "BaseException"})
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


class ShadowSite(NamedTuple):
    """An ``except`` clause that catches ahead of a later clause on the same ``try``, whose mark survives."""

    module: str
    lineno: int
    function: str
    caught: str
    shadowed_lineno: int
    shadowed_type: str

    def location(self) -> str:
        return f"{self.module}:{self.lineno} {self.function}()"


class UnresolvedType(NamedTuple):
    """An ``except`` type beside an escalating clause whose base chain leaves the builtins and ``mloda/``."""

    module: str
    lineno: int
    function: str
    name: str

    def location(self) -> str:
        return f"{self.module}:{self.lineno} {self.function}()"


class FinallySite(NamedTuple):
    """A ``finally`` that discards, undoing an escalation the ``try`` it closes would otherwise let out."""

    module: str
    lineno: int
    function: str
    discarded_lineno: int

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
    shadowed: tuple[ShadowSite, ...]
    unresolved: tuple[UnresolvedType, ...]
    finally_shadowed: tuple[FinallySite, ...]


def _call_name(node: ast.AST) -> str | None:
    """The name a call names, by function or by attribute; None for any other node and for a computed callee."""
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return node.func.id
    return node.func.attr if isinstance(node.func, ast.Attribute) else None


def _children(node: ast.AST) -> list[ast.AST]:
    """The children of ``node``, minus its decorators: those run at definition time, not when it is called."""
    found: list[ast.AST] = []
    for field, value in ast.iter_fields(node):
        if field == "decorator_list":
            continue
        for child in value if isinstance(value, list) else [value]:
            if isinstance(child, ast.AST):
                found.append(child)
    return found


def _scan(node: ast.AST) -> tuple[frozenset[str], bool]:
    """The names ``node`` calls and whether it raises, in one walk (the index scans all of ``mloda/``)."""
    names: set[str] = set()
    raises = False
    stack: list[ast.AST] = [node]
    while stack:
        current = stack.pop()
        called = _call_name(current)
        if called is not None:
            names.add(called)
        elif isinstance(current, ast.Raise) and current.exc is not None:
            raises = True
        stack.extend(_children(current))
    return frozenset(names), raises


def _escalates(node: ast.AST) -> bool:
    return _ESCALATION in _scan(node)[0]


def _keeps_the_mark(node: ast.Raise, bound: str | None) -> bool:
    """A re-raise the marker survives: bare, the caught name itself, or a replacement marked on its way out."""
    if node.exc is None:
        return True
    if bound is not None and isinstance(node.exc, ast.Name) and node.exc.id == bound:
        return True
    return _escalates(node.exc)


# A nested def or lambda runs later or never, and a nested try may catch the raise it holds.
# ast.TryStar is 3.11+, so the name is read off the module instead of written.
_DEFERRED: tuple[type[ast.AST], ...] = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_NESTED_TRY: tuple[type[ast.AST], ...] = (ast.Try, getattr(ast, "TryStar", ast.Try))
# A break or continue binds to the innermost loop around it, so one opened here leaves that loop and nothing else.
_LOOPS: tuple[type[ast.AST], ...] = (ast.For, ast.AsyncFor, ast.While)

# A typo in that attribute name would fall back to ast.Try on every version and disable the boundary in silence.
assert sys.version_info < (3, 11) or _NESTED_TRY[1] is not ast.Try, "ast.TryStar no longer resolves off ast"


def _is_suppress(expr: ast.expr) -> bool:
    if not isinstance(expr, ast.Call):
        return False
    func = expr.func
    if isinstance(func, ast.Name):
        return func.id == _SUPPRESS
    return isinstance(func, ast.Attribute) and func.attr == _SUPPRESS


def _shields(node: ast.AST) -> TypeGuard[ast.With | ast.AsyncWith]:
    """A ``suppress`` block, whatever it names: it drops exceptions, so it owes a containment decision."""
    return isinstance(node, (ast.With, ast.AsyncWith)) and any(_is_suppress(item.context_expr) for item in node.items)


def _suppresses_everything(expr: ast.expr) -> bool:
    """A ``suppress`` naming Exception or BaseException; a bare, starred or computed argument decides nothing."""
    if not _is_suppress(expr):
        return False
    return any(isinstance(arg, ast.Name) and arg.id in _BLANKET_TYPES for arg in cast(ast.Call, expr).args)


def _shields_everything(node: ast.AST) -> TypeGuard[ast.With | ast.AsyncWith]:
    """A ``suppress`` block wide enough that what raises inside it never leaves it."""
    return isinstance(node, (ast.With, ast.AsyncWith)) and any(
        _suppresses_everything(item.context_expr) for item in node.items
    )


def _evaluated_at(node: ast.AST) -> list[ast.AST]:
    """What the scans still read off a pruned boundary: a loop's else arm, a definition's defaults."""
    if isinstance(node, _LOOPS):
        # The else arm sits outside its own loop, so a break there binds to the loop around this one.
        return list(cast(ast.For, node).orelse)
    if isinstance(node, _DEFERRED):
        # Every member carries the same args field; the body runs later, the defaults run here.
        arguments = cast(ast.Lambda, node).args
        return [*arguments.defaults, *(default for default in arguments.kw_defaults if default is not None)]
    return []


def _runs_here(body: list[ast.stmt], boundaries: tuple[type[ast.AST], ...], *, shielded: bool = False) -> list[ast.AST]:
    """The nodes of ``body`` that run where they stand, pruned at ``boundaries`` and, when shielded, at a suppress."""
    found: list[ast.AST] = []
    stack: list[ast.AST] = list(body)
    while stack:
        current = stack.pop()
        if shielded and _shields_everything(current):
            continue
        if isinstance(current, boundaries):
            stack.extend(_evaluated_at(current))
            continue
        found.append(current)
        stack.extend(_children(current))
    return found


def _escapes(body: list[ast.stmt]) -> bool:
    """A return, break or continue that really leaves ``body``: a return from any depth, a break bound outside it."""
    if any(isinstance(node, ast.Return) for node in _runs_here(body, _DEFERRED)):
        return True
    return any(isinstance(node, (ast.Break, ast.Continue)) for node in _runs_here(body, _DEFERRED + _LOOPS))


def _reraises(body: Sequence[ast.AST], bound: str | None, bare_counts: bool, *, in_loops: bool = True) -> bool:
    """Does a raise the mark survives leave ``body``, with a bare one counting only where it re-raises ``bound``."""
    for node in body:
        if isinstance(node, _DEFERRED) or _shields_everything(node):
            continue
        # A loop body runs zero times on an empty iterable and a break skips the rest of it and its else arm, so
        # a raise written in a loop stands on no path out of the body around it.
        if not in_loops and isinstance(node, _LOOPS):
            continue
        if isinstance(node, ast.Raise):
            if _keeps_the_mark(node, bound) and (bare_counts or node.exc is not None):
                return True
            continue
        if isinstance(node, _NESTED_TRY):
            # The arms are read off ast.Try: TryStar carries the same ones, and 3.10 cannot name the type.
            if _nested_try_reraises(cast(ast.Try, node), bound, bare_counts, in_loops=in_loops):
                return True
            continue
        # A raise cannot sit inside an expression; a match_case is neither expression nor statement, so it stays.
        inside = [child for child in ast.iter_child_nodes(node) if not isinstance(child, ast.expr)]
        if _reraises(inside, bound, bare_counts, in_loops=in_loops):
            return True
    return False


def _nested_try_reraises(node: ast.Try, bound: str | None, bare_counts: bool, *, in_loops: bool = True) -> bool:
    """The arms of a nested try its own clauses never catch: else and finally, plus the body when it has none."""
    if _escapes(node.finalbody):
        return False
    if _reraises(node.orelse, bound, bare_counts, in_loops=in_loops) or _reraises(
        node.finalbody, bound, bare_counts, in_loops=in_loops
    ):
        return True
    if not node.handlers:
        return _reraises(node.body, bound, bare_counts, in_loops=in_loops)
    # A bare raise in a clause re-raises that clause's own exception, not the one the handler above it caught.
    return any(_reraises(clause.body, bound, False, in_loops=in_loops) for clause in node.handlers)


def _reads_a_shadowed_name(call: ast.Call, shadowed: frozenset[str]) -> bool:
    """Does any argument of ``call``, at any depth and in any form, name an exception bound below the handler."""
    if not shadowed:
        return False
    arguments: list[ast.expr] = [*call.args, *(keyword.value for keyword in call.keywords)]
    return any(
        isinstance(node, ast.Name) and node.id in shadowed for argument in arguments for node in ast.walk(argument)
    )


def _called_here(body: list[ast.stmt], bound: str | None) -> frozenset[str]:
    """The names ``body`` calls on the exception ``bound`` names, past the exceptions bound below it."""
    # Two rules: a nested clause is read only when it can re-raise the caught exception, and inside one a call
    # reading the name that clause binds gates its exception, not the caught one. An alias for it, ``failure =
    # inner`` and then a check on ``failure``, is out of reach of this or any other name-based scan.
    names: set[str] = set()
    stack: list[tuple[ast.AST, frozenset[str]]] = [(node, frozenset()) for node in body]
    while stack:
        current, shadowed = stack.pop()
        if isinstance(current, _DEFERRED):
            stack.extend((node, shadowed) for node in _evaluated_at(current))
            continue
        if isinstance(current, ast.ExceptHandler):
            if not _reraises(current.body, bound, False):
                continue
            inside = shadowed | {current.name} if current.name is not None else shadowed
            stack.extend((node, inside) for node in current.body)
            continue
        called = _call_name(current)
        if called is not None and not _reads_a_shadowed_name(cast(ast.Call, current), shadowed):
            names.add(called)
        stack.extend((node, shadowed) for node in _children(current))
    return frozenset(names)


def _handler_escalates(handler: ast.ExceptHandler) -> bool:
    """A re-raise that keeps the mark, gated on the abort check or standing on every path out of the handler."""
    if not _reraises(handler.body, handler.name, True):
        return False
    if _ABORT_CHECK in _called_here(handler.body, handler.name):
        return True
    # Discounting a break bound to a loop opened here only holds for a re-raise standing outside that loop.
    return not _escapes(handler.body) and _reraises(handler.body, handler.name, True, in_loops=False)


class _Containment(NamedTuple):
    """One place an exception can be dropped, with the header lines a trailing reason may sit on."""

    lineno: int
    col_offset: int
    header_end: int
    escalating: bool


def _containment_at(node: ast.AST) -> _Containment | None:
    """The containment this node opens: an except clause, a suppress block, or a finally that escapes."""
    if isinstance(node, ast.ExceptHandler):
        return _Containment(node.lineno, node.col_offset, node.body[0].lineno - 1, _handler_escalates(node))
    if _shields(node):
        return _Containment(node.lineno, node.col_offset, node.body[0].lineno - 1, False)
    if isinstance(node, ast.Try) and _escapes(node.finalbody):
        # ast exposes no position for the finally keyword, so the try line carries the decision.
        return _Containment(node.lineno, node.col_offset, node.body[0].lineno - 1, False)
    return None


def _swallows(node: ast.AST) -> bool:
    """Does this definition hold a containment that neither re-raises nor checks the marker."""
    for child in ast.walk(node):
        containment = _containment_at(child)
        if containment is not None and not containment.escalating:
            return True
    return False


def _own_line_and_trailing_comments(source: str) -> tuple[dict[int, tuple[int, str]], dict[int, str]]:
    """Comments per line, own-line ones with their column; tokenize, so a ``#`` inside a string is not one."""
    lines = source.splitlines()
    own_line: dict[int, tuple[int, str]] = {}
    trailing: dict[int, str] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        row, col = token.start
        text = token.string.lstrip("#").strip()
        if lines[row - 1][:col].strip():
            trailing[row] = text
        else:
            own_line[row] = (col, text)
    return own_line, trailing


def _tagged_reason(text: str, tag: str) -> str | None:
    """The reason behind ``tag`` in one comment, or None when the tag is absent or the reason is empty."""
    if not text.startswith(tag):
        return None
    reason = text[len(tag) :].strip()
    return reason or None


def _anchored_reason(
    own_line: dict[int, tuple[int, str]],
    trailing: dict[int, str],
    lineno: int,
    col_offset: int,
    tag: str,
    header_end: int | None = None,
) -> str | None:
    """The tagged reason for the site: trailing anywhere in its header, or above it at its own column."""
    for row in range(lineno, (lineno if header_end is None else header_end) + 1):
        reason = _tagged_reason(trailing.get(row, ""), tag)
        if reason is not None:
            return reason
    row = lineno - 1
    while row in own_line:
        comment_col, text = own_line[row]
        # A reason deeper than the site belongs to the block it is written in, not to the site below it.
        if comment_col == col_offset:
            reason = _tagged_reason(text, tag)
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
        column = raise_node.col_offset
        claimed = _anchored_reason(own_line, trailing, raise_node.lineno, column, _MARKED_TAG)
        contained = _anchored_reason(own_line, trailing, raise_node.lineno, column, _CONTAINED_TAG)
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


def _collect_handlers(node: ast.AST, functions: list[str], out: list[tuple[_Containment, list[str]]]) -> None:
    """Walk ``node``, recording each containment with its enclosing function chain."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _collect_handlers(child, [*functions, child.name], out)
        else:
            containment = _containment_at(child)
            if containment is not None:
                out.append((containment, functions))
            _collect_handlers(child, functions, out)


def classify_handlers(source: str, module: str, functions: frozenset[str] | None = None) -> list[HandlerSite]:
    """Classify every containment in ``source``, restricted to the ones inside the named functions."""
    tree = ast.parse(source, filename=module)
    own_line, trailing = _own_line_and_trailing_comments(source)
    found: list[tuple[_Containment, list[str]]] = []
    _collect_handlers(tree, [], found)

    sites: list[HandlerSite] = []
    for containment, chain in found:
        if not chain:
            continue
        if functions is not None and not any(name in functions for name in chain):
            continue
        # Anchored at the site itself: a reason elsewhere in the function would bless the next handler too.
        declared = _anchored_reason(
            own_line,
            trailing,
            containment.lineno,
            containment.col_offset,
            _SWALLOW_TAG,
            containment.header_end,
        )
        kind: HandlerKind
        if containment.escalating:
            kind, reason = ("misannotated", declared) if declared is not None else ("escalating", None)
        elif declared is not None:
            kind, reason = "swallowing", declared
        else:
            kind, reason = "unannotated", None
        sites.append(HandlerSite(module, containment.lineno, chain[-1], kind, reason))
    return sorted(sites)


def _collect_tries(node: ast.AST, functions: list[str], out: list[tuple[ast.Try, list[str]]]) -> None:
    """Walk ``node``, recording each ``try`` with its enclosing function chain."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _collect_tries(child, [*functions, child.name], out)
        else:
            if isinstance(child, ast.Try):
                out.append((child, functions))
            _collect_tries(child, functions, out)


def _caught_types(handler: ast.ExceptHandler) -> tuple[str, ...]:
    """The types ``handler`` catches; a bare except catches everything, so it stands for BaseException."""
    if handler.type is None:
        return ("BaseException",)
    expressions = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    # Anything but a plain name is kept verbatim, so it resolves to nothing and is reported rather than passed.
    return tuple(node.id if isinstance(node, ast.Name) else ast.unparse(node) for node in expressions)


def _ancestor_names(name: str) -> frozenset[str] | None:
    """Every name ``name`` is or inherits from, or None when the chain leaves the builtins and ``mloda/``."""
    resolved: set[str] = set()
    pending = [name]
    while pending:
        current = pending.pop()
        if current in resolved:
            continue
        resolved.add(current)
        builtin = getattr(builtins, current, None)
        if isinstance(builtin, type):
            resolved |= {ancestor.__name__ for ancestor in builtin.__mro__}
            continue
        bases = _index().class_bases.get(current)
        if bases is None:
            return None
        pending.extend(bases)
    return frozenset(resolved)


def classify_shadowing(
    source: str, module: str, functions: frozenset[str] | None = None
) -> tuple[list[ShadowSite], list[UnresolvedType]]:
    """Find the clauses of one ``try`` that catch ahead of a later clause a mark survives, and the unresolved."""
    tree = ast.parse(source, filename=module)
    tries: list[tuple[ast.Try, list[str]]] = []
    _collect_tries(tree, [], tries)

    shadowed: set[ShadowSite] = set()
    unresolved: set[UnresolvedType] = set()
    for node, chain in tries:
        if not chain:
            continue
        if functions is not None and not any(name in functions for name in chain):
            continue
        # ast.Try alone reaches here: an except* group delivers to every matching clause, so none of them shadows
        # another, and a try with no clause has nothing to compare.
        if not node.handlers:
            continue
        function = chain[-1]
        for index, later in enumerate(node.handlers):
            if not _handler_escalates(later):
                continue
            for earlier in node.handlers[:index]:
                if _handler_escalates(earlier):
                    continue
                for caught in _caught_types(earlier):
                    ancestors = _ancestor_names(caught)
                    if ancestors is None:
                        unresolved.add(UnresolvedType(module, earlier.lineno, function, caught))
                        continue
                    for guarded in _caught_types(later):
                        guarded_ancestors = _ancestor_names(guarded)
                        if guarded_ancestors is None:
                            unresolved.add(UnresolvedType(module, later.lineno, function, guarded))
                        # Either direction hides the check: a subclass takes part of the later clause's
                        # traffic, a superclass takes all of it and leaves the clause dead.
                        elif guarded in ancestors or caught in guarded_ancestors:
                            shadowed.add(ShadowSite(module, earlier.lineno, function, caught, later.lineno, guarded))
    return sorted(shadowed), sorted(unresolved)


def _finally_discards(finalbody: list[ast.stmt]) -> bool:
    """The finally drops what is in flight: it leaves, or it raises a replacement over it."""
    if _escapes(finalbody):
        return True
    return any(
        isinstance(node, ast.Raise) and not _keeps_the_mark(node, None)
        for node in _runs_here(finalbody, _DEFERRED, shielded=True)
    )


def _enclosed_escalations(body: Sequence[ast.AST]) -> list[int]:
    """The line of every escalation leaving ``body``: a marked raise, or an escalating clause of a nested try."""
    lines: list[int] = []
    for node in body:
        if isinstance(node, _DEFERRED) or _shields_everything(node):
            continue
        if isinstance(node, ast.Raise):
            if node.exc is not None and _escalates(node.exc):
                lines.append(node.lineno)
            continue
        if isinstance(node, _NESTED_TRY):
            lines.extend(_nested_try_escalations(cast(ast.Try, node)))
            continue
        inside = [child for child in ast.iter_child_nodes(node) if not isinstance(child, ast.expr)]
        lines.extend(_enclosed_escalations(inside))
    return lines


def _escalations_the_finally_encloses(node: ast.Try) -> list[int]:
    """Its own escalating clauses, plus the escalations of the arms those clauses never catch."""
    lines = [clause.lineno for clause in node.handlers if _handler_escalates(clause)]
    lines.extend(_enclosed_escalations(node.orelse))
    if not node.handlers:
        lines.extend(_enclosed_escalations(node.body))
    return lines


def _nested_try_escalations(node: ast.Try) -> list[int]:
    """The arms _nested_try_reraises reads, named instead of counted; a finally of its own discards them first."""
    if _finally_discards(node.finalbody):
        return []
    return _escalations_the_finally_encloses(node) + _enclosed_escalations(node.finalbody)


def classify_finally_shadowing(source: str, module: str, functions: frozenset[str] | None = None) -> list[FinallySite]:
    """Find the ``try`` statements whose discarding ``finally`` undoes an escalation it encloses."""
    tree = ast.parse(source, filename=module)
    tries: list[tuple[ast.Try, list[str]]] = []
    _collect_tries(tree, [], tries)

    sites: list[FinallySite] = []
    for node, chain in tries:
        if not chain:
            continue
        if functions is not None and not any(name in functions for name in chain):
            continue
        if not _finally_discards(node.finalbody):
            continue
        # ast exposes no position for the finally keyword, so the try line carries the decision.
        sites.extend(
            FinallySite(module, node.lineno, chain[-1], lineno) for lineno in _escalations_the_finally_encloses(node)
        )
    return sorted(sites)


class _Index(NamedTuple):
    """Every definition under ``mloda/``, plus the names whose call can raise and the ones that can swallow."""

    functions: dict[str, list[_Definition]]
    constructors: dict[str, list[_Definition]]
    raising: frozenset[str]
    swallowing: frozenset[str]
    class_bases: dict[str, frozenset[str]]


def _raising_names(calls_per_name: dict[str, frozenset[str]], direct: frozenset[str]) -> frozenset[str]:
    """Names with the property directly or through anything they call: one clean wrapper defeats depth-1."""
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
    """Index ``mloda/`` by name: functions, the constructors a class name runs, and what a call can raise or drop."""
    functions: dict[str, list[_Definition]] = {}
    constructors: dict[str, list[_Definition]] = {}
    calls: dict[str, set[str]] = {}
    bases: dict[str, set[str]] = {}
    direct: set[str] = set()
    direct_swallowers: set[str] = set()
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
                if _swallows(node):
                    direct_swallowers.add(node.name)
            elif isinstance(node, ast.ClassDef):
                # A call on a CLASS name is an edge into what construction runs, which no call node names.
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in _CONSTRUCTORS:
                        constructors.setdefault(node.name, []).append(_Definition(module, child))
                # Keyed by name like everything else here, so two same-named classes pool their bases.
                bases.setdefault(node.name, set()).update(
                    base.id if isinstance(base, ast.Name) else ast.unparse(base) for base in node.bases
                )
    frozen = {name: frozenset(called) for name, called in calls.items()}
    return _Index(
        functions,
        constructors,
        _raising_names(frozen, frozenset(direct)),
        _raising_names(frozen, frozenset(direct_swallowers)),
        {name: frozenset(inherited) for name, inherited in bases.items()},
    )


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
                if target.node.name in index.swallowing:
                    swallowing.add(ExternalCall(target.module, target.node.name, caller))

    sites: list[RaiseSite] = []
    handlers: list[HandlerSite] = []
    shadowed: list[ShadowSite] = []
    unresolved: list[UnresolvedType] = []
    finally_shadowed: list[FinallySite] = []
    for module in sorted({module for module, _ in reachable}):
        names = frozenset(name for reached_module, name in reachable if reached_module == module)
        source = (_REPO_ROOT / module).read_text(encoding="utf-8")
        sites.extend(classify_raises(source, module, names))
        handlers.extend(classify_handlers(source, module, names))
        module_shadowed, module_unresolved = classify_shadowing(source, module, names)
        shadowed.extend(module_shadowed)
        unresolved.extend(module_unresolved)
        finally_shadowed.extend(classify_finally_shadowing(source, module, names))

    return _Sweep(
        tuple(sorted(sites)),
        frozenset(reachable),
        tuple(sorted(external)),
        tuple(sorted(handlers)),
        tuple(sorted(swallowing)),
        tuple(sorted(shadowed)),
        tuple(sorted(unresolved)),
        tuple(sorted(finally_shadowed)),
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


def test_the_reachable_walk_does_not_collapse() -> None:
    """One reached definition per module already satisfies the module canary, so pin the walk's size too."""
    reachable = sweep().reachable

    # Vacuity floor, not a target: 86 definitions today, so pruning a helper stays a legitimate change.
    assert len(reachable) >= 70, (
        f"the walk reached only {len(reachable)} definitions; it stopped following calls somewhere"
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
    # Vacuity floor, not a target: 19 sites today, so removing one stays a legitimate change.
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


def test_the_swallowing_closure_is_transitive() -> None:
    """One wrapper that swallows nothing itself must not hide the swallow two calls below it."""
    loader = "mloda/core/abstract_plugins/plugin_loader/plugin_loader.py"
    definitions = [definition for definition in _index().functions.get("load_group", []) if definition.module == loader]

    assert definitions and not any(_swallows(definition.node) for definition in definitions), (
        "load_group now swallows in its own body; pick another transitive-only entry of that chain"
    )

    pairs = {(call.module, call.function) for call in sweep().swallowing_external}

    assert (loader, "load_group") in pairs, (
        "get_all_filtered_subclasses calls load_group, which reaches _load_plugin, whose handler swallows. A "
        f"depth-1 swallow check sees none of it. Seen instead: {sorted(pairs)}"
    )


def test_known_escalating_handlers_are_enumerated() -> None:
    """Canary: the handlers that re-raise today are all seen, so a silently empty handler sweep cannot pass."""
    match_hook = "mloda/core/abstract_plugins/components/match_hook.py"
    feature_group = "mloda/core/abstract_plugins/feature_group.py"
    mixin = "mloda/core/abstract_plugins/components/feature_chainer/feature_chain_parser_mixin.py"
    guards = "mloda/core/abstract_plugins/components/feature_chainer/feature_chain_author_guards.py"
    expected = {
        (match_hook, "call_match_hook"),
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


# A raise only escalates when the object leaving the handler is the marked one, and nothing skips it.


def test_handler_classifier_flags_a_replacement_raise() -> None:
    """A brand-new exception object carries no marker, so raising it contains the abort."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except Exception:\n"
        "        raise ValueError('brand new')\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["unannotated"]
    assert sites[0].lineno == 4


def test_handler_classifier_flags_a_replacement_raise_chained_from_the_caught_name() -> None:
    """``from exc`` chains the traceback, not the marker: the object leaving the handler is still a new one."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except Exception as exc:\n"
        "        raise ValueError('brand new') from exc\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["unannotated"]


def test_handler_classifier_accepts_a_reraise_of_the_bound_name() -> None:
    """``raise exc`` re-raises the caught object itself, marker intact."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        record(exc)\n"
        "        raise exc\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["escalating"]


def test_handler_classifier_accepts_a_raise_of_a_freshly_escalated_exception() -> None:
    """Rewrapping is fine as long as the replacement is marked on its way out."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        raise escalate_match_abort(RuntimeError('rewrapped'))\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["escalating"]


def test_handler_classifier_flags_a_reraise_the_handler_can_skip() -> None:
    """A bare raise on one path and a return on another swallows whenever the return wins."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ImportError:\n"
        "        if options is None:\n"
        "            return None\n"
        "        raise\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["unannotated"]


# Nesting scopes it: a raise the handler never runs, and a return that never leaves it, are not the handler's.


def test_handler_classifier_flags_a_reraise_inside_a_nested_try() -> None:
    """A re-raise the nested try can catch again escalates nothing out of the handler holding it."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        try:\n"
        "            if is_match_abort(exc):\n"
        "                raise\n"
        "        except Exception:  # Swallows: the nested try catches the re-raise above it.\n"
        "            pass\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "unannotated"), (8, "swallowing")]


# A nested try only shields the arms its own handlers can catch; the rest still leaves the handler holding it.


def test_handler_classifier_counts_a_reraise_in_a_nested_finally() -> None:
    """A finally runs on the way out of the nested try, past the clauses that could have caught the re-raise."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        try:\n"
        "            cleanup()\n"
        "        except OSError:  # Swallows: cleanup is best effort here.\n"
        "            pass\n"
        "        finally:\n"
        "            if is_match_abort(exc):\n"
        "                raise\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "escalating"), (7, "swallowing")]


def test_handler_classifier_counts_a_reraise_in_a_nested_try_without_a_handler() -> None:
    """A try that only cleans up catches nothing, so the re-raise in its body leaves the handler around it."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        try:\n"
        "            if is_match_abort(exc):\n"
        "                raise\n"
        "        finally:\n"
        "            release()\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "escalating")]


def test_handler_classifier_counts_a_nested_handler_reraising_the_outer_bound_name() -> None:
    """``raise exc`` names the outer handler's own exception, which no clause of the nested try is catching."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        try:\n"
        "            cleanup()\n"
        "        except OSError:  # Swallows: the OSError is dropped; exc takes its place.\n"
        "            if is_match_abort(exc):\n"
        "                raise exc\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "escalating"), (7, "swallowing")]


def test_handler_classifier_flags_a_bare_reraise_in_a_nested_handler() -> None:
    """A bare raise there re-raises that clause's own exception, not the one the outer handler caught."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        try:\n"
        "            cleanup()\n"
        "        except OSError:\n"
        "            if is_match_abort(exc):\n"
        "                raise\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "unannotated"), (7, "escalating")]


def test_handler_classifier_flags_an_abort_check_gating_only_a_bare_reraise_in_a_nested_clause() -> None:
    """That bare raise re-raises the cleanup failure, so the check beside it gates nothing of the caught one."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        try:\n"
        "            cleanup()\n"
        "        except Exception:\n"
        "            if is_match_abort(exc):\n"
        "                raise\n"
        "        if options is None:\n"
        "            return False\n"
        "        raise exc\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "unannotated"), (7, "escalating")]


def test_handler_classifier_flags_a_handler_whose_abort_check_reads_another_exception() -> None:
    """The check gates the nested clause's own exception, so it credits nothing to the re-raise the return skips."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        try:\n"
        "            cleanup()\n"
        "        except Exception as inner:  # Swallows: a cleanup failure is not the candidate's verdict.\n"
        "            if is_match_abort(inner):\n"
        "                log(inner)\n"
        "        if options is None:\n"
        "            return False\n"
        "        raise exc\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "unannotated"), (7, "swallowing")]


def test_handler_classifier_flags_an_abort_check_reading_a_nested_exception_by_keyword() -> None:
    """The check reads the nested clause's own exception whichever argument form it is passed in."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        try:\n"
        "            cleanup()\n"
        "        except Exception as inner:  # Swallows: a cleanup failure is not the candidate's verdict.\n"
        "            if is_match_abort(exc=inner):\n"
        "                raise exc\n"
        "        if options is None:\n"
        "            return False\n"
        "        raise exc\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "unannotated"), (7, "swallowing")]


def test_handler_classifier_flags_an_abort_check_reading_an_attribute_of_a_nested_exception() -> None:
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        try:\n"
        "            cleanup()\n"
        "        except Exception as inner:  # Swallows: a cleanup failure is not the candidate's verdict.\n"
        "            if is_match_abort(inner.__cause__):\n"
        "                raise exc\n"
        "        if options is None:\n"
        "            return False\n"
        "        raise exc\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "unannotated"), (7, "swallowing")]


def test_handler_classifier_counts_an_abort_check_a_later_nested_clause_rebinds_over() -> None:
    """The rebinding begins at the nested clause, so it cannot cancel the check that already ran above it."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "        try:\n"
        "            cleanup()\n"
        "        except Exception as exc:  # Swallows: a cleanup failure is not the candidate's verdict.\n"
        "            log(exc)\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "escalating"), (9, "swallowing")]


def test_handler_classifier_counts_a_reraise_in_a_nested_else() -> None:
    """The else runs only once the nested try body has passed, so its clauses never see the re-raise."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        try:\n"
        "            cleanup()\n"
        "        except OSError:  # Swallows: cleanup is best effort here.\n"
        "            pass\n"
        "        else:\n"
        "            if is_match_abort(exc):\n"
        "                raise\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "escalating"), (7, "swallowing")]


def test_handler_classifier_flags_a_reraise_a_nested_finally_discards() -> None:
    """A return in the nested finally drops whatever is in flight, so the re-raise above it leaves nothing."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        try:\n"
        "            if is_match_abort(exc):\n"
        "                raise\n"
        "        finally:\n"
        "            return False\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "unannotated"), (5, "unannotated")]


def test_handler_classifier_flags_a_reraise_inside_a_nested_def() -> None:
    """A re-raise in a function defined by the handler runs later or never, so the handler still swallows."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        def escalate_later():\n"
        "            if is_match_abort(exc):\n"
        "                raise exc\n"
        "        register(escalate_later)\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "unannotated")]


def test_handler_classifier_flags_an_abort_check_that_only_decorates_a_nested_class() -> None:
    """A check spent on a decorator gates no re-raise, so the return below it still skips the one at the end."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        @is_match_abort(exc)\n"
        "        class Report:\n"
        "            pass\n"
        "        if options is None:\n"
        "            return False\n"
        "        raise exc\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "unannotated")]


def test_handler_classifier_counts_an_abort_check_in_the_default_of_a_nested_def() -> None:
    """A default argument is evaluated where the def stands, so that check runs in the handler like any other."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        def report(marked=is_match_abort(exc)):\n"
        "            return marked\n"
        "        if options is None:\n"
        "            return False\n"
        "        raise exc\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "escalating")]


def test_handler_classifier_counts_a_return_inside_a_nested_try() -> None:
    """A return leaves the function from any depth, so it still skips the re-raise below it."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ImportError:\n"
        "        try:\n"
        "            return probe(options)\n"
        "        except OSError:  # Swallows: a missing file is a non-match for this reader only.\n"
        "            pass\n"
        "        raise\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "unannotated"), (7, "swallowing")]


def test_handler_classifier_ignores_a_return_inside_a_nested_def() -> None:
    """A return in a function the handler defines leaves that function, so the raise still stands on every path."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ImportError:\n"
        "        def fallback():\n"
        "            return None\n"
        "        register(fallback)\n"
        "        raise\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "escalating")]


def test_handler_classifier_ignores_a_break_bound_to_a_loop_it_opens() -> None:
    """That break leaves the for, not the handler, so the raise below it still stands on every path out."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError:\n"
        "        for candidate in candidates:\n"
        "            if ready(candidate):\n"
        "                break\n"
        "        raise\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "escalating")]


def test_handler_classifier_counts_a_break_that_leaves_the_handler() -> None:
    """The loop is around the try, so the break leaves the handler and skips the re-raise below it."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    for candidate in options:\n"
        "        try:\n"
        "            return probe(candidate)\n"
        "        except ValueError:\n"
        "            if options is None:\n"
        "                break\n"
        "            raise\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(5, "unannotated")]


def test_handler_classifier_counts_a_break_in_the_else_arm_of_a_loop_it_opens() -> None:
    """The else arm sits outside its own loop, so that break binds to the loop around the try and leaves."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    for candidate in options:\n"
        "        try:\n"
        "            return probe(candidate)\n"
        "        except ValueError:\n"
        "            for attempt in retries:\n"
        "                probe(attempt)\n"
        "            else:\n"
        "                break\n"
        "            raise\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(5, "unannotated")]


def test_handler_classifier_flags_a_reraise_in_the_else_arm_a_break_skips() -> None:
    """A break leaves the loop past its else arm, so the raise written there stands on one path only."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        for attempt in retries:\n"
        "            if retry(feature):\n"
        "                break\n"
        "        else:\n"
        "            raise exc\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "unannotated")]


def test_handler_classifier_flags_a_reraise_a_break_in_the_same_loop_body_skips() -> None:
    """The break leaves the loop before the raise below it in the body is ever reached."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        for attempt in retries:\n"
        "            if ready(attempt):\n"
        "                break\n"
        "            raise exc\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "unannotated")]


def test_handler_classifier_flags_a_reraise_in_a_loop_body_that_may_never_run() -> None:
    """An empty iterable runs the body zero times, so a raise written only there stands on no path out."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        for attempt in retries:\n"
        "            raise exc\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "unannotated")]


# ``except`` is not the only way to drop an exception: suppress() and an escaping finally do it too.


def test_handler_classifier_flags_a_contextlib_suppress() -> None:
    """``suppress`` discards the exception exactly as a bare except does, so it owes the same decision."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    with contextlib.suppress(Exception):\n"
        "        return probe(feature)\n"
        "    return False\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["unannotated"]
    assert sites[0].lineno == 2


def test_handler_classifier_accepts_a_swallows_comment_on_a_suppress() -> None:
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    with suppress(ValueError):  # Swallows: a malformed name is this candidate's own defect.\n"
        "        return probe(feature)\n"
        "    return False\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["swallowing"]
    assert sites[0].reason == "a malformed name is this candidate's own defect."


def test_handler_classifier_flags_a_reraise_inside_a_suppress_block() -> None:
    """The suppress drops the re-raise written in it, so the handler around it lets nothing out."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        with suppress(Exception):  # Swallows: cleanup is best effort here.\n"
        "            if is_match_abort(exc):\n"
        "                raise\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "unannotated"), (5, "swallowing")]


def test_handler_classifier_counts_a_reraise_inside_a_narrow_suppress() -> None:
    """suppress(KeyError) never catches the re-raised ValueError, so it does leave the handler."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        with suppress(KeyError):  # Swallows: a missing key is this candidate's own defect.\n"
        "            if is_match_abort(exc):\n"
        "                raise\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "escalating"), (5, "swallowing")]


def test_handler_classifier_counts_an_abort_check_gating_a_return_inside_a_suppress() -> None:
    """A suppress drops what raises in it, not what returns from it, so the check there still gates the escape."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except ValueError as exc:\n"
        "        with suppress(Exception):  # Swallows: cleanup is best effort here.\n"
        "            if not is_match_abort(exc):\n"
        "                return False\n"
        "        raise\n"
    )

    sites = _handlers(source)

    assert [(site.lineno, site.kind) for site in sites] == [(4, "escalating"), (5, "swallowing")]


def test_handler_classifier_flags_a_finally_that_returns() -> None:
    """A return in finally discards the in-flight exception, marker and all."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    finally:\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["unannotated"]
    # Anchored at the try line: ast.Try carries no lineno for the finally keyword.
    assert sites[0].lineno == 2


def test_handler_classifier_accepts_a_swallows_comment_on_a_finally_that_returns() -> None:
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    # Swallows: the cleanup verdict outranks the probe's, by this reader's contract.\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    finally:\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["swallowing"]
    assert sites[0].reason == "the cleanup verdict outranks the probe's, by this reader's contract."


# An own-line reason belongs to the site at its own column; a trailing one may sit anywhere in the site's header.


def test_handler_classifier_ignores_a_swallows_comment_indented_past_the_except() -> None:
    """A reason written as the last line of the try body sits deeper than the except and annotates nothing."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        probe(feature)\n"
        "        # Swallows: this belongs to the try body, not to the handler under it.\n"
        "    except ValueError:\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["unannotated"]


def test_handler_classifier_ignores_a_swallows_comment_left_in_an_earlier_handler_body() -> None:
    """A reason trailing the previous handler's body is indented past the next except, so it cannot annotate it."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except OSError:  # Swallows: a missing file is a non-match for this reader only.\n"
        "        return False\n"
        "        # Swallows: this trails the OSError body; the next handler needs its own reason.\n"
        "    except ValueError:\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["swallowing", "unannotated"]


def test_handler_classifier_accepts_a_trailing_comment_on_a_split_exception_tuple() -> None:
    """ruff format pushes the tag onto the closing-paren line, past the except's own lineno."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except (\n"
        "        ValueError,\n"
        "        TypeError,\n"
        "    ):  # Swallows: a malformed name is this candidate's own defect.\n"
        "        return False\n"
    )

    sites = _handlers(source)

    assert [site.kind for site in sites] == ["swallowing"]
    assert sites[0].lineno == 4
    assert sites[0].reason == "a malformed name is this candidate's own defect."


def _splice_blanket_handler(source: str, function: str) -> tuple[str, int]:
    """Insert a blanket handler at the top of ``function``'s body; in memory only, never written back to disk."""
    targets = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function
    ]
    assert len(targets) == 1, f"{function} is not a single definition in this module: {len(targets)} found"
    first = targets[0].body[0]
    pad = " " * first.col_offset
    block = [f"{pad}try:", f"{pad}    probe()", f"{pad}except Exception:", f"{pad}    return False"]
    lines = source.splitlines()
    at = first.lineno - 1
    return "\n".join([*lines[:at], *block, *lines[at:]]) + "\n", first.lineno + 2


def test_the_sweep_flags_a_blanket_handler_spliced_into_the_real_seam() -> None:
    """End to end: the real reachable set and the classifier on real source, mutated in memory only."""
    module = "mloda/core/prepare/identify_feature_group.py"
    function = "_filter_feature_group_by_criteria"
    names = frozenset(name for reached_module, name in sweep().reachable if reached_module == module)
    assert function in names, f"{function} is no longer reachable; splice into another reached definition"

    source = (_REPO_ROOT / module).read_text(encoding="utf-8")
    assert [site for site in classify_handlers(source, module, names) if site.kind == "unannotated"] == []

    mutated, lineno = _splice_blanket_handler(source, function)
    spliced = [site for site in classify_handlers(mutated, module, names) if site.lineno == lineno]

    assert [site.kind for site in spliced] == ["unannotated"], (
        f"a blanket handler spliced into the seam itself was not flagged: {spliced}"
    )
    assert spliced[0].function == function


# The clauses of one try, judged against each other: each looks well-formed alone, but the first match wins.


def _shadowing(source: str) -> tuple[list[ShadowSite], list[UnresolvedType]]:
    return classify_shadowing(source, "snippet.py", frozenset({"match_feature_group_criteria"}))


def test_no_handler_shadows_an_abort_check_on_the_same_try() -> None:
    shadowed = list(sweep().shadowed)

    assert shadowed == [], (
        "Handlers that catch ahead of a clause below them a marked exception survives:\n"
        + "\n".join(
            f"  {site.location()}: except {site.caught} shadows the {site.shadowed_type} clause on line "
            f"{site.shadowed_lineno}"
            for site in shadowed
        )
        + "\n\nPython runs the first matching clause, so the check below never sees the marked exception. Run "
        f"{_ABORT_CHECK} in the earlier clause too, or make the two catch disjoint types."
    )


def test_every_except_type_beside_an_escalating_clause_resolves() -> None:
    """An unresolvable type decides nothing, so it is reported here instead of passing the shadow check."""
    unresolved = list(sweep().unresolved)

    assert unresolved == [], (
        "Handler types on a try with an escalating clause whose base chain leaves the builtins and the classes "
        "under mloda/, so whether they shadow it is unknown:\n"
        + "\n".join(f"  {site.location()}: {site.name}" for site in unresolved)
    )


def test_shadow_classifier_flags_a_narrow_handler_above_the_abort_check() -> None:
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except KeyError:  # Swallows: a missing key is this candidate's own defect.\n"
        "        return False\n"
        "    except Exception as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "        return False\n"
    )

    shadowed, unresolved = _shadowing(source)

    assert unresolved == []
    assert [(site.lineno, site.caught, site.shadowed_lineno, site.shadowed_type) for site in shadowed] == [
        (4, "KeyError", 6, "Exception")
    ]


def test_shadow_classifier_flags_a_broad_handler_above_the_abort_check() -> None:
    """The reverse inheritance direction is worse, not safe: the clause below it can never run at all."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except Exception:  # Swallows: a broken probe is this candidate's own defect.\n"
        "        return False\n"
        "    except ValueError as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "        return False\n"
    )

    shadowed, unresolved = _shadowing(source)

    assert unresolved == []
    assert [(site.caught, site.shadowed_type) for site in shadowed] == [("Exception", "ValueError")]


def test_shadow_classifier_flags_a_handler_whose_only_reraise_sits_in_a_nested_try() -> None:
    """The earlier clause checks the marker for its own cleanup, not for the exception the clause below it wants."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except KeyError:\n"
        "        try:\n"
        "            cleanup()\n"
        "        except Exception as inner:\n"
        "            if is_match_abort(inner):\n"
        "                raise\n"
        "        return False\n"
        "    except Exception as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "        return False\n"
    )

    shadowed, unresolved = _shadowing(source)

    assert unresolved == []
    assert [(site.lineno, site.caught, site.shadowed_lineno, site.shadowed_type) for site in shadowed] == [
        (4, "KeyError", 11, "Exception")
    ]


def test_shadow_classifier_flags_a_blanket_handler_above_a_check_wrapped_in_a_nested_try() -> None:
    """The clause below only wraps its re-raise in cleanup, so the blanket clause above still hides the check."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except Exception:  # Swallows: a broken probe is this candidate's own defect.\n"
        "        return False\n"
        "    except ValueError as exc:\n"
        "        try:\n"
        "            if is_match_abort(exc):\n"
        "                raise\n"
        "        finally:\n"
        "            release()\n"
        "        return False\n"
    )

    shadowed, unresolved = _shadowing(source)

    assert unresolved == []
    assert [(site.lineno, site.caught, site.shadowed_lineno, site.shadowed_type) for site in shadowed] == [
        (4, "Exception", 6, "ValueError")
    ]


def test_shadow_classifier_resolves_a_project_exception_through_its_declared_base() -> None:
    """The shape fixed in match_parser_criteria: PropertyValueRejection subclasses ValueError statically."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except PropertyValueRejection:  # Swallows: the parser's non-match verdict.\n"
        "        return False\n"
        "    except ValueError as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "        return False\n"
    )

    shadowed, unresolved = _shadowing(source)

    assert unresolved == []
    assert [(site.caught, site.shadowed_type) for site in shadowed] == [("PropertyValueRejection", "ValueError")]


def test_shadow_classifier_accepts_an_earlier_handler_that_runs_the_abort_check() -> None:
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except PropertyValueRejection as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "        return False\n"
        "    except ValueError as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "        return False\n"
    )

    assert _shadowing(source) == ([], [])


def test_shadow_classifier_accepts_an_earlier_handler_of_an_unrelated_type() -> None:
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except KeyError:  # Swallows: a missing key is this candidate's own defect.\n"
        "        return False\n"
        "    except ValueError as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "        return False\n"
    )

    assert _shadowing(source) == ([], [])


def test_shadow_classifier_reads_every_type_of_an_earlier_tuple() -> None:
    """One shadowing member is enough; the clause catches on any of them."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except (KeyError, TypeError):  # Swallows: this candidate's own defect.\n"
        "        return False\n"
        "    except TypeError as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "        return False\n"
    )

    shadowed, unresolved = _shadowing(source)

    assert unresolved == []
    assert [(site.caught, site.shadowed_type) for site in shadowed] == [("TypeError", "TypeError")]


def test_shadow_classifier_flags_a_narrow_handler_above_a_bare_except() -> None:
    """A bare except catches everything, so every earlier clause shadows the re-raise it holds."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except KeyError:  # Swallows: a missing key is this candidate's own defect.\n"
        "        return False\n"
        "    except:\n"
        "        raise\n"
    )

    shadowed, unresolved = _shadowing(source)

    assert unresolved == []
    assert [(site.caught, site.shadowed_type) for site in shadowed] == [("KeyError", "BaseException")]


def test_shadow_classifier_reports_a_type_it_cannot_resolve() -> None:
    """A dotted type resolves to no static base chain, so it is reported rather than read as no shadow."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except third_party.Weird:  # Swallows: not ours to judge.\n"
        "        return False\n"
        "    except Exception as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "        return False\n"
    )

    shadowed, unresolved = _shadowing(source)

    assert shadowed == []
    assert [(site.lineno, site.name) for site in unresolved] == [(4, "third_party.Weird")]


def _splice_narrow_handler(source: str, function: str) -> tuple[str, int]:
    """Insert a narrow handler above ``function``'s first escalating clause; in memory only, never written back."""
    targets = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function
    ]
    assert len(targets) == 1, f"{function} is not a single definition in this module: {len(targets)} found"
    escalating = [
        handler
        for node in ast.walk(targets[0])
        if isinstance(node, ast.Try)
        for handler in node.handlers
        if _handler_escalates(handler)
    ]
    assert escalating, f"{function} holds no escalating handler to shadow; splice into another definition"
    first = escalating[0]
    pad = " " * first.col_offset
    block = [f"{pad}except KeyError:", f"{pad}    return False"]
    lines = source.splitlines()
    at = first.lineno - 1
    return "\n".join([*lines[:at], *block, *lines[at:]]) + "\n", first.lineno


def test_the_sweep_flags_a_narrow_handler_spliced_above_the_real_seam_check() -> None:
    """End to end: the real reachable set and the classifier on real source, mutated in memory only."""
    module = "mloda/core/abstract_plugins/components/match_hook.py"
    function = "call_match_hook"
    names = frozenset(name for reached_module, name in sweep().reachable if reached_module == module)
    assert function in names, f"{function} is no longer reachable; splice into another reached definition"

    source = (_REPO_ROOT / module).read_text(encoding="utf-8")
    assert classify_shadowing(source, module, names) == ([], [])

    mutated, lineno = _splice_narrow_handler(source, function)
    spliced, unresolved = classify_shadowing(mutated, module, names)

    assert unresolved == []
    assert [(site.lineno, site.caught, site.shadowed_type) for site in spliced] == [
        (lineno, "KeyError", "Exception")
    ], f"a narrow handler spliced above the seam's own abort check was not flagged: {spliced}"
    assert spliced[0].function == function


# One try again, the other pairing: a finally that discards drops every escalation it encloses.


def _finally_shadowing(source: str) -> list[FinallySite]:
    return classify_finally_shadowing(source, "snippet.py", frozenset({"match_feature_group_criteria"}))


def test_no_finally_discards_an_abort_check_on_the_same_try() -> None:
    discarded = list(sweep().finally_shadowed)

    assert discarded == [], (
        "Try statements whose finally discards an escalation the try around it would otherwise let out:\n"
        + "\n".join(
            f"  {site.location()}: the finally discards the escalation on line {site.discarded_lineno}"
            for site in discarded
        )
        + "\n\nA return, break or continue leaving finally, or a raise replacing what is in flight, drops the marked "
        f"exception the {_ABORT_CHECK} clause re-raised, so it never leaves the function. Let the finally fall off "
        "its end, or drop the escalation and declare the swallow at its site."
    )


def test_finally_classifier_flags_a_returning_finally_beside_an_abort_check() -> None:
    """The return in finally discards the marked exception the clause above it deliberately re-raised."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    # Swallows: the cleanup verdict outranks the probe's, by this reader's contract.\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except Exception as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "    finally:\n"
        "        return False\n"
    )

    sites = _finally_shadowing(source)

    # Anchored at the try line: ast exposes no position for the finally keyword.
    assert sites == [FinallySite("snippet.py", 3, "match_feature_group_criteria", 5)]
    assert sites[0].location() == "snippet.py:3 match_feature_group_criteria()"


def test_the_earlier_passes_clear_a_finally_that_discards_the_reraise() -> None:
    """Both sites are well-formed alone, which is why the pairing needs a pass of its own."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    # Swallows: the cleanup verdict outranks the probe's, by this reader's contract.\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except Exception as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "    finally:\n"
        "        return False\n"
    )

    assert [(site.lineno, site.kind) for site in _handlers(source)] == [(3, "swallowing"), (5, "escalating")]
    assert _shadowing(source) == ([], [])


def test_finally_classifier_flags_a_break_leaving_the_finally() -> None:
    """A break discards the exception in flight exactly as a return does, and so does a continue."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    for candidate in options:\n"
        "        try:\n"
        "            return probe(candidate)\n"
        "        except Exception as exc:\n"
        "            if is_match_abort(exc):\n"
        "                raise\n"
        "        finally:\n"
        "            break\n"
        "    return False\n"
    )

    assert [(site.lineno, site.discarded_lineno) for site in _finally_shadowing(source)] == [(3, 5)]


def test_finally_classifier_ignores_a_returning_finally_beside_a_swallowing_handler() -> None:
    """No clause re-raises here, so the try is the plain finally-swallow site; reporting it twice is noise."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except Exception:\n"
        "        return False\n"
        "    finally:\n"
        "        return False\n"
    )

    assert _finally_shadowing(source) == []


def test_finally_classifier_ignores_a_finally_that_does_not_escape() -> None:
    """A plain cleanup call leaves the exception in flight, so the re-raise above it still leaves the function."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except Exception as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "        return False\n"
        "    finally:\n"
        "        release()\n"
    )

    assert _finally_shadowing(source) == []


def test_finally_classifier_ignores_a_discarding_finally_with_no_escalation() -> None:
    """A handler-less try is read through its body now, so the empty case is the one holding nothing marked."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        # Contained: a missing feature is this candidate's own defect.\n"
        "        raise ValueError('no feature')\n"
        "    finally:\n"
        "        return False\n"
    )

    assert _finally_shadowing(source) == []


# What the finally encloses is what it discards: hoisting the try, or moving the raise, changes nothing.


def test_finally_classifier_flags_a_finally_hoisted_to_the_try_around_the_check() -> None:
    """Wrapping the flagged try in one that owns the finally is the refactor the gate invites, and it discards
    the same re-raise."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        # Swallows: the cleanup verdict outranks the probe's, by this reader's contract.\n"
        "        try:\n"
        "            return probe(feature)\n"
        "        except Exception as exc:\n"
        "            if is_match_abort(exc):\n"
        "                raise\n"
        "    finally:\n"
        "        return False\n"
    )

    assert _finally_shadowing(source) == [FinallySite("snippet.py", 2, "match_feature_group_criteria", 6)]


def test_finally_classifier_flags_a_marked_raise_in_the_body_of_a_discarding_try() -> None:
    """A clause is not the only escalation a finally can undo: with no clause to catch it, the raise is its own."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        if options is None:\n"
        "            raise escalate_match_abort(ValueError('no options'))\n"
        "        return probe(feature)\n"
        "    finally:\n"
        "        return False\n"
    )

    assert _finally_shadowing(source) == [FinallySite("snippet.py", 2, "match_feature_group_criteria", 4)]


def test_finally_classifier_ignores_a_marked_raise_a_suppress_in_the_body_swallows() -> None:
    """That raise never leaves the suppress, so the finally around it has no escalation to undo."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        with suppress(Exception):  # Swallows: the probe is best effort here.\n"
        "            if options is None:\n"
        "                raise escalate_match_abort(ValueError('no options'))\n"
        "        return probe(feature)\n"
        "    finally:\n"
        "        return False\n"
    )

    assert _finally_shadowing(source) == []


def test_finally_classifier_flags_a_marked_raise_a_narrow_suppress_does_not_catch() -> None:
    """suppress(KeyError) lets the marked ValueError past, so the discarding finally is what drops it."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        with suppress(KeyError):  # Swallows: a missing key is this candidate's own defect.\n"
        "            raise escalate_match_abort(ValueError('no options'))\n"
        "        return probe(feature)\n"
        "    finally:\n"
        "        return False\n"
    )

    assert _finally_shadowing(source) == [FinallySite("snippet.py", 2, "match_feature_group_criteria", 4)]


def test_finally_classifier_reads_past_a_nested_try_to_the_marked_raise_beside_it() -> None:
    """The nested try's clause decides its own body only; the raise beside it still reaches the finally."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        try:\n"
        "            release()\n"
        "        except OSError:  # Swallows: cleanup is best effort here.\n"
        "            pass\n"
        "        raise escalate_match_abort(ValueError('no options'))\n"
        "    finally:\n"
        "        return False\n"
    )

    assert _finally_shadowing(source) == [FinallySite("snippet.py", 2, "match_feature_group_criteria", 7)]


def test_finally_classifier_flags_a_marked_raise_in_the_else_arm() -> None:
    """The else runs once the body has passed, so its own clauses never see the raise and only the finally is left."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        probe(feature)\n"
        "    except OSError:  # Swallows: a missing file is a non-match for this reader only.\n"
        "        return False\n"
        "    else:\n"
        "        raise escalate_match_abort(ValueError('no options'))\n"
        "    finally:\n"
        "        return False\n"
    )

    assert _finally_shadowing(source) == [FinallySite("snippet.py", 2, "match_feature_group_criteria", 7)]


def test_finally_classifier_reports_every_escalating_clause_of_one_discarding_try() -> None:
    """One site per escalation, not one per try: the finally discards both clauses, so both are named."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except KeyError as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "    except ValueError as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "    finally:\n"
        "        return False\n"
    )

    assert [(site.lineno, site.discarded_lineno) for site in _finally_shadowing(source)] == [(2, 4), (2, 7)]


def test_a_swallows_comment_on_the_try_does_not_silence_the_finally_pass() -> None:
    """The reason answers the handler pass alone: this pass reads no comments, and the clause still re-raises."""
    header = "def match_feature_group_criteria(cls, feature, options):\n"
    reason = "    # Swallows: the cleanup verdict outranks the probe's, by this reader's contract.\n"
    body = (
        "    try:\n"
        "        return probe(feature)\n"
        "    except Exception as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "    finally:\n"
        "        return False\n"
    )

    assert _finally_shadowing(header + body) == [FinallySite("snippet.py", 2, "match_feature_group_criteria", 4)]
    # The same site, one line down: the reason changes the line numbers and nothing else.
    assert _finally_shadowing(header + reason + body) == [
        FinallySite("snippet.py", 3, "match_feature_group_criteria", 5)
    ]


# Discarding is what counts, so a return or break that never leaves the finally discards nothing.


def test_finally_classifier_ignores_a_break_bound_to_a_loop_inside_the_finally() -> None:
    """That break leaves the while, not the finally, so what is in flight is still in flight after it."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except Exception as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "    finally:\n"
        "        while pending(feature):\n"
        "            if drained(feature):\n"
        "                break\n"
        "            drain(feature)\n"
    )

    assert _finally_shadowing(source) == []
    # The same reading feeds the handler pass, which would otherwise demand a reason for a finally that drops nothing.
    assert [(site.lineno, site.kind) for site in _handlers(source)] == [(4, "escalating")]


def test_finally_classifier_ignores_a_return_inside_a_def_in_the_finally() -> None:
    """That return leaves fallback(), not the function, so it discards nothing the finally is holding."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except Exception as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "    finally:\n"
        "        def fallback():\n"
        "            return None\n"
        "        register(fallback)\n"
    )

    assert _finally_shadowing(source) == []
    assert [(site.lineno, site.kind) for site in _handlers(source)] == [(4, "escalating")]


# Leaving is not the only way out of a finally: a raise there replaces what is in flight, unless it is a bare one.


def test_finally_classifier_flags_a_finally_that_raises_a_replacement() -> None:
    """The replacement carries no marker, and a '# Contained:' reason answers only the raise pass."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except Exception as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "    finally:\n"
        "        # Contained: a failed cleanup is this candidate's own defect.\n"
        "        raise ValueError('cleanup failed')\n"
    )

    assert _finally_shadowing(source) == [FinallySite("snippet.py", 2, "match_feature_group_criteria", 4)]


def test_finally_classifier_ignores_a_bare_raise_in_the_finally() -> None:
    """A bare raise re-raises what is in flight, marker and all, so it replaces nothing."""
    source = (
        "def match_feature_group_criteria(cls, feature, options):\n"
        "    try:\n"
        "        return probe(feature)\n"
        "    except Exception as exc:\n"
        "        if is_match_abort(exc):\n"
        "            raise\n"
        "    finally:\n"
        "        if failed(feature):\n"
        "            raise\n"
    )

    assert _finally_shadowing(source) == []


def _splice_returning_finally(source: str, function: str) -> tuple[str, int, int]:
    """Insert a return at the top of the finally beside ``function``'s escalating clause; in memory only."""
    targets = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function
    ]
    assert len(targets) == 1, f"{function} is not a single definition in this module: {len(targets)} found"
    tries = [
        node
        for node in ast.walk(targets[0])
        if isinstance(node, ast.Try)
        if node.finalbody and any(_handler_escalates(clause) for clause in node.handlers)
    ]
    assert tries, f"{function} holds no finally beside an escalating clause; splice into another definition"
    node = tries[0]
    clause = next(clause for clause in node.handlers if _handler_escalates(clause))
    first = node.finalbody[0]
    lines = source.splitlines()
    at = first.lineno - 1
    block = f"{' ' * first.col_offset}return False"
    return "\n".join([*lines[:at], block, *lines[at:]]) + "\n", node.lineno, clause.lineno


def test_the_sweep_flags_a_returning_finally_spliced_into_the_real_seam() -> None:
    """End to end: the real reachable set and the classifier on real source, mutated in memory only."""
    module = "mloda/core/prepare/identify_feature_group.py"
    function = "_filter_feature_group_by_criteria"
    names = frozenset(name for reached_module, name in sweep().reachable if reached_module == module)
    assert function in names, f"{function} is no longer reachable; splice into another reached definition"

    source = (_REPO_ROOT / module).read_text(encoding="utf-8")
    assert classify_finally_shadowing(source, module, names) == []

    mutated, lineno, discarded = _splice_returning_finally(source, function)
    spliced = classify_finally_shadowing(mutated, module, names)

    assert spliced == [FinallySite(module, lineno, function, discarded)], (
        f"a return spliced into the seam's own finally was not flagged: {spliced}"
    )
