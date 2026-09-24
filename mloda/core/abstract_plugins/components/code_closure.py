"""AST canonicalization and reachable-code closure for feature-group version hashing.

``canonical_dump`` turns an AST subtree into a deterministic string that
ignores docstrings, comments, formatting and source positions. ``closure_parts``
walks the first-party code a class can reach, at function/class/constant granularity.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib.metadata
import importlib.util
import inspect
import os
import re
import sys
import types
import warnings
import weakref
from dataclasses import dataclass
from typing import Any, Final

from mloda.core.abstract_plugins.components.utils import safe_field, safe_field_with_error

_NONE_PLACEHOLDER = "none"


class _Frame:
    """One pending node or list being serialized; a manual stack frame."""

    __slots__ = ("is_list", "type_name", "field_names", "items", "results", "idx", "parent", "slot")

    def __init__(self, value: Any, parent: "_Frame | None", slot: int) -> None:
        self.parent = parent
        self.slot = slot
        self.idx = 0
        if isinstance(value, list):
            self.is_list = True
            self.type_name = ""
            self.field_names: list[str] = []
            self.items: list[Any] = value
        else:
            self.is_list = False
            self.type_name = type(value).__name__
            field_names: list[str] = []
            items: list[Any] = []
            for name in value._fields:
                raw = getattr(value, name, None)
                if name == "body" and isinstance(raw, list):
                    raw = [stmt for stmt in raw if not _is_bare_string_statement(stmt)]
                if raw is None:
                    continue
                if isinstance(raw, list) and len(raw) == 0:
                    continue
                field_names.append(name)
                items.append(raw)
            self.field_names = field_names
            self.items = items
        self.results: list[str] = [""] * len(self.items)

    def finalize(self) -> str:
        if self.is_list:
            return "[" + ",".join(self.results) + "]"
        parts = (f"{name}={value}" for name, value in zip(self.field_names, self.results))
        return self.type_name + "(" + ",".join(parts) + ")"


def _is_bare_string_statement(stmt: Any) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def _scalar_dump(value: Any) -> str:
    if isinstance(value, str):
        return "s:" + value.encode("utf-8", "surrogatepass").hex()
    if type(value) is bool:
        return "bool:" + repr(value)
    if isinstance(value, int):
        # format(..., "x") is a linear bit-shift conversion, unlike str()/repr()
        # on int, which is limited (sys.set_int_max_str_digits) for huge values.
        return "i:" + format(value, "x")
    return type(value).__name__ + ":" + repr(value)


def canonical_dump(node: Any) -> str:
    """Serializes an AST node/list into a deterministic string.

    Iterative (explicit stack): a flat expression thousands of terms deep does not raise ``RecursionError``.
    """
    root = _Frame(node, parent=None, slot=0)
    stack: list[_Frame] = [root]
    result = ""
    while stack:
        frame = stack[-1]
        if frame.idx < len(frame.items):
            value = frame.items[frame.idx]
            if value is None:
                frame.results[frame.idx] = _NONE_PLACEHOLDER
                frame.idx += 1
            elif isinstance(value, (ast.AST, list)):
                stack.append(_Frame(value, parent=frame, slot=frame.idx))
            else:
                frame.results[frame.idx] = _scalar_dump(value)
                frame.idx += 1
        else:
            text = frame.finalize()
            stack.pop()
            if stack:
                parent = stack[-1]
                parent.results[frame.slot] = text
                parent.idx += 1
            else:
                result = text
    return result


def _digest(node: ast.AST) -> str:
    return hashlib.sha256(canonical_dump(node).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Ref collection: bare Name loads and full dotted Name.attr1.attr2... chains.
# One explicit-stack pass: a maximal Attribute chain is consumed as soon as
# it is found, so its member nodes are never independently re-visited.
# ---------------------------------------------------------------------------


def _dotted_chain(node: ast.Attribute) -> str | None:
    parts: list[str] = [node.attr]
    current: ast.expr = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name) and isinstance(current.ctx, ast.Load):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _collect_refs(node: ast.AST) -> list[str]:
    """Bare ``Name`` loads and maximal dotted ``Name.attr1.attr2...`` chains within node."""
    refs: set[str] = set()
    stack: list[ast.AST] = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, ast.Attribute) and isinstance(current.ctx, ast.Load):
            chain = _dotted_chain(current)
            if chain is not None:
                refs.add(chain)
            else:
                stack.append(current.value)
            continue
        if isinstance(current, ast.Name) and isinstance(current.ctx, ast.Load):
            refs.add(current.id)
            continue
        stack.extend(ast.iter_child_nodes(current))
    return sorted(refs)


# ---------------------------------------------------------------------------
# Module index: definitions by qualname, bindings by name, imports by name.
# Built once per module object from a single parse. Digest and refs per
# candidate are computed lazily, on first reach by the walk, and memoized;
# most of a module's definitions are never reached and never canonicalized.
# ---------------------------------------------------------------------------

_MODULE_INDEX_READ_ERRORS: Final = (OSError, ImportError, SyntaxError, ValueError, RecursionError, MemoryError)
_WALK_READ_ERRORS: Final = _MODULE_INDEX_READ_ERRORS + (TypeError,)


class _Candidate:
    """One definition/binding occurrence; span (not the AST) is kept, re-parsed and memoized on first reach."""

    __slots__ = ("kind", "start_line", "end_line", "lineno", "col_offset", "lines", "digest_refs")

    def __init__(self, kind: str, node: ast.stmt, lines: list[str]) -> None:
        self.kind = kind
        decorators = getattr(node, "decorator_list", [])
        self.start_line = decorators[0].lineno if decorators else node.lineno
        self.end_line = node.end_lineno if node.end_lineno is not None else self.start_line
        self.lineno = node.lineno
        self.col_offset = node.col_offset
        self.lines = lines
        self.digest_refs: tuple[str | None, list[str]] | None = None


@dataclass(frozen=True)
class _ImportBinding:
    source_module: str
    imported_name: str | None  # None: the whole module is bound (plain ``import x``)


@dataclass(frozen=True)
class _ModuleIndex:
    definitions: dict[str, list[_Candidate]]  # qualname -> candidates, source order
    bindings: dict[str, list[_Candidate]]  # name -> candidates, source order
    imports: dict[str, list[_ImportBinding]]  # local name -> import bindings
    star_imports: list[str]  # resolved source modules of ``from x import *``


# Every ast.parse call in this module goes through _parse_quiet with this sentinel filename, so a filter
# that ignores SyntaxWarning only for that filename can replace a per-parse warnings.catch_warnings() (which
# snapshots and restores the whole global filter list, and is not thread-safe). Callers are free to prepend
# their own filters (e.g. warnings.simplefilter("always") in a test), so the check re-installs whenever
# something else has since become the front-most filter, rather than truly registering only once.
_PARSE_FILENAME: Final = "<mloda-code-closure>"
_PARSE_WARNING_FILTER: Final = ("ignore", None, SyntaxWarning, re.compile(re.escape(_PARSE_FILENAME)), 0)


def _install_parse_warning_filter() -> None:
    if not warnings.filters or warnings.filters[0][:3] != _PARSE_WARNING_FILTER[:3]:
        warnings.filterwarnings("ignore", category=SyntaxWarning, module=re.escape(_PARSE_FILENAME))


def _parse_quiet(source: str | bytes) -> ast.Module:
    _install_parse_warning_filter()
    return ast.parse(source, filename=_PARSE_FILENAME)


def _module_source(module: types.ModuleType) -> str | bytes | None:
    loader = safe_field(lambda: module.__loader__, None)
    if loader is not None and hasattr(loader, "get_source"):

        def _read() -> str | None:
            source = loader.get_source(module.__name__)
            return source if isinstance(source, str) else None

        source = safe_field(_read, None, catching=_MODULE_INDEX_READ_ERRORS)
        if source is not None:
            return source

    # No usable get_source (e.g. pytest's assertion-rewriting loader) or it
    # returned nothing: fall back to reading the file, letting ast.parse
    # decode it (handles coding cookies).
    path = safe_field(lambda: module.__file__, None)
    if not isinstance(path, str) or not path.endswith(".py"):
        return None

    def _read_bytes() -> bytes:
        with open(path, "rb") as handle:
            return handle.read()

    return safe_field(_read_bytes, None, catching=(OSError,))


def _parse_module_source(source: str | bytes) -> ast.Module | None:
    return safe_field(lambda: _parse_quiet(source), None, catching=_MODULE_INDEX_READ_ERRORS)


def _source_lines(source: str | bytes) -> list[str]:
    """Splits on "\\n" only, matching the line numbers ast/compile use (unlike str.splitlines(), which also
    splits on \\x0b, \\x0c, \\x1c-\\x1e, U+0085, U+2028, U+2029). Bytes are decoded honoring the source's own
    coding cookie (not a hardcoded utf-8), so column offsets from a re-parsed span line up with the original AST.
    """
    text = importlib.util.decode_source(source) if isinstance(source, bytes) else source
    return text.split("\n")


def _resolve_import_module(module: str | None, level: int, package: str | None) -> str | None:
    if level == 0:
        return module
    relative = "." * level + (module or "")
    return safe_field(lambda: importlib.util.resolve_name(relative, package), None)


def _index_nested(
    body: list[ast.stmt], prefix: str, definitions: dict[str, list[_Candidate]], lines: list[str]
) -> None:
    """Recursively indexes classes reached from a class/function body, including through control flow."""
    for stmt in body:
        if isinstance(stmt, ast.ClassDef):
            qualname = f"{prefix}{stmt.name}"
            definitions.setdefault(qualname, []).append(_Candidate("class", stmt, lines))
            _index_nested(stmt.body, f"{qualname}.", definitions, lines)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _index_nested(stmt.body, f"{prefix}{stmt.name}.<locals>.", definitions, lines)
        elif isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            _index_nested(stmt.body, prefix, definitions, lines)
            _index_nested(stmt.orelse, prefix, definitions, lines)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            _index_nested(stmt.body, prefix, definitions, lines)
        elif isinstance(stmt, ast.Try):
            for sub_body in (stmt.body, stmt.orelse, stmt.finalbody):
                _index_nested(sub_body, prefix, definitions, lines)
            for handler in stmt.handlers:
                _index_nested(handler.body, prefix, definitions, lines)


def _index_body(
    body: list[ast.stmt],
    class_prefix: str,
    package: str | None,
    definitions: dict[str, list[_Candidate]],
    bindings: dict[str, list[_Candidate]],
    imports: dict[str, list[_ImportBinding]],
    star_imports: list[str],
    lines: list[str],
) -> None:
    args = (class_prefix, package, definitions, bindings, imports, star_imports, lines)
    for stmt in body:
        if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            _index_body(stmt.body, *args)
            _index_body(stmt.orelse, *args)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            _index_body(stmt.body, *args)
        elif isinstance(stmt, ast.Try):
            for sub_body in (stmt.body, stmt.orelse, stmt.finalbody):
                _index_body(sub_body, *args)
            for handler in stmt.handlers:
                _index_body(handler.body, *args)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            qualname = f"{class_prefix}{stmt.name}"
            kind = "class" if isinstance(stmt, ast.ClassDef) else "function"
            candidate = _Candidate(kind, stmt, lines)
            definitions.setdefault(qualname, []).append(candidate)
            bindings.setdefault(stmt.name, []).append(candidate)
            if isinstance(stmt, ast.ClassDef):
                _index_nested(stmt.body, f"{qualname}.", definitions, lines)
            else:
                _index_nested(stmt.body, f"{qualname}.<locals>.", definitions, lines)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            names = {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            if names:
                candidate = _Candidate("stmt", stmt, lines)
                for name in names:
                    bindings.setdefault(name, []).append(candidate)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.asname:
                    imports.setdefault(alias.asname, []).append(_ImportBinding(alias.name, None))
                else:
                    head = alias.name.split(".")[0]
                    imports.setdefault(head, []).append(_ImportBinding(head, None))
        elif isinstance(stmt, ast.ImportFrom):
            source_module = _resolve_import_module(stmt.module, stmt.level, package)
            if source_module is None:
                continue
            for alias in stmt.names:
                if alias.name == "*":
                    star_imports.append(source_module)
                else:
                    local = alias.asname or alias.name
                    imports.setdefault(local, []).append(_ImportBinding(source_module, alias.name))


def _build_module_index(module: types.ModuleType) -> _ModuleIndex | None:
    source = _module_source(module)
    if source is None:
        return None
    tree = _parse_module_source(source)
    if tree is None:
        return None
    definitions: dict[str, list[_Candidate]] = {}
    bindings: dict[str, list[_Candidate]] = {}
    imports: dict[str, list[_ImportBinding]] = {}
    star_imports: list[str] = []
    package = safe_field(lambda: module.__package__, None)
    lines = _source_lines(source)
    _index_body(tree.body, "", package, definitions, bindings, imports, star_imports, lines)
    return _ModuleIndex(definitions=definitions, bindings=bindings, imports=imports, star_imports=star_imports)


_module_index_cache: "weakref.WeakKeyDictionary[types.ModuleType, tuple[tuple[int, int] | None, _ModuleIndex | None]]" = weakref.WeakKeyDictionary()


def _module_file_stat(module: types.ModuleType) -> tuple[int, int] | None:
    path = safe_field(lambda: module.__file__, None)
    if not path:
        return None

    def _stat() -> tuple[int, int]:
        info = os.stat(path)
        return (info.st_mtime_ns, info.st_size)

    return safe_field(_stat, None, catching=(OSError,))


def _get_module_index(modname: str) -> _ModuleIndex | None:
    module = sys.modules.get(modname)
    if module is None:
        return None
    stat_key = _module_file_stat(module)
    cached = _module_index_cache.get(module)
    if cached is not None and cached[0] == stat_key:
        return cached[1]
    index = _build_module_index(module)
    _module_index_cache[module] = (stat_key, index)
    return index


_KIND_NODE_TYPES: Final[dict[str, tuple[type[ast.AST], ...]]] = {
    "class": (ast.ClassDef,),
    "function": (ast.FunctionDef, ast.AsyncFunctionDef),
    "stmt": (ast.Assign, ast.AnnAssign, ast.AugAssign),
}


def _parse_span_node(candidate: _Candidate) -> ast.AST | None:
    """Re-parses the candidate's span and picks the node at its own (lineno, col_offset), not the span's first
    node of the right kind: a same-line sibling (``A = 1; B = 2``) or a one-line ``else:`` body would otherwise
    hash as its neighbor.
    """
    text = "\n".join(candidate.lines[candidate.start_line - 1 : candidate.end_line])
    tree = _parse_module_source(text)
    line_offset = 0
    if tree is None:
        tree = _parse_module_source("if True:\n" + text)
        line_offset = 1
    if tree is None:
        return None
    target_lineno = candidate.lineno - candidate.start_line + 1 + line_offset
    for node in ast.walk(tree):
        if (
            isinstance(node, _KIND_NODE_TYPES[candidate.kind])
            and getattr(node, "lineno", None) == target_lineno
            and getattr(node, "col_offset", None) == candidate.col_offset
        ):
            return node
    return None


def _resolve_candidate(candidate: _Candidate) -> tuple[str | None, list[str]]:
    if candidate.digest_refs is None:
        node = _parse_span_node(candidate)
        candidate.digest_refs = (None, []) if node is None else (_digest(node), _collect_refs(node))
    return candidate.digest_refs


def _parse_named_node(source: str, name: str, node_types: tuple[type[ast.AST], ...]) -> ast.AST | None:
    tree = safe_field(lambda: _parse_quiet(source), None, catching=_MODULE_INDEX_READ_ERRORS)
    if tree is None:
        tree = safe_field(lambda: _parse_quiet("if True:\n" + source), None, catching=_MODULE_INDEX_READ_ERRORS)
    if tree is None:
        return None
    for node in ast.walk(tree):
        if isinstance(node, node_types) and getattr(node, "name", None) == name:
            return node
    return None


def _root_identity_ok(modname: str, qualname: str, cls: type[Any]) -> bool:
    mod = sys.modules.get(modname)
    if mod is None:
        return False
    current: Any = mod
    for part in qualname.split("."):
        if part == "<locals>":
            return False
        namespace = safe_field(lambda c=current: vars(c), None)  # type: ignore[misc]
        if namespace is None:
            return False
        current = namespace.get(part)
        if current is None:
            return False
    return current is cls


def _unwrap_descriptor(value: Any) -> Any:
    if type(value) in (classmethod, staticmethod):
        return safe_field(lambda: value.__func__, None)
    if type(value) is property:
        return value.fget
    return value


def _code_of(func: Any) -> Any:
    return safe_field(lambda: func.__code__, None)


def _method_first_lines(cls: type[Any]) -> set[int]:
    """``co_firstlineno`` of every function reachable through ``vars(cls)`` (unwrapping classmethod/
    staticmethod/property), used to pick the live candidate among same-qualname ClassDef definitions.
    """
    lines: set[int] = set()
    namespace: dict[str, Any] = safe_field(lambda: dict(vars(cls)), {})
    for value in namespace.values():
        code = _code_of(_unwrap_descriptor(value))
        if code is not None:
            lines.add(code.co_firstlineno)
    return lines


def _enclosing_qualname(qualname: str) -> str | None:
    if "." not in qualname:
        return None
    return qualname.rsplit(".", 1)[0]


def _top(modname: str) -> str:
    return modname.split(".", 1)[0]


_CORE_SURFACE_PREFIXES: Final = ("mloda.core", "mloda.user", "mloda.provider", "mloda.steward")


def _is_core_surface(modname: str) -> bool:
    return any(modname == prefix or modname.startswith(prefix + ".") for prefix in _CORE_SURFACE_PREFIXES)


# ---------------------------------------------------------------------------
# Dependency names and versions. Never imports: reads only modules already in sys.modules.
# ---------------------------------------------------------------------------

_IGNORED_TOPS: Final = ("mloda_plugins", "__main__", "__mp_main__")


def _has_file(module: types.ModuleType) -> bool:
    return safe_field(lambda: module.__file__, None, catching=(AttributeError,)) is not None


def _dunder_version(module: types.ModuleType | None) -> str | None:
    if module is None:
        return None
    value = safe_field(lambda: module.__version__, None, catching=(AttributeError,))
    return value if isinstance(value, str) else None


def _namespace_dependency_name_version(module_name: str) -> tuple[str, str | None]:
    """Walks the dotted path to the first regular (non-namespace) package; that is the dependency name."""
    parts = module_name.split(".")
    prefix = parts[0]
    for part in parts[1:]:
        mod = sys.modules.get(prefix)
        if mod is not None and _has_file(mod):
            return prefix, _dunder_version(mod)
        prefix = f"{prefix}.{part}"
    return prefix, _dunder_version(sys.modules.get(prefix))


def _dependency_name_version(module_name: str, top: str) -> tuple[str, str | None]:
    # Always tried first, whether or not top is already imported: importlib.metadata.version imports nothing,
    # so a TYPE_CHECKING-only (or try/except-guarded) dependency hashes the same whether or not it is installed.
    version = safe_field(
        lambda: importlib.metadata.version(top),
        None,
        catching=(importlib.metadata.PackageNotFoundError, ValueError),
    )
    if version is not None:
        return top, version
    top_module = sys.modules.get(top)
    if top_module is None:
        return top, None
    if not _has_file(top_module):
        return _namespace_dependency_name_version(module_name)
    return top, _dunder_version(top_module)


def _is_ignored_module_name(module_name: str) -> bool:
    """True for stdlib, builtins, the core surface, mloda_plugins, __main__/__mp_main__ (never a real dependency)."""
    top = _top(module_name)
    return top in sys.stdlib_module_names or _is_core_surface(module_name) or top in _IGNORED_TOPS


@functools.cache
def dependency_entry(module_name: str) -> str | None:
    """Third-party dependency descriptor ``dep:name==version``/``dep:name`` for module_name, else None.

    None for stdlib, builtins, the core surface, mloda_plugins, __main__/__mp_main__. Never imports,
    only reads sys.modules; never calls importlib.metadata.packages_distributions or resolve_plugin_version.
    """
    if _is_ignored_module_name(module_name):
        return None
    top = _top(module_name)
    name, version = _dependency_name_version(module_name, top)
    return f"dep:{name}=={version}" if version else f"dep:{name}"


def _root_fallback_source(cls: type[Any]) -> str:
    # Local import: base_feature_group_version imports closure_parts, so importing it back here at
    # module level would be circular. Both modules are already loaded by the time this call happens.
    from mloda.core.abstract_plugins.components.base_feature_group_version import _resolve_class_source_text

    return _resolve_class_source_text(cls)


# ---------------------------------------------------------------------------
# Walk: an explicit worklist (LIFO stack of pending tasks) replaces the former
# resolve_ref -> add_value -> add_def -> resolve_ref mutual recursion, so a
# long same-module call chain cannot raise RecursionError. Parts are a set
# that gets sorted at the end, so task order does not matter.
# ---------------------------------------------------------------------------

_Task = tuple[Any, ...]


def _task_key(task: _Task) -> str:
    """Deterministic identity of a task for a contained-failure ``nosrc:`` part: the tag plus the task's own
    string fields (module names, ref/binding names, qualnames), never a live object's repr or an exception
    message (both can embed a memory address and vary between runs of identical source).
    """
    tag = task[0]
    if tag == "def":
        obj = task[1]
        modname = safe_field(lambda: obj.__module__, "?")
        qualname = safe_field(lambda: obj.__qualname__, "?")
        return f"def:{modname}:{qualname}"
    if tag == "value":
        return f"value:{task[2]}:{task[3]}"
    return ":".join([tag, *(str(field) for field in task[1:])])


class _Walker:
    """Worklist walk over the module index, driven by live values for name/attribute resolution."""

    def __init__(self, tops: frozenset[str], include_dependencies: bool) -> None:
        self._tops = tops
        self._include_dependencies = include_dependencies
        self.parts: set[str] = set()
        self._seen: set[tuple[str, str]] = set()
        self._queue: list[_Task] = []
        self._index_cache: dict[str, _ModuleIndex | None] = {}
        self._dependency_by_top: dict[str, str | None] = {}

    def _module_index(self, modname: str) -> _ModuleIndex | None:
        """_get_module_index memoized per walk: a ref-heavy walk would otherwise re-stat the same module's
        file once per reference into it.
        """
        if modname not in self._index_cache:
            self._index_cache[modname] = _get_module_index(modname)
        return self._index_cache[modname]

    def in_scope(self, modname: str) -> bool:
        top = _top(modname)
        if top in sys.stdlib_module_names or modname == "builtins":
            return False
        if _is_core_surface(modname):
            return False
        return top in self._tops

    def _add_dependency(self, modname: str) -> None:
        if not self._include_dependencies:
            return
        # Reuses the first dependency_entry() result seen for a top-level package within this walk: a single
        # class can reference many submodules of the same third-party package, each an expensive cache miss
        # for dependency_entry's own (per-exact-module-name) cache.
        top = _top(modname)
        if top not in self._dependency_by_top:
            self._dependency_by_top[top] = dependency_entry(modname)
        entry = self._dependency_by_top[top]
        if entry is not None:
            self.parts.add(entry)

    def _enqueue_refs(self, modname: str, refs: list[str]) -> None:
        for ref in refs:
            self._queue.append(("ref", modname, ref))

    def _accept_candidate(self, modname: str, qualname: str, candidate: _Candidate) -> None:
        digest, refs = _resolve_candidate(candidate)
        if digest is None:
            self.parts.add(f"nosrc:{modname}:{qualname}")
            return
        self.parts.add(f"{modname}:{qualname}:{digest}")
        self._enqueue_refs(modname, refs)

    def add_root(self, cls: type[Any]) -> None:
        modname = cls.__module__
        qualname = cls.__qualname__
        key = (modname, qualname)
        if key in self._seen:
            return
        self._seen.add(key)
        if not self.in_scope(modname):
            self._add_dependency(modname)
            return
        idx = self._module_index(modname)
        if idx is not None:
            candidates = [c for c in idx.definitions.get(qualname, []) if c.kind == "class"]
            if "<locals>" in qualname:
                if len(candidates) == 1:
                    self._accept_candidate(modname, qualname, candidates[0])
                    return
            elif candidates and _root_identity_ok(modname, qualname, cls):
                # Several if/else (or try/except) definitions can share this qualname; pick the one whose
                # span contains a live method's co_firstlineno. None matched (no methods, or file changed
                # underneath): accept all candidates, which is over-inclusive but safe.
                method_lines = _method_first_lines(cls)
                matched = [c for c in candidates if any(c.start_line <= ln <= c.end_line for ln in method_lines)]
                for candidate in matched or candidates:
                    self._accept_candidate(modname, qualname, candidate)
                return
        source = _root_fallback_source(cls)  # raises SOURCE_INTROSPECTION_ERRORS on failure
        node = _parse_named_node(source, cls.__name__, (ast.ClassDef,))
        if node is None:
            self.parts.add(f"nosrc:{modname}:{qualname}")
            return
        digest = _digest(node)
        refs = _collect_refs(node)
        self.parts.add(f"{modname}:{qualname}:{digest}")
        self._enqueue_refs(modname, refs)

    def _accept_def_by_qualname(self, modname: str, qualname: str, idx: _ModuleIndex) -> None:
        key = (modname, qualname)
        if key in self._seen:
            return
        self._seen.add(key)
        for candidate in idx.definitions.get(qualname, []):
            self._accept_candidate(modname, qualname, candidate)

    def _handle_def(self, obj: Any) -> None:
        modname = safe_field(lambda: obj.__module__, None)
        qualname = safe_field(lambda: obj.__qualname__, None)
        if not modname or not qualname:
            return
        key = (modname, qualname)
        if key in self._seen:
            return
        idx = self._module_index(modname)
        if idx is not None and idx.definitions.get(qualname):
            self._accept_def_by_qualname(modname, qualname, idx)
            return
        if idx is not None:
            parent = _enclosing_qualname(qualname)
            if parent is not None and idx.definitions.get(parent):
                self._accept_def_by_qualname(modname, parent, idx)
                return
        self._seen.add(key)
        source = safe_field(lambda: inspect.getsource(obj), None, catching=_WALK_READ_ERRORS)
        if source is not None:
            name = qualname.rsplit(".", 1)[-1]
            node_types = (ast.ClassDef,) if issubclass(type(obj), type) else (ast.FunctionDef, ast.AsyncFunctionDef)
            node = _parse_named_node(source, name, node_types)
            if node is not None:
                digest = _digest(node)
                refs = _collect_refs(node)
                self.parts.add(f"{modname}:{qualname}:{digest}")
                self._enqueue_refs(modname, refs)
                return
        self.parts.add(f"nosrc:{modname}:{qualname}")

    def _handle_binding(self, modname: str, name: str) -> None:
        key = (modname, f"={name}")
        if key in self._seen:
            return
        self._seen.add(key)
        idx = self._module_index(modname)
        if idx is None:
            return
        candidates = idx.bindings.get(name)
        if candidates:
            for candidate in candidates:
                digest, refs = _resolve_candidate(candidate)
                if digest is None:
                    self.parts.add(f"nosrc:{modname}:={name}")
                    continue
                self.parts.add(f"{modname}:={name}:{digest}")
                self._enqueue_refs(modname, refs)
            return
        for entry in idx.imports.get(name, []):
            if entry.imported_name is not None and self.in_scope(entry.source_module):
                self._queue.append(("binding", entry.source_module, entry.imported_name))
        for star_source in idx.star_imports:
            if self.in_scope(star_source):
                self._queue.append(("binding", star_source, name))

    def _handle_module(self, modname: str) -> None:
        key = (modname, "*")
        if key in self._seen:
            return
        self._seen.add(key)
        idx = self._module_index(modname)
        if idx is None:
            self.parts.add(f"nosrc:{modname}:*")
            return
        for qualname in idx.definitions:
            if "." not in qualname:
                self._accept_def_by_qualname(modname, qualname, idx)
        for name in idx.bindings:
            self._queue.append(("binding", modname, name))
        # A module used whole also reaches the names it re-exports through its own imports/star imports
        # (e.g. ``dispatch/__init__.py: from .impl import double``, read back via getattr(dispatch, "double")).
        for entries in idx.imports.values():
            for entry in entries:
                if entry.imported_name is not None and self.in_scope(entry.source_module):
                    self._queue.append(("binding", entry.source_module, entry.imported_name))
        for star_source in idx.star_imports:
            if self.in_scope(star_source):
                self._queue.append(("module", star_source))

    def _handle_value(self, value: Any, ref_modname: str, ref_name: str, optional_dependency_sentinel: bool) -> None:
        """Classifies value by type(value), never isinstance()/value.__class__: a lazy proxy's __class__ can be
        a property that runs arbitrary (and possibly raising) user code.
        """
        vtype = type(value)
        if vtype is types.FunctionType:
            value = safe_field(lambda: inspect.unwrap(value), value, catching=(ValueError,))
            vtype = type(value)

        if vtype is types.ModuleType:
            modname = safe_field(lambda: value.__name__, "")
            if not modname:
                return
            if self.in_scope(modname):
                self._queue.append(("module", modname))
            else:
                self._add_dependency(modname)
            return
        if vtype in (types.BuiltinFunctionType, types.BuiltinMethodType):
            return
        if vtype is types.FunctionType or issubclass(vtype, type):
            def_modname = safe_field(lambda: value.__module__, None)
            if not def_modname:
                return
            if not self.in_scope(def_modname):
                self._add_dependency(def_modname)
                return
            qualname = safe_field(lambda: value.__qualname__, "")
            if "<lambda>" in qualname or "<locals>" in qualname:
                mod = sys.modules.get(ref_modname)
                namespace = safe_field(lambda: vars(mod), None) if mod is not None else None
                if namespace is not None and namespace.get(ref_name) is value:
                    self._queue.append(("binding", ref_modname, ref_name))
                    return
            self._queue.append(("def", value))
            return
        if value is None and optional_dependency_sentinel:
            # A bare `None` bound to a name that also has a static out-of-scope import binding is how a
            # try/except ImportError fallback marks "dependency unavailable" (e.g. `x = None`); that
            # assignment's own source must not affect the hash (it would otherwise differ only depending on
            # whether the optional dependency is installed). A first-party `X = compute()` that merely
            # happens to return None is not this sentinel and is still walked below.
            return
        self._queue.append(("binding", ref_modname, ref_name))

    def _handle_ref(self, ref_modname: str, ref: str) -> None:
        head, *rest = ref.split(".")
        idx = self._module_index(ref_modname)
        optional_dependency_sentinel = False
        if idx is not None:
            for binding in idx.imports.get(head, []):
                if not self.in_scope(binding.source_module) and not _is_ignored_module_name(binding.source_module):
                    self._add_dependency(binding.source_module)
                    optional_dependency_sentinel = True
                    # A static import binding out of scope (e.g. an optional accelerated dependency) does not
                    # short-circuit: the live namespace value below still gets resolved and, when it turns out
                    # to be a first-party fallback (try/except ImportError), still gets walked.
        mod = sys.modules.get(ref_modname)
        if mod is None:
            return
        namespace = safe_field(lambda: vars(mod), None)
        if namespace is None or head not in namespace:
            return
        value = namespace[head]
        owner_mod, owner_name = ref_modname, head
        for attr in rest:
            if type(value) is not types.ModuleType:
                break
            modname = safe_field(lambda: value.__name__, "")
            if not modname:
                return
            if not self.in_scope(modname):
                self._add_dependency(modname)
                return
            mod_vars = safe_field(lambda: vars(value), None)
            if mod_vars is None or attr not in mod_vars:
                return
            owner_mod, owner_name = modname, attr
            value = mod_vars[attr]
        self._queue.append(("value", value, owner_mod, owner_name, optional_dependency_sentinel))

    def _dispatch(self, task: _Task) -> None:
        tag = task[0]
        if tag == "ref":
            self._handle_ref(task[1], task[2])
        elif tag == "value":
            self._handle_value(task[1], task[2], task[3], task[4])
        elif tag == "def":
            self._handle_def(task[1])
        elif tag == "binding":
            self._handle_binding(task[1], task[2])
        else:
            self._handle_module(task[1])

    def run(self) -> None:
        while self._queue:
            task = self._queue.pop()
            _, error = safe_field_with_error(functools.partial(self._dispatch, task), None)
            if error is not None:
                # A single task raising (e.g. a live value's own code misbehaving) must not abort the whole
                # walk; only a root with no source may raise out of closure_parts. The recorded part is
                # deterministic (task tag plus its own string fields): never the exception message or a
                # repr of a live object, which can embed a memory address and vary across runs.
                self.parts.add(f"nosrc:{_task_key(task)}")


def closure_parts(leaf_class: type[Any], include_dependencies: bool) -> list[str]:
    """Reachable first-party code of leaf_class as sorted parts; ``dep:`` entries only when include_dependencies."""
    leaf_modname = leaf_class.__module__
    if _top(leaf_modname) in sys.stdlib_module_names or leaf_modname == "builtins":
        # E.g. a class built with type() under ABCMeta gets __module__ == "abc": an empty closure would
        # otherwise hash to the same value for every such class instead of raising.
        raise OSError(f"{leaf_class!r}'s own module {leaf_modname!r} is stdlib/builtins: no source available")
    tops = frozenset({_top(leaf_modname), "mloda"})
    walker = _Walker(tops, include_dependencies)
    for cls in leaf_class.__mro__:
        walker.add_root(cls)
    walker.run()
    return sorted(walker.parts)
