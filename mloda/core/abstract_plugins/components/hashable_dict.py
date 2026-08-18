from collections.abc import Callable
from typing import Any

from mloda.core.abstract_plugins.components.utils import unhashable_part


class _CycleMarker:
    """Stand-in for a container already on the recursion path.

    Its own type keeps it distinct from every normalized user value, so a literal ``"<cycle>"``
    cannot collide with a real back-reference. Hash is constant so it stays stable across processes.
    """

    __slots__ = ()

    def __hash__(self) -> int:
        return hash("mloda-deep-hashable-cycle")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _CycleMarker)

    def __repr__(self) -> str:
        return "<cycle>"


# A cyclic value hashes instead of raising RecursionError. Mirrors the id() visited guard in
# Feature._reduce. _deep_equal mirrors this normalization for the == that such a collision reaches;
# residual: set values and containers whose type overrides __eq__ (including subclasses of a
# registered node) fall back to plain == and reset the path. A cycle routed through a nested Feature
# also resets the path and still raises RecursionError: Feature equality reads more fields than a
# dict target can express, so it is not a registered node and carries its own guard in _reduce.
_CYCLE = _CycleMarker()

# Nodes (Options, HashableDict) register themselves here: hashable_dict cannot import options,
# since options imports it.
_NODE_KINDS: dict[type, Callable[[Any], dict[Any, Any]]] = {}
_NODE_TYPES: tuple[type, ...] = ()


def register_deep_node(kind: type, target: Callable[[Any], dict[Any, Any]]) -> None:
    """Values of ``kind`` are walked into via ``target``, so the visited path survives the nesting.

    ``target`` must return the same dict object on every call for the same value: termination
    depends on that dict's id landing in the visited path, and a rebuilt dict recurses unbounded.
    Re-registering a kind replaces its entry.
    """
    global _NODE_TYPES
    _NODE_KINDS[kind] = target
    _NODE_TYPES = tuple(_NODE_KINDS)


def _node_target(value: Any, dunder: str) -> tuple[type, dict[Any, Any]] | None:
    """The registered kind and the dict it is walked as, or None when its own dunder must decide."""
    if not isinstance(value, _NODE_TYPES):
        return None
    for kind, target in _NODE_KINDS.items():
        if isinstance(value, kind) and getattr(type(value), dunder) is getattr(kind, dunder):
            return kind, target(value)
    return None


# _deep_hashable's output is hash-only and deliberately untagged: collisions between unrelated
# types are legal for its equality-checked callers (Options.__hash__, HashableDict.__hash__, each
# paired with a _deep_equal tiebreak), since tagging would break hash/eq consistency for subclasses
# with inherited dunders. Feature.similarity_hash/base_similarity_hash (feature.py) are the
# exception: they feed this hash straight into ExecutionPlan's dict-keyed grouping with no equality
# tiebreak, so a collision there would silently merge unrelated feature groups.
# Feature._reduce's output (feature.py) is an equality key instead, so it must stay tagged: its
# ("feature", ...), ("options", ...), ("hashable_dict", ...) markers are load-bearing, not decoration.
def _deep_hashable(value: Any, seen: frozenset[int] = frozenset()) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        if id(value) in seen:
            return _CYCLE
        seen = seen | {id(value)}
    if isinstance(value, dict):
        items = [(k, _deep_hashable(v, seen)) for k, v in value.items()]
        # Mixed-type keys (e.g. reader class plus str) are unorderable; fall back to a
        # type-robust deterministic sort. The common orderable case stays unchanged.
        try:
            return tuple(sorted(items))
        except TypeError:
            return tuple(
                sorted(items, key=lambda kv: (kv[0].__class__.__module__, kv[0].__class__.__qualname__, repr(kv[0])))
            )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_hashable(item, seen) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_hashable(item, seen) for item in value)
    # A node normalizes to the same shape as its plain dict: collisions between unequal values are
    # legal, and not tagging the type keeps hash consistent with equality for inherited dunders.
    node = _node_target(value, "__hash__")
    if node is not None:
        return _deep_hashable(node[1], seen)
    # A leaf whose hash raises TypeError is coerced to repr so grouping never crashes; any other raise
    # propagates. filter_parameter rejects broadly instead.
    # Residual constraint: two values that are __eq__-equal but unhashable must have
    # repr consistent with equality, else they over-split into separate groups (a
    # rare, non-crashing tradeoff).
    if unhashable_part(value, catching=(TypeError,)) is not None:
        return repr(value)
    return value


_CONTAINER_KINDS = (dict, list, tuple)


def _container_kind(value: Any) -> type | None:
    """The base container type value is walked as, or None when its own __eq__ must decide."""
    # Exact types cannot override __eq__, so they skip the isinstance scan; this is the hot path.
    if type(value) in _CONTAINER_KINDS:
        return type(value)
    for kind in _CONTAINER_KINDS:
        if isinstance(value, kind) and type(value).__eq__ is kind.__eq__:
            return kind
    return None


def _deep_equal(a: Any, b: Any) -> bool:
    """Compare two values structurally, matching a back-reference only against another back-reference."""
    return _walk_equal(a, b, set(), set())


def _walk_equal(a: Any, b: Any, path_a: set[int], path_b: set[int]) -> bool:
    # _container_kind's exact-type fast path runs first; no registered kind is a dict/list/tuple
    # subclass, so consulting the node registry only when it finds nothing is equivalent.
    kind = _container_kind(a)
    if kind is None:
        node_a = _node_target(a, "__eq__")
        if node_a is not None:
            node_b = _node_target(b, "__eq__")
            # Only matching kinds are walked; a differing or absent node keeps its own type identity.
            if node_b is not None and node_b[0] is node_a[0]:
                return _walk_equal(node_a[1], node_b[1], path_a, path_b)
        return a is b or bool(a == b)
    if _container_kind(b) is not kind:
        return a is b or bool(a == b)
    on_a, on_b = id(a) in path_a, id(b) in path_b
    if on_a or on_b:
        # A back-reference matches only another back-reference, as _deep_hashable's marker does.
        return on_a and on_b
    if len(a) != len(b):
        return False
    path_a.add(id(a))
    path_b.add(id(b))
    if kind is dict:
        # Keys keep their own hash and __eq__; only values are walked.
        equal = all(k in b and _walk_equal(v, b[k], path_a, path_b) for k, v in a.items())
    else:
        equal = all(_walk_equal(x, y, path_a, path_b) for x, y in zip(a, b))
    path_a.discard(id(a))
    path_b.discard(id(b))
    return equal


class HashableDict:
    def __init__(self, data: dict[Any, Any]) -> None:
        self.data = data

    def __hash__(self) -> int:
        return hash(_deep_hashable(self.data))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HashableDict):
            return False
        return _deep_equal(self.data, other.data)

    def items(self) -> Any:
        return self.data.items()


register_deep_node(HashableDict, lambda node: node.data)
