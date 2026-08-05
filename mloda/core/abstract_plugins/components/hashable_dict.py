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
# residual: set values, containers whose type overrides __eq__, and cycles routed through a nested
# Options/HashableDict (where the path resets) are still plain ==.
_CYCLE = _CycleMarker()


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
    kind = _container_kind(a)
    if kind is None or _container_kind(b) is not kind:
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
