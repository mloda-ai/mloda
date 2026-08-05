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
# Feature._reduce. _deep_equal carries the matching guard for the == that such a collision reaches.
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


def _deep_equal(a: Any, b: Any, seen: frozenset[tuple[int, int]] = frozenset()) -> bool:
    """Compare two values structurally, treating a repeated id pair on the recursion path as equal."""
    if a is b:
        return True
    if type(a) is not type(b) or type(a) not in (dict, list, tuple):
        return bool(a == b)
    pair = (id(a), id(b))
    if pair in seen:
        return True
    seen = seen | {pair}
    if len(a) != len(b):
        return False
    if type(a) is dict:
        # Keys keep their own hash and __eq__; only values are walked.
        return all(k in b and _deep_equal(v, b[k], seen) for k, v in a.items())
    return all(_deep_equal(x, y, seen) for x, y in zip(a, b))


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
