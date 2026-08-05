from typing import Any

from mloda.core.abstract_plugins.components.utils import unhashable_part


# Stand-in for a container already on the recursion path, so a cyclic value hashes
# instead of raising RecursionError. Mirrors the id() visited guard in Feature._reduce.
_CYCLE = ("<cycle>",)


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


class HashableDict:
    def __init__(self, data: dict[Any, Any]) -> None:
        self.data = data

    def __hash__(self) -> int:
        return hash(_deep_hashable(self.data))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HashableDict):
            return False
        return self.data == other.data

    def items(self) -> Any:
        return self.data.items()
