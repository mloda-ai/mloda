from typing import Any


def _make_hashable(value: Any) -> Any:
    if isinstance(value, dict):
        items = [(k, _make_hashable(v)) for k, v in value.items()]
        # Mixed-type keys (e.g. reader class plus str) are unorderable; fall back to a
        # type-robust deterministic sort. The common orderable case stays unchanged.
        try:
            return tuple(sorted(items))
        except TypeError:
            return tuple(
                sorted(items, key=lambda kv: (kv[0].__class__.__module__, kv[0].__class__.__qualname__, repr(kv[0])))
            )
    if isinstance(value, (list, tuple)):
        return tuple(_make_hashable(item) for item in value)
    if isinstance(value, set):
        return frozenset(_make_hashable(item) for item in value)
    # Unhashable non-container leaves fall back to repr so grouping never crashes.
    # Residual constraint: two values that are __eq__-equal but unhashable must have
    # repr consistent with equality, else they over-split into separate groups (a
    # rare, non-crashing tradeoff).
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


class HashableDict:
    def __init__(self, data: dict[Any, Any]) -> None:
        self.data = data

    def __hash__(self) -> int:
        return hash(_make_hashable(self.data))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HashableDict):
            return False
        return self.data == other.data

    def items(self) -> Any:
        return self.data.items()
