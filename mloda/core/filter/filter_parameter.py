from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

from mloda.core.abstract_plugins.components.utils import unhashable_part


@runtime_checkable
class FilterParameter(Protocol):
    @property
    def value(self) -> Any | None: ...

    @property
    def values(self) -> list[Any] | None: ...

    @property
    def min_value(self) -> Any | None: ...

    @property
    def max_value(self) -> Any | None: ...

    @property
    def max_exclusive(self) -> bool: ...


def _normalize_collections(value: Any) -> Any:
    """Normalize collection values so the frozen dataclass stays hashable.

    Sets/frozensets become frozenset, not a repr-sorted tuple: repr isn't cross-type-equality-safe
    (1 == True), frozenset already is.
    """
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (set, frozenset)):
        return frozenset(value)
    if isinstance(value, list):
        return tuple(value)
    return value


@dataclass(frozen=True)
class FilterParameterImpl:
    _raw: tuple[tuple[str, Any], ...]

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "FilterParameterImpl":
        # Checked before the sort below, which would otherwise raise a TypeError naming nothing.
        for key in params:
            if not isinstance(key, str):
                raise ValueError(f"Filter parameter key {key!r} is not a string.")
        normalized = {k: _normalize_collections(v) for k, v in params.items()}
        # This site rejects what still does not hash; _deep_hashable in hashable_dict coerces instead.
        for key, value in normalized.items():
            culprit = unhashable_part(value)
            if culprit is not None:
                raise ValueError(
                    f"Filter parameter '{key}' holds an unhashable {culprit}; "
                    "filter values must be hashable: scalars, or lists, sets or tuples of hashables."
                )
        return cls(_raw=tuple(sorted(normalized.items())))

    @property
    def value(self) -> Any | None:
        return self._get("value")

    @property
    def values(self) -> list[Any] | None:
        # Stored as a tuple or frozenset for hashability; hand out the declared list type. A
        # frozenset iterates in hash-seed order, so it's re-sorted here to stay deterministic.
        stored = self._get("values")
        if isinstance(stored, frozenset):
            return sorted(stored, key=repr)
        if isinstance(stored, tuple):
            return list(stored)
        return cast(list[Any] | None, stored)

    @property
    def min_value(self) -> Any | None:
        return self._get("min")

    @property
    def max_value(self) -> Any | None:
        return self._get("max")

    @property
    def max_exclusive(self) -> bool:
        return cast(bool, self._get("max_exclusive", False))

    def _get(self, key: str, default: Any = None) -> Any:
        for k, v in self._raw:
            if k == key:
                return v
        return default
