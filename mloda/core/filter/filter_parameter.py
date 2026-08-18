from dataclasses import dataclass
from typing import Any, Optional, Protocol, cast, runtime_checkable

from mloda.core.abstract_plugins.components.utils import unhashable_part


@runtime_checkable
class FilterParameter(Protocol):
    @property
    def value(self) -> Optional[Any]: ...

    @property
    def values(self) -> Optional[list[Any]]: ...

    @property
    def min_value(self) -> Optional[Any]: ...

    @property
    def max_value(self) -> Optional[Any]: ...

    @property
    def max_exclusive(self) -> bool: ...


def _normalize_collections(value: Any) -> Any:
    """Normalize collection values so the frozen dataclass stays hashable.

    Lists become tuples, order preserved. Sets/frozensets become frozensets: frozenset equality and
    hashing are already order- and representation-independent, so cross-type-equal elements (1 and
    True) normalize the same regardless of which concrete type built either side, unlike a repr-sorted
    tuple. str/bytes stay scalar and are never exploded.
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
    def value(self) -> Optional[Any]:
        return self._get("value")

    @property
    def values(self) -> Optional[list[Any]]:
        # Stored as a tuple or frozenset for hashability; hand out the declared list type. Filter
        # engines rely on this: PySpark's Column.isin only unwraps list/set, so either would silently
        # break it if handed out raw.
        stored = self._get("values")
        if isinstance(stored, (tuple, frozenset)):
            return list(stored)
        return cast(Optional[list[Any]], stored)

    @property
    def min_value(self) -> Optional[Any]:
        return self._get("min")

    @property
    def max_value(self) -> Optional[Any]:
        return self._get("max")

    @property
    def max_exclusive(self) -> bool:
        return cast(bool, self._get("max_exclusive", False))

    def _get(self, key: str, default: Any = None) -> Any:
        for k, v in self._raw:
            if k == key:
                return v
        return default
