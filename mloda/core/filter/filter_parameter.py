from dataclasses import dataclass
from typing import Any, Optional, Protocol, cast, runtime_checkable

from mloda.core.abstract_plugins.components.utils import safe_field


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


def _make_hashable(value: Any) -> Any:
    """Normalize collection values to tuples so the frozen dataclass stays hashable.

    Sets are ordered deterministically, str/bytes stay scalar and are never exploded.
    """
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(value, key=repr))
    if isinstance(value, list):
        return tuple(value)
    return value


def _unhashable_type(value: Any) -> str | None:
    """Name of the first part of `value` that does not hash, None when the whole value hashes."""
    # Probe the real hash, not isinstance(value, Hashable): a __hash__ that raises reports as hashable.
    if safe_field(lambda: isinstance(hash(value), int), False):
        return None
    if isinstance(value, tuple):
        for element in value:
            found = _unhashable_type(element)
            if found is not None:
                return found
    return type(value).__name__


@dataclass(frozen=True)
class FilterParameterImpl:
    _raw: tuple[tuple[str, Any], ...]

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "FilterParameterImpl":
        # Checked before the sort below, which would otherwise raise a TypeError naming nothing.
        for key in params:
            if not isinstance(key, str):
                raise ValueError(f"Filter parameter key {key!r} is not a string.")
        normalized = {k: _make_hashable(v) for k, v in params.items()}
        for key, value in normalized.items():
            culprit = _unhashable_type(value)
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
        # Stored as a tuple for hashability; hand out the declared list type. Filter engines rely
        # on this: PySpark's Column.isin only unwraps list/set, so a tuple would silently break it.
        stored = self._get("values")
        if isinstance(stored, tuple):
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
