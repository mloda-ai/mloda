from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, cast, runtime_checkable


@runtime_checkable
class FilterParameter(Protocol):
    @property
    def value(self) -> Optional[Any]: ...

    @property
    def values(self) -> Optional[List[Any]]: ...

    @property
    def min_value(self) -> Optional[Any]: ...

    @property
    def max_value(self) -> Optional[Any]: ...

    @property
    def max_exclusive(self) -> bool: ...


@dataclass(frozen=True)
class FilterParameterImpl:
    _raw: Tuple[Tuple[str, Any], ...]

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "FilterParameterImpl":
        return cls(_raw=tuple(sorted(params.items())))

    @property
    def value(self) -> Optional[Any]:
        return self._get("value")

    @property
    def values(self) -> Optional[List[Any]]:
        return cast(Optional[List[Any]], self._get("values"))

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
