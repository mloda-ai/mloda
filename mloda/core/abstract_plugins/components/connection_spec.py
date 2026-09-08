"""Picklable connection recipes for the ComputeFramework connection resolution seam.

ConnectionSpec is what travels across a process boundary; ConnectionSource wraps either
a live connection object or a spec, and drops the live object on pickle.
"""

import threading
from typing import Any


def _framework_name(framework: type[Any] | str) -> str:
    return framework if isinstance(framework, str) else framework.__name__


class ConnectionSpec:
    """Picklable recipe a ComputeFramework opens via open_connection(); a valid
    DataAccessCollection connection entry."""

    def __init__(self, framework: type[Any] | str, **params: Any) -> None:
        if not isinstance(framework, (str, type)):
            raise TypeError(f"ConnectionSpec framework must be a class or a str, got {type(framework).__name__}.")
        self.framework = framework
        self.params: dict[str, Any] = dict(params)

    def matches(self, cfw_class: type[Any]) -> bool:
        if isinstance(self.framework, str):
            return any(base.__name__ == self.framework for base in cfw_class.__mro__)
        return issubclass(cfw_class, self.framework)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConnectionSpec):
            return False
        return self.framework == other.framework and self.params == other.params

    def __hash__(self) -> int:
        return hash(_framework_name(self.framework))

    def __repr__(self) -> str:
        name = _framework_name(self.framework)
        params_str = ", ".join(f"{key}='***'" for key in self.params)
        if params_str:
            return f"ConnectionSpec({name}, {params_str})"
        return f"ConnectionSpec({name})"


class ConnectionSource:
    """Where a framework instance gets its connection: the live object in the registering process, the spec anywhere."""

    def __init__(self, live: Any | None = None, spec: ConnectionSpec | None = None) -> None:
        self.live = live
        self.spec = spec
        self.live_dropped: bool = False
        self._lock = threading.Lock()
        self._opened: Any | None = None

    @classmethod
    def from_entry(cls, entry: Any) -> "ConnectionSource":
        if isinstance(entry, ConnectionSpec):
            return cls(spec=entry)
        return cls(live=entry)

    def open_for(self, cfw_class: type[Any]) -> Any | None:
        """Return the live handle: `self.live` if set, else a per-source memoized handle opened
        once via `cfw_class.open_connection(self.spec)`. No spec and no live returns None."""
        with self._lock:
            if self.live is not None:
                return self.live
            if self.spec is None:
                return None
            if self._opened is None:
                self._opened = cfw_class.open_connection(self.spec)
            return self._opened

    def __getstate__(self) -> dict[str, Any]:
        return {"live": None, "spec": self.spec, "live_dropped": self.live_dropped or self.live is not None}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._opened = None
