"""Picklable connection recipes for the ComputeFramework connection resolution seam.

ConnectionSpec is what travels across a process boundary; ConnectionSource wraps either
a live connection object or a spec, and drops the live object on pickle.
"""

from typing import Any


def _framework_name(framework: type[Any] | str) -> str:
    return framework if isinstance(framework, str) else framework.__name__


class ConnectionSpec:
    """Picklable recipe a ComputeFramework opens via open_connection(); a valid DataAccessCollection connection entry."""

    def __init__(self, framework: type[Any] | str, **params: Any) -> None:
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
        params_str = ", ".join(f"{key}={value!r}" for key, value in self.params.items())
        if params_str:
            return f"ConnectionSpec({name}, {params_str})"
        return f"ConnectionSpec({name})"


class ConnectionSource:
    """Where a framework instance gets its connection: the live object in the registering process, the spec anywhere."""

    def __init__(self, live: Any | None = None, spec: ConnectionSpec | None = None) -> None:
        self.live = live
        self.spec = spec
        self.live_dropped: bool = False

    @classmethod
    def from_entry(cls, entry: Any) -> "ConnectionSource":
        if isinstance(entry, ConnectionSpec):
            return cls(spec=entry)
        return cls(live=entry)

    def __getstate__(self) -> dict[str, Any]:
        return {"live": None, "spec": self.spec, "live_dropped": self.live_dropped or self.live is not None}
