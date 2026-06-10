"""Typed credential slot for ``DataAccessCollection`` (issue #511).

Wraps exactly one credential mapping so a single credential dict can no longer
be mistaken for a ``{handle: value}`` registry. One type serves four user
groups: notebook users get a constructor with no nesting to get wrong, multi-
source production users keep a homogeneous handle registry, plugin authors
keep receiving plain dicts (the wrapper is unwrapped at registration), and ops
users get a value-redacting ``repr`` that keeps secrets out of logs and
tracebacks. Full rationale: docs/docs/in_depth/named-data-access-handles.md.
"""

from typing import Any


class Credential:
    """One credential mapping built from a dict, keyword fields, or both."""

    def __init__(self, mapping: dict[str, Any] | None = None, /, **fields: Any) -> None:
        if mapping is not None and not isinstance(mapping, dict):
            raise TypeError(f"Credential expects a dict/mapping, got {type(mapping).__name__}.")
        merged = dict(mapping or {})
        merged.update(fields)
        if not merged:
            raise ValueError("Credential requires at least one field.")
        self._data: dict[str, Any] = merged

    @property
    def data(self) -> dict[str, Any]:
        """Return a shallow copy of the credential mapping."""
        return dict(self._data)

    def __repr__(self) -> str:
        redacted = ", ".join(f"{key}='***'" for key in self._data)
        return f"Credential({redacted})"
