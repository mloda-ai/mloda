"""Typed credential slot for ``DataAccessCollection`` (issue #511).

Wraps exactly one credential mapping so a single credential dict can no longer
be mistaken for a ``{handle: value}`` registry. One type serves four user
groups: notebook users get a constructor with no nesting to get wrong, multi-
source production users keep a homogeneous handle registry, plugin authors
receive a dict subclass, and ops users get a value-redacting ``repr`` that
keeps secrets out of logs and tracebacks. Full rationale:
docs/docs/in_depth/named-data-access-handles.md.
"""

from typing import Any

from mloda.core.abstract_plugins.components.credential_scrub import redact_mapping
from mloda.core.abstract_plugins.components.hashable_dict import _deep_equal, _deep_hashable


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
        redacted = ", ".join(f"{key}={value!r}" for key, value in redact_mapping(self._data).items())
        return f"Credential({redacted})"

    def __hash__(self) -> int:
        return hash(_deep_hashable(self._data))

    # Deliberately not a registered deep node: Feature._reduce must keep a Credential
    # distinct from an equal plain dict.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Credential):
            return False
        return _deep_equal(self._data, other._data)


class RegisteredCredential(dict[str, Any]):
    """A registered credential mapping whose repr/str redact values; item access stays raw."""

    def __repr__(self) -> str:
        return repr(redact_mapping(self))
