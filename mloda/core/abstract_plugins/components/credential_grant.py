"""Hold the credentials a step declared for the duration of its execution, masked in errors and logs."""

from collections.abc import Mapping
from contextvars import ContextVar, Token
from typing import Any, Protocol

from mloda.core.abstract_plugins.components.credential_leaves import (
    CredentialKeyPolicy,
    credential_identity_literals,
    credential_secret_literals,
)
from mloda.core.abstract_plugins.components.credential_scrub import redact_escaping_exception, registered_secrets_scope


class CredentialLookup:
    """The opened credentials of one step; refuses every read once closed."""

    def __init__(self, values: Mapping[str, Any], policy: CredentialKeyPolicy | None = None) -> None:
        self._data: dict[str, Any] = dict(values)
        self.policy = policy
        self._closed = False
        self._reads: dict[str, int] = {}
        self._counter = 0

    @property
    def closed(self) -> bool:
        return self._closed

    def credential(self, handle: str) -> Any:
        if self._closed:
            raise RuntimeError(f"Credential '{handle}' is no longer available: the step has ended.")
        if handle not in self._data:
            raise KeyError(f"Credential '{handle}' was not declared by this step.")
        return self._data[handle]

    def record_read(self, handle: str) -> None:
        self._counter += 1
        self._reads[handle] = self._counter

    def read_mark(self) -> int:
        return self._counter

    def read_since(self, mark: int) -> tuple[str, ...]:
        return tuple(sorted(handle for handle, seq in self._reads.items() if seq > mark))

    def declared(self) -> tuple[str, ...]:
        return tuple(sorted(self._data))

    def close(self) -> None:
        self._data = {}
        self._closed = True

    def _values(self) -> list[Any]:
        return list(self._data.values())


class CredentialGrant(Protocol):
    def open(self) -> CredentialLookup: ...


class StaticGrant:
    """A grant over already resolved values; open() hands them over once."""

    def __init__(self, values: Mapping[str, Any], policy: CredentialKeyPolicy | None = None) -> None:
        self._values: dict[str, Any] = dict(values)
        self._policy = policy

    def open(self) -> CredentialLookup:
        values, self._values = self._values, {}
        return CredentialLookup(values, self._policy)


_GRANT_HOLDER: ContextVar[CredentialLookup | None] = ContextVar("mloda_credential_grant", default=None)


class grant_scope:  # noqa: N801
    """Open a grant, mask its secret leaves and expose it for the block (a class: see registered_secrets_scope)."""

    def __init__(self, grant: CredentialGrant | None) -> None:
        self._grant = grant
        self._frames: list[
            tuple[CredentialLookup, registered_secrets_scope, redact_escaping_exception, Token[Any]]
        ] = []

    def __enter__(self) -> None:
        lookup = CredentialLookup({}) if self._grant is None else self._grant.open()
        leaves: set[str] = set()
        identity: set[str] = set()
        for value in lookup._values():
            leaves |= credential_secret_literals(value, lookup.policy)
            identity |= credential_identity_literals(value)
        secrets = registered_secrets_scope(frozenset(leaves), frozenset(identity))
        secrets.__enter__()
        redaction = redact_escaping_exception()
        redaction.__enter__()
        self._frames.append((lookup, secrets, redaction, _GRANT_HOLDER.set(lookup)))

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        lookup, secrets, redaction, token = self._frames.pop()
        _GRANT_HOLDER.reset(token)
        try:
            lookup.close()
            redaction.__exit__(exc_type, exc, tb)
        finally:
            secrets.__exit__(exc_type, exc, tb)


def credential_read_mark() -> int:
    lookup = _GRANT_HOLDER.get()
    return 0 if lookup is None else lookup.read_mark()


def credentials_read_since(mark: int) -> tuple[str, ...]:
    lookup = _GRANT_HOLDER.get()
    return () if lookup is None else lookup.read_since(mark)


def _read_granted(handle: str) -> Any:
    """Test-only reader of the current step's granted credential."""
    lookup = _GRANT_HOLDER.get()
    if lookup is None:
        raise RuntimeError(f"Credential '{handle}' is not available outside a step.")
    return lookup.credential(handle)
