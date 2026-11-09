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
        self._data: dict[str, Any] | None = dict(values)
        self.policy = policy

    @property
    def closed(self) -> bool:
        return self._data is None

    def credential(self, handle: str) -> Any:
        data = self._data
        if data is None:
            raise RuntimeError(f"Credential '{handle}' is no longer available: the step has ended.")
        if handle not in data:
            raise KeyError(f"Credential '{handle}' was not declared by this step.")
        return data[handle]

    def declared(self) -> tuple[str, ...]:
        data = self._data
        return () if data is None else tuple(sorted(data))

    def snapshot(self) -> Mapping[str, Any] | None:
        """The values, or None once closed; the dict is never mutated, callers must not expose it."""
        return self._data

    def close(self) -> None:
        self._data = None

    def _values(self) -> list[Any]:
        data = self._data
        return [] if data is None else list(data.values())


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
            identity |= credential_identity_literals(value, lookup.policy)
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


_READ_RECORDERS: ContextVar[tuple[set[str], ...]] = ContextVar("mloda_credential_read_recorders", default=())


class read_recorder:  # noqa: N801
    """Collect the handles read inside the block, alongside any enclosing recorders; a fresh set per enter."""

    def __init__(self) -> None:
        self._tokens: list[Token[tuple[set[str], ...]]] = []

    def __enter__(self) -> set[str]:
        recorder: set[str] = set()
        self._tokens.append(_READ_RECORDERS.set((*_READ_RECORDERS.get(), recorder)))
        return recorder

    def __exit__(self, *exc_info: object) -> None:
        _READ_RECORDERS.reset(self._tokens.pop())


def record_credential_read(handle: str) -> None:
    for recorder in _READ_RECORDERS.get():
        recorder.add(handle)


def _read_granted(handle: str) -> Any:
    """Test-only reader of the current step's granted credential."""
    lookup = _GRANT_HOLDER.get()
    if lookup is None:
        raise RuntimeError(f"Credential '{handle}' is not available outside a step.")
    return lookup.credential(handle)
