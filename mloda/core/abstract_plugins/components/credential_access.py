"""Run-time read of a declared credential, open only around calculate_feature and the artifact saver."""

import copy
import functools
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextvars import ContextVar, Token
from typing import Any, NoReturn

from mloda.core.abstract_plugins.components.credential_grant import (
    record_credential_read,
    _GRANT_HOLDER,
    CredentialLookup,
    StaticGrant,
    grant_scope,
)


class CredentialReadRefused(Exception):
    """A credential read outside its window, or of a handle the step did not declare."""


class CredentialFieldMissing(KeyError):
    """A credential field that does not exist; names the handle and field, never a value."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class ResolvedCredential(MutableMapping[str, Any]):
    """A private copy of one credential; redacted in repr, refuses pickling and copying."""

    __slots__ = ("handle", "version", "expires_at", "store", "_fields")
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        handle: str,
        fields: Mapping[str, Any],
        *,
        version: str | None = None,
        expires_at: Any | None = None,
        store: str | None = None,
    ) -> None:
        self.handle = handle
        self.version = version
        self.expires_at = expires_at
        self.store = store
        self._fields: dict[str, Any] = copy.deepcopy(dict(fields))

    def __getitem__(self, key: str) -> Any:
        if key not in self._fields:
            raise CredentialFieldMissing(f"Credential '{self.handle}' has no field '{key}'.")
        return self._fields[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._fields[key] = value

    def __delitem__(self, key: str) -> None:
        if key not in self._fields:
            raise CredentialFieldMissing(f"Credential '{self.handle}' has no field '{key}'.")
        del self._fields[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __repr__(self) -> str:
        return f"ResolvedCredential(handle={self.handle!r}, fields={sorted(self._fields)!r})"

    __str__ = __repr__

    def __reduce__(self) -> NoReturn:
        raise TypeError(f"Credential '{self.handle}' must not be pickled or copied.")


class _ReadableWindow:
    """One readable window over a lookup; copied contexts hold it and see nothing once it closes."""

    __slots__ = ("_lookup",)

    def __init__(self, lookup: CredentialLookup | None) -> None:
        self._lookup: CredentialLookup | None = lookup

    def values(self) -> Mapping[str, Any] | None:
        lookup = self._lookup
        return None if lookup is None else lookup.snapshot()

    def close(self) -> None:
        self._lookup = None


_READABLE: ContextVar[_ReadableWindow | None] = ContextVar("mloda_credential_readable", default=None)


class _ReadableScope:
    """Set the readable window for a block; a class, not @contextmanager (see registered_secrets_scope)."""

    def __init__(self, hide: bool) -> None:
        self._hide = hide
        self._frames: list[tuple[_ReadableWindow | None, Token[_ReadableWindow | None]]] = []

    def __enter__(self) -> None:
        window = None if self._hide else _ReadableWindow(_GRANT_HOLDER.get())
        self._frames.append((window, _READABLE.set(window)))

    def __exit__(self, *exc_info: object) -> None:
        window, token = self._frames.pop()
        _READABLE.reset(token)
        if window is not None:
            window.close()


def readable_credentials() -> _ReadableScope:
    """Make the held grant readable for the block."""
    return _ReadableScope(hide=False)


def hidden_credentials() -> _ReadableScope:
    """Make reads refuse inside the block."""
    return _ReadableScope(hide=True)


def restore_readable(method: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap method so it runs with the readable lookup as it is now, not as it is when called."""
    captured = _READABLE.get()

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        token = _READABLE.set(captured)
        try:
            return method(*args, **kwargs)
        finally:
            _READABLE.reset(token)

    return wrapper


def readable(method: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap method so it runs with the grant readable; keeps __self__."""

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with readable_credentials():
            return method(*args, **kwargs)

    if hasattr(method, "__self__"):
        wrapper.__self__ = method.__self__  # type: ignore[attr-defined]
    return wrapper


class credentials_scope:  # noqa: N801
    """Unit-test helper: grant mapping and open the readable window for the block."""

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        self._mapping = copy.deepcopy(dict(mapping))

    def __enter__(self) -> None:
        self._grant = grant_scope(StaticGrant(self._mapping))
        self._window = readable_credentials()
        self._grant.__enter__()
        self._window.__enter__()

    def __exit__(self, *exc_info: Any) -> None:
        self._window.__exit__(*exc_info)
        self._grant.__exit__(*exc_info)


def read_credential(handle: str) -> ResolvedCredential:
    """The declared credential for handle; readable only inside calculate_feature or an artifact saver."""
    window = _READABLE.get()
    values = None if window is None else window.values()
    if values is None:
        raise CredentialReadRefused(
            f"Credential '{handle}' can only be read inside calculate_feature or an artifact saver "
            "of a step that declares it."
        )
    if handle not in values:
        names = ", ".join(repr(name) for name in sorted(values))
        raise CredentialReadRefused(f"Credential '{handle}' was not declared by this step; declared: {names}.")
    resolved = ResolvedCredential(handle, values[handle])
    record_credential_read(handle)
    return resolved
