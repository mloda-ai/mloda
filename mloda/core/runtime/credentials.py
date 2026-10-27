"""Private credential transport and per-hook access, separate from plan and run metadata."""

from collections.abc import Mapping
from contextvars import ContextVar, Token
from typing import Any, cast

from mloda.core.abstract_plugins.components.credential import RegisteredCredential
from mloda.core.abstract_plugins.components.credential_scrub import register_selected_secrets, selected_secrets_scope
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection, _resolve_handle


class RuntimeCredentialRegistry:
    """Snapshot credential handles and kinds without carrying live connections to workers."""

    def __init__(self, collection: DataAccessCollection | None) -> None:
        self.credentials = {} if collection is None else dict(collection.credentials)
        self.kinds = {} if collection is None else collection.handles()

    def read(self, handle: str) -> RegisteredCredential:
        return cast(RegisteredCredential, _resolve_handle("credentials", self.credentials, self.kinds, handle))


_RUNTIME_CREDENTIALS: ContextVar[RuntimeCredentialRegistry | None] = ContextVar(
    "mloda_runtime_credentials", default=None
)
_HOOK_CREDENTIALS: ContextVar[RuntimeCredentialRegistry | None] = ContextVar("mloda_hook_credentials", default=None)


def current_runtime_credentials() -> RuntimeCredentialRegistry | None:
    return _RUNTIME_CREDENTIALS.get()


class runtime_credentials_scope:  # noqa: N801
    """Restore both access and selected literals when a hook, nested run or preparation ends."""

    def __init__(self, credentials: RuntimeCredentialRegistry | None, *, hook: bool = False) -> None:
        self._credentials = credentials
        self._hook = hook
        self._tokens: list[tuple[Token[RuntimeCredentialRegistry | None], Token[RuntimeCredentialRegistry | None]]] = []
        self._secrets = selected_secrets_scope()

    def __enter__(self) -> None:
        self._tokens.append(
            (
                _RUNTIME_CREDENTIALS.set(self._credentials),
                _HOOK_CREDENTIALS.set(self._credentials if self._hook else None),
            )
        )
        self._secrets.__enter__()

    def __exit__(self, *exc_info: object) -> None:
        self._secrets.__exit__(*exc_info)
        runtime_token, hook_token = self._tokens.pop()
        _HOOK_CREDENTIALS.reset(hook_token)
        _RUNTIME_CREDENTIALS.reset(runtime_token)


def _collect_literals(value: Any, literals: set[str], seen: set[int]) -> None:
    if isinstance(value, (str, bytes)):
        if not value:
            return
        if isinstance(value, bytes):
            variants = {value.decode("latin1"), value.decode("utf8", "replace"), repr(value), repr(value)[2:-1]}
        else:
            variants = {value, repr(value)[1:-1]}
        literals.update(variants)
        literals.update(text.replace("\\", "\\\\") for text in variants)
        return
    if isinstance(value, (Mapping, list, tuple, set, frozenset)) and id(value) not in seen:
        seen.add(id(value))
        for item in value.values() if isinstance(value, Mapping) else value:
            _collect_literals(item, literals, seen)


def read_credential(handle: str) -> RegisteredCredential:
    credentials = _HOOK_CREDENTIALS.get()
    if credentials is None:
        raise RuntimeError("Credentials can only be read inside an active feature group runtime hook.")
    if not isinstance(handle, str):
        raise TypeError("A runtime credential handle must be a string.")
    credential = credentials.read(handle)
    literals: set[str] = set()
    _collect_literals(credential, literals, set())
    register_selected_secrets(frozenset(literals))
    return credential
