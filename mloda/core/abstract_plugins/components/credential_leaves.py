"""Walk a registered credential and classify each str/bytes leaf as secret or known-safe."""

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, NamedTuple

from mloda.core.abstract_plugins.components.credential import Credential
from mloda.core.abstract_plugins.components.credential_scrub import _SECRET_KEY

# Normalized (lowercase, "_" and "-" removed); "tablename" must match ReadDBFG.TABLE_KEY.
_SAFE_CREDENTIAL_KEYS = frozenset(
    {
        "host",
        "hostname",
        "server",
        "serverhostname",
        "port",
        "database",
        "db",
        "dbname",
        "schema",
        "catalog",
        "warehouse",
        "account",
        "role",
        "project",
        "dataset",
        "location",
        "region",
        "bucket",
        "container",
        "namespace",
        "cluster",
        "endpoint",
        "endpointurl",
        "httppath",
        "driver",
        "dialect",
        "scheme",
        "protocol",
        "sslmode",
        "authenticator",
        "applicationname",
        "clientid",
        "tenantid",
        "user",
        "username",
        "sqlite",
        "path",
        "file",
        "folder",
        "tablename",
        "table",
        "databasename",
        "schemaname",
        "bucketname",
        "projectid",
        "datasetid",
        "filepath",
    }
)


MIN_REDACT_LENGTH = 8


def normalize_credential_key(key: str) -> str:
    return key.lower().replace("_", "").replace("-", "")


class CredentialKeyPolicy(NamedTuple):
    """Normalized keys a plugin or steward declared public or secret; secret wins over public and base."""

    public: frozenset[str]
    secret: frozenset[str]


def credential_key_names(label: str, value: Any) -> tuple[str, ...]:
    """The items of an iterable of non-empty str; a bare str or any other item is a ValueError."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{label} must be an iterable of key names, got {type(value).__name__}.")
    items = tuple(value)
    if not all(isinstance(item, str) and item for item in items):
        raise ValueError(f"{label} must hold non-empty str key names.")
    return items


class CredentialLeaf(NamedTuple):
    path: str
    leaf: Any
    secret: bool
    shaped: bool


def _is_shaped(key: Any) -> bool:
    return isinstance(key, str) and _SECRET_KEY.fullmatch(key) is not None


def _is_safe(key: Any, policy: CredentialKeyPolicy | None) -> bool:
    if not isinstance(key, str):
        return False
    normalized = normalize_credential_key(key)
    if policy is None:
        return normalized in _SAFE_CREDENTIAL_KEYS
    return normalized not in policy.secret and (normalized in _SAFE_CREDENTIAL_KEYS or normalized in policy.public)


def iter_credential_leaves(value: Any, policy: CredentialKeyPolicy | None = None) -> Iterator[CredentialLeaf]:
    """Yield every str/bytes leaf; secret unless its nearest key is safe and no key on the way is credential-shaped."""
    visited: set[tuple[int, bool, bool]] = set()
    stack: list[tuple[Any, str, Any, bool]] = [(value, "", None, False)]
    while stack:
        node, path, key, ancestor_shaped = stack.pop()
        shaped = ancestor_shaped or _is_shaped(key)
        key_secret = not _is_safe(key, policy)
        if isinstance(node, (str, bytes, bytearray)):
            yield CredentialLeaf(path, node, shaped or key_secret, shaped)
            continue
        if isinstance(node, Credential):
            guard = (id(node), shaped, key_secret)
            if guard in visited:
                continue
            visited.add(guard)
            node = node.data
        elif isinstance(node, (Mapping, list, tuple, set, frozenset)):
            guard = (id(node), shaped, key_secret)
            if guard in visited:
                continue
            visited.add(guard)
        else:
            continue
        children: list[tuple[Any, str, Any, bool]]
        if isinstance(node, Mapping):
            children = [(v, f"{path}.{k}" if path else str(k), k, shaped) for k, v in node.items()]
        else:
            children = [(v, f"{path}[{i}]", key, shaped) for i, v in enumerate(node)]
        stack.extend(reversed(children))


def _literals_where(
    value: Any, predicate: Callable[[CredentialLeaf], bool], policy: CredentialKeyPolicy | None = None
) -> frozenset[str]:
    return frozenset(
        item.leaf
        for item in iter_credential_leaves(value, policy)
        if predicate(item) and isinstance(item.leaf, str) and len(item.leaf) >= MIN_REDACT_LENGTH and item.leaf.strip()
    )


def credential_secret_literals(value: Any, policy: CredentialKeyPolicy | None = None) -> frozenset[str]:
    """The secret str leaves of one credential value that are long enough and not blank to mask by exact match."""
    return _literals_where(value, lambda item: item.secret, policy)


def credential_identity_literals(value: Any) -> frozenset[str]:
    """The credential-shaped str leaves of one credential value that are long enough and not blank."""
    return _literals_where(value, lambda item: item.shaped)
