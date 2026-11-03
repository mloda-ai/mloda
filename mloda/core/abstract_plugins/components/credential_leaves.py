"""Walk a registered credential and classify each str/bytes leaf as secret or known-safe."""

from collections.abc import Iterator, Mapping
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
    }
)


class CredentialLeaf(NamedTuple):
    path: str
    leaf: Any
    secret: bool
    shaped: bool


def _is_shaped(key: Any) -> bool:
    return isinstance(key, str) and _SECRET_KEY.fullmatch(key) is not None


def _is_safe(key: Any) -> bool:
    return isinstance(key, str) and key.lower().replace("_", "").replace("-", "") in _SAFE_CREDENTIAL_KEYS


def iter_credential_leaves(value: Any) -> Iterator[CredentialLeaf]:
    """Yield every str/bytes leaf; secret unless its nearest key is safe and no key on the way is credential-shaped."""
    visited: set[tuple[int, bool, bool]] = set()
    stack: list[tuple[Any, str, Any, bool]] = [(value, "", None, False)]
    while stack:
        node, path, key, ancestor_shaped = stack.pop()
        shaped = ancestor_shaped or _is_shaped(key)
        key_secret = not _is_safe(key)
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
