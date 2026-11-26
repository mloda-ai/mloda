"""Test-only reader of the current step's granted credential."""

from typing import Any

from mloda.core.abstract_plugins.components.credential_grant import _GRANT_HOLDER


def _read_granted(handle: str) -> Any:
    lookup = _GRANT_HOLDER.get()
    if lookup is None:
        raise RuntimeError(f"Credential '{handle}' is not available outside a step.")
    return lookup.credential(handle)
