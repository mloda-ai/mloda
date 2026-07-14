"""The single place mloda core reaches an optional backend: at the point of use, never at module import."""

from __future__ import annotations

import importlib
import sys
from typing import Any


def require(module: str, reason: str) -> Any:
    """Import an optional backend at the point of use, or raise ImportError naming the extra to install.

    The install hint is only rewritten for a genuine absence of the distribution itself. A module that is
    found but fails to load (broken .so, ABI mismatch, missing transitive dependency) raises its own error,
    since telling the user to install what they already have hides the real fault.
    """
    distribution = module.partition(".")[0]
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != distribution and not (exc.name or "").startswith(f"{distribution}."):
            raise
        raise ImportError(
            f"{distribution} is required for {reason}. Install it with: pip install 'mloda[{distribution}]'"
        ) from exc


def loaded(module: str) -> Any | None:
    """The module if it is already imported, else None. Never triggers the import of an absent module.

    An import in flight in ANOTHER thread publishes its module to sys.modules before running its body, so the
    raw entry can be half-built. Resolving it through an import blocks on the per-module import lock until
    that body has finished. A same-thread reentrant (circular) import owns the lock and is not covered.
    """
    found = sys.modules.get(module)
    if found is None:
        return None
    if getattr(getattr(found, "__spec__", None), "_initializing", True):
        # Still being executed, or unknowable: take the slow but safe path. This is the check CPython 3.11+
        # does inside _find_and_load, hand-applied so 3.10 skips the process-global import lock too.
        return importlib.import_module(module)
    return found
