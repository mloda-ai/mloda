"""Single source of truth for the installed mloda package version."""

from __future__ import annotations

# Redundant alias form: makes ``importlib`` an explicit re-export so callers (and tests)
# may patch ``mloda.core.version.importlib.metadata.version`` under mypy --strict.
import importlib as importlib
import importlib.metadata

# Package metadata cannot change within a process, so the lookup is memoized:
# repeated importlib.metadata parsing was a measured hot path (docs enumeration).
_mloda_version_cache: str | None = None


def get_mloda_version() -> str:
    """Returns the installed 'mloda' version, or "0.0.0" when the distribution is not installed."""
    global _mloda_version_cache
    if _mloda_version_cache is not None:
        return _mloda_version_cache
    try:
        _mloda_version_cache = importlib.metadata.version("mloda")
    except importlib.metadata.PackageNotFoundError:
        _mloda_version_cache = "0.0.0"
    return _mloda_version_cache
