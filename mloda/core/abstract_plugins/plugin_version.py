"""Resolve the installed distribution version of the package owning a module."""

import importlib.metadata
from collections.abc import Mapping

from mloda.core.abstract_plugins.components.utils import safe_field

_packages_distributions_cache: Mapping[str, list[str]] | None = None


def resolve_plugin_version(module_name: str) -> str | None:
    """Distribution version of the package owning `module_name`'s top-level package, else None."""
    global _packages_distributions_cache
    if _packages_distributions_cache is None:
        _packages_distributions_cache = importlib.metadata.packages_distributions()

    distributions = _packages_distributions_cache.get(module_name.split(".")[0])
    if not distributions:
        return None

    return safe_field(lambda: importlib.metadata.version(distributions[0]), None, field="plugin_version")
