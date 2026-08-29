"""Resolve the installed distribution version of the package owning a module."""

import functools
import importlib.metadata
from collections.abc import Mapping

from mloda.core.abstract_plugins.components.utils import safe_field

_packages_distributions_cache: Mapping[str, list[str]] | None = None


def _read_distribution(dist_name: str) -> importlib.metadata.Distribution | None:
    return safe_field(lambda: importlib.metadata.distribution(dist_name), None, field="plugin_version")


def _distribution_owning(module_name: str, distributions: list[str]) -> str:
    """Pick the distribution whose file manifest lists module_name; a shared namespace
    (e.g. "mloda") can map to several distributions, so the first candidate alone is
    not reliable ownership. Falls back to the first candidate when nothing matches.
    """
    module_path = module_name.replace(".", "/")
    for dist_name in distributions:
        dist = _read_distribution(dist_name)
        files = () if dist is None or dist.files is None else dist.files
        if any(str(f).removesuffix(".py") == module_path for f in files):
            return dist_name
    return distributions[0]


def _read_packages_distributions() -> Mapping[str, list[str]]:
    empty: Mapping[str, list[str]] = {}
    return safe_field(lambda: importlib.metadata.packages_distributions(), empty, field="plugin_version")


@functools.lru_cache(maxsize=None)
def resolve_plugin_version(module_name: str) -> str | None:
    """Distribution version of the package owning `module_name`, else None."""
    global _packages_distributions_cache
    if _packages_distributions_cache is None:
        _packages_distributions_cache = _read_packages_distributions()

    distributions = _packages_distributions_cache.get(module_name.split(".")[0])
    if not distributions:
        return None

    dist_name = _distribution_owning(module_name, distributions)
    return safe_field(lambda: importlib.metadata.version(dist_name), None, field="plugin_version")
