"""Resolve the installed distribution version of the package owning a module."""

import functools
import importlib.metadata
from collections.abc import Mapping

from mloda.core.abstract_plugins.components.utils import safe_field


@functools.cache
def _read_distribution(dist_name: str) -> importlib.metadata.Distribution | None:
    return safe_field(lambda: importlib.metadata.distribution(dist_name), None, field="plugin_version")


def _owns_module(entry: str, module_path: str) -> bool:
    if entry == f"{module_path}/__init__.py":
        return True
    prefix = f"{module_path}."
    if not entry.startswith(prefix):
        return False
    # Keep the leading "." so a compiled extension's tag is matched by suffix; ".pyi" stubs never own a module.
    remainder = entry[len(module_path) :]
    return "/" not in remainder and remainder.endswith((".py", ".so", ".pyd"))


def _distribution_owning(module_name: str, distributions: list[str]) -> str | None:
    """First distribution whose file manifest lists module_name, else None."""
    module_path = module_name.replace(".", "/")
    for dist_name in distributions:
        dist = _read_distribution(dist_name)
        files = None if dist is None else safe_field(lambda: dist.files, None)
        if files is None:
            continue
        if any(_owns_module(str(f), module_path) for f in files):
            return dist_name
    return None


@functools.cache
def _read_packages_distributions() -> Mapping[str, list[str]]:
    empty: Mapping[str, list[str]] = {}
    return safe_field(lambda: importlib.metadata.packages_distributions(), empty, field="plugin_version")


@functools.cache
def resolve_plugin_version(module_name: str) -> str | None:
    """Distribution version of the package owning `module_name`, else None."""
    distributions = _read_packages_distributions().get(module_name.split(".")[0])
    if not distributions:
        return None

    dist_name = distributions[0] if len(set(distributions)) == 1 else _distribution_owning(module_name, distributions)
    if dist_name is None:
        return None

    return safe_field(lambda: importlib.metadata.version(dist_name), None, field="plugin_version")
