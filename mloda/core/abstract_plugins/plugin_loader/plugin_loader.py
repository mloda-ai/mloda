import importlib
import importlib.metadata
import inspect
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

import logging
from types import ModuleType
from typing import Any, ClassVar, Optional

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.function_extender import Extender
from mloda.core.abstract_plugins.plugin_registry.plugin_policy import PluginPolicyViolationError
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import (
    PluginRegistry,
    PluginSource,
    register_module_plugins,
    warn_policy_denied,
)

logger = logging.getLogger(__name__)

# Missing modules NOT in this set are treated as genuine errors and re-raised; the set is intentionally
# generous because under-inclusion breaks `tox -e nopyarrow`, while over-inclusion is harmless (it only
# ever causes a skip when that dependency is genuinely absent).
OPTIONAL_PLUGIN_DEPENDENCIES: frozenset[str] = frozenset(
    {
        "pyarrow",
        "pandas",
        "polars",
        "numpy",
        "scipy",
        "duckdb",
        "pyiceberg",
        "pyspark",
        "opentelemetry",
        "sklearn",
        "joblib",
        "nltk",
        "anthropic",
        "openai",
        "google",
        "yaml",
    }
)

ENTRY_POINT_GROUPS: dict[str, type[Any]] = {
    "mloda.feature_groups": FeatureGroup,
    "mloda.compute_frameworks": ComputeFramework,
    "mloda.extenders": Extender,
}


class PluginLoader:
    _disabled_groups: ClassVar[set[str]] = set()
    _cached_loader: ClassVar[Optional["PluginLoader"]] = None
    _cache_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def disable_auto_load(cls, group: str) -> None:
        cls._disabled_groups.add(group)

    def __init__(self) -> None:
        """
        Initialize the PluginLoader with the base plugin package and dependencies.

        The PluginLoader is responsible for loading plugins from a specific package and its subpackages.

        You can choose from:
        - load_all_plugins
        - load_group
        - load_matching

        You can show loaded modules with:
        - list_loaded_modules

        You can show dependencies with:
        - display_plugin_graph

        """
        self.base_package = "mloda_plugins"
        self.dependencies = {
            "input_data": "feature_group",  # Depends on input_data
            "feature_group": None,  # root group
            "compute_framework": None,  # root group
            "function_extender": None,  # root group
        }
        self.plugins: dict[str, ModuleType] = {}
        self.plugin_graph: dict[str, list[str]] = {}  # Graph representation of plugins

    @classmethod
    def all(cls, force_reload: bool = False) -> "PluginLoader":
        """Return a fully loaded PluginLoader, building and caching it on first use.

        The result is cached: repeated calls return the same loader without redoing the load work.

        Pass ``force_reload=True`` (or call ``reset_cache()`` first) to rebuild, for example after the
        default plugin registry has been cleared/restored or to pick up newly installed entry points.
        This is the supported way to re-populate registry state.

        Must not be called re-entrantly from within plugin import / entry-point loading: the cache lock
        is non-reentrant, so a re-entrant call during the initial build would deadlock. Treat the returned
        loader as read-only (do not mutate its ``plugins``/``plugin_graph``), since it is shared across callers.
        """
        if not force_reload and cls._cached_loader is not None:
            return cls._cached_loader
        with cls._cache_lock:
            if force_reload or cls._cached_loader is None:
                plugin_loader = cls()
                plugin_loader.load_all_plugins()
                plugin_loader.load_entry_points()
                cls._cached_loader = plugin_loader
            return cls._cached_loader

    @classmethod
    def reset_cache(cls) -> None:
        with cls._cache_lock:
            cls._cached_loader = None

    def load_entry_points(self, group: str | None = None) -> list[str]:
        """Discover installed entry-point manifests and register their plugin classes."""
        if group is not None and group not in ENTRY_POINT_GROUPS:
            valid = ", ".join(ENTRY_POINT_GROUPS)
            raise ValueError(f"Unknown entry-point group '{group}'. Valid groups are: {valid}.")
        groups = [group] if group is not None else list(ENTRY_POINT_GROUPS)
        keys: list[str] = []
        for group_name in groups:
            base_type = ENTRY_POINT_GROUPS[group_name]
            for entry_point in importlib.metadata.entry_points(group=group_name):
                try:
                    manifest = entry_point.load()
                except ModuleNotFoundError as e:
                    root = e.name.split(".")[0] if e.name else None
                    if root in OPTIONAL_PLUGIN_DEPENDENCIES:
                        logger.debug(
                            "Skipping entry point %s: missing optional dependency %s", entry_point.name, e.name
                        )
                        continue
                    raise
                keys.extend(self._register_manifest(entry_point.name, group_name, base_type, manifest))
        return sorted(set(keys))

    def _register_manifest(self, label: str, group_name: str, base_type: type[Any], manifest: Any) -> list[str]:
        """Validate a manifest sequence and register its concrete classes."""
        if isinstance(manifest, (str, bytes)) or not isinstance(manifest, Sequence):
            raise TypeError(
                f"Entry point '{label}' must resolve to a list or tuple of plugin classes, got {manifest!r}."
            )
        registry = PluginRegistry.default()
        keys: list[str] = []
        for cls in manifest:
            if not isinstance(cls, type):
                raise TypeError(f"Entry point '{label}' manifest contains a non-class item: {cls!r}.")
            if not issubclass(cls, base_type):
                raise TypeError(
                    f"Entry-point group '{group_name}' requires subclasses of {base_type.__name__}; "
                    f"got {cls.__qualname__}."
                )
            if inspect.isabstract(cls):
                continue
            try:
                key = registry.register(cls, source=PluginSource.ENTRY_POINT, replace=False)
            except PluginPolicyViolationError:
                warn_policy_denied(registry, f"{cls.__module__}:{cls.__qualname__}")
                continue
            keys.append(key)
        return keys

    def __repr__(self) -> str:
        return f"PluginLoader plugins: {self.plugins}"

    def load_matching(self, group_name: str, pattern: str) -> None:
        """Load only files within a group whose filename matches a glob pattern."""
        group_path = self._get_group_path(group_name)
        if not group_path.is_dir():
            raise ValueError(f"Group '{group_name}' does not exist in the package '{self.base_package}'")
        for item in group_path.rglob("*.py"):
            if item.name == "__init__.py":
                continue
            if not item.match(pattern):
                continue
            relative_path = item.relative_to(group_path.parent).with_suffix("")
            module_path = ".".join(relative_path.parts)
            self._load_plugin(module_path)

    def load_group(self, group_name: str) -> None:
        """
        Load all plugins within a specific group (subfolder), including subgroups (nested directories).
        Args:
            group_name (str): The name of the group (folder) to load (e.g., "input_data").
        """
        group_path = self._get_group_path(group_name)
        if not group_path.is_dir():
            raise ValueError(f"Group '{group_name}' does not exist in the package '{self.base_package}'")

        self._load_plugins_from_path(group_path)

    def _load_plugins_from_path(self, group_path: Path) -> None:
        """
        Loads plugins from a given path, recursively traversing subdirectories.
        """
        for item in group_path.rglob("*.py"):  # Finds all .py files in the directory
            if item.name == "__init__.py":
                continue  # Skip __init__.py
            relative_path = item.relative_to(self._get_group_path("")).with_suffix("")  # Relative path without .py
            module_path = ".".join(relative_path.parts)  # Convert to module path
            self._load_plugin(module_path)

    def _load_plugin(self, module_path: str) -> None:
        """Internal function to load a plugin."""
        full_module_name = f"{self.base_package}.{module_path}"
        if full_module_name in sys.modules:
            cached_module = sys.modules[full_module_name]
            self.plugins[full_module_name] = cached_module
            self._add_plugin_to_graph(full_module_name)
            register_module_plugins(cached_module, source="loader")
            return

        try:
            module = importlib.import_module(full_module_name)
        except ModuleNotFoundError as e:
            if e.name and (e.name == self.base_package or e.name.startswith(self.base_package + ".")):
                raise
            root = e.name.split(".")[0] if e.name else None
            if root in OPTIONAL_PLUGIN_DEPENDENCIES:
                logger.debug("Skipping plugin %s: missing optional dependency %s", full_module_name, e.name)
                return
            raise

        self.plugins[full_module_name] = module
        self._add_plugin_to_graph(full_module_name)
        register_module_plugins(module, source="loader")

    def load_all_plugins(self) -> None:
        """Load all groups (top-level folders) and their plugins."""
        base_path = self._get_group_path("")
        for item in base_path.iterdir():
            if item.is_dir() and (item / "__init__.py").exists():
                self.load_group(item.name)

    def _add_plugin_to_graph(self, plugin_name: str) -> None:
        """
        Add a plugin to the graph.
        If the plugin is part of a group, add an edge from the parent group to the plugin.
        """
        if plugin_name not in self.plugin_graph:
            self.plugin_graph[plugin_name] = []

        # Check group dependencies
        group_name = self._get_group_from_plugin(plugin_name)
        parent_group = self.dependencies.get(group_name)
        if parent_group:
            parent_group_plugin = f"{self.base_package}.{parent_group}"
            self.plugin_graph[parent_group_plugin] = self.plugin_graph.get(parent_group_plugin, [])
            self.plugin_graph[parent_group_plugin].append(plugin_name)

    def _get_group_from_plugin(self, plugin_name: str) -> str:
        """
        Extract the group name from the plugin name.
        Assumes the plugin name is in the format '<base_package>.<group_name>.<plugin>'.
        """
        parts = plugin_name.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid plugin name: {plugin_name}")
        return parts[1]

    def _get_group_path(self, group_name: str) -> Path:
        """Get the filesystem path to a group (folder)."""
        base_package = importlib.import_module(self.base_package).__file__

        if base_package is None:
            raise ValueError(f"Base package '{self.base_package}' not found")

        package_path = Path(base_package).parent
        return package_path / group_name

    def display_plugin_graph(self, plugin_category: Optional[str] = None) -> list[str]:
        """Display the plugin graph."""

        _list_plugins_dependencies: list[str] = []

        for plugin, dependencies in self.plugin_graph.items():
            if plugin_category is not None:
                if plugin_category not in plugin:
                    continue

            _list_plugins_dependencies.append(f"{plugin} -> {dependencies}")

        if len(_list_plugins_dependencies) == 0:
            raise ValueError(f"No plugins found for category {plugin_category}")
        return _list_plugins_dependencies

    def list_loaded_modules(self, plugin_category: Optional[str] = None) -> list[str]:
        """List all loaded modules (plugins)."""

        _list_plugins_dependencies: list[str] = []

        for plugin in self.plugins.keys():
            if plugin_category is not None:
                if plugin_category not in plugin:
                    continue

            _list_plugins_dependencies.append(plugin)

        return _list_plugins_dependencies
