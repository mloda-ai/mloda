import importlib
import sys
from pathlib import Path

import logging
from types import ModuleType
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PluginLoader:
    def __init__(self) -> None:
        """
        Initialize the PluginLoader with the base plugin package and dependencies.

        The PluginLoader is responsible for loading plugins from a specific package and its subpackages.

        You can chose from:
        - load_all_plugins
        - load_group

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
        self.plugins: Dict[str, ModuleType] = {}
        self.plugin_graph: Dict[str, List[str]] = {}  # Graph representation of plugins

    @staticmethod
    def all() -> "PluginLoader":
        plugin_loader = PluginLoader()
        plugin_loader.load_all_plugins()
        return plugin_loader

    def __repr__(self) -> str:
        return f"PluginLoader plugins: {self.plugins}"

    def load_group(self, group_name: str) -> None:
        """
        Load all plugins within a specific group (subfolder), including subgroups (nested directories).
        Args:
            group_name (str): The name of the group (folder) to load (e.g., "input_data").
        """
        group_path = self._get_group_path(group_name)
        if not group_path.is_dir():
            raise ValueError(f"Group '{group_name}' does not exist in the package '{self.base_package}'")

        # Traverse the group directory and its subdirectories recursively
        for item in group_path.rglob("*.py"):  # Finds all .py files in the directory
            if item.name == "__init__.py":
                continue  # Skip __init__.py
            relative_path = item.relative_to(group_path.parent).with_suffix("")  # Relative path without .py
            module_path = ".".join(relative_path.parts)  # Convert to module path
            self._load_plugin(module_path)

    def _load_plugin(self, module_path: str) -> None:
        """Internal function to load a plugin."""
        full_module_name = f"{self.base_package}.{module_path}"
        if full_module_name in sys.modules:
            self.plugins[full_module_name] = sys.modules[full_module_name]
            self._add_plugin_to_graph(full_module_name)
            return

        module = importlib.import_module(full_module_name)
        self.plugins[full_module_name] = module
        self._add_plugin_to_graph(full_module_name)

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

    def display_plugin_graph(self, plugin_category: Optional[str] = None) -> List[str]:
        """Display the plugin graph."""

        _list_plugins_dependencies: List[str] = []

        for plugin, dependencies in self.plugin_graph.items():
            if plugin_category is not None:
                if plugin_category not in plugin:
                    continue

            _list_plugins_dependencies.append(f"{plugin} -> {dependencies}")

        if len(_list_plugins_dependencies) == 0:
            raise ValueError(f"No plugins found for category {plugin_category}")
        return _list_plugins_dependencies

    def list_loaded_modules(self, plugin_category: Optional[str] = None) -> List[str]:
        """List all loaded modules (plugins)."""

        _list_plugins_dependencies: List[str] = []

        for plugin in self.plugins.keys():
            if plugin_category is not None:
                if plugin_category not in plugin:
                    continue

            _list_plugins_dependencies.append(plugin)

        return _list_plugins_dependencies
