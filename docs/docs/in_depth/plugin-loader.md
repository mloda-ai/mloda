# PluginLoader

The PluginLoader dynamically loads and manages plugins from the `mloda_plugins` package.

## Quick Start

```python
from mloda_core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader

# Load all plugins
loader = PluginLoader.all()

# Or load specific groups
loader = PluginLoader()
loader.load_group("feature_group")
```

## Main Methods

### `PluginLoader.all()`
Creates a loader and loads all available plugins.

### `load_group(group_name: str)`
Loads plugins from a specific group folder.

### `load_all_plugins()`
Loads all plugin groups.

### `list_loaded_modules(plugin_category: Optional[str])`
Lists loaded plugin modules, optionally filtered by category.

### `display_plugin_graph(plugin_category: Optional[str])`
Shows plugin dependencies as a graph.

## Example

```python
# Create loader and load plugins
loader = PluginLoader()
loader.load_all_plugins()

# List loaded modules
modules = loader.list_loaded_modules()

# Show dependencies
graph = loader.display_plugin_graph()
```

Plugins are automatically discovered from `.py` files in these directories.
