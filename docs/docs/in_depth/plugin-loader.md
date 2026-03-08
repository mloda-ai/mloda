# PluginLoader

The PluginLoader dynamically loads and manages plugins from the `mloda_plugins` package.

## Quick Start

```python
from mloda.user import PluginLoader

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
Loads all plugins from a specific group folder, including nested subdirectories. Supports slash-separated paths for subdirectories (e.g. `"feature_group/input_data/read_files"`).

### `load_matching(group_name: str, pattern: str)`
Loads only files within a group whose filename matches a glob pattern (e.g. `"*transformer*"`). Useful when only a subset of files in a group is needed.

### `load_all_plugins()`
Loads all plugin groups.

### `disable_auto_load(group: str)`
Prevents a group from being auto-loaded lazily. Call this before any plugin discovery if you want full control over what is loaded.

```python
PluginLoader.disable_auto_load("feature_group/input_data/read_files")
PluginLoader.disable_auto_load("compute_framework")
```

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
