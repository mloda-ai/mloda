# PluginLoader

The PluginLoader dynamically loads and manages plugins from the `mloda_plugins` package, and discovers plugins shipped by separately installed packages via [entry points](#entry-points).

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
Creates a loader, loads all bundled plugins, then folds in installed [entry points](#entry-points).

### `load_group(group_name: str)`
Loads all plugins from a specific group folder, including nested subdirectories. Supports slash-separated paths for subdirectories (e.g. `"feature_group/input_data/read_files"`).

### `load_matching(group_name: str, pattern: str)`
Loads only files within a group whose filename matches a glob pattern (e.g. `"*transformer*"`). Useful when only a subset of files in a group is needed.

### `load_all_plugins()`
Loads all plugin groups.

### `load_entry_points(group: str | None = None)`
Discovers plugins from installed packages that declare [entry points](#entry-points). Pass a group name to restrict discovery to one group; `None` loads all three. Returns the sorted list of registry keys it registered.

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

## Entry Points

Separately installed packages publish plugins through three entry-point groups, one per plugin base type:

- `mloda.feature_groups` for FeatureGroup classes
- `mloda.compute_frameworks` for ComputeFramework classes
- `mloda.extenders` for Extender classes

### Manifest Convention

A package declares one entry per group it ships plugins for. The entry points at a module attribute, the manifest: a plain sequence of plugin classes. The entry-point name (`my-pkg` below) is a package label only, never a registry key; registered classes keep the usual `module:qualname` keys.

```toml title="pyproject.toml"
[project.entry-points."mloda.feature_groups"]
my-pkg = "my_pkg.manifest:FEATURE_GROUPS"

[project.entry-points."mloda.compute_frameworks"]
my-pkg = "my_pkg.manifest:COMPUTE_FRAMEWORKS"
```

The manifest module lists the concrete classes explicitly:

```python title="my_pkg/manifest.py"
from mloda.provider import FeatureGroup

class CustomerChurnFeatureGroup(FeatureGroup):
    """Shipped by my-pkg; registers under its module:qualname key."""

FEATURE_GROUPS = [CustomerChurnFeatureGroup]
```

### Loading

Discovery is lazy: installing a package does nothing by itself. Manifests are imported and registered only when `load_entry_points()` runs, either directly or as the final step of `PluginLoader.all()`. Discovered classes register into the default registry with provenance `source="entry_point"` (see [Plugin Registry](plugin_registry.md)).

```python
from mloda.user import PluginLoader

loader = PluginLoader()

# All three groups; returns the sorted registry keys that were registered.
keys = loader.load_entry_points()
assert keys == sorted(keys)

# One group only; unknown group names raise ValueError.
fg_keys = loader.load_entry_points(group="mloda.feature_groups")
```

### Behavior Notes

- **Validation is loud.** A manifest that is not a sequence of classes, or that contains classes of the wrong base type for its group, raises `TypeError` naming the entry point.
- **Abstract classes are skipped** silently; manifests may list a shared abstract base alongside its concrete subclasses.
- **Collisions raise.** If a different class already holds a manifest class's `module:qualname` key, loading raises `PluginRegistryCollisionError`. Loading the same manifests twice is idempotent.
- **Missing optional dependencies skip.** If importing a manifest fails on a module listed in `OPTIONAL_PLUGIN_DEPENDENCIES` (pandas, polars, duckdb, ...), only that entry point is skipped; any other `ModuleNotFoundError` is re-raised.
- **Policies apply.** Entry-point registrations denied by an installed [plugin policy](plugin_registry.md#governance) are skipped with one warning per key.
