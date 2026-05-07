# Feature Group Resolution Errors

## Multiple Feature Groups Error

### The Problem

```
ValueError: Multiple feature groups found for feature '<feature_name>':
  - FeatureGroupA (...) [domain: ...]
  - FeatureGroupB (...) [domain: ...]
```

This error occurs when multiple distinct feature groups claim they can handle the same feature. Each feature must resolve to exactly one feature group to prevent conflicts.

If you previously hit this in a notebook because of a redefined feature group (re-running a cell that defines `class MyFG(FeatureGroup): ...`), that case is now auto-deduplicated. If this error still appears, it points to a real conflict between two distinct classes (different `(module, qualname)`).

### Solutions

#### 1. Use PluginCollector to enable or disable feature groups
Control which feature groups are loaded to prevent conflicts:

``` python
from mloda.user import PluginCollector

# Disable specific conflicting feature groups
collector = PluginCollector.disabled_feature_groups({ConflictingFeatureGroupA, ConflictingFeatureGroupB})

# Or enable only specific feature groups you need
collector = PluginCollector.enabled_feature_groups({RequiredFeatureGroupA, RequiredFeatureGroupB})
```

#### 2. Avoid loading all plugins
You can load plugins by importing the module as a class.

But you can also import just plugins from subfolders.

```python
from mloda.user import PluginLoader

plugin_loader = PluginLoader()
plugin_loader.load_group("feature_group") # load plugins only from mloda_plugins.feature_group
```

#### 3. Use Domains to Separate Feature Groups
```python
@classmethod
def get_domain(cls):
    return "sales"  # Makes this FG only handle 'sales' domain features
```

## FeatureGroup Redefinition Errors

### The Problem

```
ValueError: FeatureGroup redefined with different source code:
  - MyFG (__main__) source hash 5a3f0c12
  - MyFG (__main__) source hash b1e052a3
Set PluginCollector(...).set_allow_redefinition() to keep only the most recently defined version of each class.
If you are running this in a notebook, restart the kernel to clear stale class definitions.
```

This error occurs when the same feature group class name has been redefined with different source code in a long-lived Python namespace. Common triggers:

- Re-running a Jupyter cell that defines `class MyFG(FeatureGroup): ...` after editing the class body. IPython's `Out[N]` history holds a strong reference to the old class object, so it stays alive in `FeatureGroup.__subclasses__()`.
- Calling `importlib.reload` on a module that defines feature group classes.

If both versions of the class have **identical** source code (e.g., re-running a cell without edits), mloda silently keeps one. The error fires only when the source actually differs.

### Solutions

#### 1. Take the most recent version (iterative development)

For rapid iteration in notebooks, opt in to "newest wins" using the builder method on `PluginCollector`:

```python
from mloda.provider import FeatureGroup
from mloda.user import PluginCollector


class SomeFG(FeatureGroup):
    pass


plugin_collector = PluginCollector().set_allow_redefinition()
# Compose with disable/enable filters as needed:
plugin_collector = (
    PluginCollector.disabled_feature_groups({SomeFG})
    .set_allow_redefinition()
)
```

Pass this collector through to `mloda.run_all`, `resolve_feature`, or any other entry point that accepts a `plugin_collector` argument.

#### 2. Restart the kernel

Restarting the Jupyter kernel clears `Out[N]` history and unloads all stale class objects. Use this when you want a fully clean state and do not need to preserve other notebook state.

## Related Documentation

- [Feature Group Matching](../feature-group-matching.md)
- [PROPERTY_MAPPING Configuration](../property-mapping.md)
- [Feature Group Testing](../feature-group-testing.md)
