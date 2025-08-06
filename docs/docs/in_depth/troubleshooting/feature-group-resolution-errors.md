# Feature Group Resolution Errors

## Multiple Feature Groups Error

### The Problem

```
ValueError: Multiple feature groups {<feature_groups>} found for feature name: <feature_name>
```

This error occurs when multiple feature groups claim they can handle the same feature. Each feature must resolve to exactly one feature group to prevent conflicts.

If you are running this in a notebook, please restart the kernel to clear any cached plugins. 
If you experience this multiple times, please open an [issue](https://github.com/mloda-ai/mloda/issues/) or [contact the maintainers](mailto:info@mloda.ai), so that we prioritize this.


### Solutions

#### 1. Use PlugInCollector to enable or disable feature groups
Control which feature groups are loaded to prevent conflicts:

``` python
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector

# Disable specific conflicting feature groups
collector = PlugInCollector.disabled_feature_groups({ConflictingFeatureGroupA, ConflictingFeatureGroupB})

# Or enable only specific feature groups you need
collector = PlugInCollector.enabled_feature_groups({RequiredFeatureGroupA, RequiredFeatureGroupB})
```

#### 2. Avoid loading all plugins
You can load plugins by importing the module as a class.

But you can also import just plugins from subfolders.

```python
from mloda_core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader

plugin_loader = PluginLoader()
plugin_loader.load_group("feature_group") # load plugins only from mloda_plugins.feature_group
```

#### 3. Use Domains to Separate Feature Groups
```python
@classmethod
def get_domain(cls):
    return "sales"  # Makes this FG only handle 'sales' domain features
```

## Related Documentation

- [Feature Group Matching](../feature-group-matching.md)
- [PROPERTY_MAPPING Configuration](../property-mapping.md)
- [Feature Group Testing](../feature-group-testing.md)
