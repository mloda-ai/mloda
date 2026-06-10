# Plugin Registry

The plugin registry is an explicit catalog of FeatureGroup, ComputeFramework, and Extender classes, keyed by a string name (default: `"module:qualname"`). The [PluginLoader](plugin-loader.md) feeds it automatically: every plugin class defined in a loaded module is registered with provenance `source="loader"`; manual registrations carry `source="manual"`. Registries are plain instances, and the engine and catalog APIs use the process-global default, `PluginRegistry.default()`.

## Registered vs Ad-Hoc Classes

Two kinds of plugin classes exist at runtime:

- **Registered**: classes defined in modules loaded by `PluginLoader`, plus anything registered manually via `register()`.
- **Ad-hoc**: any other subclass, for example a notebook cell, a test double, or a class in a module that was imported but not loaded.

Registration is additive. Defining a subclass and importing its module keeps working exactly as before:

- **Catalog APIs**: `list_registered()` reports only registered classes. The `get_*_docs()` functions keep walking all subclasses by default and accept `registered_only=True` to restrict the result to the registry.
- **Engine resolution**: while strict mode is `"off"` or `"warn"` (see below), the engine resolves registered plugins plus local ad-hoc subclasses. Only `"strict"` filters resolution to registered FeatureGroups.

```python
from mloda.provider import FeatureGroup
from mloda.user import PluginLoader, list_registered

PluginLoader.all()

registered = list_registered(FeatureGroup)
assert registered

class AdHocFeatureGroup(FeatureGroup):
    """Defined locally, for example in a notebook cell; never auto-registered."""

assert AdHocFeatureGroup not in list_registered(FeatureGroup)
```

## Contract Tests over Registered Plugins

`list_registered()` returns the classes of one plugin base type, sorted by registry key. That makes project-wide contract tests a simple loop: load your plugins, then assert a property on every registered class.

```python
from mloda.provider import FeatureGroup
from mloda.user import PluginLoader, list_registered

PluginLoader.all()

for fg in list_registered(FeatureGroup):
    name = fg.get_class_name()
    assert isinstance(name, str)
    assert name
```

Replace the assertion with whatever your project requires of every plugin: a docstring, a naming convention, a version scheme.

## Manual Registration (Notebooks, Third-Party Packages)

Classes that do not live in loaded plugin modules opt in with `register()`. Registering the same class twice is a no-op. Registering a *different* class under an already taken key raises `PluginRegistryCollisionError`; this happens when a notebook cell that defines a class is edited and re-run, because the re-run creates a new class object under the same key. Pass `replace=True` to accept the newest definition. `unregister()` removes an entry by key.

```python
from mloda.provider import FeatureGroup
from mloda.user import PluginRegistry, register
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistryCollisionError

class MyFeatureGroup(FeatureGroup):
    """First definition."""

key = register(MyFeatureGroup)
assert PluginRegistry.default().get(key) is MyFeatureGroup

class MyFeatureGroup(FeatureGroup):
    """Edited and re-run, as in a notebook cell: a new class under the same key."""

try:
    register(MyFeatureGroup)
    raise AssertionError("expected a collision")
except PluginRegistryCollisionError:
    pass

register(MyFeatureGroup, replace=True)
assert PluginRegistry.default().get(key) is MyFeatureGroup

PluginRegistry.default().unregister(key)
```

`register()` also accepts `name=` for a custom key, for example `register(MyFeatureGroup, name="my_pkg:MyFeatureGroup")`.

## Strict Mode

Strict mode controls how the engine treats unregistered FeatureGroups during resolution. It is tri-state:

- `"off"` (default): resolution behaves as before; the registry is informational.
- `"warn"`: resolution is unchanged, but every unregistered FeatureGroup is logged with a warning naming the class.
- `"strict"`: resolution only considers FeatureGroups registered in the default registry; unregistered ad-hoc classes are filtered out.

Set it per run on the `PluginCollector` you pass into the API (`plugin_collector=` keyword):

```python
from mloda.user import PluginCollector

plugin_collector = PluginCollector().set_strict_mode("warn")
```

Or process-wide via the environment variable, which also applies when no `PluginCollector` is passed:

```bash
export MLODA_PLUGIN_REGISTRY_STRICT=warn
```

An explicit `set_strict_mode()` overrides the environment variable; invalid values raise a `ValueError`.

Recommended rollout for deployments: stay on `"off"`, switch CI or staging to `"warn"` and watch the logs for "not registered" warnings, then move to `"strict"` once they are gone. This repository's own CI (tox) runs with `MLODA_PLUGIN_REGISTRY_STRICT=warn`.

## Test Isolation for Plugin Authors

Manual registrations mutate the process-global default registry, so tests can leak state into each other. mloda ships a pytest fixture that snapshots the default registry before each test and restores it afterwards. Re-export it from your `conftest.py`:

```python title="conftest.py"
from mloda.core.abstract_plugins.plugin_registry.fixtures import isolated_plugin_registry

__all__ = ["isolated_plugin_registry"]
```

Then request it in tests that register plugins:

```python title="test_my_plugin.py"
def test_my_feature_group_registration(isolated_plugin_registry):
    key = isolated_plugin_registry.register(MyFeatureGroup)
    assert isolated_plugin_registry.get(key) is MyFeatureGroup
```

The fixture yields `PluginRegistry.default()`, so registrations made through it (or through plain `register()`) are rolled back on teardown.

## Related Documentation

- [Plugin Loader](plugin-loader.md)
- [Discover Plugins](discover-plugins.md)
