# Plugin Registry

The plugin registry is an explicit catalog of FeatureGroup, ComputeFramework, and Extender classes, keyed by a string name (default: `"module:qualname"`). The [PluginLoader](plugin-loader.md) feeds it automatically: every concrete plugin class defined in a loaded module is registered with provenance `source="loader"`, manual registrations carry `source="manual"`, and classes discovered from installed [entry-point manifests](plugin-loader.md#entry-points) carry `source="entry_point"`. Provenance is the `PluginSource` enum on each entry (`get_entry(key).source`); `register()` accepts the enum or its string value and rejects anything else with a `ValueError`. Abstract classes (`inspect.isabstract`) are infrastructure, never plugins: the loader skips them, and strict and warn modes ignore them. Registries are plain instances, and the engine and catalog APIs use the process-global default, `PluginRegistry.default()`.

## Registered vs Ad-Hoc Classes

Two kinds of plugin classes exist at runtime:

- **Registered**: classes defined in modules loaded by `PluginLoader`, plus anything registered manually via `register_plugin()`.
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

Classes that do not live in loaded plugin modules opt in with `register_plugin()`. Registering the same class twice is a no-op. Registering a *different* class under an already taken key raises `PluginRegistryCollisionError`; this happens when a notebook cell that defines a class is edited and re-run, because the re-run creates a new class object under the same key. Pass `replace=True` to accept the newest definition. `unregister()` removes an entry by key.

```python
from mloda.provider import FeatureGroup
from mloda.steward import PluginRegistry
from mloda.user import PluginRegistryCollisionError, register_plugin

class MyFeatureGroup(FeatureGroup):
    """First definition."""

key = register_plugin(MyFeatureGroup)
assert PluginRegistry.default().get(key) is MyFeatureGroup

class MyFeatureGroup(FeatureGroup):
    """Edited and re-run, as in a notebook cell: a new class under the same key."""

try:
    register_plugin(MyFeatureGroup)
    raise AssertionError("expected a collision")
except PluginRegistryCollisionError:
    pass

register_plugin(MyFeatureGroup, replace=True)
assert PluginRegistry.default().get(key) is MyFeatureGroup

PluginRegistry.default().unregister(key)
```

`register_plugin()` also accepts `name=` for a custom key, for example `register_plugin(MyFeatureGroup, name="my_pkg:MyFeatureGroup")`. A class holds exactly one key: registering an already registered class under a second key raises `PluginRegistryCollisionError` naming the existing key; rename by unregistering the existing key first.

The registry does not watch `importlib.reload()`: reloading a module creates new class objects while the registry keeps the old ones, so strict mode and `registered_only=True` would treat the reloaded classes as unregistered. After a reload, re-run the `PluginLoader` scope that covers the module (loading is idempotent and re-registers) or call `register_plugin(..., replace=True)` for the affected classes.

## Strict Mode

Strict mode controls how the engine treats unregistered plugins during resolution. It is tri-state:

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

An explicit `set_strict_mode()` overrides the environment variable; invalid values raise a `ValueError`. Environment variable values are case-insensitive (`WARN` works); `set_strict_mode()` accepts only the exact lowercase values.

Three behavioral notes:

- Strict mode governs engine resolution only. The catalog APIs (`get_*_docs()`, `list_registered()`) never consult strict mode: the docs functions keep walking all subclasses unless you pass `registered_only=True`, so in strict mode the catalog can show plugins the engine will not resolve. Pass `registered_only=True` to keep catalog and engine views consistent.
- In `"warn"` mode, each unregistered class is reported once per process, not once per run, so long-running services do not repeat identical warnings forever.
- In `"strict"` mode, FeatureGroups explicitly enabled on the `PluginCollector` but missing from the registry are dropped with a warning naming them; if strict filtering removes every FeatureGroup, the run fails with an error that points to `register_plugin()` and the environment variable.

Strict mode covers all three plugin kinds:

- **FeatureGroups**: resolution filtering as described above; this behavior is unchanged.
- **ComputeFrameworks**: requested frameworks must be registered; under `"strict"` unregistered ones are dropped (the run fails if none survive). Frameworks bundled in `mloda_plugins` ship with mloda and are exempt as infrastructure, like abstract classes.
- **Extenders**: function-extender instances passed into the runner must have registered classes; under `"strict"` unregistered ones are dropped with a warning naming them, under `"warn"` they are logged once per process.

Recommended rollout for deployments: stay on `"off"`, switch CI or staging to `"warn"` and watch the logs for "not registered" warnings, then move to `"strict"` once they are gone. Packages that declare [entry points](plugin-loader.md#entry-points) register automatically, so the usual path is: add entry-point manifests to your installed plugin packages, watch the warn logs go quiet, then ramp to strict. This repository's own CI (tox) runs with `MLODA_PLUGIN_REGISTRY_STRICT=warn`.

## Governance

Stewards can install a `PluginPolicy` on a registry to control what may register. All governance symbols (`PluginPolicy`, `ApprovalStatus`, `PluginPolicyViolationError`, `PluginRegistry`) are exported from `mloda.steward`.

### Plugin Policy

`PluginPolicy` is a frozen dataclass evaluated deny-first on every registration: `denied_keys` first, then `denied_module_prefixes`, then the allowlist (`allowed_keys` admits individual keys; `allowed_module_prefixes`, when not `None`, requires the class module to match a prefix), then `require_approval`. `allows(key, module, approval)` exposes the decision directly.

Module-prefix matching is boundary-aware on both the denied and allowed sides: the prefix `"acme.plugins"` matches the module `acme.plugins` itself and submodules such as `acme.plugins.churn`, but never a sibling like `acme.plugins_evil`. The trailing-dot form `"acme.plugins."` matches submodules only. For `allowed_module_prefixes`, `None` (the default) leaves modules unrestricted, while the empty tuple `()` denies everything that is not in `allowed_keys`.

Enforcement is uniform across provenance: a denied `register()` raises `PluginPolicyViolationError` naming the would-be key, whether the source is manual, loader, or entry-point. The discovery funnels (the loader's module scans and entry-point loading) catch that denial themselves: they skip the denied class, leave it out of their returned keys, and log one warning per denied key per registry instance, so an installed policy never breaks `PluginLoader.all()`.

```python
from mloda.provider import FeatureGroup
from mloda.steward import ApprovalStatus, PluginPolicy, PluginRegistry

class GovernedFeatureGroup(FeatureGroup):
    """Allowed below via its canonical module prefix."""

registry = PluginRegistry()
policy = PluginPolicy(allowed_module_prefixes=(GovernedFeatureGroup.__module__,))
registry.set_policy(policy)

# Outside the allowlist: allows() is False, and register() of such a
# class raises PluginPolicyViolationError, regardless of source.
assert not policy.allows("acme.plugins:Alien", "acme.plugins", ApprovalStatus.UNREVIEWED)

# Inside the allowlist: registration succeeds.
key = registry.register(GovernedFeatureGroup)
assert registry.get(key) is GovernedFeatureGroup
```

`set_policy(None)` removes the policy; the installed policy is readable via the `policy` property.

### Approval Metadata

Every entry carries an `owner` (default `None`) and an `ApprovalStatus` (`unreviewed`, `approved`, `rejected`; default `unreviewed`). `register()` accepts `owner=` and `approval=`, and `set_approval()` updates an existing entry. A policy with `require_approval=True` admits only plugins registered as approved.

```python
from mloda.provider import FeatureGroup
from mloda.steward import ApprovalStatus, PluginPolicy, PluginRegistry

class ReviewedFeatureGroup(FeatureGroup):
    """Registered pre-approved by the data platform team."""

registry = PluginRegistry()
registry.set_policy(PluginPolicy(require_approval=True))

key = registry.register(ReviewedFeatureGroup, owner="data-platform", approval="approved")
entry = registry.get_entry(key)
assert entry.owner == "data-platform"
assert entry.approval is ApprovalStatus.APPROVED

# Enforcement is registration-time only: rejecting later does not evict the entry.
registry.set_approval(key, "rejected")
assert registry.get(key) is ReviewedFeatureGroup

# The steward removes it explicitly.
registry.unregister(key)
assert registry.get(key) is None
```

### Registration-Time Enforcement

A policy is checked only when `register()` runs. Installing or tightening a policy later does not evict entries that are already registered: audit them via `snapshot()` and remove offenders with `unregister()`. The same applies to approval: `set_approval(key, "rejected")` records the decision but keeps the entry until the steward unregisters it.

The policy gates registration only, not import: `PluginLoader.all()` imports entry-point manifest modules before the policy applies, so installing a package implies trusting its import side effects. A policy keeps denied plugins out of the registry; it is not a sandbox for untrusted code.

### Dual-Import Caveat

A module importable under two paths (for example `my_pkg.churn` via the installed package and `churn` via a stray `sys.path` entry) produces two distinct class objects and therefore two registry keys. A key-based allowlist cannot tell them apart. Allowlist by canonical module prefix (`allowed_module_prefixes=("my_pkg.",)`) so only classes imported under the canonical path register.

### Registry Injection

By default the engine consults `PluginRegistry.default()`. Multi-tenant processes can give each run its own registry via `PluginCollector().set_registry()`; strict and warn modes then consult the injected registry instead of the process-global default.

```python
from mloda.steward import PluginRegistry
from mloda.user import PluginCollector

tenant_registry = PluginRegistry()
plugin_collector = PluginCollector().set_strict_mode("strict").set_registry(tenant_registry)
```

Pass the collector into the API via the `plugin_collector=` keyword, as with strict mode.

## Test Isolation for Plugin Authors

Manual registrations mutate the process-global default registry, so tests can leak state into each other. mloda ships a pytest plugin (registered through the `pytest11` entry point in its packaging) that provides the `isolated_plugin_registry` fixture: it snapshots the default registry and its installed policy before each test and restores both afterwards, also resetting the policy-denial warning dedup. Any project with mloda installed gets the fixture automatically; no `conftest.py` re-export is needed. Request it in tests that register plugins:

```python title="test_my_plugin.py"
def test_my_feature_group_registration(isolated_plugin_registry):
    key = isolated_plugin_registry.register(MyFeatureGroup)
    assert isolated_plugin_registry.get(key) is MyFeatureGroup
```

The fixture yields `PluginRegistry.default()`, so registrations made through it (or through plain `register_plugin()`) are rolled back on teardown.

## Related Documentation

- [Plugin Loader](plugin-loader.md)
- [Discover Plugins](discover-plugins.md)
