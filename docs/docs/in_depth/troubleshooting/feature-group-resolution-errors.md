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

#### 4. Scope the feature to a specific feature group

When two different feature group classes legitimately share a column name (for example,
two sources that both declare a join key like `subject_token`), disabling one of them is
not an option: you need both enabled, but a single request for the shared column is
ambiguous. Scope the feature to the source it should come from:

```python
from mloda.user import Feature

# By class name
Feature("subject_token", feature_group="ClaimsReader")

# By class object
Feature("subject_token", feature_group=ClaimsReader)

# Or via options
Feature("subject_token", options={"feature_group": "ClaimsReader"})
```

The feature then resolves only against the named feature group class, so the
"Multiple feature groups found" collision does not occur. This also works inside a
derived feature group's `input_features`, which is the typical place you need it: a
derived feature group reading a subset of one source's columns can pull in the shared
join key from that same source.

Prefer the `feature_group` parameter over the options form. The options form keeps the
`feature_group` key inside the feature's options, which feeds into feature grouping and
into the options the resolved feature group receives, so a scoped feature may no longer
share a read with its sibling features. The parameter form leaves options untouched.

The scope matches on the class name only. Two feature group classes that share the same
class name in different modules still collide and raise "Multiple feature groups found".

```python
class NewestClaimPerSubject(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature("claim_date"),
            Feature("dx_code"),
            Feature("subject_token", feature_group=ClaimsReader),
        }
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

**Option A** — set the override flag on a fresh collector:

```python
from mloda.provider import FeatureGroup
from mloda.user import PluginCollector


class SomeFG(FeatureGroup):
    pass


plugin_collector = PluginCollector().set_allow_redefinition()
```

**Option B** — compose with disable/enable filters:

```python
plugin_collector = (
    PluginCollector.disabled_feature_groups({SomeFG})
    .set_allow_redefinition()
)
```

Pass this collector through to `mloda.run_all`, `resolve_feature`, or any other entry point that accepts a `plugin_collector` argument.

#### 2. Restart the kernel

Restarting the Jupyter kernel clears `Out[N]` history and unloads all stale class objects. Use this when you want a fully clean state and do not need to preserve other notebook state.

## EmptyResultError

### The Problem

```
EmptyResultError: Result carries no schema (no columns): MyFeatureGroup. ...
```

This error fires when a **final** requested feature group returns a result with
*no schema* (zero columns) and `allow_empty_result()` is `False` (the default).
Note that **zero rows is not an error**: a well-typed frame with the right
columns and no rows is a valid result and passes through. Only a result that
carries no columns at all is rejected.

In practice you hit this on the PythonDict framework, whose empty value (`[]`)
cannot carry a schema, or when a feature genuinely produces a column-less result.

If the schema-less result is genuine (a bug, missing data, wrong filter), this
error is correct behavior: it protects callers from silently receiving output
they cannot interpret.

### When to fix it

Override `allow_empty_result` only when a schema-less empty result is a
legitimate answer for your domain, for example: graph traversals that may find no
path, search queries that may match nothing, authorization filters that may deny
all rows, or agent memory that may have no relevant entries. This is most often
needed on the PythonDict framework.

``` python
class MyFeatureGroup(FeatureGroup):
    @classmethod
    def allow_empty_result(cls) -> bool:
        return True
```

With the override in place, empty data flows through to the caller without
raising. The public import `from mloda.provider import EmptyResultError` (a
`ValueError` subclass) supports a typed `except` when you invoke framework or
plugin code directly. When calling `mloda.run_all`, worker errors are wrapped
in a generic `Exception` whose message embeds the original traceback, so match
on the `EmptyResultError` name in the raised exception's message instead.

### Intermediates are exempt

The check applies only to features that were explicitly requested by the caller.
Intermediate feature groups, meaning those whose output feeds another feature
group rather than the final result, are never subject to `EmptyResultError`,
even if `allow_empty_result()` is `False`.

For full details on how the guard works and the schema-presence gate, see
[Allowing Empty Results](../compute-framework-integration.md#allowing-empty-results).

## Related Documentation

- [Feature Group Matching](../feature-group-matching.md)
- [PROPERTY_MAPPING Configuration](../property-mapping.md)
- [Feature Group Testing](../feature-group-testing.md)
