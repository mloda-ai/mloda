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

#### 4. Scope a feature to one source (shared keys across sources)
When two enabled sources declare the same column (for example a shared join key), requesting that column by bare name is ambiguous. Scope the request to one source. The scope is resolution-only and does not affect feature identity.

``` python
# class object, collision-proof
Feature("subject_token", feature_group=ClaimsReader)

# class-name string form, the only form a JSON config can carry
Feature("subject_token", feature_group="ClaimsReader")
```

The same scope in a JSON config ([feature config](../feature-config.md)):

``` json
[
    {"name": "subject_token", "feature_group": "ClaimsReader"}
]
```

The config form takes the class-name string only, because JSON cannot express a class object; the class-object form is Python-only.

Both forms match the scoped class and its subclasses, preferring the most specific one. Naming an abstract family base therefore selects the concrete subclass, so a config can scope to the family without naming a compute-framework-specific class:

``` json
[
    {"name": "age__mean_aggr", "feature_group": "AggregatedFeatureGroup"}
]
```

Caveats:

- The scope narrows candidates; it does not break ties between them. If the run enables two compute frameworks whose concrete subclasses both match, the family base stays ambiguous and raises, exactly as the bare name does. Enable one framework for the run, or pin `compute_frameworks` on the `Feature` (Python only).
- The class object is collision-proof; the string form is not. Two classes with the same name in different modules both match the string, so the request stays ambiguous and raises.
- Neither form pins one exact class: a subclass of the named class is preferred over it. To resolve to a specific implementation, name that implementation.
- The root `FeatureGroup` base is rejected in either form: it names no family.
- The scope is resolution-only and excluded from Feature identity, so two requests for the same column name scoped to different sources compare equal. Requesting both in one features list raises `ValueError: Duplicate feature setup: <name>` rather than silently dropping one, so you are told, not surprised. Inside a single `input_features()` returning a set literal, a second same-name feature with a different scope is silently deduplicated by the Python set itself before the engine ever sees it, so never scope the same name twice within one feature group. To read the same column from two sources side by side, give them distinct derived feature names.

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
*no schema* (zero columns). There is no opt-in or opt-out: the rule applies
uniformly on every compute framework. Note that **zero rows is not an error**: a
well-typed frame with the right columns and no rows is a valid result and
passes through. Only a result that carries no columns at all is rejected.

On the PythonDict framework, whose native representation is a columnar
`dict[str, list]`, the schema is the set of keys: `{"col": []}` carries a
schema, while `{}` (zero columns) is the only schema-less value and raises.
Returning a row-oriented `[]` ("nothing to ingest": an empty source directory, a
query with no hits) also carries no columns and therefore raises. Return
`{"my_feature": []}` instead.

If the schema-less result is genuine (a bug, missing data, wrong filter), this
error is correct behavior: it protects callers from silently receiving output
they cannot interpret.

### When to fix it

If an empty result is a legitimate answer for your domain, for example: graph
traversals that may find no path, search queries that may match nothing,
authorization filters that may deny all rows, or agent memory that may have no
relevant entries, return a *schema-bearing* empty result: keep the columns and
drop the rows. On PythonDict that is a dict with empty column lists; on the
other frameworks it is an empty typed table or frame with the right columns.

``` python
class MyFeatureGroup(FeatureGroup):
    @classmethod
    def calculate_feature(cls, data, features):
        ...
        return {"my_feature": []}  # schema-bearing, zero rows: valid
```

A schema-bearing empty result flows through to the caller without raising. The
public import `from mloda.provider import EmptyResultError` (a `ValueError`
subclass) supports a typed `except` when you invoke framework or plugin code
directly. When calling `mloda.run_all`, worker errors are wrapped in a generic
`Exception` whose message embeds the original traceback, so match on the
`EmptyResultError` name in the raised exception's message instead.

### Intermediates are exempt

The check applies only to features that were explicitly requested by the caller.
Intermediate feature groups, meaning those whose output feeds another feature
group rather than the final result, are never subject to `EmptyResultError`.

For full details on how the guard works and the schema-presence gate, see
[Empty Results](../compute-framework-integration.md#empty-results).

## Related Documentation

- [Feature Group Matching](../feature-group-matching.md)
- [PROPERTY_MAPPING Configuration](../property-mapping.md)
- [Feature Group Testing](../feature-group-testing.md)
