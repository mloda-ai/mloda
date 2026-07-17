# Discover Plugins

mloda provides functions to discover, inspect, and debug plugins at runtime. These are available from `mloda.steward`.

## Resolving Feature Names

Use `resolve_feature()` to find which FeatureGroup handles a specific feature name:

``` python
from mloda.steward import resolve_feature

# Check which FeatureGroup handles a feature
result = resolve_feature("timestamp_unix")
if result.feature_group:
    print(f"Handled by: {result.feature_group.__name__}")
else:
    print(f"Resolution failed: {result.error}")
```

This is useful for:
- Debugging feature resolution issues
- Understanding which plugin handles a feature
- Identifying conflicts when multiple FeatureGroups match

`resolve_feature` runs the **same matcher over the same candidate universe as a
run**: it delegates to the engine's `IdentifyFeatureGroupClass` and builds its
plugin universe the way the engine does (availability, strict-registry mode, and
`PluginCollector` policy all apply). So the error it reports for an unresolvable
feature is the error a run would raise, and a feature it resolves is one a run
resolves. The remaining differences are presentation-only: `resolve_feature`
never raises (matching failures land in `result.error`, not an exception), and
it returns a structured `ResolvedFeature` instead of a plan.

This covers a broken framework declaration too: a plugin that raises while
declaring its frameworks (`compute_framework_rule` /
`compute_framework_definition`) aborts the environment build on both paths. A run
raises that failure; `resolve_feature` reports the same failure in `result.error`
with no candidates, because the build never reaches matching. The difference stays
presentation-only.

Its environment is a **standalone default**, not a replay of a specific run:
`ResolvedFeature.environment` reads `"standalone-default"`. The candidate
universe defaults to every installed compute framework unless you narrow it (see
[Expressing the full request](#expressing-the-full-request)).

### Expressing the full request

A run resolves a `Feature`, not just a name, so `resolve_feature` accepts one
too. Pass a `Feature` as the single source of truth for name, `options`, domain,
compute-framework pin, and scope; passing `options` or `feature_group` alongside
a `Feature` raises `TypeError`.

``` python
from mloda.user import Feature, Options

result = resolve_feature(Feature("sales__sum_aggr", options=Options(group={"partition_by": ["customer_id"]})))
```

Every engine input is expressible, either on the `Feature` or as a keyword-only
argument to `resolve_feature`:

- `options`, domain, compute-framework pin, scope: on the `Feature` (or, for the
  string form, `options=` and `feature_group=`).
- `links`: a set of `Link` objects, threaded to the matcher so link-gated groups
  resolve.
- `data_access_collection`: threaded to the matcher so reader / input-data groups
  resolve.
- `plugin_collector`: restricts the FeatureGroups considered and threads its
  `allow_redefinition` flag into deduplication.
- `compute_frameworks`: restricts the candidate universe's framework set
  (default: all installed frameworks).

### Option-Gated Feature Groups

Matching and the compute framework split both run under the options you pass.
A FeatureGroup whose `match_feature_group_criteria` or
`supports_compute_framework` reads an option therefore only resolves when you
supply it. Without `options` the evaluation uses empty `Options`, and such a
group reports no candidates:

``` python
from mloda.steward import resolve_feature
from mloda.user import Options, PluginLoader

PluginLoader.all()

# The group-by aggregation FeatureGroups in mloda-registry match only when partition_by is set.
result = resolve_feature("sales__sum_aggr", options=Options(group={"partition_by": ["customer_id"]}))
print(result.feature_group, result.supported_compute_frameworks)
```

Every argument after `feature` (`options`, `plugin_collector`, `feature_group`,
`links`, `data_access_collection`, `compute_frameworks`) is keyword-only, so
pass them by name.

### Scoping to a Feature Group

When several FeatureGroups match the same name, such as the framework-specific
`AggregatedFeatureGroup` subclasses, narrow the resolution with the keyword-only
`feature_group` argument. It takes a FeatureGroup subclass or its class-name
string (the string matches the named class and its subclasses), the same forms
as `Feature(..., feature_group=...)`:

``` python
result = resolve_feature("sales__sum_aggr", feature_group="PandasAggregatedFeatureGroup")
```

Scoped failures append `Scoped to feature group: '<Name>'.` to `result.error`,
mirroring the scoped engine error, so a scoped run can be debugged with the
same scope.

When exactly one FeatureGroup resolves, `ResolvedFeature` also reports
`subtype` (resolved from the name, the passed options, or the key's declared
default, e.g. `"sum"` for `sales__sum_aggr`) and `subtype_family` (the
parametric family name, `"ntile"` for `ntile_2`); both are `None` when
nothing resolves.

### Inspecting Candidates

When resolution reached matching but ended ambiguous (multiple FeatureGroups
matched), `candidates` lists the classes that matched. When the environment
build itself failed (for example a redefinition conflict), no matching runs,
so `candidates` is empty and the error text carries the details:

``` python
result = resolve_feature("my_feature")
if result.error:
    print(f"Error: {result.error}")
    print(f"Candidates: {[fg.__name__ for fg in result.candidates]}")
```

## Inspecting Feature Groups

Get documentation for available feature groups:

``` python
from mloda.steward import get_feature_group_docs

# Get all feature groups
all_fgs = get_feature_group_docs()

# Filter by name
fgs = get_feature_group_docs(name="timestamp")

# Filter by compute framework
fgs = get_feature_group_docs(compute_framework="PandasDataframe")
```

Each `FeatureGroupInfo` also carries the subtype declaration: `subtype_key`
(the `SUBTYPES.key`, `None` for shape-B and undeclared families), `subtypes`
(the sorted universe), `parametric_subtypes`, `subtype_support` (supported
subtypes per framework; empty for abstract bases) and `subtype_error` (set
when the declaration is invalid). See
[Declaring capability per subtype](compute-framework-integration.md#declaring-capability-per-subtype).

## Inspecting Compute Frameworks

Get documentation for compute frameworks:

``` python
from mloda.steward import get_compute_framework_docs

# List every framework (is_available flags whether its backend library is installed)
frameworks = get_compute_framework_docs()

# Filter to available frameworks only
available_frameworks = get_compute_framework_docs(available_only=True)
```

## Inspecting Extenders

Get documentation for extenders:

``` python
from mloda.steward import get_extender_docs

# Get all extenders
extenders = get_extender_docs()

# Filter by wrapped function type
extenders = get_extender_docs(wraps="formula")
```

## Graceful Degradation Policy

The discovery functions above walk every live plugin subclass and introspect it.
Plugins can be broken, exotic, or built at runtime (via `type()`, notebook
re-execution, or `importlib.reload`), so introspecting one class can fail. The
guarded catalog reads follow one shared contract so that a broken plugin
degrades a field instead of sinking the catalog call.

Every failure a plugin can cause falls into one of three tiers:

- **Annotate.** A single descriptive field cannot be read, but the class is still
  a valid catalog entry. The entry is listed with a sentinel value and discovery
  continues. Examples: `get_feature_group_docs` reports `version="unavailable"`
  when source introspection fails (`_safe_version`); `get_compute_framework_docs`
  reports `expected_data_framework="unavailable"`, `has_merge_engine=False`,
  `has_filter_engine=False`, and `is_available=False` when the corresponding call
  raises; `get_extender_docs` reports `wraps=[]` when the extender cannot be
  instantiated. A framework that degrades to `is_available=False` is excluded by
  `available_only=True`, exactly as a genuinely unavailable one would be.
- **Skip.** An entry that is not meant to be documented is dropped silently:
  classes defined in `__main__`, and the abstract bases (`get_extender_docs`
  skips `Extender` and `_CompositeExtender` explicitly; the FeatureGroup and
  ComputeFramework roots never appear because `get_all_subclasses` omits them).
  Redefinition duplicates are collapsed rather than listed twice (see below).
- **Propagate.** Errors that are not a single plugin's introspection fault are
  left to surface: bad arguments to the discovery function itself, and failures
  in mloda's own catalog machinery. These are bugs to fix, not conditions to
  paper over.

### Redefinition conflicts

When the same `(module, qualname)` FeatureGroup is defined more than once with
differing source (typical in notebooks and reload-heavy sessions), the catalog
and resolution paths intentionally differ:

- **`get_*_docs` degrade.** Documentation is a best-effort read-only view, so a
  conflict collapses to the most recently defined (live) class and listing
  continues. There is nothing to execute, so ambiguity is harmless.
- **`resolve_feature` fails closed.** Resolution feeds execution, so the
  conflict fails closed like any other environment-build failure: it returns
  `feature_group=None` with the conflict described in `error` and no
  candidates. The failure is projected from the build error itself, which names
  the conflicting classes in the message, and no matching runs.

### Broken framework declarations

When a FeatureGroup raises while declaring its frameworks
(`compute_framework_rule` / `compute_framework_definition`), the catalog and
resolution paths again differ:

- **`get_*_docs` degrade.** Listing is a read-only view, so the class stays a
  catalog entry with `compute_frameworks=[]`. This is the annotate tier's
  `safe_field` guard on that one field.
- **`resolve_feature` fails closed.** Resolution feeds execution, so it builds its
  environment the way a run does: the broken declaration aborts the build. It
  returns `feature_group=None` with the provider's failure in `error` and no
  candidates, the same failure a run raises.

Failing closed is process-wide, not group-local: the build declares every
FeatureGroup's frameworks, so one broken plugin fails *every* resolution and run,
including features it has nothing to do with. The way out is to scope it out of
the universe. Both filters below drop the class before its declaration is ever
consulted, so the build succeeds and the feature is an ordinary no-match:

``` python
# One broken third-party plugin, and every call reports its failure:
resolve_feature("timestamp_unix").error
# "Failed to build the plugin environment: RuntimeError: ..."

# Scope it out, and resolution works again:
resolve_feature("timestamp_unix", plugin_collector=PluginCollector().disabled_feature_groups({BrokenFG}))
```

Strict mode is the other filter: an unregistered broken plugin is dropped before
its declaration is read. Both apply identically to a run, so a scope that rescues
`resolve_feature` rescues the run too.

### Shared helper

The annotate tier is a single shared helper,
`safe_field(read, fallback, catching=(Exception,), field="")` in
`mloda.core.abstract_plugins.components.utils`: it calls the `read` thunk and
returns `fallback` if the read raises one of `catching`, so a catalog function or
info field reaches for one helper instead of re-deriving a `try/except`.

Fallbacks are base-class-derived rather than sentinels, so a degraded entry stays
usable: a broken `description()` falls back to the class docstring (or `__name__`),
which keeps it findable via `search=`, and a broken `prefix()` to `"<__name__>_"`.
The reads that feed a filter (`get_class_name()`, `description()`, `version()`)
also reject a non-str return, so a wrong type degrades too instead of sinking the
call at the filter.

Logging is opt-in via `field`: the labelled `get_feature_group_docs` reads warn on
swallow, where a raise does mean a broken plugin. The unlabelled guards stay silent
because degrading there is by design (source introspection of `type()`-built
classes, an availability probe without its optional backend, Iceberg's deliberate
`NotImplementedError` from `merge_engine()`).

`get_compute_framework_docs` uses the default broad `catching` (any failure
degrades the field), while the narrower named guards `_safe_version` (in
`plugin_docs.py`) and `_safe_class_source_hash` (one layer down in
`accessible_plugins.py`) pass the shared `SOURCE_INTROSPECTION_ERRORS` constant
(from `base_feature_group_version.py`) so anything else still propagates. The helper is
deliberately per-field (annotate the entry) rather than per-class (skip the whole
entry), because the catalog's job is to list a degraded class, not hide it.

## Related Documentation

- [Plugin Loader](plugin-loader.md)
- [Feature Group Matching](feature-group-matching.md)
- [Feature Group Resolution Errors](troubleshooting/feature-group-resolution-errors.md)
