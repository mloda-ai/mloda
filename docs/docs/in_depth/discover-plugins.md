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

`options` and `plugin_collector` are keyword-only, so pass them by name.

When exactly one FeatureGroup resolves, `ResolvedFeature` also reports the
feature's subtype: `subtype` is the value resolved from the name or the passed
options (e.g. `"sum"` for `sales__sum_aggr`), and `subtype_family` is the
parametric family name when the subtype is a parametric instance (`"ntile"`
for `ntile_2`), else `None`. Both stay `None` for families without a subtype
dimension or when nothing resolves.

### Inspecting Candidates

When resolution fails due to conflicts, check the candidates:

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

Each `FeatureGroupInfo` also documents the family's subtype declaration:
`subtype_key` (the discriminator option key, `None` without one), `subtypes`
(the sorted universe), `parametric_subtypes` (family names like `ntile`),
`subtype_support` (sorted supported subtypes per framework from
`compute_framework_definition()`: the declared rule, or every loaded framework
when the family declares no `compute_framework_rule`; empty for abstract
bases) and `subtype_error` (set when
the class declares support outside its universe, so a misdeclared capability
is distinguishable from a legitimately empty matrix). See
[Declaring capability per subtype](compute-framework-integration.md#declaring-capability-per-subtype).

## Inspecting Compute Frameworks

Get documentation for compute frameworks:

``` python
from mloda.steward import get_compute_framework_docs

# List every framework (is_available flags whether each is importable)
frameworks = get_compute_framework_docs()

# Filter to importable frameworks only
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
- **`resolve_feature` surfaces the conflict.** Resolution feeds execution, which
  must be unambiguous to be safe, so it returns `feature_group=None` with the
  conflict described in `error` and the conflicting classes that match the
  requested feature name in `candidates` rather than silently picking one.

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
