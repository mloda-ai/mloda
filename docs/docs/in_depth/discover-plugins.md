# Discover Plugins

mloda provides functions to discover, inspect, and debug plugins at runtime. These are available from `mloda.steward`.

## Resolving Feature Names

Use `resolve_feature()` to find which FeatureGroup handles a specific feature name:

```python
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

### Inspecting Candidates

When resolution fails due to conflicts, check the candidates:

```python
result = resolve_feature("my_feature")
if result.error:
    print(f"Error: {result.error}")
    print(f"Candidates: {[fg.__name__ for fg in result.candidates]}")
```

## Inspecting Feature Groups

Get documentation for available feature groups:

```python
from mloda.steward import get_feature_group_docs

# Get all feature groups
all_fgs = get_feature_group_docs()

# Filter by name
fgs = get_feature_group_docs(name="timestamp")

# Filter by compute framework
fgs = get_feature_group_docs(compute_framework="PandasDataframe")
```

## Inspecting Compute Frameworks

Get documentation for compute frameworks:

```python
from mloda.steward import get_compute_framework_docs

# Get all available frameworks
frameworks = get_compute_framework_docs()

# Include unavailable frameworks
all_frameworks = get_compute_framework_docs(available_only=False)
```

## Inspecting Extenders

Get documentation for extenders:

```python
from mloda.steward import get_extender_docs

# Get all extenders
extenders = get_extender_docs()

# Filter by wrapped function type
extenders = get_extender_docs(wraps="formula")
```

## Related Documentation

- [Plugin Loader](plugin-loader.md)
- [Feature Group Matching](feature-group-matching.md)
- [Feature Group Resolution Errors](troubleshooting/feature-group-resolution-errors.md)
