# Missing Feature Configuration Patterns

## Overview

This document identifies feature configuration patterns currently used throughout the mloda project that are **NOT yet supported** by the `mloda_plugins/config/feature/` plugin (as of Phase 6 completion).

## Current Support Status

### ✅ What's Currently Supported
- **Simple string features**: `"age"`, `"weight"`
- **Object features with flat options**: `{"name": "weight", "options": {"imputation_method": "mean"}}`
- **JSON format parsing**
- **Pydantic schema validation**
- **Basic loader converting config → Feature objects**

### ❌ What's Missing

---

## 1. Chained Features (String-based with `__` syntax)

### Priority: 🔴 **HIGH** - Core mloda pattern used extensively

### Description
Features can be chained using double underscore (`__`) to create pipelines where one feature group's output becomes another's input.

### Pattern
```
{operation}__{source_feature}
{operation1}__{operation2}__{operation3}__{base_feature}
```

### Examples from Codebase

**Simple chaining:**
```python
features = [
    "standard_scaled__weight",           # Scale the weight feature
    "onehot_encoded__state",             # One-hot encode state
    "mean_imputed__income"               # Impute missing income values
]
```

**Multi-level chaining:**
```python
features = [
    "max_aggr__standard_scaled__mean_imputed__age",  # 3-level chain
    "robust_scaled__mean_imputed__weight"            # 2-level chain
]
```

### Found In
- `README.md:43,156,195,212-214`
- `docs/docs/examples/sklearn_integration_basic.ipynb` (extensively)
- `tests/test_examples/sklearn_integration/test_sklearn_example_cells.py`
- Throughout test suite

### Gap in Current Implementation
- `FeatureConfig` model only has `name: str` and `options: Dict`
- Parser doesn't recognize or handle `__` syntax
- No way to express feature dependencies/chains in JSON config

### Suggested JSON Representation
```json
[
  {
    "name": "standard_scaled__mean_imputed__age",
    "mloda_source": "age"
  }
]
```

---

## 2. Multi-Column Access Pattern (`~` syntax)

### Priority: 🟡 **MEDIUM** - Used for multi-output transformations

### Description
Some transformations (like one-hot encoding) produce multiple columns. The `~` syntax allows accessing specific output columns.

### Pattern
```
{feature_name}~{column_index}
```

### Examples from Codebase

```python
features = [
    "onehot_encoded__state",      # All one-hot columns
    "onehot_encoded__state~0",    # Only first column
    "onehot_encoded__state~1",    # Only second column
    "onehot_encoded__state~2"     # Only third column
]
```

### Found In
- `docs/docs/examples/sklearn_integration_basic.ipynb:282,335`
- `docs/docs/in_depth/feature-chain-parser.md:228-238`
- Test files for encoding feature groups

### Gap in Current Implementation
- No support for `~` character in feature names
- No way to specify column indexing in JSON config

### Suggested JSON Representation
```json
[
  {
    "name": "onehot_encoded__state",
    "column_index": 0
  },
  {
    "name": "onehot_encoded__state",
    "column_selector": "~1"
  }
]
```

---

## 3. Nested Feature Objects as Sources

### Priority: 🟡 **MEDIUM** - Advanced pattern for complex dependencies

### Description
Features can have other `Feature` objects (not just strings) as their source, creating explicit dependency graphs.

### Pattern
```python
Feature(
    name="derived",
    options=Options(
        context={
            DefaultOptionKeys.mloda_source_feature: Feature("base_feature")
        }
    )
)
```

### Examples from Codebase

```python
# String-based source feature
string_source = "identifier1__Sales"

# Config-based feature using Feature object as source
config_target = Feature(
    name="string_to_config",
    options=Options(
        group={"property2": "value2"},
        context={
            DefaultOptionKeys.mloda_source_feature: string_source,
            "ident": "identifier2"
        }
    )
)

# Feature using another Feature object
nested_feature = Feature(
    name="config_feature2",
    options=Options(
        context={
            DefaultOptionKeys.mloda_source_feature: config_target  # Feature object!
        }
    )
)
```

### Found In
- `tests/test_plugins/integration_plugins/chainer/context/test_mixed_string_config_features.py:31-73`
- `tests/test_plugins/feature_group/experimental/test_geo_distance_feature_group/`

### Gap in Current Implementation
- `FeatureConfig.options` is `Dict[str, Any]` but JSON can't serialize Feature objects
- No recursive parsing of nested Feature dependencies
- Loader doesn't handle Feature objects in options

### Suggested JSON Representation
```json
[
  {
    "name": "base_feature",
    "options": {}
  },
  {
    "name": "derived_feature",
    "options": {
      "mloda_source_feature": "@base_feature"
    }
  }
]
```

**Or with inline definition:**
```json
[
  {
    "name": "derived_feature",
    "source": {
      "name": "base_feature",
      "options": {"some_option": "value"}
    }
  }
]
```

---

## 4. Options with Group/Context Separation

### Priority: 🔴 **HIGH** - New architecture pattern for performance

### Description
The `Options` class separates parameters into:
- **`group`**: Affects Feature Group resolution/splitting (performance-critical)
- **`context`**: Metadata that doesn't affect splitting
- **`options`**: Convenient flat dict if you don't care about the separation

This separation is crucial for mloda's dependency resolution system.

### Pattern
```python
Feature(
    name="feature_name",
    options=Options(
        group={"data_source": "production"},      # Affects resolution
        context={"aggregation_type": "sum"}        # Doesn't affect resolution
    )
)
```

### Examples from Codebase

```python
# From test_mixed_string_config_features.py
config_feature = Feature(
    name="config_feature1",
    options=Options(
        group={
            "property2": "value2",  # Group parameter - affects splitting
        },
        context={
            DefaultOptionKeys.mloda_source_feature: "Sales",
            "ident": "identifier2",           # Context parameter
            "property3": "opt_val1"           # Optional context
        }
    )
)
```

### Found In
- `mloda_core/abstract_plugins/components/options.py:10-100`
- `docs/docs/in_depth/feature-chain-parser.md:29-50,207-225`
- `tests/test_core/test_abstract_plugins/test_components/test_options.py`
- Throughout integration tests

### Gap in Current Implementation
- `FeatureConfig` has flat `options: Dict[str, Any]` (which is convenient but doesn't support separation)
- No concept of `group` vs `context` in schema
- Loader creates Features with flat options only

### Suggested JSON Representation
```json
[
  {
    "name": "feature_name",
    "group_options": {
      "data_source": "production"
    },
    "context_options": {
      "aggregation_type": "sum",
      "window_size": 7
    }
  }
]
```

**Or nested structure:**
```json
[
  {
    "name": "feature_name",
    "options": {
      "group": {
        "data_source": "production"
      },
      "context": {
        "aggregation_type": "sum"
      }
    }
  }
]
```

---

## 5. Compute Framework Specification

### Priority: 🟡 **MEDIUM** - Used for mixed-framework scenarios

### Description
Features can specify which compute framework (Pandas, Polars, PyArrow, Spark, DuckDB) they should use.

### Pattern
```python
Feature(name="feature_name", compute_framework="PandasDataframe")
```

### Examples from Codebase

```python
features = [
    Feature("age", compute_framework="PandasDataframe"),
    Feature("weight", compute_framework="PolarsDataframe"),
    Feature("order_id", compute_framework="PyarrowTable")
]

result = mlodaAPI.run_all(features)
```

### Found In
- `README.md:172-177`
- `docs/docs/examples/mloda_basics/1_ml_mloda_intro.ipynb:263-268`
- Test files for multi-framework integration

### Gap in Current Implementation
- `FeatureConfig` doesn't have `compute_framework` field
- No validation for framework names
- Can't express per-feature framework preference in config

### Suggested JSON Representation
```json
[
  {
    "name": "age",
    "compute_framework": "PandasDataframe"
  },
  {
    "name": "weight",
    "compute_framework": "PolarsDataframe",
    "options": {
      "imputation_method": "mean"
    }
  }
]
```

---

## 6. Multiple Source Features (Sets/Frozensets)

### Priority: 🟢 **LOW** - Advanced use cases

### Description
Some feature groups require multiple source features (e.g., distance calculations between two points, joins, aggregations over multiple features).

### Pattern
```python
Feature(
    name="distance",
    options=Options(
        context={
            DefaultOptionKeys.mloda_source_feature: frozenset(["point_a", "point_b"])
        }
    )
)
```

### Examples from Codebase

```python
# Geo distance calculation
feature = Feature(
    name="euclidean_distance",
    options=FeatureName(
        context={
            "distance_type": "euclidean",
            DefaultOptionKeys.mloda_source_feature: frozenset(["point_a", "point_b"])
        }
    )
)

# Multiple features as set
input_features = {Feature("lat"), Feature("lon")}
```

### Found In
- `tests/test_plugins/feature_group/experimental/test_geo_distance_feature_group/`
- `mloda_plugins/feature_group/experimental/geo_distance/base.py:125`
- `tests/test_plugins/compute_framework/test_non_root_merges_*.py`

### Gap in Current Implementation
- Options can only have single source feature
- No array/set support in current schema
- Parser doesn't handle multiple sources

### Suggested JSON Representation
```json
[
  {
    "name": "distance_feature",
    "mloda_sources": ["point_a", "point_b"],
    "options": {
      "distance_type": "euclidean"
    }
  }
]
```

---

## 7. Feature with `initial_requested_data` Flag

### Priority: 🟢 **LOW** - Internal implementation detail

### Description
Features can be marked as initially requested vs. derived. This is used internally for dependency tracking.

### Pattern
```python
Feature(name="feature", options={...}, initial_requested_data=True)
```

### Examples from Codebase

```python
# From test artifacts
return Features([
    Feature(name=f_name, options=options, initial_requested_data=True)
    for f_name in feature_list
])
```

### Found In
- `tests/test_core/test_artifacts/test_artifacts.py:80`
- `tests/test_core/test_integration/test_core/test_runner_*.py`
- `tests/test_core/test_filter/test_filter_integration.py:99`

### Gap in Current Implementation
- Not exposed in config schema
- Likely an internal flag not needed in user configs

### Action
- Probably doesn't need config support
- May be set automatically by loader

---

## Summary Table

| Pattern | Priority | Use Frequency | Implementation Complexity |
|---------|----------|---------------|---------------------------|
| Chained features (`__`) | 🔴 HIGH | Very High | Medium |
| Multi-column access (`~`) | 🟡 MEDIUM | Medium | Low |
| Nested Feature sources | 🟡 MEDIUM | Medium | High |
| Group/Context options | 🔴 HIGH | High | Medium |
| Compute framework spec | 🟡 MEDIUM | Medium | Low |
| Multiple source features | 🟢 LOW | Low | Medium |
| `initial_requested_data` | 🟢 LOW | Low (Internal) | N/A |

---

## Recommended Implementation Priority

### Phase 7: Chained Feature Support
**Highest priority** - This is the core mloda pattern used everywhere.

### Phase 8: Group/Context Options
**High priority** - New architecture for performance optimization.

### Phase 9: Multi-Column Access
**Medium priority** - Needed for one-hot encoding and similar transformations.

### Phase 10: Additional Features
- Compute framework specification
- Nested feature sources
- Multiple source features

---

## References

- Current implementation: `mloda_plugins/config/feature/`
- Feature chaining docs: `docs/docs/in_depth/feature-chain-parser.md`
- Options architecture: `mloda_core/abstract_plugins/components/options.py`
- Integration tests: `tests/test_plugins/integration_plugins/chainer/`
