# Feature Configuration Plugin

The Feature Configuration Plugin enables declarative feature engineering through JSON/YAML configuration files, making feature definitions more maintainable and accessible.

## Overview

Instead of defining features in Python code, you can now define them in configuration files:

```json
[
    "age",
    {"name": "scaled_age", "mloda_source": "age", "options": {"method": "standard"}},
    {"name": "distance", "mloda_sources": ["latitude", "longitude"]}
]
```

## Supported Feature Patterns

### 1. Simple String Features

The simplest way to define a feature - just use the column name:

```json
["age", "salary", "department"]
```

### 2. Features with Options

Configure feature behavior with options:

```json
{
    "name": "weight",
    "options": {
        "imputation_method": "mean",
        "scaling": "minmax"
    }
}
```

### 3. Chained Features

Create multi-step transformation pipelines using `__` syntax:

```json
{
    "name": "standard_scaled__mean_imputed__age",
    "mloda_source": "age"
}
```

This represents: `age` → mean imputation → standard scaling

### 4. Group/Context Options Separation

Separate performance-critical options (group) from runtime options (context) for optimization:

```json
{
    "name": "production_feature",
    "group_options": {
        "data_source": "production",
        "cache_enabled": true
    },
    "context_options": {
        "aggregation_type": "sum",
        "window_size": 7
    }
}
```

**Benefits:**
- `group_options`: Used for grouping/caching (e.g., data source, computation method)
- `context_options`: Runtime-specific settings (e.g., window size, thresholds)

### 5. Multi-Column Access (~syntax)

Access specific columns from multi-output transformations:

```json
{
    "name": "onehot_encoded__state",
    "column_index": 0
}
```

This creates feature name: `onehot_encoded__state~0`

### 6. Feature References (@syntax)

Reference other features as sources, creating dependency chains:

```json
[
    {"name": "scaled_age", "mloda_source": "age", "options": {"method": "standard"}},
    {"name": "log_scaled_age", "mloda_source": "@scaled_age", "options": {"transformation": "log"}}
]
```

The `@` prefix tells the loader to resolve `scaled_age` to the actual Feature object.

**Benefits:**
- Forward references supported (can reference features defined later)
- Nested references work (A → B → C)
- Type-safe: references are Feature objects, not strings

### 7. Multiple Source Features

Features requiring multiple input columns:

```json
{
    "name": "distance_from_center",
    "mloda_sources": ["latitude", "longitude"],
    "options": {"distance_type": "euclidean"}
}
```

Common use cases:
- Distance calculations (lat/lon)
- Multi-column aggregations
- Complex transformations requiring multiple inputs

## JSON Schema

The configuration follows a strict Pydantic schema. Get the full schema:

```python
import json
from mloda_plugins.config.feature.models import feature_config_schema

schema = feature_config_schema()
print(json.dumps(schema, indent=2))
```

### FeatureConfig Schema

```text
class FeatureConfig(BaseModel):
    name: str                                    # Required: Feature name
    options: Dict[str, Any] = {}                 # Legacy options (flat dict)
    group_options: Optional[Dict[str, Any]] = None    # Performance options
    context_options: Optional[Dict[str, Any]] = None  # Runtime options
    mloda_source: Optional[str] = None           # Single source (string or @reference)
    mloda_sources: Optional[List[str]] = None    # Multiple sources
    column_index: Optional[int] = None           # Column selector index
```

### Validation Rules

1. **Options Mutual Exclusion**: Cannot use both `options` and `group_options`/`context_options`
2. **Source Mutual Exclusion**: Cannot use both `mloda_source` and `mloda_sources`

## Usage

### Loading from JSON String

```python
from mloda_plugins.config.feature.loader import load_features_from_config

config_str = '''[
    "age",
    {"name": "weight", "options": {"imputation": "mean"}}
]'''

features = load_features_from_config(config_str, format="json")
```

### Loading from File

```python
# Example: Load from file (file must exist)
# from pathlib import Path
# config_path = Path("features.json")
# with open(config_path) as f:
#     config_str = f.read()
# features = load_features_from_config(config_str, format="json")
```

### Integration with mlodaAPI

```python
# Example: Integration with mlodaAPI (requires setup)
# from mloda_core.mloda_api import mlodaAPI
# features = load_features_from_config(config_str, format="json")
# api = mlodaAPI()
# result = api.run_all(features, data)
```

## Complete Example

See [`tests/test_plugins/config/feature/test_config_features.json`](../../tests/test_plugins/config/feature/test_config_features.json) for a comprehensive example demonstrating all supported patterns.

## Migration Guide

### Before (Python Code)

```text
features = [
    Feature(name="age"),
    Feature(name="weight", options=Options(group={"imputation": "mean"})),
    Feature(name="scaled_age", options=Options(
        context={"mloda_source_feature": "age"},
        group={"method": "standard"}
    ))
]
```

### After (JSON Configuration)

```json
[
    "age",
    {"name": "weight", "options": {"imputation": "mean"}},
    {"name": "scaled_age", "mloda_source": "age", "options": {"method": "standard"}}
]
```

## Best Practices

1. **Use group/context separation** for production features to enable better caching
2. **Leverage feature references** (`@syntax`) to create maintainable dependency chains
3. **Keep configuration files small** - split into multiple files for large projects
4. **Validate configurations** using the JSON schema before deployment
5. **Document custom options** specific to your feature implementations

## Performance Considerations

- **Group options**: Used for grouping features with similar computation requirements
- **Context options**: Evaluated at runtime, allow dynamic behavior
- **Feature references**: Resolved once at load time (no runtime overhead)
- **Multiple sources**: Converted to immutable frozensets for hashability

## See Also

- [Tutorial: Feature Configuration](../tutorials/feature_configuration.md)
- [Examples](../examples/feature_config_examples.json)
- [FAQ](../faq.md)
