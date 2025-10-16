# Tutorial: Feature Configuration

This tutorial guides you through using the Feature Configuration Plugin to define features declaratively with JSON instead of Python code.

## Prerequisites

- mloda installed (`pip install mloda`)
- Basic understanding of feature engineering concepts
- Familiarity with JSON format

## Step 1: Your First Feature Configuration

Let's start with the simplest possible configuration:

```json
["age", "salary", "department"]
```

Save this as `features.json` and load it:

```python
from mloda_plugins.config.feature.loader import load_features_from_config

# Load configuration from JSON string
config_str = '["age", "salary", "department"]'

features = load_features_from_config(config_str, format="json")
print(f"Loaded {len(features)} features")
# Output: Loaded 3 features
```

**Result**: You now have 3 features ready to use with mlodaAPI!

## Step 2: Adding Options to Features

Features often need configuration. Add options:

```json
[
    "age",
    {
        "name": "weight",
        "options": {
            "imputation_method": "mean",
            "handle_outliers": true
        }
    }
]
```

```python
config_str = '''[
    "age",
    {
        "name": "weight",
        "options": {
            "imputation_method": "mean",
            "handle_outliers": true
        }
    }
]'''

features = load_features_from_config(config_str, format="json")

# Access feature options
weight_feature = features[1]
print(weight_feature.options.get("imputation_method"))
# Output: mean
```

**Use Case**: Configure how each feature should be processed without changing code.

## Step 3: Creating Transformation Chains

Build multi-step transformations using chained features:

```json
[
    "age",
    {
        "name": "standard_scaled__mean_imputed__age",
        "mloda_source": "age"
    }
]
```

This represents: `age` → mean imputation → standard scaling

```python
config_str = '''[
    "age",
    {
        "name": "standard_scaled__mean_imputed__age",
        "mloda_source": "age"
    }
]'''

features = load_features_from_config(config_str, format="json")

# The chained feature automatically uses "age" as its source
chained_feature = features[1]
print(chained_feature.name.name)
# Output: standard_scaled__mean_imputed__age
```

**Pattern**: Use `__` to separate transformation steps, read right-to-left.

## Step 4: Referencing Other Features

Create feature dependencies using the `@` syntax:

```json
[
    {"name": "scaled_age", "mloda_source": "age", "options": {"method": "standard"}},
    {"name": "log_scaled_age", "mloda_source": "@scaled_age", "options": {"transformation": "log"}},
    {"name": "binned_log_scaled", "mloda_source": "@log_scaled_age", "options": {"bins": 10}}
]
```

```python
config_str = '''[
    {"name": "scaled_age", "mloda_source": "age", "options": {"method": "standard"}},
    {"name": "log_scaled_age", "mloda_source": "@scaled_age", "options": {"transformation": "log"}},
    {"name": "binned_log_scaled", "mloda_source": "@log_scaled_age", "options": {"bins": 10}}
]'''

features = load_features_from_config(config_str, format="json")

# Verify reference resolution
log_feature = features[1]
source = log_feature.options.context.get("mloda_source_feature")
print(type(source).__name__)
# Output: Feature
print(source.name.name)
# Output: scaled_age
```

**Benefits**:
- Type-safe: References are resolved to Feature objects
- Forward references work (can reference features defined later)
- Creates clear dependency chains

## Step 5: Working with Multiple Sources

Some features need multiple input columns:

```json
[
    "latitude",
    "longitude",
    {
        "name": "distance_from_center",
        "mloda_sources": ["latitude", "longitude"],
        "options": {
            "center_lat": 40.7128,
            "center_lon": -74.0060,
            "distance_type": "haversine"
        }
    }
]
```

```python
config_str = '''[
    "latitude",
    "longitude",
    {
        "name": "distance_from_center",
        "mloda_sources": ["latitude", "longitude"],
        "options": {
            "center_lat": 40.7128,
            "center_lon": -74.0060,
            "distance_type": "haversine"
        }
    }
]'''

features = load_features_from_config(config_str, format="json")

distance_feature = features[2]
sources = distance_feature.options.context.get("mloda_source_features")
print(type(sources).__name__)
# Output: frozenset
print(sorted(sources))
# Output: ['latitude', 'longitude']
```

**Common Use Cases**:
- Distance calculations (lat/lon)
- Multi-column aggregations (sales + revenue + profit)
- Feature interactions (age × income)

## Step 6: Separating Group and Context Options

For production systems, separate performance options from runtime options:

```json
{
    "name": "production_feature",
    "group_options": {
        "data_source": "production",
        "cache_enabled": true,
        "computation_method": "optimized"
    },
    "context_options": {
        "aggregation_type": "sum",
        "window_size": 7,
        "min_samples": 100
    }
}
```

**Why Separate?**
- `group_options`: Used for grouping/caching similar features (performance)
- `context_options`: Runtime behavior (flexibility)

```python
config_str = '''[{
    "name": "production_feature",
    "group_options": {
        "data_source": "production",
        "cache_enabled": true,
        "computation_method": "optimized"
    },
    "context_options": {
        "aggregation_type": "sum",
        "window_size": 7,
        "min_samples": 100
    }
}]'''

features = load_features_from_config(config_str, format="json")
feature = features[0]
print(feature.options.group.get("data_source"))
# Output: production
print(feature.options.context.get("window_size"))
# Output: 7
```

## Step 7: Multi-Column Access with Column Selectors

Access specific columns from multi-output features:

```json
[
    {
        "name": "onehot_encoded__state",
        "column_index": 0
    },
    {
        "name": "onehot_encoded__state",
        "column_index": 1
    },
    {
        "name": "onehot_encoded__state",
        "column_index": 2
    }
]
```

```python
config_str = '''[
    {
        "name": "onehot_encoded__state",
        "column_index": 0
    },
    {
        "name": "onehot_encoded__state",
        "column_index": 1
    },
    {
        "name": "onehot_encoded__state",
        "column_index": 2
    }
]'''

features = load_features_from_config(config_str, format="json")

# Feature names automatically get ~{index} suffix
print([f.name.name for f in features])
# Output: ['onehot_encoded__state~0', 'onehot_encoded__state~1', 'onehot_encoded__state~2']
```

**Use Case**: One-hot encoding creates multiple columns; select which ones you need.

## Complete Example: Real-World Pipeline

```json
[
    "customer_id",
    "age",
    "income",
    "state",
    "latitude",
    "longitude",

    {
        "name": "age_scaled",
        "mloda_source": "age",
        "options": {"method": "standard"}
    },
    {
        "name": "income_imputed",
        "mloda_source": "income",
        "options": {"method": "median"}
    },
    {
        "name": "income_scaled",
        "mloda_source": "@income_imputed",
        "options": {"method": "robust"}
    },
    {
        "name": "age_income_interaction",
        "mloda_sources": ["age", "income"],
        "options": {"interaction_type": "multiply"}
    },
    {
        "name": "distance_from_office",
        "mloda_sources": ["latitude", "longitude"],
        "options": {
            "center_lat": 40.7128,
            "center_lon": -74.0060
        }
    },
    {
        "name": "state_encoded",
        "mloda_source": "state",
        "column_index": 0
    }
]
```

```python
# Example: Complete pipeline (requires full setup)
# from mloda_plugins.config.feature.loader import load_features_from_config
# from mloda_core.api.request import mlodaAPI
#
# with open("complete_features.json") as f:
#     config_str = f.read()
#
# features = load_features_from_config(config_str, format="json")
#
# result = mlodaAPI.run_all(
#     features=features,
#     compute_frameworks={PandasDataframe},
#     data_access_collection=DataAccessCollection(files={"customers.csv"})
# )
#
# processed_data = result[0]
# print(f"Created {len(processed_data.columns)} features!")
```

## Common Patterns and Recipes

### Pattern 1: Feature Family

Group related features together:

```json
[
    {"name": "age_binned", "mloda_source": "age", "options": {"bins": [0, 18, 35, 50, 65, 100]}},
    {"name": "age_scaled", "mloda_source": "age", "options": {"method": "standard"}},
    {"name": "age_normalized", "mloda_source": "age", "options": {"method": "minmax"}},
    {"name": "age_log", "mloda_source": "age", "options": {"transformation": "log1p"}}
]
```

### Pattern 2: Transformation Pipeline

Build progressive transformations:

```json
[
    "raw_sales",
    {"name": "sales_imputed", "mloda_source": "raw_sales", "options": {"method": "forward_fill"}},
    {"name": "sales_smoothed", "mloda_source": "@sales_imputed", "options": {"window": 7}},
    {"name": "sales_scaled", "mloda_source": "@sales_smoothed", "options": {"method": "robust"}},
    {"name": "sales_momentum", "mloda_source": "@sales_scaled", "options": {"periods": 14}}
]
```

### Pattern 3: Feature Crosses

Create feature interactions:

```json
[
    "age",
    "income",
    "education_years",
    {
        "name": "earning_potential",
        "mloda_sources": ["age", "income", "education_years"],
        "options": {"aggregation": "weighted_mean", "weights": [0.3, 0.5, 0.2]}
    }
]
```

## Troubleshooting

### Issue 1: Feature Not Found

**Error**: `KeyError: 'scaled_age'`

**Solution**: Make sure the referenced feature is defined:
```json
[
    {"name": "scaled_age", "mloda_source": "age"},
    {"name": "derived", "mloda_source": "@scaled_age"}  // Correct: scaled_age exists
]
```

### Issue 2: Circular References

**Error**: `RuntimeError: Circular reference detected`

**Solution**: Avoid circular dependencies:
```json
// ❌ Bad
[
    {"name": "feature_a", "mloda_source": "@feature_b"},
    {"name": "feature_b", "mloda_source": "@feature_a"}
]

// ✅ Good
[
    {"name": "base", "mloda_source": "age"},
    {"name": "feature_a", "mloda_source": "@base"},
    {"name": "feature_b", "mloda_source": "@base"}
]
```

### Issue 3: Validation Errors

**Error**: `ValidationError: Cannot use both 'mloda_source' and 'mloda_sources'`

**Solution**: Use one or the other, not both:
```json
// ❌ Bad
{"name": "feat", "mloda_source": "age", "mloda_sources": ["age", "income"]}

// ✅ Good - single source
{"name": "feat", "mloda_source": "age"}

// ✅ Good - multiple sources
{"name": "feat", "mloda_sources": ["age", "income"]}
```

## Performance Tips

1. **Use group_options for caching**: Features with same group_options can be cached together
2. **Minimize feature references**: Each reference adds a dependency resolution step
3. **Batch similar transformations**: Group features with similar options
4. **Leverage frozensets**: Multiple sources are automatically converted to immutable frozensets for better performance

## Next Steps

- Explore the [complete example](../examples/feature_config_examples.json)
- Read the [full documentation](../plugins/feature_config.md)
- Check out [mlodaAPI integration](../in_depth/mloda-api.md)
- Learn about [compute frameworks](../chapter1/compute-frameworks.md)

## Questions?

- [FAQ](../faq.md)
- [GitHub Issues](https://github.com/mloda-ai/mloda/issues)
- [Community Discord](https://mloda.ai/discord)
