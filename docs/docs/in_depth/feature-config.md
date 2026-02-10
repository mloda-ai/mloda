# Feature Configuration from JSON

## Overview

The `load_features_from_config` function enables loading feature configurations from JSON strings. This is the primary interface for **AI agents and LLMs** to request data from mloda - agents generate JSON, mloda executes it.

Use cases:

- **LLM Tool Functions** - LLMs generate JSON feature requests without writing Python code
- Feature configurations stored externally (files, databases, APIs)
- Dynamic feature definitions at runtime
- Configuration-driven pipelines

## Basic Usage

``` python
from mloda.user import load_features_from_config, mloda

config = '''
[
    "simple_feature",
    {"name": "configured_feature", "options": {"param": "value"}}
]
'''

features = load_features_from_config(config)
result = mloda.run_all(features, compute_frameworks=["PandasDataFrame"])
```

## JSON Format

The configuration must be a JSON array. Each item can be:

### 1. Simple String

A plain feature name string:

```json
["feature_name"]
```

### 2. Feature Object

An object with `name` and optional configuration:

```json
[
    {
        "name": "feature_name",
        "options": {"key": "value"}
    }
]
```

### 3. Mixed Configuration

Combine strings and objects:

```json
[
    "simple_feature",
    {"name": "configured_feature", "in_features": ["source_feature"]}
]
```

## FeatureConfig Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Feature name |
| `options` | object | No | Simple options dict (cannot be used with group_options/context_options) |
| `in_features` | array | No | Source feature names for chained features |
| `group_options` | object | No | Group parameters (affect Feature Group resolution) |
| `context_options` | object | No | Context parameters (metadata, doesn't affect resolution) |
| `propagate_context_keys` | array | No | Context keys that propagate to dependent features |
| `column_index` | integer | No | Index for multi-output features (adds `~N` suffix) |

## Configuration Approaches

### Simple Options

Use `options` for simple key-value configuration:

```json
[
    {
        "name": "my_feature",
        "options": {
            "window_size": 7,
            "aggregation": "sum"
        }
    }
]
```

### Modern Group/Context Options

For explicit separation of group and context parameters:

```json
[
    {
        "name": "my_feature",
        "group_options": {
            "data_source": "production"
        },
        "context_options": {
            "aggregation_type": "sum"
        }
    }
]
```

Note: `options` and `group_options`/`context_options` are mutually exclusive.

## Feature Chaining with in_features

Define dependent features using `in_features`:

```json
[
    {
        "name": "aggregated_sales",
        "in_features": ["raw_sales"],
        "context_options": {
            "aggregation_type": "sum"
        }
    }
]
```

Multiple source features:

```json
[
    {
        "name": "distance_feature",
        "in_features": ["point_a", "point_b"]
    }
]
```

## Multi-Column Features

Access specific columns from multi-output features using `column_index`:

```json
[
    {
        "name": "pca_result",
        "column_index": 0
    }
]
```

This produces a feature named `pca_result~0`.

## Context Propagation

By default, context parameters are local to each feature and do not propagate through feature chains. Use `propagate_context_keys` to specify which context keys should flow to dependent features:

``` json
[
    {
        "name": "my_feature",
        "context_options": {
            "session_id": "abc123",
            "window_function": "sum"
        },
        "propagate_context_keys": ["session_id"]
    }
]
```

In this example, `session_id` propagates to any features that depend on `my_feature`, while `window_function` stays local.

## Complete Example

``` python
from mloda.user import load_features_from_config, mloda

config = '''
[
    "customer_id",
    {
        "name": "sales_aggregated",
        "in_features": ["daily_sales"],
        "context_options": {
            "aggregation_type": "sum",
            "window_days": 7
        }
    },
    {
        "name": "encoded_category",
        "in_features": ["category"],
        "column_index": 0
    }
]
'''

features = load_features_from_config(config)

result = mloda.run_all(
    features,
    compute_frameworks=["PandasDataFrame"],
    api_data={"customer_data": {"customer_id": [1, 2, 3]}}
)
```
