# Feature Configuration from JSON

## Overview

The `load_features_from_config` function enables loading feature configurations from JSON strings. This is the primary interface for **AI agents and LLMs** to request data from mloda - agents generate JSON, mloda executes it.

Use cases:

- **LLM Tool Functions** - LLMs generate JSON feature requests without writing Python code
- Feature configurations stored externally (files, databases, APIs)
- Dynamic feature definitions at runtime
- Configuration-driven pipelines

## Basic Usage

```python
from mloda.user import load_features_from_config, mloda

config = '''
[
    "customer_id",
    {
        "name": "sales__sum_aggr",
        "in_features": ["sales"],
        "context_options": {
            "report": "weekly"
        }
    }
]
'''

features = load_features_from_config(config)
result = mloda.run_all(
    features,
    compute_frameworks=["PandasDataFrame"],
    api_data={"customer_data": {"customer_id": [1, 2, 3], "sales": [10.0, 20.0, 30.0]}},
)
```

The config only builds the `Feature` objects; every name in it still needs a source, here `api_data`. `sales__sum_aggr` resolves to the aggregation Feature Group, which reads its source from the name; `in_features` states the same source explicitly, and `context_options` rides along as metadata.

## JSON Format

The configuration must be a JSON array. Each item is a feature name string or a feature object with `name` and optional fields (below); strings and objects can be mixed.

## FeatureConfig Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Feature name |
| `options` | object | No | Simple options dict (cannot be used with group_options/context_options) |
| `in_features` | array | No | Array of source feature names for chained features |
| `group_options` | object | No | Group parameters (affect Feature Group resolution) |
| `context_options` | object | No | Context parameters (metadata, doesn't affect resolution) |
| `propagate_context_keys` | array | No | Context keys pushed to the feature's direct input features |
| `cascade_context_keys` | array | No | Context keys pushed to every input feature down the chain |
| `column_index` | integer | No | Index for multi-output features (adds `~N` suffix) |
| `keep_index` | boolean | No | Return the key columns with the result (default `false`) |
| `index` | array of strings | No | Unique key columns of the feature; the columns `keep_index` returns |
| `implementation` | string or array | No | Framework implementation of an operation group (next to `feature_group`) |
| `feature_group` | string | No | Feature Group class name the feature resolves to (scope read by feature resolution and filter matching) |

## Configuration Approaches

### Options

Use `options` for simple key-value configuration, or `group_options` and `context_options` to separate group and context parameters:

```json
[
    {"name": "my_feature", "options": {"window_size": 7, "aggregation": "sum"}},
    {
        "name": "my_other_feature",
        "group_options": {"data_source": "production"},
        "context_options": {"aggregation_type": "sum"}
    }
]
```

`options` and `group_options`/`context_options` are mutually exclusive.

### Feature Group Scope

When two enabled sources declare the same column (for example a shared join key), the bare name is ambiguous and resolution fails with "Multiple feature groups found". Use `feature_group` to scope the request to one source:

```json
[
    {"name": "subject_token", "feature_group": "ClaimsReader"},
    {"name": "claims_amount"},
    {"name": "labs_result"}
]
```

The config form takes a non-empty class-name string only (JSON cannot express a class object); it matches the named class and its subclasses. The scope rules, ties and caveats are in [Feature Group resolution errors](troubleshooting/feature-group-resolution-errors.md#4-scope-a-feature-to-one-source-shared-keys-across-sources).

For an operation group such as `AggregatedFeatureGroup`, add an `implementation` key next to `feature_group` (a string, or a list of strings for a chain) to choose the framework implementation, e.g. `{"name": "sales__sum_aggr", "implementation": "PandasAggregation"}`. See [Compute Framework Integration](compute-framework-integration.md#pinning-an-implementation).

`keep_index` (boolean, default `false`) and `index` (non-empty array of column names) are top-level fields that set `Feature.keep_index` and `Feature.index`, e.g. `{"name": "income", "keep_index": true, "index": ["customer_id"]}`. A nested `in_features` dict rejects both.

`feature_group` is a top-level field next to `name`: writing it inside `options`, `group_options`, or `context_options` is rejected with a validation error.

## Worked Example: Window, Rank, and Percentile Features

Row-preserving operations (window aggregation, rank, percentile) cannot be requested by a bare name: the Feature Group only matches when the request also carries the partition/order options its matcher needs. The feature name encodes the operation (`{source}__{operation}`); the matcher then requires those options to be present. It reads them via `options.get` (group first, then context), so they resolve from either side, but `context_options` is the right home.

```json
[
    {
        "name": "steps__sum_window",
        "context_options": {"partition_by": ["subject_id"]}
    },
    {
        "name": "price__last_window",
        "context_options": {"partition_by": ["region"], "order_by": "timestamp"}
    },
    {
        "name": "sales__row_number_ranked",
        "context_options": {"partition_by": ["region"], "order_by": "sales"}
    },
    {
        "name": "sales__p95_percentile",
        "context_options": {"partition_by": ["region"]}
    }
]
```

Key names per operation (from the registry `data_operations` packages):

| Operation | Name pattern | Required `context_options` |
|-----------|--------------|----------------------------|
| Window aggregation | `{source}__{agg}_window` (`sum`, `avg`, `first`, `last`, ...) | `partition_by` (list); `order_by` (string) is required for order-dependent aggregations like `first`/`last` |
| Rank | `{source}__{rank_type}_ranked` (`row_number`, `dense_rank`, `ntile_N`, ...) | `partition_by` (list), `order_by` (string) |
| Percentile | `{source}__p{N}_percentile` (e.g. `p50`, `p95`) | `partition_by` (list) |

Use `context_options` (not `group_options`) for these: the partition/order are operation parameters, not identity that should split the Feature Group.

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

On a feature whose name carries its source, a declared `in_features` must list the name's direct sources (the part before the last `__`; for `age__mean_imputed__standard_scaled` that is `age__mean_imputed`), otherwise the run raises. Omit `in_features` to let the name decide.

The `in_features` order is kept, and duplicates are kept and count toward `MAX_IN_FEATURES`.

### Nested in_features feature dict

Inside `options`, `group_options` or `context_options`, an `in_features` key can hold a feature dict instead of a name, which defines the source feature inline:

```json
[
    {
        "name": "scaled_sales",
        "options": {
            "scaler_type": "minmax",
            "in_features": {
                "name": "aggregated_sales",
                "feature_group": "SalesAggregator",
                "options": {"in_features": "raw_sales"}
            }
        }
    }
]
```

A nested dict supports `name`, `options`, `in_features` and `feature_group` only, in `options`, `group_options` and `context_options` alike. A nested feature built from `group_options` stays in the group options, one built from `context_options` stays in the context options. In a nested `in_features` dict, the top-level-only fields (`column_index`, `keep_index`, `index`, `group_options`, `context_options`, `propagate_context_keys`, `cascade_context_keys`) are rejected, and so is an unknown key. A feature dict must be the direct value of `in_features`: an `in_features` list holds source feature names, not feature dicts.

The container decides identity: a nested source feature declared under `group_options` is part of the feature's identity, so it affects Options hashing and equality and therefore Feature Group splitting, while one declared under `context_options` is not.

The top-level `in_features` array cannot be combined with an `in_features` key inside `options`, `group_options` or `context_options`, and `in_features` cannot be a key of `group_options` and `context_options` at once: declare the source features in one place.

## Multi-Column Features

`column_index` selects one column of a multi-output feature: `{"name": "pca_result", "column_index": 0}` produces `pca_result~0`.

## Context Propagation

By default, context parameters are local to each feature and do not propagate through feature chains. `propagate_context_keys` pushes the listed context keys to the feature's direct input features (one hop); `cascade_context_keys` pushes them through every input down the chain:

```json
[
    {
        "name": "my_feature",
        "context_options": {"session_id": "abc123", "window_function": "sum"},
        "propagate_context_keys": ["session_id"],
        "cascade_context_keys": ["window_function"]
    }
]
```

Here `session_id` reaches the direct inputs of `my_feature` and `window_function` every input down the chain. A context key not listed stays local. See [Context propagation](property-mapping.md#context-propagation) for the rules.
