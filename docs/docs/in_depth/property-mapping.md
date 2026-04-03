# PROPERTY_MAPPING Configuration

## Overview

PROPERTY_MAPPING defines parameter validation and classification for modern feature groups using the unified parser approach.

Each entry maps a parameter name to a dict. The dict's top-level keys are valid option values; all metadata (classification flags, validators, defaults, explanation strings) lives inside a nested `"_meta"` sub-dict. This separation prevents collisions between valid values and metadata key names.

## Basic Structure

``` python
from mloda.provider import DefaultOptionKeys

PROPERTY_MAPPING = {
    "parameter_name": {
        "value1": "Description of value1",
        "value2": "Description of value2",
        "_meta": {
            DefaultOptionKeys.context: True,  # Parameter classification
            DefaultOptionKeys.strict_validation: True,  # Validation mode
        },
    },
    DefaultOptionKeys.in_features: {
        "_meta": {
            "explanation": "Source feature description",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,  # Flexible validation
        },
    },
}
```

## Parameter Classification

``` python
# Context parameter (doesn't affect Feature Group splitting)
"aggregation_type": {
    "sum": "Sum aggregation",
    "_meta": {DefaultOptionKeys.context: True},
}

# Group parameter (affects Feature Group splitting)
"data_source": {
    "production": "Production data",
    "_meta": {DefaultOptionKeys.group: True},
}

# Order-by parameter (defines sort order for sequential operations)
DefaultOptionKeys.order_by: {
    "_meta": {
        "explanation": "Column(s) controlling row order for rank, offset, or frame_aggregate",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: False,
    },
}
```

## Validation Modes

### Strict Validation (Default: False)
``` python
"algorithm_type": {
    "kmeans": "K-means clustering",
    "dbscan": "DBSCAN clustering", 
    "_meta": {DefaultOptionKeys.strict_validation: True},  # Only listed values allowed
}
```

### Custom Validation Functions

Use `validation_function` with `strict_validation=True` to validate individual
parsed elements. The parser unpacks lists and calls the function on each element:

``` python
"window_size": {
    "_meta": {
        "explanation": "Size of time window",
        DefaultOptionKeys.validation_function: lambda x: isinstance(x, int) and x > 0,
        DefaultOptionKeys.strict_validation: True,
    },
}
```

### Type Validators

Use `type_validator` to validate the shape or composite type of the raw option
value before any list unpacking. Unlike `validation_function`, it does not
require `strict_validation`. The validator receives the value exactly as stored
in Options and must return a truthy value for the match to succeed:

``` python
def _is_list_of_strings(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)

"partition_by": {
    "_meta": {
        "explanation": "List of columns to partition by",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: False,
        DefaultOptionKeys.type_validator: _is_list_of_strings,
    },
}
```

When both `validation_function` and `type_validator` are present on the same
entry, `validation_function` runs first (during property mapping validation on
each parsed element), then `type_validator` runs on the raw value. Validators
must be pure functions with no side effects.

### Default Values
``` python
"method": {
    "linear": "Linear interpolation",
    "cubic": "Cubic interpolation",
    "_meta": {DefaultOptionKeys.default: "linear"},  # Default if not specified
}
```

## Usage in Feature Groups

``` python
class MyFeatureGroup(FeatureGroup):
    PROPERTY_MAPPING = {
        "operation_type": {
            "sum": "Sum operation",
            "avg": "Average operation",
            "_meta": {
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: True,
            },
        },
        DefaultOptionKeys.in_features: {
            "_meta": {
                "explanation": "Source feature",
                DefaultOptionKeys.context: True,
            },
        },
    }
    
    @classmethod
    def match_feature_group_criteria(cls, feature_name, options, data_access_collection=None):
        return FeatureChainParser.match_configuration_feature_chain_parser(
            feature_name, options, property_mapping=cls.PROPERTY_MAPPING
        )
```

## Validation Examples

``` python
# Valid - "sum" is in mapping
Options(context={"operation_type": "sum"})

# Invalid with strict validation - "custom" not in mapping  
Options(context={"operation_type": "custom"})  # Raises ValueError

# Valid with flexible validation - any value allowed
Options(context={"in_features": "any_feature_name"})
```

## Conditional Requirements with `required_when`

Use `DefaultOptionKeys.required_when` to declare options that are only required under certain conditions. Attach a predicate callable to a PROPERTY_MAPPING entry. The predicate receives an effective `Options` object and returns `True` when the option is required.

This works for both configuration-based and string-based feature creation. For string-based features, the operation value parsed from the feature name is merged into the effective options before predicate evaluation, so predicates see values from both sources.

``` python
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys

_ORDER_DEPENDENT = {"first", "last"}

def _needs_order_by(options: Options) -> bool:
    return options.get("aggregation_type") in _ORDER_DEPENDENT

PROPERTY_MAPPING = {
    "aggregation_type": {
        "sum": "Sum", "avg": "Average", "first": "First", "last": "Last",
        "_meta": {
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
    },
    "order_by": {
        "_meta": {
            "explanation": "Column to order by within each partition",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: False,
            DefaultOptionKeys.required_when: _needs_order_by,
        },
    },
}
```

When the predicate returns `True` and the option is absent, `match_feature_group_criteria` returns `False`. When the predicate returns `False`, the option is not required. Entries with `required_when` are treated as optional by the base parser.

### Predicate Contract

The predicate must satisfy:

- **Signature:** `(Options) -> bool`
- **Must be callable.** Non-callable values are skipped with a warning log.
- **Must not raise exceptions.** Exceptions from predicates propagate uncaught.
- **Must be a pure function** (no side effects).
- Non-bool truthy return values are treated as `True`.

## Context Propagation

By default, context parameters are local: they do not flow through feature dependency chains. This is correct for feature-specific config like aggregation types.

For cross-cutting metadata (session IDs, environment flags) that should flow through chains, use `propagate_context_keys` on `Options`:

``` python
Options(
    context={"session_id": "abc", "window_function": "sum"},
    propagate_context_keys=frozenset({"session_id"}),  # only session_id flows to dependents
)
```

Only the specified keys propagate. Everything else stays local. Group propagation is unchanged.
