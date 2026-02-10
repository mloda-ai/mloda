# PROPERTY_MAPPING Configuration

## Overview

PROPERTY_MAPPING defines parameter validation and classification for modern feature groups using the unified parser approach.

## Basic Structure

``` python
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

PROPERTY_MAPPING = {
    "parameter_name": {
        "value1": "Description of value1",
        "value2": "Description of value2",
        DefaultOptionKeys.context: True,  # Parameter classification
        DefaultOptionKeys.strict_validation: True,  # Validation mode
    },
    DefaultOptionKeys.in_features: {
        "explanation": "Source feature description",
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: False,  # Flexible validation
    },
}
```

## Parameter Classification

``` python
# Context parameter (doesn't affect Feature Group splitting)
"aggregation_type": {
    "sum": "Sum aggregation",
    DefaultOptionKeys.context: True,
}

# Group parameter (affects Feature Group splitting)  
"data_source": {
    "production": "Production data",
    DefaultOptionKeys.group: True,
}
```

## Validation Modes

### Strict Validation (Default: False)
``` python
"algorithm_type": {
    "kmeans": "K-means clustering",
    "dbscan": "DBSCAN clustering", 
    DefaultOptionKeys.strict_validation: True,  # Only listed values allowed
}
```

### Custom Validation Functions
``` python
"window_size": {
    "explanation": "Size of time window",
    DefaultOptionKeys.validation_function: lambda x: isinstance(x, int) and x > 0,
    DefaultOptionKeys.strict_validation: True,
}
```

### Default Values
``` python
"method": {
    "linear": "Linear interpolation",
    "cubic": "Cubic interpolation",
    DefaultOptionKeys.default: "linear",  # Default if not specified
}
```

## Usage in Feature Groups

``` python
class MyFeatureGroup(FeatureGroup):
    PROPERTY_MAPPING = {
        "operation_type": {
            "sum": "Sum operation",
            "avg": "Average operation",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature",
            DefaultOptionKeys.context: True,
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

## Context Propagation

By default, context parameters are local — they don't flow through feature dependency chains. This is correct for feature-specific config like aggregation types.

For cross-cutting metadata (session IDs, environment flags) that should flow through chains, use `propagate_context_keys` on `Options`:

``` python
Options(
    context={"session_id": "abc", "window_function": "sum"},
    propagate_context_keys=frozenset({"session_id"}),  # only session_id flows to dependents
)
```

Only the specified keys propagate — everything else stays local. Group propagation is unchanged.