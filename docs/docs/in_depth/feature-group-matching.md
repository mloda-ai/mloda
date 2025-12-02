# Feature Group Matching Criteria

## Overview

The mloda framework uses a sophisticated matching system to determine which feature group should handle a given feature. The modern approach supports both traditional string-based matching and configuration-based matching through the unified `FeatureChainParser`.

## Matching Process

When a feature is requested, the system checks all available feature groups to find the one that should handle the feature. This is done through the `match_feature_group_criteria` method in each feature group, which now typically uses the unified parser approach.

## Modern Unified Matching

The recommended approach uses `FeatureChainParser.match_configuration_feature_chain_parser` which provides:

### 1. Dual Approach Support

```python
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_core.abstract_plugins.components.options import Options

@classmethod
def match_feature_group_criteria(cls, feature_name, options, data_access_collection=None):
    return FeatureChainParser.match_configuration_feature_chain_parser(
        feature_name,
        options,
        property_mapping=cls.PROPERTY_MAPPING,  # Configuration-based matching
        prefix_patterns=[cls.PREFIX_PATTERN],   # String-based matching (pattern defaults to CHAIN_SEPARATOR)
    )
```

### 2. PROPERTY_MAPPING Configuration

The `PROPERTY_MAPPING` defines how configuration-based features are validated:

```python
PROPERTY_MAPPING = {
    "aggregation_type": {
        "sum": "Sum aggregation",
        "avg": "Average aggregation",
        "max": "Maximum aggregation",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
    },
    DefaultOptionKeys.in_features: {
        "explanation": "Source feature for aggregation",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: False,
    },
}
```

### 3. Validation Modes

#### Strict Validation
When `mloda_strict_validation: True`, parameter values must be in the mapping:

```python
# This will match
options = Options(context={"aggregation_type": "sum"})  # "sum" is in mapping

# This will fail validation
options = Options(context={"aggregation_type": "custom"})  # "custom" not in mapping
```

#### Flexible Validation
When `mloda_strict_validation: False` (default), any value is accepted:

```python
# Both will match
options = Options(context={"in_features": "sales"})      # Any value OK
options = Options(context={"in_features": "custom_feature"})  # Any value OK
```

#### Custom Validation Functions
For complex validation beyond simple value lists:

```python
PROPERTY_MAPPING = {
    "window_size": {
        "explanation": "Size of the time window",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
        DefaultOptionKeys.mloda_validation_function: lambda x: isinstance(x, int) and x > 0,
    },
}
```

## Legacy Default Matching Criteria

For feature groups not yet modernized, the default matching criteria still apply:

1. **Root Feature with Matching Input Data**: The feature group is a root feature (has no dependencies) and its input data matches the feature.

2. **Class Name Match**: The feature name exactly matches the feature group's class name.
   ``` python
   feature_name == FeatureGroup.get_class_name()
   ```

3. **Prefix Match**: The feature name starts with the feature group's class name as a prefix.
   ``` python
   feature_name.startswith(FeatureGroup.prefix())  # Default prefix is "ClassName_"
   ```

4. **Explicitly Supported**: The feature name is in the set of explicitly supported feature names.
   ``` python
   feature_name in FeatureGroup.feature_names_supported()
   ```

## Matching Examples

### Modern Feature Group (Aggregation)

``` python
# String-based matching
feature = Feature("sales__sum_aggr")  # Matches via pattern

# Configuration-based matching
feature = Feature(
    "placeholder",
    Options(context={
        "aggregation_type": "sum",
        "in_features": "sales"
    })
)  # Matches via PROPERTY_MAPPING validation
```

### Parameter Classification Impact

The group/context parameter separation affects matching behavior:

``` python
# These create different Feature Group instances (different group parameters)
feature1 = Feature("placeholder", Options(
    group={"data_source": "production"},
    context={"aggregation_type": "sum", "in_features": "sales"}
))

feature2 = Feature("placeholder", Options(
    group={"data_source": "staging"},  # Different group parameter
    context={"aggregation_type": "sum", "in_features": "sales"}
))

# These create the same Feature Group instance (same group, different context)
feature3 = Feature("placeholder", Options(
    group={"data_source": "production"},
    context={"aggregation_type": "sum", "in_features": "sales"}
))

feature4 = Feature("placeholder", Options(
    group={"data_source": "production"},  # Same group parameter
    context={"aggregation_type": "avg", "in_features": "revenue"}  # Different context
))
```

## Migration Path

When modernizing a feature group:

1. **Add PROPERTY_MAPPING** with parameter definitions
2. **Update match_feature_group_criteria** to use unified parser
3. **Classify parameters** as group vs context appropriately
4. **Test both approaches** work correctly
5. **Update documentation** and examples
