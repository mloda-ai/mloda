# Feature Chain Parser

## Overview

The Feature Chain Parser system enables feature groups to work with both traditional string-based feature names and modern configuration-based feature creation. This unified approach provides flexibility while maintaining backward compatibility.

## Key Concepts

### Feature Chaining

Feature chaining allows feature groups to be composed, where the output of one feature group becomes the input to another. This is reflected in the feature name using a double underscore pattern:

```
{operation}__{source_feature}
```

For example:
- `sum_aggr__sales` - Simple feature
- `max_aggr__sum_7_day_window__mean_imputed__price` - Chained feature

### Unified Parser Architecture

The modernized `FeatureChainParser` provides a unified approach through the `match_configuration_feature_chain_parser` method that handles:

- **String-based features**: Traditional pattern matching with regex
- **Configuration-based features**: Modern approach using Options and PROPERTY_MAPPING
- **Dual validation**: Features can be validated using either or both approaches

### Options Architecture: Group vs Context Parameters

The new `Options` class separates parameters into two categories:

- **Group Parameters**: Affect Feature Group resolution and splitting (stored in `options.group`)
- **Context Parameters**: Metadata that doesn't affect splitting (stored in `options.context`)

``` python
from mloda_core.abstract_plugins.components.options import Options
from typing import Optional

# New Options architecture
options = Options(
    group={
        "data_source": "production",  # Affects Feature Group splitting
    },
    context={
        "aggregation_type": "sum",    # Doesn't affect splitting
        "mloda_source_features": "sales"
    }
)
```

### Configuration-Based Feature Creation

Modern feature creation uses the Options architecture:

``` python
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options

# Traditional string-based approach:
feature = Feature("sum_aggr__sales")

# Modern configuration-based approach:
feature = Feature(
    "placeholder",  # Will be replaced during processing
    Options(
        context={
            "aggregation_type": "sum",
            "mloda_source_features": "sales"
        }
    )
)
```

## Modern Implementation in Feature Groups

### 1. Define PROPERTY_MAPPING Configuration

The modern approach uses `PROPERTY_MAPPING` to define parameter validation and classification:

``` python
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_core.abstract_plugins.components.feature_name import FeatureName

class MyFeatureGroup(AbstractFeatureGroup):
    PATTERN = "__"
    PREFIX_PATTERN = [r"^([a-zA-Z_]+)_operation__"]
    
    PROPERTY_MAPPING = {
        # Feature-specific parameter
        "operation_type": {
            "sum": "Sum aggregation",
            "avg": "Average aggregation", 
            "max": "Maximum aggregation",
            DefaultOptionKeys.mloda_context: True,  # Context parameter
            DefaultOptionKeys.mloda_strict_validation: True,  # Strict validation
        },
        # Source feature parameter
        DefaultOptionKeys.mloda_source_features: {
            "explanation": "Source feature for the operation",
            DefaultOptionKeys.mloda_context: True,  # Context parameter
            DefaultOptionKeys.mloda_strict_validation: False,  # Flexible validation
        },
    }
```

### 2. Update match_feature_group_criteria

Replace old pattern-only matching with unified parser:

``` python
@classmethod
def match_feature_group_criteria(cls, feature_name, options, data_access_collection=None):
    return FeatureChainParser.match_configuration_feature_chain_parser(
        feature_name, 
        options, 
        property_mapping=cls.PROPERTY_MAPPING,
        pattern=cls.PATTERN, 
        prefix_patterns=cls.PREFIX_PATTERN
    )
```

### 3. Modernize input_features Method

Handle both string-based and configuration-based features:

``` python
def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
    """Extract source feature from either configuration-based options or string parsing."""
    
    # Try string-based parsing first
    _, source_feature = FeatureChainParser.parse_feature_name(
        feature_name, self.PATTERN, self.PREFIX_PATTERN
    )
    if source_feature is not None:
        return {Feature(source_feature)}
    
    # Fall back to configuration-based approach
    source_features = options.get_source_features()
    if len(source_features) != 1:
        raise ValueError(
            f"Expected exactly one source feature, but found {len(source_features)}: {source_features}"
        )
    return set(source_features)
```

### 4. Update calculate_feature Method

Support dual approach in feature processing:

``` python
def calculate_feature(self, features, options):
    for feature in features.features:
        # Try configuration-based approach first
        try:
            source_features = feature.options.get_source_features()
            source_feature = next(iter(source_features))
            source_feature_name = source_feature.get_name()
            
            # Extract parameters from options
            operation_type = feature.options.get("operation_type")
            
        except (ValueError, StopIteration):
            # Fall back to string-based approach for legacy features
            operation_type, source_feature_name = FeatureChainParser.parse_feature_name(
                feature.name, self.PATTERN, self.PREFIX_PATTERN
            )
        
        # Process using extracted values
        # ... implementation logic
```

### 5. Advanced PROPERTY_MAPPING Features

#### Validation Functions

For complex validation beyond simple value lists:

``` python
PROPERTY_MAPPING = {
    "dimension": {
        "explanation": "Number of dimensions for reduction",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
        DefaultOptionKeys.mloda_validation_function: lambda x: isinstance(x, int) and x > 0,
    },
}
```

#### Default Values

Specify default values for optional parameters:

``` python
PROPERTY_MAPPING = {
    "window_size": {
        "7": "7-day window",
        "30": "30-day window",
        DefaultOptionKeys.mloda_default: "7",  # Default value
        DefaultOptionKeys.mloda_context: True,
    },
}
```

#### Group vs Context Classification

``` python
PROPERTY_MAPPING = {
    # Group parameter - affects Feature Group resolution
    "data_source": {
        "production": "Production data",
        "staging": "Staging data", 
        DefaultOptionKeys.mloda_group: True,  # Explicit group parameter
        DefaultOptionKeys.mloda_strict_validation: True,
    },
    # Context parameter - doesn't affect resolution
    "algorithm_type": {
        "kmeans": "K-means clustering",
        "dbscan": "DBSCAN clustering",
        DefaultOptionKeys.mloda_context: True,  # Context parameter
        DefaultOptionKeys.mloda_strict_validation: False,  # Flexible validation
    },
}
```

## Multiple Result Columns with ~ Pattern

Some feature groups produce multiple result columns from a single input feature. The `~` pattern allows accessing individual columns:

``` python
# OneHot encoding creates multiple columns
base_feature = "onehot_encoded__category"  # Creates all columns
specific_column = "onehot_encoded__category~0"  # Access first column
another_column = "onehot_encoded__category~1"  # Access second column
```

**Implementation Note**: Feature groups handle this pattern in their `input_features()` method to extract the base feature name, and in `calculate_feature()` to create the appropriately named result columns.

## Benefits

- **Consistent Naming**: Enforces naming conventions across feature groups
- **Composability**: Enables building complex features through chaining
- **Configuration-Based Creation**: Simplifies feature creation in client code
- **Validation**: Ensures feature names follow expected patterns
- **Multi-Column Support**: Handle transformations that produce multiple result columns
