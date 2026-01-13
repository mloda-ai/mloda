# Feature Chain Parser

## Overview

The Feature Chain Parser system enables feature groups to work with both traditional string-based feature names and modern configuration-based feature creation. This unified approach provides flexibility while maintaining backward compatibility.

> **For AI Agents:** Feature chaining enables LLMs to declare complex data pipelines through simple naming conventions. Instead of writing pipeline code, agents can request `user_query__validated__retrieved__pii_redacted` - mloda resolves the full chain automatically.

## Key Concepts

### Separator System

mloda uses three separator characters in feature names, each with a specific purpose:

| Separator | Constant | Purpose | Example |
|-----------|----------|---------|---------|
| `__` | `CHAIN_SEPARATOR` | Separates chained transformations (source→suffix) | `price__mean_imputed` |
| `~` | `COLUMN_SEPARATOR` | Separates multi-column output index | `feature__pca~0` |
| `&` | `INPUT_SEPARATOR` | Separates multiple input features | `point1&point2__distance` |

These constants are available from the `mloda.provider` facade:

```python
from mloda.provider import (
    CHAIN_SEPARATOR,    # "__"
    COLUMN_SEPARATOR,   # "~"
    INPUT_SEPARATOR,    # "&"
)
```

### Feature Chaining

Feature chaining allows feature groups to be composed, where the output of one feature group becomes the input to another. This is reflected in the feature name using the chain separator (`__`):

```
{in_feature}__{operation}
```

For example:
- `sales__sum_aggr` - Simple feature
- `price__mean_imputed__sum_7_day_window__max_aggr` - Chained feature

### Multi-Feature Input

Some feature groups require multiple input features. These are separated using the input separator (`&`):

```
{feature1}&{feature2}__{operation}
```

For example:
- `point1&point2__haversine_distance` - GeoDistance with two points
- `age&income&score__cluster_kmeans_3` - Clustering with multiple features

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
from mloda.user import Options
from typing import Optional

# New Options architecture
options = Options(
    group={
        "data_source": "production",  # Affects Feature Group splitting
    },
    context={
        "aggregation_type": "sum",    # Doesn't affect splitting
        "in_features": "sales"
    }
)
```

### Configuration-Based Feature Creation

Modern feature creation uses the Options architecture:

``` python
from mloda.user import Feature, Options

# Traditional string-based approach:
feature = Feature("sales__sum_aggr")

# Modern configuration-based approach:
feature = Feature(
    "placeholder",  # Will be replaced during processing
    Options(
        context={
            "aggregation_type": "sum",
            "in_features": "sales"
        }
    )
)
```

## FeatureChainParserMixin

The `FeatureChainParserMixin` provides default implementations for common feature chain parsing operations. Feature groups that use feature chaining should inherit from this mixin to reduce boilerplate code.

### Basic Usage

``` python
from mloda.provider import FeatureGroup, FeatureChainParserMixin
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

class MyFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__my_operation$"

    # In-feature constraints
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1  # Or None for unlimited

    PROPERTY_MAPPING = {
        "operation_type": {
            "sum": "Sum operation",
            "avg": "Average operation",
            DefaultOptionKeys.mloda_context: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature",
            DefaultOptionKeys.mloda_context: True,
        },
    }

    # input_features() inherited from FeatureChainParserMixin
    # match_feature_group_criteria() inherited from FeatureChainParserMixin
```

### Mixin Configuration

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `PREFIX_PATTERN` | `str` | Required | Regex pattern for matching feature names |
| `PROPERTY_MAPPING` | `dict` | Required | Parameter validation configuration |
| `MIN_IN_FEATURES` | `int` | `1` | Minimum required in_features |
| `MAX_IN_FEATURES` | `int \| None` | `None` | Maximum allowed in_features (None = unlimited) |
| `IN_FEATURE_SEPARATOR` | `str` | `"&"` | Separator for multiple in_features |

### Customization Hooks

#### 1. Custom Validation with `_validate_string_match()`

Override this hook when you need custom validation for string-based feature names:

``` python
class ClusteringFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    @classmethod
    def _validate_string_match(cls, feature_name: str, operation_config: str, in_feature: str) -> bool:
        """Validate clustering-specific patterns."""
        if FeatureChainParser.is_chained_feature(feature_name):
            try:
                cls.parse_clustering_prefix(feature_name)
            except ValueError:
                return False
        return True
```

#### 2. Custom `input_features()` Method

Override when you need to add additional input features (e.g., time filter):

``` python
class TimeWindowFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        # Try string-based parsing first
        _, in_feature = FeatureChainParser.parse_feature_name(feature_name.name, [self.PREFIX_PATTERN])
        if in_feature is not None:
            time_filter_feature = Feature(self.get_reference_time_column(options))
            return {Feature(in_feature), time_filter_feature}

        # Fall back to configuration-based approach
        in_features = options.get_in_features()
        time_filter_feature = Feature(self.get_reference_time_column(options))
        return set(in_features) | {time_filter_feature}
```

#### 3. Custom `match_feature_group_criteria()` Method

Override for complex pre-check logic that can't be captured by the hook:

``` python
class SklearnPipelineFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    @classmethod
    def match_feature_group_criteria(cls, feature_name, options, data_access_collection=None) -> bool:
        """Custom matching with mutual exclusivity validation."""
        has_pipeline_name = options.get(cls.PIPELINE_NAME)
        has_pipeline_steps = options.get(cls.PIPELINE_STEPS)

        # Pre-check: require pattern in name if no config provided
        if has_pipeline_name is None and has_pipeline_steps is None:
            if "sklearn_pipeline_" not in str(feature_name):
                return False

        # Use base matching
        base_match = FeatureChainParser.match_configuration_feature_chain_parser(...)

        # Post-check: mutual exclusivity
        if base_match and has_pipeline_name and has_pipeline_steps:
            return False
        return base_match
```

## Modern Implementation in Feature Groups

### 1. Define PROPERTY_MAPPING Configuration

The modern approach uses `PROPERTY_MAPPING` to define parameter validation and classification:

``` python
from mloda.provider import FeatureGroup
from mloda.user import FeatureName
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

class MyFeatureGroup(FeatureGroup):
    PREFIX_PATTERN = r"__([a-zA-Z_]+)_operation$"

    PROPERTY_MAPPING = {
        # Feature-specific parameter
        "operation_type": {
            "sum": "Sum aggregation",
            "avg": "Average aggregation", 
            "max": "Maximum aggregation",
            DefaultOptionKeys.context: True,  # Context parameter
            DefaultOptionKeys.strict_validation: True,  # Strict validation
        },
        # Source feature parameter
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature for the operation",
            DefaultOptionKeys.context: True,  # Context parameter
            DefaultOptionKeys.strict_validation: False,  # Flexible validation
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
        prefix_patterns=[cls.PREFIX_PATTERN],
    )
```

### 3. Modernize input_features Method

Handle both string-based and configuration-based features:

``` python
def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
    """Extract source feature from either configuration-based options or string parsing."""

    # Try string-based parsing first
    _, in_feature = FeatureChainParser.parse_feature_name(
        feature_name, [self.PREFIX_PATTERN]
    )
    if in_feature is not None:
        return {Feature(in_feature)}

    # Fall back to configuration-based approach
    in_features = options.get_in_features()
    if len(in_features) != 1:
        raise ValueError(
            f"Expected exactly one in_feature, but found {len(in_features)}: {in_features}"
        )
    return set(in_features)
```

### 4. Update calculate_feature Method

Support dual approach in feature processing:

``` python
def calculate_feature(self, features, options):
    for feature in features.features:
        # Try configuration-based approach first
        try:
            in_features = feature.options.get_in_features()
            in_feature = next(iter(in_features))
            in_feature_name = in_feature.get_name()
            
            # Extract parameters from options
            operation_type = feature.options.get("operation_type")
            
        except (ValueError, StopIteration):
            # Fall back to string-based approach for legacy features
            operation_type, in_feature_name = FeatureChainParser.parse_feature_name(
                feature.name, [cls.PREFIX_PATTERN]
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
        DefaultOptionKeys.context: True,
        DefaultOptionKeys.strict_validation: True,
        DefaultOptionKeys.validation_function: lambda x: isinstance(x, int) and x > 0,
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
        DefaultOptionKeys.default: "7",  # Default value
        DefaultOptionKeys.context: True,
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
        DefaultOptionKeys.group: True,  # Explicit group parameter
        DefaultOptionKeys.strict_validation: True,
    },
    # Context parameter - doesn't affect resolution
    "algorithm_type": {
        "kmeans": "K-means clustering",
        "dbscan": "DBSCAN clustering",
        DefaultOptionKeys.context: True,  # Context parameter
        DefaultOptionKeys.strict_validation: False,  # Flexible validation
    },
}
```

## Multiple Result Columns with ~ Pattern

Some feature groups produce multiple result columns from a single input feature. mloda provides utilities to work with these patterns seamlessly.

### Producer Side: Creating Multi-Column Outputs

Use `apply_naming_convention()` to create properly named columns:

``` python
from mloda.provider import FeatureGroup, FeatureSet

class MultiColumnProducer(FeatureGroup):
    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Compute results (e.g., from sklearn OneHotEncoder)
        result = encoder.transform(data)  # Returns 2D numpy array (n_samples, n_features)

        # Automatically apply naming convention
        feature_name = features.get_name_of_one_feature().name
        named_columns = cls.apply_naming_convention(result, feature_name)
        # Returns: {"category__onehot_encoded~0": data, "~1": data, "~2": data}

        return named_columns
```

### Consumer Side: Discovering Multi-Column Features

Use `resolve_multi_column_feature()` to automatically discover columns:

``` python
class MultiColumnConsumer(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        # Request base feature without ~N suffix
        return {Feature("category__onehot_encoded")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Automatically discover all matching columns
        columns = cls.resolve_multi_column_feature(
            "category__onehot_encoded",
            set(data.columns)
        )
        # Returns: ["category__onehot_encoded~0", "~1", "~2"]

        # Process all discovered columns
        result = sum(data[col] for col in columns)

        feature_name = features.get_name_of_one_feature().name
        return {feature_name: result}
```

### Manual Column Access (Legacy)

For backwards compatibility, you can still access specific columns:

``` python
# Manual specification of specific columns
base_feature = "category__onehot_encoded"  # Creates all columns
specific_column = "category__onehot_encoded~0"  # Access first column
another_column = "category__onehot_encoded~1"  # Access second column
```

**Recommended**: Use automatic discovery (`resolve_multi_column_feature`) instead of manual enumeration for cleaner, more maintainable code.

## Benefits

- **Consistent Naming**: Enforces naming conventions across feature groups
- **Composability**: Enables building complex features through chaining
- **Configuration-Based Creation**: Simplifies feature creation in client code
- **Validation**: Ensures feature names follow expected patterns
- **Multi-Column Support**: Handle transformations that produce multiple result columns
