# Feature Groups

## Overview

Feature Groups are a core component of the mloda framework that define feature dependencies and calculations. They transform input data into meaningful features for analysis and machine learning using a modern configuration-based architecture.

## Key Concepts

### Description and Versioning

Feature Groups now include built-in description and versioning capabilities:

#### Description

- `description()`: Returns a human-readable description of a feature group
- Uses the class's docstring or falls back to the class name
- Helps users understand the purpose and functionality

#### Versioning

- `version()`: Generates a composite version identifier
- Combines mloda package version, module name, and source code hash
- Tracks changes, manages compatibility, and aids debugging

### AbstractFeatureGroup

The base class for all feature groups in mloda, providing a common interface and functionality:

#### Core Methods (To Implement)
- `calculate_feature`: Contains the logic to transform input features into output features

#### Meta Data Methods
- `description`: Returns a human-readable description of the feature group
- `version`: Returns a composite version identifier for the feature group

#### Feature identification
- `match_feature_group_criteria`: Determines whether this feature group matches given criteria using the unified parser approach
- `feature_names_supported`: Returns a set of feature names explicitly supported
- `prefix`: Returns the prefix used for feature names

#### Feature input data
- `input_data`: Returns the input data class used for this feature group
- `input_features`: Defines the input features required by this feature group

#### Feature setup
- `input_data`: Returns the input data class used for this feature group
- `get_domain`: Returns the domain for this feature group
- `return_data_type_rule`: Specifies a fixed return data type for this feature group
- `index_columns`: Specifies the index columns used for merging or joining data
- `compute_framework_rule`: Defines the rule for determining the compute framework
- `artifact`: Returns the artifact associated with this feature group
- `supports_index`: Indicates whether this feature group supports a given index
- `set_feature_name`: Allows modification of the feature name based on configuration

#### Quality Methods (Can Override)
- `validate_input_features`: Validates the input data (optional)
- `validate_output_features`: Validates the output data (optional)

#### Utility Methods (Final)
- `load_artifact`: Loads an artifact associated with a FeatureSet
- `get_class_name`: Returns the name of the class
- `is_root`: Determines if this is a root feature (no dependencies)
- `compute_framework_definition`: Determines supported compute frameworks

## Modern Architecture

### PROPERTY_MAPPING Configuration System

All feature groups now use the `PROPERTY_MAPPING` configuration system for parameter definition and validation:

```python
PROPERTY_MAPPING = {
    # Feature-specific parameters (e.g., AGGREGATION_TYPE)
    FEATURE_PARAMETER: {
        **VALID_VALUES_DICT,  # All supported values as valid options
        DefaultOptionKeys.mloda_context: True,  # Mark as context parameter
        DefaultOptionKeys.mloda_strict_validation: True,  # Enable strict validation
        DefaultOptionKeys.mloda_validation_function: lambda x: x in VALID_VALUES_DICT,  # Custom validation
    },
    # Source feature parameter
    DefaultOptionKeys.mloda_source_feature: {
        "explanation": "Source feature description",
        DefaultOptionKeys.mloda_context: True,  # Mark as context parameter
        DefaultOptionKeys.mloda_strict_validation: False,  # Flexible validation
    },
}
```

### Context vs Group Parameter Classification

Parameters are classified into two categories that affect feature group behavior:

#### Context Parameters
- **Don't affect Feature Group resolution/splitting**
- Include feature-specific parameters (aggregation_type, algorithm_type, etc.)
- Include `mloda_source_feature`
- Marked with `DefaultOptionKeys.mloda_context: True`

#### Group Parameters  
- **Affect Feature Group resolution/splitting**
- Include data source isolation parameters
- Include environment-specific parameters
- Not marked as context parameters (default behavior)

### Unified Parser Integration

All feature groups use the unified `match_configuration_feature_chain_parser` approach:

```python
def match_feature_group_criteria(cls, feature_name: FeatureName, options: Options) -> bool:
    return FeatureChainParser.match_configuration_feature_chain_parser(
        feature_name, options, 
        property_mapping=cls.PROPERTY_MAPPING,
        pattern=cls.PATTERN, 
        prefix_patterns=[cls.PREFIX_PATTERN]
    )
```

### Dual Approach Support

Feature groups support both creation methods seamlessly:

#### String-Based Creation (Legacy)
```python
# Traditional approach using feature names
feature = Feature("sum_aggr__sales")
```

#### Configuration-Based Creation (Modern)
```python
# Modern approach using Options with proper parameter classification
feature = Feature(
    "placeholder",  # Name will be generated
    Options(
        group={
            # Group parameters (affect resolution)
        },
        context={
            "aggregation_type": "sum",
            DefaultOptionKeys.mloda_source_feature: "sales",
            # Other context parameters
        }
    )
)
```

## Usage

Feature Groups are used to:

- Define dependencies between features
- Implement transformation logic using modern configuration
- Specify data types and constraints through PROPERTY_MAPPING
- Manage feature metadata with proper parameter classification

Custom Feature Groups can be created by inheriting from `AbstractFeatureGroup` and implementing the required methods with PROPERTY_MAPPING configuration.

## Creating New Feature Groups

### Modern Implementation Pattern

When creating new feature groups, follow this modernized pattern:

1. **Define PROPERTY_MAPPING** with parameter classification
2. **Update match_feature_group_criteria** to use unified parser
3. **Implement dual approach support** in calculate_feature and input_features methods
4. **Add proper validation functions** for complex parameters

### Aggregated Feature Group Pattern

For aggregation operations on features:

**Naming Convention**:
- Use `{aggregation_type}_aggr__{mloda_source_feature}` format (note the double underscore)
- Example: `sum_aggr__sales`, `avg_aggr__price`

**Modern Configuration**:
```python
PROPERTY_MAPPING = {
    AGGREGATION_TYPE: {
        **AGGREGATION_TYPES_DICT,
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
    },
    DefaultOptionKeys.mloda_source_feature: {
        "explanation": "Source feature for aggregation",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: False,
    },
}
```

**Dual Creation Examples**:
```python
# String-based
feature = Feature("sum_aggr__sales")

# Configuration-based
feature = Feature(
    "placeholder",
    Options(context={
        "aggregation_type": "sum",
        DefaultOptionKeys.mloda_source_feature: "sales"
    })
)
```

### Time Window Feature Group Pattern

For time-based window operations:

**Naming Convention**:
- Use `{window_function}_{window_size}_{time_unit}_window__{mloda_source_feature}` format
- Example: `avg_3_day_window__temperature`, `max_5_day_window__humidity`

**Modern Configuration**:
```python
PROPERTY_MAPPING = {
    WINDOW_FUNCTION: {
        **WINDOW_FUNCTIONS_DICT,
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
    },
    WINDOW_SIZE: {
        "explanation": "Size of time window (positive integer)",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
        DefaultOptionKeys.mloda_validation_function: lambda x: isinstance(x, int) and x > 0,
    },
    TIME_UNIT: {
        **TIME_UNITS_DICT,
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
    },
    DefaultOptionKeys.mloda_source_feature: {
        "explanation": "Source feature for window operation",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: False,
    },
}
```

### Missing Value Feature Group Pattern

For handling missing values in data:

**Naming Convention**:
- Use `{imputation_method}_imputed__{mloda_source_feature}` format
- Examples: `mean_imputed__income`, `median_imputed__age`, `constant_imputed__category`

**Modern Configuration**:
```python
PROPERTY_MAPPING = {
    IMPUTATION_METHOD: {
        **IMPUTATION_METHODS_DICT,
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
    },
    CONSTANT_VALUE: {
        "explanation": "Constant value for imputation (when using constant method)",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: False,
    },
    DefaultOptionKeys.mloda_source_feature: {
        "explanation": "Source feature for imputation",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: False,
    },
}
```

### Clustering Feature Group Pattern

For clustering operations:

**Naming Convention**:
- Use `cluster_{algorithm}_{k_value}__{mloda_source_features}` format
- Example: `cluster_kmeans_5__customer_behavior`, `cluster_dbscan_auto__sensor_readings`

**Modern Configuration**:
```python
PROPERTY_MAPPING = {
    ALGORITHM: {
        **CLUSTERING_ALGORITHMS_DICT,
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
    },
    K_VALUE: {
        "explanation": "Number of clusters or 'auto' for automatic determination",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
        DefaultOptionKeys.mloda_validation_function: lambda x: x == "auto" or (isinstance(x, int) and x > 0),
    },
    DefaultOptionKeys.mloda_source_feature: {
        "explanation": "Source features for clustering (multiple features supported)",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: False,
    },
}
```

### Dimensionality Reduction Feature Group Pattern

For dimensionality reduction operations:

**Naming Convention**:
- Use `{algorithm}_{dimension}d__{mloda_source_features}` format
- Example: `pca_10d__customer_metrics`, `tsne_2d__product_attributes`

**Result Columns**:
- Uses multiple result columns pattern: `{feature_name}~dim{i+1}`
- Example: `pca_2d__customer_metrics~dim1`, `pca_2d__customer_metrics~dim2`

**Modern Configuration**:
```python
PROPERTY_MAPPING = {
    ALGORITHM: {
        **DIMENSIONALITY_REDUCTION_ALGORITHMS_DICT,
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,  
    },
    DIMENSION: {
        "explanation": "Target dimension (positive integer)",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
        DefaultOptionKeys.mloda_validation_function: lambda x: isinstance(x, int) and x > 0,
    },
    DefaultOptionKeys.mloda_source_feature: {
        "explanation": "Source features for dimensionality reduction",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: False,
    },
}
```

### Geo Distance Feature Group Pattern

For calculating distances between geographic points:

**Naming Convention**:
- Use `{distance_type}_distance__{point1_feature}__{point2_feature}` format
- Examples: `haversine_distance__customer_location__store_location`, `euclidean_distance__origin__destination`

**Modern Configuration**:
```python
PROPERTY_MAPPING = {
    DISTANCE_TYPE: {
        **DISTANCE_TYPES_DICT,
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
    },
    DefaultOptionKeys.mloda_source_feature: {
        "explanation": "Two source features representing geographic points",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: False,
        DefaultOptionKeys.mloda_validation_function: lambda x: (
            isinstance(x, str) or  # Individual strings when parser iterates
            (isinstance(x, (list, tuple, frozenset, set)) and len(x) == 2)  # Collections
        ),
    },
}
```

### Forecasting Feature Group Pattern

For time series forecasting:

**Naming Convention**:
- Use `{algorithm}_forecast_{horizon}{time_unit}__{mloda_source_feature}` format
- Examples: `linear_forecast_7day__sales`, `randomforest_forecast_24hr__energy_consumption`

**Modern Configuration**:
```python
PROPERTY_MAPPING = {
    ALGORITHM: {
        **FORECASTING_ALGORITHMS_DICT,
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
    },
    HORIZON: {
        "explanation": "Forecast horizon (positive integer)",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
        DefaultOptionKeys.mloda_validation_function: lambda x: isinstance(x, int) and x > 0,
    },
    TIME_UNIT: {
        **TIME_UNITS_DICT,
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
    },
    DefaultOptionKeys.mloda_source_feature: {
        "explanation": "Source feature for forecasting",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: False,
    },
}
```

### Node Centrality Feature Group Pattern

For calculating centrality metrics in graphs:

**Naming Convention**:
- Use `{centrality_type}_centrality__{mloda_source_feature}` format
- Examples: `degree_centrality__user`, `betweenness_centrality__product`, `pagerank_centrality__website`

**Modern Configuration**:
```python
PROPERTY_MAPPING = {
    CENTRALITY_TYPE: {
        **CENTRALITY_TYPES_DICT,
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
    },
    GRAPH_TYPE: {
        **GRAPH_TYPES_DICT,
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
    },
    DefaultOptionKeys.mloda_source_feature: {
        "explanation": "Source feature representing graph nodes",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: False,
    },
}
```

### Sklearn Pipeline Feature Group Pattern

For sklearn pipeline transformations:

**Naming Convention**:
- Use `sklearn_pipeline_{pipeline_name}__{mloda_source_feature}` format
- Example: `sklearn_pipeline_scaling__income`, `sklearn_pipeline_preprocessing__features`

**Modern Configuration**:
```python
PROPERTY_MAPPING = {
    PIPELINE_NAME: {
        **SKLEARN_PIPELINES_DICT,
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
    },
    PIPELINE_STEPS: {
        "explanation": "Custom pipeline steps (mutually exclusive with PIPELINE_NAME)",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: False,
        DefaultOptionKeys.mloda_validation_function: lambda x: isinstance(x, (frozenset, list, tuple)),
    },
    PIPELINE_PARAMS: {
        "explanation": "Pipeline parameters",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: False,
    },
    DefaultOptionKeys.mloda_source_feature: {
        "explanation": "Source feature for sklearn pipeline transformation",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: False,
    },
}
```

### Text Cleaning Feature Group Pattern

For text normalization and cleaning:

**Naming Convention**:
- Use `text_clean_{operations}__{mloda_source_feature}` format
- Example: `text_clean_normalize_stopwords__content`, `text_clean_punctuation__description`

**Modern Configuration**:
```python
PROPERTY_MAPPING = {
    OPERATIONS: {
        **TEXT_OPERATIONS_DICT,
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: True,
    },
    DefaultOptionKeys.mloda_source_feature: {
        "explanation": "Source feature containing text to clean",
        DefaultOptionKeys.mloda_context: True,
        DefaultOptionKeys.mloda_strict_validation: False,
    },
}
```

### Combined Feature Group Pattern

Feature groups can be composed to create complex features by chaining multiple transformations:

**Composability**:
- Feature groups can use the output of other feature groups as their input
- Allows building complex features through a series of simpler transformations
- The unified parser handles extraction of source features in the chain

**Chaining Example**:
```python
# Complex feature chain: price → mean_imputed__price → sum_7_day_window__mean_imputed__price → max_aggr__sum_7_day_window__mean_imputed__price

# Step 1: Missing value imputation
imputed_feature = Feature(
    "placeholder",
    Options(context={
        "imputation_method": "mean",
        DefaultOptionKeys.mloda_source_feature: "price"
    })
)
# Results in: mean_imputed__price

# Step 2: Time window aggregation  
window_feature = Feature(
    "placeholder", 
    Options(context={
        "window_function": "sum",
        "window_size": 7,
        "time_unit": "day",
        DefaultOptionKeys.mloda_source_feature: "mean_imputed__price"
    })
)
# Results in: sum_7_day_window__mean_imputed__price

# Step 3: Final aggregation
final_feature = Feature(
    "placeholder",
    Options(context={
        "aggregation_type": "max", 
        DefaultOptionKeys.mloda_source_feature: "sum_7_day_window__mean_imputed__price"
    })
)
# Results in: max_aggr__sum_7_day_window__mean_imputed__price
```

### Multiple Result Columns Pattern

For feature groups that return multiple related columns:

**Naming Convention**:
- Use `{feature_name}~{column_suffix}` format (note the tilde separator)
- Example: `temperature~mean`, `temperature~max`, `temperature~min`

**Implementation Tips**:
- Framework automatically identifies and selects columns matching the pattern
- Used by dimensionality reduction and other multi-output feature groups

## Implementation Guidelines

### Modern Feature Group Implementation

When implementing new feature groups:

1. **Define PROPERTY_MAPPING** with proper parameter classification:
   ```python
   PROPERTY_MAPPING = {
       # Context parameters (don't affect resolution)
       FEATURE_PARAMETER: {
           **VALID_VALUES_DICT,
           DefaultOptionKeys.mloda_context: True,
           DefaultOptionKeys.mloda_strict_validation: True,
           DefaultOptionKeys.mloda_validation_function: custom_validator,  # Optional
       },
   }
   ```

2. **Update match_feature_group_criteria** to use unified parser:
   ```python
   @classmethod
   def match_feature_group_criteria(cls, feature_name: FeatureName, options: Options) -> bool:
       return FeatureChainParser.match_configuration_feature_chain_parser(
           feature_name, options,
           property_mapping=cls.PROPERTY_MAPPING,
           pattern=cls.PATTERN,
           prefix_patterns=[cls.PREFIX_PATTERN]
       )
   ```

3. **Implement dual approach support** in calculate_feature:
   ```python
   def calculate_feature(self, features: FeatureCollection) -> DataCreator:
       for feature in features.features:
           # Try configuration-based approach first
           try:
               source_features = feature.options.get_source_features()
               source_feature = next(iter(source_features))
               param_value = feature.options.get("parameter_name")
           except (ValueError, StopIteration):
               # Fall back to string-based approach
               param_value, source_feature_name = FeatureChainParser.parse_feature_name(
                   feature.name, self.PATTERN, [self.PREFIX_PATTERN]
               )
           # Process using extracted values
   ```

4. **Update input_features** for dual support:
   ```python
   def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
       # Try string-based parsing first
       _, source_feature = FeatureChainParser.parse_feature_name(
           feature_name, self.PATTERN, [self.PREFIX_PATTERN]
       )
       if source_feature is not None:
           return {Feature(source_feature)}
       
       # Fall back to configuration-based approach
       source_features = options.get_source_features()
       if len(source_features) != 1:
           raise ValueError(f"Expected exactly one source feature, got {len(source_features)}")
       return set(source_features)
   ```

### Validation Functions

For complex parameter validation, use custom validation functions:

```python
# Simple validation
DefaultOptionKeys.mloda_validation_function: lambda x: isinstance(x, int) and x > 0

# Complex validation  
def validate_pipeline_steps(steps):
    if isinstance(steps, frozenset):
        return all(isinstance(step, tuple) and len(step) == 2 for step in steps)
    return isinstance(steps, (list, tuple))

DefaultOptionKeys.mloda_validation_function: validate_pipeline_steps
```

### Testing Pattern

When testing modernized feature groups:

```python
def test_dual_approach():
    # Test string-based creation
    string_feature = Feature("sum_aggr__sales")
    
    # Test configuration-based creation
    config_feature = Feature(
        "placeholder",
        Options(context={
            "aggregation_type": "sum",
            DefaultOptionKeys.mloda_source_feature: "sales"
        })
    )
    
    # Both should produce equivalent results
    assert process_feature(string_feature) == process_feature(config_feature)
```

## Modernization Status

### ✅ Completed Feature Groups

All feature groups have been successfully modernized to use the new PROPERTY_MAPPING configuration system:

- **Aggregated Feature Group**: Sum, avg, min, max aggregations with context parameter classification
- **Clustering Feature Group**: K-means, DBSCAN, hierarchical clustering with algorithm and k-value parameters
- **Data Quality (Missing Value) Feature Group**: Mean, median, mode, constant imputation methods
- **Dimensionality Reduction Feature Group**: PCA, t-SNE with algorithm and dimension parameters  
- **Forecasting Feature Group**: Linear, random forest, SVR forecasting with horizon and time unit parameters
- **Geo Distance Feature Group**: Haversine, euclidean, manhattan distance calculations
- **Node Centrality Feature Group**: Degree, betweenness, closeness, eigenvector, pagerank centrality
- **Sklearn Pipeline Feature Group**: Predefined and custom sklearn pipelines with mutual exclusivity validation
- **Text Cleaning Feature Group**: Text normalization, stopword removal, punctuation cleaning
- **Time Window Feature Group**: Time-based window operations with function, size, and time unit parameters

### Key Benefits Achieved

- **Modern Architecture**: All feature groups use unified PROPERTY_MAPPING configuration
- **Proper Parameter Classification**: Context parameters don't affect Feature Group splitting
- **Dual Approach Support**: Both string-based and configuration-based creation work seamlessly
- **Consistent Validation**: Built-in validation rules with custom validation function support
- **Improved Maintainability**: Unified parser approach reduces code duplication

## Changelog

### 2025-01-08: Feature Group Modernization Completion
- **BREAKING CHANGE**: Completed modernization of all feature groups to use PROPERTY_MAPPING configuration
- Removed `configurable_feature_chain_parser` approach entirely (non-backwards compatible)
- All feature groups now use unified `match_configuration_feature_chain_parser` approach
- Added proper context vs group parameter classification across all feature groups
- Implemented dual approach support (string-based + configuration-based) in all feature groups
- Added custom validation functions for complex parameter validation
- Updated all compute framework implementations (Pandas, PyArrow, Polars) to support modernized approach

### 2025-04-20: Added NodeCentralityFeatureGroup
- Implemented NodeCentralityFeatureGroup with Pandas support
- Added support for multiple centrality metrics (degree, betweenness, closeness, eigenvector, pagerank)
- Added support for both directed and undirected graphs
- Added integration tests demonstrating feature usage and configuration-based creation

### 2025-04-19: Added ForecastingFeatureGroup
- Implemented ForecastingFeatureGroup with Pandas support
- Added support for multiple forecasting algorithms
- Added artifact support for saving and loading trained models

### 2025-04-19: Added Multiple Result Columns Support
- Added `identify_naming_convention` method to ComputeFrameWork
- Updated compute framework implementations to support the new naming convention
- Created documentation for the multiple result columns pattern

### 2025-04-18: Added GeoDistanceFeatureGroup
- Implemented GeoDistanceFeatureGroup with Pandas support
- Added support for haversine, euclidean, and manhattan distance calculations
- Added comprehensive unit and integration tests

### 2025-04-18: Added TextCleaningFeatureGroup
- Added support for text normalization, stopword removal, punctuation removal, etc.
- Added behavior note: different options create different feature sets in results

### 2025-04-17: Added ClusteringFeatureGroup
- Implemented ClusteringFeatureGroup with Pandas support
- Added support for K-means, DBSCAN, hierarchical, spectral, and affinity clustering
- Added automatic determination of optimal cluster count

### 2025-04-15: Added Feature Chain Parser
- Created `FeatureChainParser` utility class to handle feature name parsing and chaining
- Refactored TimeWindowFeatureGroup, MissingValueFeatureGroup, and AggregatedFeatureGroup to use the parser
- Improved maintainability by centralizing the chaining logic in one place

### 2025-04-15: Updated Feature Naming Conventions
- Added double underscore before source feature names in all feature groups
- Updated naming patterns to `{operation}__{mloda_source_feature}` format
- Created integration test for combined feature groups demonstrating composability

### 2025-04-15: Standardized Source Feature Terminology
- Renamed `get_source_feature` method to `mloda_source_feature` in all feature groups

### 2025-04-14: Added Missing Value Imputation Feature Group
- Created pattern for handling missing values in features
- Implemented multiple imputation methods: mean, median, mode, constant, ffill, bfill

### 2025-04-14: Added Time Window Feature Group
- Created pattern for implementing time-based window operations on features
- Integrated with global filter functionality

### 2025-04-03: Added PyArrow Aggregated Feature Group
- Implemented PyArrow version of the aggregated feature group

### 2025-04-03: Added Aggregated Feature Group
- Created pattern for implementing aggregation operations on features
- Implemented for Pandas with support for sum, min, max, avg, etc.

### 2025-03-30: Added Description and Versioning
- Added `description()` method to provide human-readable descriptions of feature groups
- Added `version()` method to generate composite version identifiers
- Created `FeatureGroupVersion` class to handle versioning logic
