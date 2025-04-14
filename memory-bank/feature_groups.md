# Feature Groups

## Overview

Feature Groups are a core component of the mloda framework that define feature dependencies and calculations. They transform input data into meaningful features for analysis and machine learning.

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
- `match_feature_group_criteria`: Determines whether this feature group matches given criteria
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

## Usage

Feature Groups are used to:

- Define dependencies between features
- Implement transformation logic
- Specify data types and constraints
- Manage feature metadata

Custom Feature Groups can be created by inheriting from `AbstractFeatureGroup` and implementing the required methods.

## Creating New Feature Groups

### Time Window Feature Group Pattern

When implementing feature groups that perform time-based window operations:

**Naming Convention**:
- Use `{window_function}_{window_size}_{time_unit}_window_{source_feature_name}` format
- Example: `avg_3_day_window_temperature`, `max_5_day_window_humidity`

### Missing Value Feature Group Pattern

When implementing feature groups that handle missing values in data:

**Naming Convention**:
- Use `{imputation_method}_imputed_{source_feature_name}` format
- Examples: `mean_imputed_income`, `median_imputed_age`, `constant_imputed_category`

### Aggregated Feature Group Pattern

When implementing feature groups that perform aggregation operations:

1. **Structure**: 
   - Create a base abstract class that defines the interface
   - Implement concrete classes for specific compute frameworks (e.g., Pandas)
   - Organize in a dedicated folder with separate files

2. **Naming Convention**:
   - Use `{aggregation_type}_aggr_{source_feature_name}` format
   - Example: `sum_aggr_sales`, `avg_aggr_price`

3. **Implementation Tips**:
   - Handle edge cases in feature name validation (empty prefix/suffix)
   - Use DataCreator with a set of feature names, not a dictionary
   - For testing, expect multiple DataFrames in results (one per feature group)

## Changelog

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
