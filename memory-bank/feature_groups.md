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
- Use `{window_function}_{window_size}_{time_unit}_window__{mloda_source_feature}` format (note the double underscore)
- Example: `avg_3_day_window__temperature`, `max_5_day_window__humidity`

### Missing Value Feature Group Pattern

When implementing feature groups that handle missing values in data:

**Naming Convention**:
- Use `{imputation_method}_imputed__{mloda_source_feature}` format (note the double underscore)
- Examples: `mean_imputed__income`, `median_imputed__age`, `constant_imputed__category`

### Aggregated Feature Group Pattern

When implementing feature groups that perform aggregation operations:

1. **Structure**: 
   - Create a base abstract class that defines the interface
   - Implement concrete classes for specific compute frameworks (e.g., Pandas)
   - Organize in a dedicated folder with separate files

2. **Naming Convention**:
   - Use `{aggregation_type}_aggr__{mloda_source_feature}` format (note the double underscore)
   - Example: `sum_aggr__sales`, `avg_aggr__price`

3. **Implementation Tips**:
   - Handle edge cases in feature name validation (empty prefix/suffix)
   - Use DataCreator with a set of feature names, not a dictionary
   - For testing, expect multiple DataFrames in results (one per feature group)

### Combined Feature Group Pattern

Feature groups can be composed to create complex features by chaining multiple transformations:

1. **Composability**:
   - Feature groups can use the output of other feature groups as their input
   - This allows for building complex features through a series of simpler transformations

2. **Naming Convention**:
   - Chain the naming patterns of the feature groups being composed
   - Example: `max_aggr__sum_7_day_window__mean_imputed__price`
     - First imputes missing values in price using mean imputation: `mean_imputed__price`
     - Then applies a 7-day time window sum: `sum_7_day_window__mean_imputed__price`
     - Finally applies a max aggregation: `max_aggr__sum_7_day_window__mean_imputed__price`

3. **Implementation Tips**:
   - Ensure each feature group in the chain correctly parses its portion of the feature name
   - Test the feature chain with different data scenarios to verify correct behavior
   - Use integration tests to validate the entire feature chain

## Changelog

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
