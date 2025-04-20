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

### Feature Chain Parser and Configuration

The feature chain parsing system consists of two main components:

#### FeatureChainParser

The `FeatureChainParser` is a utility class that handles the common aspects of feature name parsing and chaining across different feature group types:

1. **Purpose**:
   - Centralizes the logic for parsing feature names with chaining support
   - Provides a consistent approach for all feature groups
   - Makes feature group classes more focused on their specific functionality

2. **Key Methods**:
   - `extract_source_feature`: Extracts the source feature from a feature name based on a prefix pattern
   - `validate_feature_name`: Validates that a feature name matches the expected pattern
   - `is_chained_feature`: Checks if a feature name follows the chaining pattern
   - `get_prefix_part`: Extracts the prefix part from a feature name

3. **Usage**:
   - Each feature group defines a `PREFIX_PATTERN` that matches its naming convention
   - Feature groups use the `FeatureChainParser` methods with their specific pattern
   - This approach simplifies the implementation of new feature groups that support chaining

#### FeatureChainParserConfiguration

The `FeatureChainParserConfiguration` extends the feature chain parsing system to support configuration-based feature creation:

1. **Purpose**:
   - Enables creating features from options rather than explicit feature names
   - Simplifies feature creation in client code
   - Supports dynamic feature creation at runtime

2. **Key Methods**:
   - `parse_keys`: Returns the set of option keys used for parsing
   - `parse_from_options`: Creates a feature name from options
   - `create_feature_without_options`: Creates a feature with the parsed name and removes the parsed options

3. **Integration with Feature Groups**:
   - Feature groups implement `configurable_feature_chain_parser()` to return their parser configuration
   - The Engine automatically uses this configuration to parse features with the appropriate options
   - Implemented for all feature groups using a unified approach

4. **Unified Implementation**:
   - A factory function `create_configurable_parser` creates a configured parser class

5. **Usage Example**:

   **AggregatedFeatureGroup**:
   ```python
   feature = Feature(
       "PlaceHolder",  # Placeholder name, will be replaced
       Options({
           AggregatedFeatureGroup.AGGREGATION_TYPE: "sum",
           DefaultOptionKeys.mloda_source_feature: "Sales"
       })
   )
   
   # The Engine will automatically parse this into a feature with name "sum_aggr__Sales"
   ```
### Clustering Feature Group Pattern

When implementing feature groups that perform clustering operations:

**Naming Convention**:
- Use `cluster_{algorithm}_{k_value}__{mloda_source_features}` format (note the double underscore)
- Example: `cluster_kmeans_5__customer_behavior`, `cluster_dbscan_auto__sensor_readings`

### Geo Distance Feature Group Pattern

When implementing feature groups that calculate distances between geographic points:

**Naming Convention**:
- Use `{distance_type}_distance__{point1_feature}__{point2_feature}` format (note the double underscores)
- Examples: `haversine_distance__customer_location__store_location`, `euclidean_distance__origin__destination`

**Implementation Tips**:
- Support different distance metrics (haversine for geographic coordinates, euclidean and manhattan for Cartesian coordinates)
- Use numpy for efficient distance calculations
- Integrate with FeatureChainParserConfiguration for configuration-based creation

### DimensionalityReductionFeatureGroup Pattern

When implementing feature groups that perform dimensionality reduction operations:

**Naming Convention**:
- Use `{algorithm}_{dimension}d__{mloda_source_features}` format (note the double underscore)
- Example: `pca_10d__customer_metrics`, `tsne_2d__product_attributes`

**Result Columns**:
- Uses the multiple result columns pattern with naming convention `{feature_name}~dim{i+1}`
- Example: `pca_2d__customer_metrics~dim1`, `pca_2d__customer_metrics~dim2`

### Combined Feature Group Pattern

Feature groups can be composed to create complex features by chaining multiple transformations:

1. **Composability**:
   - Feature groups can use the output of other feature groups as their input
   - This allows for building complex features through a series of simpler transformations
   - The `FeatureChainParser` handles the extraction of source features in the chain

2. **Naming Convention**:
   - Chain the naming patterns of the feature groups being composed
   - Example: `max_aggr__sum_7_day_window__mean_imputed__price`
     - First imputes missing values in price using mean imputation: `mean_imputed__price`
     - Then applies a 7-day time window sum: `sum_7_day_window__mean_imputed__price`
     - Finally applies a max aggregation: `max_aggr__sum_7_day_window__mean_imputed__price`

3. **Implementation Tips**:
   - Use the `FeatureChainParser` to handle the parsing of chained feature names

### DimensionalityReductionFeatureGroup Pattern

When implementing feature groups that perform dimensionality reduction operations:

**Naming Convention**:
- Use `{algorithm}_{dimension}d__{mloda_source_features}` format (note the double underscore)
- Example: `pca_10d__customer_metrics`, `tsne_2d__product_attributes`

### Multiple Result Columns Pattern

When implementing feature groups that return multiple related columns:

**Naming Convention**:
- Use `{feature_name}~{column_suffix}` format (note the tilde separator)
- Example: `temperature~mean`, `temperature~max`, `temperature~min`

**Implementation Tips**:
- When consuming these features, access them using the naming convention pattern
- The framework automatically identifies and selects columns that match the pattern

### ForecastingFeatureGroup Pattern

When implementing feature groups that perform time series forecasting:

**Naming Convention**:
- Use `{algorithm}_forecast_{horizon}{time_unit}__{mloda_source_feature}` format (note the double underscore)
- Examples: `linear_forecast_7day__sales`, `randomforest_forecast_24hr__energy_consumption`

**Supported Algorithms**:
- Linear regression, ridge regression, lasso regression
- Random forest regression, gradient boosting regression
- Support vector regression, k-nearest neighbors regression

**Supported Time Units**:
- second, minute, hour, day, week, month, year

**Implementation Tips**:
- Requires a datetime column for time-based operations
- Automatically creates time-based features (hour, day of week, month, etc.)
- Supports artifact saving/loading to reuse trained models
- Integrates with FeatureChainParserConfiguration for configuration-based creation

### NodeCentralityFeatureGroup Pattern

When implementing feature groups that calculate centrality metrics for nodes in a graph:

**Naming Convention**:
- Use `{centrality_type}_centrality__{mloda_source_feature}` format (note the double underscore)
- Examples: `degree_centrality__user`, `betweenness_centrality__product`, `pagerank_centrality__website`

**Supported Centrality Types**:
- `degree`: Measures the number of connections a node has
- `betweenness`: Measures how often a node lies on the shortest path between other nodes
- `closeness`: Measures how close a node is to all other nodes
- `eigenvector`: Measures the influence of a node in a network
- `pagerank`: A variant of eigenvector centrality used by Google

**Graph Types**:
- `directed`: A graph where edges have direction
- `undirected`: A graph where edges have no direction (default)

**Implementation Tips**:
- Requires edge data with source and target columns
- Optionally supports weighted edges with a weight column
- Integrates with FeatureChainParserConfiguration for configuration-based creation
- Implements matrix-based centrality calculations without requiring external graph libraries

## Changelog

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

### 2025-04-19: Unified Feature Chain Parser Configuration
- Unified the implementation of configurable_feature_chain_parser across all feature groups
- Enhanced the create_configurable_parser function to handle list values and required keys
- Fixed template key case sensitivity issues in TimeWindowFeatureGroup
- Improved validation rules to provide consistent error messages
- Updated all feature groups to use the unified implementation
- Added comprehensive tests to verify the unified implementation

### 2025-04-18: Added GeoDistanceFeatureGroup
- Implemented GeoDistanceFeatureGroup with Pandas support
- Added support for haversine, euclidean, and manhattan distance calculations
- Integrated with FeatureChainParserConfiguration for configuration-based creation
- Added comprehensive unit and integration tests

### 2025-04-18: Added TextCleaningFeatureGroup
- Added support for text normalization, stopword removal, punctuation removal, etc.
- Added behavior note: different options create different feature sets in results

### 2025-04-17: Added ClusteringFeatureGroup
- Implemented ClusteringFeatureGroup with Pandas support
- Added support for K-means, DBSCAN, hierarchical, spectral, and affinity clustering
- Added automatic determination of optimal cluster count

### 2025-04-17: Added Feature Chain Parser Configuration
- Created `FeatureChainParserConfiguration` class for configuration-based feature creation
- Moved feature_chain_parser.py to core components for better organization
- Enhanced AggregatedFeatureGroup with configuration-based creation
- Added integration tests demonstrating the new functionality
- Updated Engine to automatically parse features with appropriate options

### 2025-04-15: Added Feature Chain Parser
- Created `FeatureChainParser` utility class to handle feature name parsing and chaining
- Refactored TimeWindowFeatureGroup, MissingValueFeatureGroup, and AggregatedFeatureGroup to use the parser
- Improved maintainability by centralizing the chaining logic in one place

### 2025-04-15: Refactored Feature Group Chaining
- Extracted chaining capability from feature group classes to improve maintainability
- Simplified implementation of TimeWindowFeatureGroup, MissingValueFeatureGroup, and AggregatedFeatureGroup

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
