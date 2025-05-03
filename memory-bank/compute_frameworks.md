# Compute Frameworks

## Overview

Compute Frameworks define the technology stack for executing feature transformations in mloda. They decouple feature definitions from specific computation technologies, enabling flexibility for online/offline computation, testing, and migrations between environments.

## Key Concepts

### ComputeFrameWork

The base class for all compute frameworks:

#### Core Methods
- `expected_data_framework`: Defines the data type (e.g., pandas.DataFrame, pyarrow.Table)
- `transform`: Converts data between formats
- `select_data_by_column_names`: Extracts specific columns from the data
- `identify_naming_convention`: Supports multiple result columns pattern

#### Supporting Components

- **MergeEngine**: Handles joining/merging operations between datasets
  - Each compute framework provides its own implementation (e.g., PandasMergeEngine)
  - Supports various join types: inner, left, right, outer, append, union

- **FilterEngine**: Manages data filtering operations
  - Framework-specific implementations (e.g., PandasFilterEngine)
  - Applies filters during different stages: initial read, calculation, final output

## Implementations

### PandasDataframe
- Based on pandas DataFrame
- Optimized for data manipulation and analysis
- Provides rich data transformation capabilities
- Suitable for development and smaller datasets

### PyarrowTable
- Based on Apache Arrow Tables
- Memory-efficient columnar format
- Optimized for performance and interoperability
- Better suited for larger datasets and production environments

## Framework Transformers

Enable seamless conversion between different data representations:

- `BaseTransformer`: Abstract base class defining the transformation interface
- `ComputeFrameworkTransformer`: Registry of available transformers
- `PandasPyarrowTransformer`: Bidirectional conversion between Pandas and PyArrow

The transformation process is automatic when data needs to move between frameworks:
1. System identifies source and target framework types
2. Looks up appropriate transformer in the registry
3. Calls transformation method to convert the data

## Integration with Feature Groups

Feature groups specify which compute frameworks they support through the `compute_framework_rule` method:

```python
@classmethod
def compute_framework_rule(cls):
    return {PandasDataframe}  # Support only Pandas
    # Or return True to support all available frameworks
```

Feature groups often follow a layered architecture:
```
AbstractFeatureGroup
  └── BaseFeatureGroup (e.g., ClusteringFeatureGroup)
        ├── PandasImplementation
        └── PyArrowImplementation
```

This pattern allows:
- Base class to define common interface and functionality
- Framework-specific classes to implement optimized calculations
- Seamless switching between different compute technologies

When feature groups depend on features from different compute frameworks, the system:
1. Calculates the source feature using its required framework
2. Automatically transforms the data to the target framework
3. Calculates the dependent feature using its framework

## Framework Selection Process

The system selects a compatible compute framework through three levels:

1. **Feature Definition**: `Feature("id", options={"compute_framework": "PyarrowTable"})`
2. **Feature Group Definition**: Limits which frameworks a feature group supports
3. **API Request**: Specifies which frameworks to use for the entire request

This flexibility allows precise control over which technologies are used for specific computations while maintaining consistent feature definitions.

## Testing

Compute frameworks are tested at multiple levels:

1. **Unit Tests**: Verify individual framework functionality
   - Data transformation capabilities (dict to table, arrays to columns)
   - Merge operations with different join types (inner, left, right, outer, append, union)
   - Column selection and naming conventions
   - All tests pass for both PandasDataframe and PyarrowTable implementations

2. **Integration Tests**: Verify interactions between frameworks
   - Automatic transformation between frameworks
   - Execution planning with multiple frameworks
   - Framework selection based on feature group requirements

## Changelog

### 2025-04-20: Added Multiple Result Columns Support
- Added `identify_naming_convention` method to ComputeFrameWork
- Updated implementations to support the new naming convention

### 2025-04-15: Enhanced Framework Transformers Documentation
- Added comprehensive documentation for framework transformers

### 2025-04-10: Improved Framework Selection Logic
- Enhanced the framework selection process with better validation

### 2025-04-05: Enhanced PyArrow Support
- Improved PyarrowTable implementation with better error handling

### 2025-04-03: Added PyArrow Aggregated Feature Group
- Implemented PyArrow version of the aggregated feature group

### 2025-03-30: Initial Implementation
- Created base ComputeFrameWork abstract class
- Implemented PandasDataframe and PyarrowTable compute frameworks
