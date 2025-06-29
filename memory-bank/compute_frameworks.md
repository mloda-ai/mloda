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

### PythonDict
- Based on native Python data structures (List[Dict[str, Any]])
- Dependency-free implementation using only Python standard library
- Simple and lightweight for basic data operations
- Ideal for environments with minimal dependencies or educational purposes
- Includes bidirectional transformation with PyArrow through PythonDictPyarrowTransformer

### PolarsDataframe
- Based on Polars DataFrame, a fast DataFrame library implemented in Rust
- High-performance data processing with lazy evaluation capabilities
- Memory-efficient columnar operations with excellent performance characteristics
- Optimized for large datasets and complex analytical workloads
- Includes bidirectional transformation with PyArrow through PolarsTransformer
- Supports all standard operations: filtering, merging, transformations

### DuckDBFramework
- Based on DuckDB Relations for SQL-based analytical operations
- Provides SQL interface for complex analytical queries and transformations
- Optimized for OLAP (Online Analytical Processing) workloads
- Requires framework connection object for stateful database connections
- Ideal for analytical workloads, SQL-based transformations, and data warehousing
- Does not support mloda framework inherent multiprocessing (DuckDB multiprocessing still works)

### IcebergFramework
- Based on Apache Iceberg Tables for data lake management
- Supports schema evolution, time travel, and versioned datasets
- Optimized for large-scale analytics and data lake scenarios
- Requires catalog connection object for table operations
- Uses PyArrow as interchange format for compatibility with other mloda frameworks
- Includes bidirectional transformation with PyArrow through IcebergPyarrowTransformer
- Supports filtering operations through IcebergFilterEngine
- Ideal for data lake scenarios with evolving schemas and large-scale analytics

### SparkFramework
- Based on Apache Spark DataFrames for distributed data processing
- Leverages Spark's distributed computing capabilities for large-scale data operations
- Requires SparkSession as framework connection object (auto-creates local session if not provided)
- Optimized for big data processing, distributed computing, and scalable analytics
- Includes bidirectional transformation with PyArrow through SparkPyarrowTransformer
- Supports all standard operations: filtering (SparkFilterEngine), merging (SparkMergeEngine), transformations
- Requires PySpark installation and Java 8+ environment with JAVA_HOME configured
- Ideal for large datasets, distributed processing, and production big data workflows
- Note: Does not support mloda framework inherent multiprocessing (Spark's own distributed processing is used)

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

### 2025-06-29: Implemented Spark Compute Framework
- Added complete Spark framework implementation using Apache Spark DataFrames


### 2025-06-11: Implemented Iceberg Compute Framework
- Added Iceberg framework implementation using Apache Iceberg Tables

### 2025-06-07: Implemented Polars Compute Framework
- Added complete Polars framework implementation using Polars DataFrame

### 2025-06-03: Implemented PythonDict Compute Framework
- Added complete PythonDict framework implementation using List[Dict[str, Any]] data structure

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
