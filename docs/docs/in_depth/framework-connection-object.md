# Framework Connection Object

## Overview

The Framework Connection Object is a key concept in mloda's compute framework system that enables stateful compute frameworks to maintain persistent connections and share resources across different operations. This is particularly important for frameworks like DuckDB, Spark, or database connections that require maintaining state between operations.

## What is a Framework Connection Object?

A Framework Connection Object is an optional parameter that some compute frameworks use to:

- **Maintain persistent connections** to databases or compute engines
- **Share state** between different operations within the same compute session
- **Ensure data consistency** across merge operations and transformations
- **Optimize performance** by reusing connections and avoiding repeated setup costs

## When is it Required?

### Stateful Frameworks

Some compute frameworks require a connection object to function properly:

- **DuckDB**: Requires a `DuckDBPyConnection` to maintain database state
- **Spark**: Requires a `SparkSession` to manage cluster resources and distributed computing
- **Iceberg**: Requires a catalog connection for table operations
- **Database frameworks**: Need database connections for query execution

### Stateless Frameworks

Other frameworks don't require connection objects:

- **Pandas**: Operations are performed on in-memory DataFrames
- **PyArrow**: Tables are self-contained data structures
- **Polars**: DataFrames and LazyFrames are independent objects
- **PythonDictFramework**: Simple Python data structures

## How it Works

### In ComputeFramework Base Class

The base `ComputeFramework` class provides methods to manage connection objects:

``` python
class ComputeFramework:
    def __init__(self) -> None:
        # Connection object for frameworks that need persistent connections
        self.framework_connection_object: Optional[Any] = None
    
    def set_framework_connection_object(self, framework_connection_object: Optional[Any] = None) -> None:
        """
        Some compute frameworks (e.g., DuckDB, Spark) require sharing their connection
        with merge engines to ensure data consistency. Override this method in
        subclasses that need to provide a connection object.
        """
        self.framework_connection_object = None
    
    def get_framework_connection_object(self) -> Any:
        """This method retrieves the connection object set by `set_framework_connection_object`."""
        return self.framework_connection_object
```

### In Framework Transformers

Framework transformers receive the connection object as a parameter:

``` python
@classmethod
def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
    """
    Transform data from the secondary framework to the primary framework.
    
    Args:
        data: Data in the secondary framework format
        framework_connection_object: Optional connection object for stateful frameworks
    
    Returns:
        Any: Transformed data in the primary framework format
    """
```

### Using DuckDB with mloda API

``` python
from mloda.user import mloda
from mloda.user import DataAccessCollection
import duckdb

# Create DuckDB connection
connection = duckdb.connect()

# Set up data access
data_access_collection = DataAccessCollection(initialized_connection_object={connection})

# Run with DuckDB framework
result = mloda.run_all(
    ["feature1", "feature2"],
    compute_frameworks=["DuckDBFramework"],
    data_access_collection=data_access_collection
)
```

### Using Spark with mloda API

``` python
from mloda.user import mloda
from mloda.user import DataAccessCollection
from pyspark.sql import SparkSession

# Create SparkSession
spark = SparkSession.builder \
    .appName("MLoda-Spark-Application") \
    .master("local[*]") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Set up data access
data_access_collection = DataAccessCollection(initialized_connection_objects={spark})

# Run with Spark framework
result = mloda.run_all(
    ["feature1", "feature2"],
    compute_frameworks=["SparkFramework"],
    data_access_collection=data_access_collection
)
```

### DuckDB Transformer Example

The `DuckDBPyarrowTransformer` requires a connection object for PyArrow → DuckDB transformations:

``` python
class DuckDBPyarrowTransformer(BaseTransformer):
    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
        """Transform a PyArrow Table to a DuckDB relation."""
        if framework_connection_object is None:
            raise ValueError("A DuckDB connection object is required for this transformation.")
        
        if not isinstance(framework_connection_object, duckdb.DuckDBPyConnection):
            raise ValueError(f"Expected a DuckDB connection object, got {type(framework_connection_object)}")
        
        return framework_connection_object.from_arrow(data)
```

## Implementation Guidelines

### For Framework Developers

When implementing a new compute framework that requires a connection object:

1. **Override `set_framework_connection_object`**:
``` python
def set_framework_connection_object(self, framework_connection_object: Optional[Any] = None) -> None:
    if framework_connection_object is not None:
        if not isinstance(framework_connection_object, ExpectedConnectionType):
            raise ValueError(f"Expected connection type, got {type(framework_connection_object)}")
        self.framework_connection_object = framework_connection_object
```

2. **Pass connection objects** to merge engines and other components that need them.

### For Transformer Developers

When implementing transformers for stateful frameworks:

1. **Accept the connection object parameter**:
``` python
@classmethod
def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
    # Implementation here
```

2. **Validate the connection object** when required:
``` python
if framework_connection_object is None:
    raise ValueError("Connection object is required for this transformation.")
```

3. **Use the connection object** for the transformation:
``` python
return framework_connection_object.from_other_format(data)
```
## Troubleshooting

### Common Issues

1. **Missing Connection Object**:
   ```
   ValueError: A DuckDB connection object is required for this transformation.
   ```
   **Solution**: Ensure you're providing a connection object when using stateful frameworks. Currently, parent features are not receiving the connection details automatic. Thus, you might need to give them a second time to root features!

## Summary

The Framework Connection Object is a powerful feature that enables mloda to work with stateful compute frameworks while maintaining the flexibility to work with stateless frameworks as well. By understanding when and how to use connection objects, you can:

- **Integrate database-backed compute frameworks** like DuckDB
- **Maintain state consistency** across operations
- **Optimize performance** through connection reuse
- **Build robust applications** with proper error handling

This design allows mloda to support both simple, stateless frameworks (like Pandas) and complex, stateful frameworks (like DuckDB) within the same unified interface.
