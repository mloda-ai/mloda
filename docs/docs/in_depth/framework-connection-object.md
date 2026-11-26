# Framework Connection Object

## Overview

The Framework Connection Object is a key concept in mloda's compute framework system that enables stateful compute frameworks to maintain persistent connections and share resources across different operations. This is particularly important for frameworks like DuckDB, Spark, or database connections that require maintaining state between operations.

## Which Frameworks Need One

A connection object is an optional per-framework handle that stateful frameworks use to share one connection or session across steps. Each framework's `connection_requirement()` says how it gets one:

- **Required** (`ConnectionRequirement.REQUIRED`): DuckDB and SQLite need a connection in the `DataAccessCollection`; a step can also take the connection its parent relation carries.
- **Self-managed** (`SELF_MANAGED`): Spark uses a supplied `SparkSession`, or creates a local one (`getOrCreate` reuses any existing session) when none is supplied; Iceberg uses a supplied catalog when given (a table is also accepted).
- **None** (`NONE`): Pandas, PyArrow, Polars and PythonDictFramework hold their data in memory.

## How it Works

`ComputeFramework` holds the object in `framework_connection_object`, set through `set_framework_connection_object` and read through `get_framework_connection_object`. Frameworks that need one override the setter (see [Implementation Guidelines](#implementation-guidelines)). Framework transformers receive it as the `framework_connection_object` argument of `transform_other_fw_to_fw`; see [Framework Transformers](framework-transformers.md).

### Using DuckDB with mloda API

The API examples below are sketches, not executed: `feature1` and `feature2` stand for columns your own
tables or source groups provide, so a connection alone does not make the request resolve.

```py
from mloda.user import mloda
from mloda.user import DataAccessCollection
import duckdb

# Create DuckDB connection
connection = duckdb.connect()

# Set up data access
data_access_collection = DataAccessCollection(connections={connection})

# Run with DuckDB framework
result = mloda.run_all(
    ["feature1", "feature2"],
    compute_frameworks=["DuckDBFramework"],
    data_access_collection=data_access_collection
)
```

Connections live only in the `DataAccessCollection`. With one matching connection no selection is needed; with several, name each in a feature's options (see [Selecting a connection](named-data-access-handles.md#selecting-a-connection-connection_handle)).

On DuckDB 2.0, an `execute()` result on your connection that was not fetched to the end (even `fetchone()` on a one-row result) blocks reading any relation mloda returns on that connection: use `fetchall()` or a separate `connection.cursor()`.

### Using Spark with mloda API

```py
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
data_access_collection = DataAccessCollection(connections={spark})

# Run with Spark framework
result = mloda.run_all(
    ["feature1", "feature2"],
    compute_frameworks=["SparkFramework"],
    data_access_collection=data_access_collection
)
```

### DuckDB Transformer Example

The `DuckDBPyArrowTransformer` requires a connection object for PyArrow → DuckDB transformations. The shared `transform_other_fw_to_fw` lives in its base class `SqlBasePyArrowTransformer`, which validates the connection and delegates to the framework-specific hooks:

```py
class DuckDBPyArrowTransformer(SqlBasePyArrowTransformer):
    @classmethod
    def _convert_to_arrow(cls, data: Any) -> Any:
        return data.to_arrow_table()

    @classmethod
    def _convert_to_native(cls, data: Any, connection: Any) -> Any:
        return DuckdbRelation.from_arrow(connection, data)

    @classmethod
    def _validate_connection(cls, connection: Any) -> None:
        if not isinstance(connection, duckdb.DuckDBPyConnection):
            raise ValueError(f"Expected a DuckDB connection object, got {type(connection)}")
```

## Implementation Guidelines

### For Framework Developers

When implementing a new compute framework that requires a connection object:

1. **Override `set_framework_connection_object`**:

```py
def set_framework_connection_object(self, framework_connection_object: Any | None = None) -> None:
    if framework_connection_object is not None:
        if not isinstance(framework_connection_object, ExpectedConnectionType):
            raise ValueError(f"Expected connection type, got {type(framework_connection_object)}")
        self.framework_connection_object = framework_connection_object
```

2. **Pass connection objects** to merge engines and other components that need them.

3. **Declare the modes the connection can survive.** A live connection or session cannot be pickled into a spawned worker. Override `supported_parallelization_modes` to leave out `ParallelizationMode.MULTIPROCESSING`, keeping `THREADING` only if the handle is thread-safe (DuckDB and SQLite declare `{ParallelizationMode.SYNC}` alone). The framework's steps then stay in the parent process under a MULTIPROCESSING run and take their connection exactly as in a SYNC run. Skipping this raises when a MULTIPROCESSING run starts, before any worker starts.

### For Transformer Developers

Accept `framework_connection_object` in `transform_other_fw_to_fw` and raise `ValueError` when the target framework needs one and it is `None`, as `SqlBasePyArrowTransformer` does.

## Troubleshooting

### Common Issues

1. **Missing Connection Object**:
   ```
   ValueError: A connection object is required for this transformation.
   ```
   **Solution**: Put the connection in the `DataAccessCollection`. mloda binds it to every step that needs it, roots included, so nothing is repeated per feature. With several connections, name one per feature with the `connection_handle` option (see [Selecting a connection](named-data-access-handles.md#selecting-a-connection-connection_handle)).

2. **Connection object in a feature option**: a connection object under any feature option (for example the group's class name) raises at planning, naming the key.
   **Solution**: Move it to the `DataAccessCollection` and select it with `connection_handle`.

3. **Two handles in one step**: parents on different handles never share a DuckDB or SQLite consumer, and a consumer pinned to that framework fails at planning naming both handles.
   **Solution**: Give the consumer one `connection_handle`, or run the branches as separate features.
