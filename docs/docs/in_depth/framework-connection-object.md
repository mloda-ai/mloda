# Framework Connection Object

DuckDB, SQLite, Spark, and Iceberg each need a persistent connection object to run. Pandas, PyArrow, and Polars are stateless and never need one.

## Registering a connection

Register on `DataAccessCollection` in one of two shapes. A live object works only in the process that registered it, for SYNC and THREADING runs. A `ConnectionSpec(Framework, **params)` recipe is picklable: every process, including each MULTIPROCESSING worker, opens its own from it.

A MatchData feature group whose `match_data_access` checks connection entries with isinstance will not match a ConnectionSpec; it must accept both shapes.

```py
import duckdb
from mloda.user import DataAccessCollection

data_access_collection = DataAccessCollection(connections={duckdb.connect()})
```

```py
from mloda.user import ConnectionSpec, DataAccessCollection
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework

DataAccessCollection(connections={ConnectionSpec(DuckDBFramework, database="/data/warehouse.duckdb")})
```

```py
from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import SparkFramework

DataAccessCollection(connections={ConnectionSpec(SparkFramework, app_name="demo", master="local[2]")})
```

```py
from mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework import IcebergFramework

DataAccessCollection(connections={ConnectionSpec(IcebergFramework, name="lake", type="rest", uri="https://...")})
```

Params pass straight through: to `duckdb.connect`, `sqlite3.connect`, the `SparkSession` builder (`app_name`, `master`, `config`), and pyiceberg's `load_catalog`.

Spark's config merges over the framework's adaptive defaults, and an unknown Spark param key raises ValueError.

## Resolution rule

The runtime resolves one connection entry per transform-destination framework at setup. A framework binds it lazily, calling `ensure_connection()` right before it needs the handle: during a transform, a join, or materializing flight-server results. `open_connection(spec)` is the only per-framework hook, a classmethod that opens a new handle from the spec.

## Multiprocessing

A live connection never crosses a process boundary; it is dropped when a framework instance is pickled. A `ConnectionSpec` does cross: each worker, and the parent, opens its own connection from it. A handle opened from a spec is shared by every framework instance of that class in one process (worker or parent) until it exits. The caller keeps ownership of a live object it registered.

A MULTIPROCESSING-eligible step that resolved a live connection instead of a spec fails before any worker starts:

```text
DuckDBFramework resolved a live DuckDBPyConnection from DataAccessCollection, but
TransformFrameworkStep (uuid=...) can run in a spawned worker process, and a live connection
never crosses that boundary.
Resolution: register a ConnectionSpec(DuckDBFramework, ...) so each process opens its own
connection, or run without ParallelizationMode.MULTIPROCESSING.
```

DuckDB and SQLite declare themselves SYNC-only, so this never fires for them by default. Declare MULTIPROCESSING support on a subclass and pair it with a `ConnectionSpec` to run them in workers:

```py
from mloda.user import ParallelizationMode

class WorkerDuckDBFramework(DuckDBFramework):
    @classmethod
    def supported_parallelization_modes(cls) -> set[ParallelizationMode]:
        return {ParallelizationMode.SYNC, ParallelizationMode.THREADING, ParallelizationMode.MULTIPROCESSING}
```

## For framework authors

Override `set_framework_connection_object` to validate and bind a live object; raise on `None`, never self-construct a connection there:

```py
def set_framework_connection_object(self, framework_connection_object=None) -> None:
    if framework_connection_object is None:
        raise ValueError("A DuckDB connection object is required.")
    self.framework_connection_object = framework_connection_object
```

Override `open_connection(spec)` to build a new connection from `spec.params`:

```py
@classmethod
def open_connection(cls, spec: ConnectionSpec | None) -> Any | None:
    return None if spec is None else duckdb.connect(**spec.params)
```

Transformers still receive the resolved handle as `framework_connection_object`:

```py
@classmethod
def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Any | None = None) -> Any:
    """Transform data into this framework's native format, using the connection if needed."""
```

```py
if framework_connection_object is None:
    raise ValueError("A connection object is required for this transformation.")
return framework_connection_object.from_other_format(data)
```

## Changed behaviour

`SparkFramework.set_framework_connection_object(None)` now raises instead of building a local session. Register a `ConnectionSpec(SparkFramework)`, or rely on the default the framework opens when nothing is registered.

`IcebergFramework.set_framework_connection_object(None)` now raises instead of silently doing nothing.

`ComputeFramework.convert_flight_server_data_back` is an instance method. Call it on the framework instance, not the class.
