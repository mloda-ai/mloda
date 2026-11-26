# Framework Transformers

## Overview

Framework transformers are a key component of mloda's compute framework system, enabling seamless conversion between different data representations. They allow feature groups to work with multiple compute frameworks by providing transformations between framework representations, one-way or two-way.

## Core Components

`BaseTransformer` is the abstract interface for converting data between two compute frameworks. `ComputeFrameworkTransformer` keeps the registry of transformers and finds the one for a framework pair.
### Built-in transformers

| Transformer | Frameworks | Direction | Connection |
| --- | --- | --- | --- |
| `PandasPyArrowTransformer` | Pandas DataFrame, PyArrow Table | two-way | no |
| `PolarsPyArrowTransformer` | Polars DataFrame, PyArrow Table | two-way | no |
| `PolarsLazyPyArrowTransformer` | Polars LazyFrame, PyArrow Table | two-way | no |
| `PythonDictPyArrowTransformer` | columnar Python dict (`dict[str, list]`), PyArrow Table | two-way | no |
| `DuckDBPyArrowTransformer` | DuckDB Relation, PyArrow Table | two-way | required for PyArrow to DuckDB |
| `SqlitePyArrowTransformer` | SQLite relation, PyArrow Table | two-way | required for PyArrow to SQLite |
| `SparkPyArrowTransformer` | Spark DataFrame, PyArrow Table | two-way | optional `SparkSession` (the framework supplies or creates one); without one the transformer uses the active session or raises |
| `IcebergPyArrowTransformer` | Iceberg table, PyArrow Table | two-way | Iceberg to PyArrow; PyArrow to Iceberg passes the table through (catalog unused) |
| `FileSourcePyArrowTransformer`, `FileSourceDictTransformer` | file source, PyArrow Table or columnar Python dict | one-way | no |

DuckDB and SQLite share `SqlBasePyArrowTransformer`, which validates the connection object (see [Framework Connection Object](framework-connection-object.md)) because their relations belong to one database connection.

## How Framework Transformers Work

### Registration Process

1. On first use in a process, `ComputeFrameworkTransformer` auto-loads only `*transformer*` files from the `compute_framework` group (heavy backends once their library is imported, refreshed on a lookup miss), then collects every `BaseTransformer` subclass
2. Each transformer is registered in a mapping from framework pairs to transformer classes
3. Each direction is registered only if the transformer implements that direction's hook: two-way transformers register both edges, one-way transformers register only the edge whose hook they implement

### Transformation Process

The system looks up the transformer for the source and target framework types and calls the hook for that direction.

## Creating Custom Transformers

To create a custom transformer for a new pair of frameworks:

1. Subclass `BaseTransformer`
2. Implement the four required methods:
   - `framework()` - Return the primary framework type
   - `other_framework()` - Return the secondary framework type
   - `import_fw()` - Import the primary framework module
   - `import_other_fw()` - Import the secondary framework module
3. Implement the direction hooks the transformer supports:
   - `transform_fw_to_other_fw()` - Transform from primary to secondary; registers the forward edge
   - `transform_other_fw_to_fw()` - Transform from secondary to primary; registers the reverse edge

**One-way transformers**: implement only the direction you support. Either hook may be omitted, and only an implemented hook's edge is registered (omit both and nothing is registered, with a warning).

- Never override an unsupported hook with `raise NotImplementedError`: registration keys off the override itself, not its body, so the dead edge is registered, and a chain routes through it and crashes instead of reporting "no chain exists".
- Override detection resolves through the MRO, so a subclass of a two-way transformer (for example `DuckDBPyArrowTransformer`) inherits both overrides and cannot drop a direction by omission.
- A transformer into a framework whose `_positional_append_needs_pinned_frame()` is `True` must build from materialized data, since mloda treats its output as pinned.
- `FileSourcePyArrowTransformer` and `FileSourceDictTransformer` are the reference one-way implementations; both omit `transform_other_fw_to_fw()`. They read csv, parquet, json, feather and orc.

Example of a two-way transformer:

```python
from typing import Any

from mloda.provider import BaseTransformer


class CustomFramework: ...  # stand-ins for your two framework types


class OtherFramework: ...


class CustomTransformer(BaseTransformer):
    @classmethod
    def framework(cls) -> Any:
        return CustomFramework

    @classmethod
    def other_framework(cls) -> Any:
        return OtherFramework

    @classmethod
    def import_fw(cls) -> None:
        import custom_framework

    @classmethod
    def import_other_fw(cls) -> None:
        import other_framework

    @classmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        # Convert from CustomFramework to OtherFramework
        return other_framework.from_custom(data)

    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Any | None = None) -> Any:
        # Convert from OtherFramework to CustomFramework
        return custom_framework.from_other(data)
```

## Integration with ComputeFramework

The `ComputeFramework` class uses the transformer system to convert data between different frameworks when needed. This happens in several scenarios:

1. When data needs to be transformed to match the expected framework of a feature group
2. When data needs to be uploaded to a flight server (always converted to PyArrow Table)
3. When data is retrieved from a flight server and needs to be converted back
