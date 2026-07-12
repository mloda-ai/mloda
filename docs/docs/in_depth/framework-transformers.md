# Framework Transformers

## Overview

Framework transformers are a key component of mloda's compute framework system, enabling seamless conversion between different data representations. They allow feature groups to work with multiple compute frameworks by providing transformations between framework representations, one-way or two-way.

## Core Components

### BaseTransformer

The `BaseTransformer` is an abstract base class that defines the interface for transforming data between different compute frameworks. It provides the foundation for all framework transformers in the system.

```python
class BaseTransformer:
    """
    Abstract base class for transforming data between different compute frameworks.
    """
```

Key features:
- Defines a consistent interface for all transformers
- Handles the logic for determining transformation direction
- Provides methods for checking if required frameworks are available
- Manages the actual transformation process

### ComputeFrameworkTransformer

The `ComputeFrameworkTransformer` manages the registry of available transformers and provides methods to find the appropriate transformer for a given pair of frameworks.

```python
class ComputeFrameworkTransformer:
    """
    Manages transformations between different compute frameworks.
    """
```

Key features:
- Maintains a registry of available transformers
- Auto-loads transformer files from the `compute_framework` group on first use if no `BaseTransformer` subclasses are registered yet
- Provides a lookup mechanism to find the appropriate transformer for any framework pair

To suppress auto-loading:
```python
from mloda.core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader
PluginLoader.disable_auto_load("compute_framework")
```

### PandasPyArrowTransformer

The `PandasPyArrowTransformer` is a concrete implementation of `BaseTransformer` that handles conversions between Pandas DataFrames and PyArrow Tables.

``` python
class PandasPyArrowTransformer(BaseTransformer):
    """
    Transformer for converting between Pandas DataFrame and PyArrow Table.
    """
```

Key features:
- Converts Pandas DataFrames to PyArrow Tables and vice versa
- Handles metadata properly during transformations
- Ensures clean conversion by removing framework-specific metadata

### DuckDBPyArrowTransformer

The `DuckDBPyArrowTransformer` is a concrete implementation of `BaseTransformer`, via `SqlBasePyArrowTransformer`, that handles conversions between DuckDB Relations and PyArrow Tables.

``` python
class DuckDBPyArrowTransformer(SqlBasePyArrowTransformer):
    """
    Transformer for converting between DuckDB relations and PyArrow Table.
    """
```

Key features:
- Converts DuckDB Relations to PyArrow Tables and vice versa
- Leverages DuckDB's native PyArrow integration for efficient zero-copy operations
- Requires a DuckDB connection object for PyArrow → DuckDB transformations
- Uses DuckDB's `to_arrow_table()` and `from_arrow()` methods for optimal performance

**Important**: The DuckDB transformer requires a connection object when transforming from PyArrow to DuckDB. This is because DuckDB relations must be associated with a specific database connection.

### SparkPyArrowTransformer

The `SparkPyArrowTransformer` is a concrete implementation of `BaseTransformer` that handles conversions between Spark DataFrames and PyArrow Tables.

``` python
class SparkPyArrowTransformer(BaseTransformer):
    """
    Transformer for converting between Spark DataFrame and PyArrow Table.
    """
```

Key features:
- Converts Spark DataFrames to PyArrow Tables and vice versa
- Leverages Spark's native PyArrow integration for efficient conversions
- Can use a provided SparkSession or auto-create one if needed
- Handles distributed data processing through Spark's execution engine
- Requires PySpark installation and Java 8+ environment

**Note**: The Spark transformer can optionally use a SparkSession connection object. If none is provided, it will attempt to get the active session or create a new local session.

## How Framework Transformers Work

### Registration Process

1. During initialization, `ComputeFrameworkTransformer` checks for existing `BaseTransformer` subclasses
2. If none are found, it auto-loads only `*transformer*` files from the `compute_framework` group
3. Each transformer is registered in a mapping from framework pairs to transformer classes
4. Each direction is registered only if the transformer implements that direction's hook: two-way transformers register both edges, one-way transformers register only the edge whose hook they implement

### Transformation Process

When data needs to be transformed from one framework to another:

1. The system identifies the source and target framework types
2. It looks up the appropriate transformer in the registry
3. It determines the direction of transformation (primary to secondary or vice versa)
4. It calls the appropriate transformation method on the transformer
5. The transformer converts the data to the target framework format

### Example Flow

```
Pandas DataFrame → PandasPyArrowTransformer → PyArrow Table
```

Or in the reverse direction:

```
PyArrow Table → PandasPyArrowTransformer → Pandas DataFrame
```

For DuckDB transformations:

```
DuckDB Relation → DuckDBPyArrowTransformer → PyArrow Table
```

```
PyArrow Table → DuckDBPyArrowTransformer → DuckDB Relation (requires connection)
```

For Spark transformations:

```
Spark DataFrame → SparkPyArrowTransformer → PyArrow Table
```

```
PyArrow Table → SparkPyArrowTransformer → Spark DataFrame (optional SparkSession)
```

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
- `FileSourcePyArrowTransformer` and `FileSourceDictTransformer` are the reference one-way implementations; both omit `transform_other_fw_to_fw()`.

Example of a two-way transformer:

```python
from typing import Any, Optional

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
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
        # Convert from OtherFramework to CustomFramework
        return custom_framework.from_other(data)
```

## Integration with ComputeFramework

The `ComputeFramework` class uses the transformer system to convert data between different frameworks when needed. This happens in several scenarios:

1. When data needs to be transformed to match the expected framework of a feature group
2. When data needs to be uploaded to a flight server (always converted to PyArrow Table)
3. When data is retrieved from a flight server and needs to be converted back

The transformation is handled automatically by the `apply_compute_framework_transformer` method:

``` python
def apply_compute_framework_transformer(self, data: Any) -> Any:
    _from_fw = type(data)
    _to_fw = self.expected_data_framework()
    transformer_cls = self.transformer.transformer_map.get((_from_fw, _to_fw), None)
    if transformer_cls is not None:
        return transformer_cls.transform(_from_fw, _to_fw, data)
    return None
```

## Benefits of Framework Transformers

- **Decoupling**: Feature groups can be defined independently of specific compute frameworks
- **Flexibility**: The system can work with multiple data representations
- **Extensibility**: New frameworks can be added by implementing new transformers
- **Transparency**: Transformations happen automatically without user intervention

## Conclusion

Framework transformers are a critical part of mloda's flexibility, allowing the system to work with multiple compute frameworks and data representations. By providing a clean interface for data conversion, they enable feature groups to be defined once and used with different technologies, supporting use cases like online/offline computation, testing, and migrations between environments.
