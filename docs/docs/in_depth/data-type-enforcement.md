# Data Type Enforcement

mloda supports optional data type declarations on Features, enabling runtime validation that computed data matches declared types.

## Declaring Feature Types

Use typed constructors to declare the expected data type:

```python
from mloda.user import Feature

# Typed features - will be validated at runtime
feature_int = Feature.int32_of("user_count")
feature_double = Feature.double_of("price")
feature_str = Feature.str_of("name")

# Untyped feature - no validation
feature_any = Feature.not_typed("legacy_column")
```

Available typed constructors:
- `int32_of()`, `int64_of()` - Integer types
- `float_of()`, `double_of()` - Floating point types
- `str_of()` - String type
- `boolean_of()` - Boolean type
- `date_of()`, `timestamp_millis_of()`, `timestamp_micros_of()` - Date/time types
- `decimal_of()`, `binary_of()` - Other types

## Validation Behavior

!!! note "Changed in 0.7.0"
    Type validation now runs on **all** bundled compute frameworks through a uniform
    `_extract_column_data_type` hook. In 0.6.x, type extraction relied on a single
    PyArrow-schema path, so typed features running on frameworks that did not expose that
    schema were silently left unvalidated. After upgrading, those features may raise a
    `DataTypeMismatchError` for the first time where validation previously did nothing.

### Default (Lenient) Mode

By default, validation allows compatible type conversions within categories:

| Declared Type | Compatible Actual Types |
|---------------|------------------------|
| INT64 | INT32, INT64 |
| DOUBLE | INT32, INT64, FLOAT, DOUBLE |
| TIMESTAMP_MICROS | TIMESTAMP_MILLIS, TIMESTAMP_MICROS |

Cross-category mismatches (e.g., STRING declared but INT64 returned) raise `DataTypeMismatchError`.

### Strict Mode

Enable strict validation per-feature via options:

```python
feature = Feature.int32_of(
    "exact_count",
    options={"strict_type_enforcement": True}
)
```

In strict mode, only exact type matches or standard widening conversions are allowed.

## Per-Framework Precision Support

Not every backend's native type system can distinguish every precision mloda declares. The table below shows which precisions each bundled framework can extract from data. For the rest, the framework's `_extract_column_data_type` returns the widest type in the family (still correct under lenient mode), and strict-mode tests for the affected precision are skipped with a clear reason.

| Framework | INT32 / INT64 | FLOAT / DOUBLE | TIMESTAMP_MILLIS / MICROS |
|---|---|---|---|
| Pandas | yes | yes | yes |
| Polars (eager / lazy) | yes | yes | yes |
| PyArrow | yes | yes | yes |
| DuckDB | yes | yes | yes |
| Spark | yes | yes | no (only `TimestampType` exists) |
| Iceberg | yes | yes | no (only `TimestampType` exists) |
| SQLite | no (INTEGER affinity) | no (REAL affinity) | no (stored as TEXT) |
| PythonDict | no (`type.__name__` is "int") | no (Python float is 64-bit) | no (`datetime.datetime` is microsecond) |

## Execution Plan Grouping

Features with different explicit data types are separated into different execution groups at plan time. This allows type-specific processing paths.

Untyped features (`data_type=None`) are "lenient" and can be grouped with any typed features, preserving compatibility with index columns and legacy code.

```python
# These will be in DIFFERENT execution groups
Feature.int32_of("amount")
Feature.int64_of("amount")

# This can join ANY group (lenient)
Feature.not_typed("id")
```

## Database Reader Type Awareness

When reading from databases (e.g., SQLite), declared types are used to build the PyArrow schema:

```python
# Declared type is used for schema, not inferred from data
feature = Feature.int64_of(
    "user_id",
    options={"sqlite": "/path/to/db.sqlite"}
)
```

## Error Handling

Type mismatches raise `DataTypeMismatchError`:

```python title="Error handling example"
from mloda.user import mloda
from mloda.user import Feature
from mloda.provider import DataTypeMismatchError

try:
    result = mloda.run_all([Feature.str_of("numeric_column")])
except DataTypeMismatchError as e:
    print(f"Feature '{e.feature_name}': declared {e.declared.name}, got {e.actual.name}")
```
