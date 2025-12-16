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
import mloda
from mloda.user import Feature
from mloda.provider import DataTypeMismatchError

try:
    result = mloda.run_all([Feature.str_of("numeric_column")])
except DataTypeMismatchError as e:
    print(f"Feature '{e.feature_name}': declared {e.declared.name}, got {e.actual.name}")
```
