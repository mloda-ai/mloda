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

## Declaring a Feature Group's Output Type

The section above is the *data user* declaring a type on a `Feature`. A *feature group* can also
declare the type it produces, via `return_data_type_rule` — the provider-side counterpart. mloda
reconciles the two at planning time.

```python
from mloda.provider import FeatureGroup
from mloda.user import DataType


class UserCount(FeatureGroup):
    @classmethod
    def return_data_type_rule(cls, feature) -> DataType | None:
        return DataType.INT64
```

The rule returns either:

- a concrete `DataType` — the group always emits this type;
- `None` (the default) — no fixed type / polymorphic.

Reconciliation with the user's declared type, at planning:

- rule returns a concrete `DataType` that **differs** from the user's declared type → planning raises a mismatch (loud, early);
- rule returns `None` → the user's declared type (if any) stands;
- rule returns a concrete type and the user declared none → the rule's type is used.

A feature that ends up with `data_type=None` is then skipped by the compute-time validator (below);
a concrete declared type flows through to runtime checking against the actual column dtype.

### Errors are not swallowed

`return_data_type_rule` runs *after* the feature group has been selected for the feature, so a rule
that raises is a failure of a committed component, not a non-applicable candidate. mloda does **not**
catch it — the exception propagates and fails planning. If your rule does work that can legitimately
fail to determine a type, return `None` for that case rather than letting it raise.

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

## Implementing the Hook in a Custom Compute Framework

Enforcement is driven by a single override point on `ComputeFramework`:

```python
from typing import Any, Optional

def _extract_column_data_type(self, data: Any, column_name: str) -> Optional[DataType]:
    ...
```

When a typed feature is produced, mloda calls this hook for the column and compares
the returned `DataType` against the feature's declared type, raising
`DataTypeMismatchError` on an incompatible pairing (per the lenient/strict rules above).

The contract:

- **Return a `DataType`** to have the column validated. Map the framework's native type
  (e.g. `pa.DataType`, `pd.api.types`, `polars.DataType`, a pyspark/pyiceberg type) directly
  to the unified `DataType` enum: no string round-trip.
- **Return `None` to skip the column.** The validator treats `None` as a graceful no-op and
  performs no check for that column.

!!! warning "The base implementation is a silent no-op"
    `ComputeFramework._extract_column_data_type` returns `None` by default. A custom framework
    that does not override it ships with **type enforcement effectively disabled**: declared
    feature types are silently never checked. Override the hook to opt your framework in.

A custom framework imports `DataType` from the public API:

```python
from mloda.user import DataType
from mloda.provider import ComputeFramework


class MyFramework(ComputeFramework):
    def _extract_column_data_type(self, data: Any, column_name: str) -> Optional[DataType]:
        native = data.schema.field(column_name).type  # framework-specific access
        if native == ...:
            return DataType.INT64
        # Unknown / unmappable type: skip rather than guess.
        return None
```

See `PythonDictFramework._extract_column_data_type` for a complete reference override, and
the table below for how each bundled framework handles precisions its backend cannot
distinguish (it returns the widest type in the family rather than `None`, so lenient-mode
validation still applies).

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
