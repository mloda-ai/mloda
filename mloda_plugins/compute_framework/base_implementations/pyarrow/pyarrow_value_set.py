from typing import Any

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]


def value_set(column: Any, values: Any) -> Any:
    """Build a pyarrow array of values, typed to match column when every value is null."""
    values = list(values)
    non_null = [v for v in values if v is not None]
    if non_null:
        return pa.array(values)
    column_type = column.type
    target_type = column_type.value_type if pa.types.is_dictionary(column_type) else column_type
    return pa.array(values, type=target_type)
