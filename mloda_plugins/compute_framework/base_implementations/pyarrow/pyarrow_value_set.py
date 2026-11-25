from datetime import datetime
from typing import Any

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]

from mloda_plugins.compute_framework.base_implementations.pyarrow.arrow_build import arrow_array, is_gap


def value_set(column: Any, values: Any) -> Any:
    """Build a pyarrow array of values, typed to match column when every value is null."""
    values = list(values)
    non_null = [v for v in values if not is_gap(v)]
    column_type = column.type
    target_type = column_type.value_type if pa.types.is_dictionary(column_type) else column_type
    if non_null:
        aware_column = pa.types.is_timestamp(target_type) and target_type.tz is not None
        if aware_column and all(isinstance(v, datetime) and v.tzinfo is not None for v in non_null):
            return arrow_array(values, type=target_type)
        return arrow_array(values)
    return pa.array(values, type=target_type)
