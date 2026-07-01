"""Column-semantics introspector for pyspark DataFrames (epic #518, Phase 1).

Spark exposes no sub-type unit, so ``unit`` is always None. Timezone awareness
maps to the Spark type: ``TimestampType`` => aware, ``TimestampNTZType`` => naive.
"""

from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics

try:
    from pyspark.sql.types import (
        ByteType,
        ShortType,
        IntegerType,
        LongType,
        FloatType,
        DoubleType,
        DecimalType,
        TimestampType,
        TimestampNTZType,
        DateType,
    )

    _NUMERIC_TYPES: tuple[type, ...] = (
        ByteType,
        ShortType,
        IntegerType,
        LongType,
        FloatType,
        DoubleType,
        DecimalType,
    )
    _TEMPORAL_TYPES: tuple[type, ...] = (TimestampType, TimestampNTZType, DateType)
except ImportError:
    TimestampType = None
    _NUMERIC_TYPES = ()
    _TEMPORAL_TYPES = ()


def column_semantics(sdf: Any, column: str) -> ColumnSemantics:
    """Return the observed semantics of ``column`` in a pyspark DataFrame."""
    data_type = sdf.schema[column].dataType

    is_numeric = isinstance(data_type, _NUMERIC_TYPES)
    is_temporal = isinstance(data_type, _TEMPORAL_TYPES)
    is_ordered = is_numeric or is_temporal
    is_tz_aware = TimestampType is not None and isinstance(data_type, TimestampType)

    return ColumnSemantics(
        is_ordered=is_ordered,
        is_temporal=is_temporal,
        is_numeric=is_numeric,
        unit=None,
        is_tz_aware=is_tz_aware,
    )
