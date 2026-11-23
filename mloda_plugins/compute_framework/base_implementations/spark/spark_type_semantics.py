"""Column-semantics introspector for pyspark DataFrames (epic #518, Phase 1).

Spark exposes no sub-type unit, so ``unit`` is always None. A ``TimestampType`` column is
aware only when a transfer recorded its zone in the field metadata; otherwise it stays
naive, as Spark's session-tz instants would raise false positives on naive filter bounds.
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


TIME_ZONE_METADATA_KEY = "mloda.time_zone"


def time_zone(sdf: Any, column: str) -> str | None:
    """Zone recorded in the field metadata of ``column``, else None."""
    zone = sdf.schema[column].metadata.get(TIME_ZONE_METADATA_KEY)
    return None if zone is None else str(zone)


def is_float_column(sdf: Any, column: str) -> bool:
    """Return True if column is FloatType or DoubleType (the Spark types that can hold NaN)."""
    return isinstance(sdf.schema[column].dataType, (FloatType, DoubleType))


def column_semantics(sdf: Any, column: str) -> ColumnSemantics:
    """Return the observed semantics of ``column`` in a pyspark DataFrame."""
    data_type = sdf.schema[column].dataType

    is_numeric = isinstance(data_type, _NUMERIC_TYPES)
    is_temporal = isinstance(data_type, _TEMPORAL_TYPES)
    is_ordered = is_numeric or is_temporal
    tz = time_zone(sdf, column) if isinstance(data_type, TimestampType) else None

    return ColumnSemantics(
        is_ordered=is_ordered,
        is_temporal=is_temporal,
        is_numeric=is_numeric,
        unit=None,
        tz=tz,
    )
