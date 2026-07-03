"""Column-semantics helper for arrow-typed SQL-family frameworks (epic #518, Phase 1).

Single source of truth for deriving :class:`ColumnSemantics` from a pyarrow
``DataType``. Backs both duckdb and sqlite; sqlite stores datetimes as ISO TEXT,
so it may pass ``is_string_storage=True``. Value-inspection of string storage is
deferred to a later phase, so the flag does not trigger any value scan yet.
"""

from typing import TYPE_CHECKING, Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda.core.abstract_plugins.components.contract.value_inspection import iso8601_string_semantics
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import is_ordered_arrow_type

if TYPE_CHECKING:
    import pyarrow as pa


def column_semantics_from_arrow(
    arrow_type: "pa.DataType",
    is_string_storage: bool = False,
    value_sample: list[Any] | None = None,
) -> ColumnSemantics:
    """Derive :class:`ColumnSemantics` from a pyarrow ``DataType``.

    ``is_string_storage`` is accepted so string-backed backends (e.g. sqlite) can
    pass it. When the arrow type is a string type and ``value_sample`` is provided,
    ISO-8601 datetime/date strings are classified as temporal via value-inspection.
    """
    import pyarrow as pa

    if pa.types.is_string(arrow_type) and value_sample is not None:
        inspected = iso8601_string_semantics(value_sample)
        if inspected is not None:
            return inspected

    is_numeric = bool(
        pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type) or pa.types.is_decimal(arrow_type)
    )
    is_temporal = bool(pa.types.is_temporal(arrow_type))
    is_ordered = is_ordered_arrow_type(arrow_type)

    unit: str | None = None
    is_tz_aware = False
    if pa.types.is_timestamp(arrow_type):
        unit = arrow_type.unit
        is_tz_aware = arrow_type.tz is not None

    return ColumnSemantics(
        is_ordered=is_ordered,
        is_temporal=is_temporal,
        is_numeric=is_numeric,
        unit=unit,
        is_tz_aware=is_tz_aware,
    )
