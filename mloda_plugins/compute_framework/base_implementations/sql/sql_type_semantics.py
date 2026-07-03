"""Column-semantics helper for arrow-typed SQL-family frameworks (epic #518, Phase 1).

Single source of truth for deriving :class:`ColumnSemantics` from a pyarrow
``DataType``. Backs both duckdb and sqlite; sqlite stores datetimes as ISO TEXT,
so it may pass ``is_string_storage=True``. When a ``value_sample`` is provided for a
string-typed column, its ISO-8601 values are classified as temporal via
``iso8601_string_semantics``.
"""

from typing import TYPE_CHECKING, Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda.core.abstract_plugins.components.contract.value_inspection import iso8601_string_semantics
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import is_ordered_arrow_type

if TYPE_CHECKING:
    import pyarrow as pa


def is_string_like_arrow_type(arrow_type: "pa.DataType") -> bool:
    """True for string, large_string and (where available) string_view arrow types.

    ``pa.types.is_string`` is False for large_string/string_view, so value-inspection
    would otherwise skip those column types. ``is_string_view`` is guarded because older
    pyarrow releases may lack it.
    """
    import pyarrow as pa

    is_string_view = getattr(pa.types, "is_string_view", None)
    return bool(
        pa.types.is_string(arrow_type)
        or pa.types.is_large_string(arrow_type)
        or (is_string_view is not None and is_string_view(arrow_type))
    )


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

    if is_string_like_arrow_type(arrow_type) and value_sample is not None:
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
