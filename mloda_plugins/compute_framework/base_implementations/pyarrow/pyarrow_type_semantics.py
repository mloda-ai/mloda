"""Column-semantics introspector for pyarrow tables (epic #518, Phase 1).

Delegates to the shared arrow helper so arrow-type semantics have a single
source of truth.
"""

from typing import TYPE_CHECKING

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda_plugins.compute_framework.base_implementations.sql.sql_type_semantics import column_semantics_from_arrow

# pa appears only in the signature below, so it never has to exist at runtime.
if TYPE_CHECKING:
    import pyarrow as pa


def column_semantics(table: "pa.Table", column: str) -> ColumnSemantics:
    """Return the observed semantics of ``column`` in a pyarrow table."""
    arrow_type = table.schema.field(column).type
    return column_semantics_from_arrow(arrow_type)
