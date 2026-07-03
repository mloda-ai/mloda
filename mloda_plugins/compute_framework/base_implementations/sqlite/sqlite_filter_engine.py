from typing import Any

import pyarrow as pa

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda_plugins.compute_framework.base_implementations.sql import sql_type_semantics
from mloda_plugins.compute_framework.base_implementations.sql.sql_base_filter_engine import SqlBaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_value_sample import sample_string_values


class SqliteFilterEngine(SqlBaseFilterEngine):
    provides_column_semantics = True

    @classmethod
    def _column_semantics(cls, data: Any, column: str) -> ColumnSemantics:
        arrow_type = data.types[data.columns.index(column)]
        value_sample = sample_string_values(data, column) if pa.types.is_string(arrow_type) else None
        return sql_type_semantics.column_semantics_from_arrow(
            arrow_type, is_string_storage=True, value_sample=value_sample
        )

    @classmethod
    def _build_regex_condition(cls, column_name: str, value: str) -> tuple[str, tuple[Any, ...]]:
        return f"{quote_ident(column_name)} REGEXP ?", (value,)
