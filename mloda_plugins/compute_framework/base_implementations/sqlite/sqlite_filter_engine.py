from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda_plugins.compute_framework.base_implementations.sql import sql_type_semantics
from mloda_plugins.compute_framework.base_implementations.sql.sql_base_filter_engine import SqlBaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident


class SqliteFilterEngine(SqlBaseFilterEngine):
    @classmethod
    def _column_semantics(cls, data: Any, column: str) -> ColumnSemantics:
        return sql_type_semantics.column_semantics_from_arrow(
            data.types[data.columns.index(column)], is_string_storage=True
        )

    @classmethod
    def _build_regex_condition(cls, column_name: str, value: str) -> tuple[str, tuple[Any, ...]]:
        return f"{quote_ident(column_name)} REGEXP ?", (value,)
