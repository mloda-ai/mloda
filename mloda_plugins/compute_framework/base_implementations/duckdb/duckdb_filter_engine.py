from typing import Any

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda_plugins.compute_framework.base_implementations.duckdb import duckdb_type_semantics
from mloda_plugins.compute_framework.base_implementations.sql.sql_base_filter_engine import SqlBaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident


class DuckDBFilterEngine(SqlBaseFilterEngine):
    provides_column_semantics = True

    @classmethod
    def _column_semantics(cls, data: Any, column: str) -> ColumnSemantics:
        return duckdb_type_semantics.column_semantics(data, column)

    @classmethod
    def _build_regex_condition(cls, column_name: str, value: str) -> tuple[str, tuple[Any, ...]]:
        return f"regexp_matches({quote_ident(column_name)}, ?)", (value,)
