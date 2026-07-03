from typing import Any

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
        # Cost note: value-sampling runs one LIMIT 100 query per string-typed column during
        # filter validation, including genuine string keys where the result is discarded;
        # a per-relation sample cache is a possible future optimization.
        is_string_like = sql_type_semantics.is_string_like_arrow_type(arrow_type)
        value_sample = sample_string_values(data, column) if is_string_like else None
        return sql_type_semantics.column_semantics_from_arrow(
            arrow_type, is_string_storage=True, value_sample=value_sample
        )

    @classmethod
    def _build_regex_condition(cls, column_name: str, value: str) -> tuple[str, tuple[Any, ...]]:
        return f"{quote_ident(column_name)} REGEXP ?", (value,)
