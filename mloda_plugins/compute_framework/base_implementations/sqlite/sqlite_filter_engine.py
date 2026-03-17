from typing import Any, Tuple

from mloda_plugins.compute_framework.base_implementations.sql.sql_base_filter_engine import SqlBaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident


class SqliteFilterEngine(SqlBaseFilterEngine):
    @classmethod
    def _apply_filter(cls, data: Any, condition: str, params: Tuple[Any, ...] = ()) -> Any:
        return data.filter(condition, params)

    @classmethod
    def _build_regex_condition(cls, column_name: str, value: str) -> Tuple[str, Tuple[Any, ...]]:
        return f"{quote_ident(column_name)} REGEXP ?", (value,)
