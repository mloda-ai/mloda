from typing import Any

from mloda_plugins.compute_framework.base_implementations.sql.sql_base_filter_engine import SqlBaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident


class DuckDBFilterEngine(SqlBaseFilterEngine):
    @classmethod
    def _build_regex_condition(cls, column_name: str, value: str) -> tuple[str, tuple[Any, ...]]:
        return f"regexp_matches({quote_ident(column_name)}, ?)", (value,)
