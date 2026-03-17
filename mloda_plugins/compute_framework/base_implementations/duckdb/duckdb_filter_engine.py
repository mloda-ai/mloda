from typing import Any, Tuple

from mloda_plugins.compute_framework.base_implementations.sql.sql_base_filter_engine import SqlBaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import inline_params, quote_ident


class DuckDBFilterEngine(SqlBaseFilterEngine):
    @classmethod
    def _apply_filter(cls, data: Any, condition: str, params: Tuple[Any, ...] = ()) -> Any:
        conn = getattr(cls._thread_local, "connection", None)
        if conn is None or not params:
            if params:
                condition = inline_params(condition, params)
            return data.filter(condition)

        temp_name = f"_flt_{id(data)}"
        arrow_data = data.arrow()
        conn.register(temp_name, arrow_data)
        cursor = conn.execute(f'SELECT * FROM "{temp_name}" WHERE {condition}', list(params))  # nosec
        result_arrow = cursor.fetch_arrow_table()
        conn.unregister(temp_name)
        return conn.from_arrow(result_arrow)

    @classmethod
    def _build_regex_condition(cls, column_name: str, value: str) -> Tuple[str, Tuple[Any, ...]]:
        return f"regexp_matches({quote_ident(column_name)}, ?)", (value,)
