from typing import Any

import pytest

from mloda._testing.compute_framework.mask_suite import SqlMaskEngineSuite
from mloda._testing.declarations import Cause
from mloda._testing.framework_adapter import FrameworkAdapter, SqliteAdapter
from mloda.provider import BaseMaskEngine
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation


class TestSqliteSqlMaskEngine(SqlMaskEngineSuite):
    adapter_class = SqliteAdapter
    switched_off = {
        "test_is_in_decimal": Cause.BACKEND,
        "test_is_in_decimal_unrepresentable_values_match_nothing": Cause.BACKEND,
        "test_decimal_comparison_null_row_is_false": Cause.BACKEND,
    }

    def evaluate_mask(self, adapter: FrameworkAdapter, mask: Any, data: SqliteRelation) -> list[bool]:
        conn = data.connection
        table_name = data.table_name
        sql = f"SELECT CASE WHEN {mask} THEN 1 ELSE 0 END AS match FROM {quote_ident(table_name)}"  # nosec
        rows = conn.execute(sql).fetchall()
        return [bool(row[0]) for row in rows]

    def select_rows(self, adapter: FrameworkAdapter, mask: Any, data: SqliteRelation) -> Any:
        return data.filter(mask)

    @pytest.mark.skip(reason="SQLite has no decimal storage type; a decimal column cannot be inserted")
    def test_is_in_decimal(self, engine: type[BaseMaskEngine], decimal_data: Any) -> None: ...

    @pytest.mark.skip(reason="SQLite has no decimal storage type; a decimal column cannot be inserted")
    def test_is_in_decimal_unrepresentable_values_match_nothing(
        self, engine: type[BaseMaskEngine], decimal_data: Any
    ) -> None: ...

    @pytest.mark.skip(reason="SQLite has no decimal storage type; a decimal column cannot be inserted")
    def test_decimal_comparison_null_row_is_false(self, engine: type[BaseMaskEngine], decimal_data: Any) -> None: ...
