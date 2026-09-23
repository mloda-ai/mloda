from typing import Any

import pyarrow as pa
import pytest

from mloda.provider import BaseMaskEngine
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_mask_engine import (
    SqliteMaskEngine,
)
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation
from tests.test_plugins.compute_framework.base_implementations.sql_mask_engine_test_mixin import (
    SqlMaskEngineTestMixin,
)


class TestSqliteSqlMaskEngine(SqlMaskEngineTestMixin):
    mask_engine_class = SqliteMaskEngine

    @pytest.fixture
    def sample_data(self, connection: Any) -> Any:
        return SqliteRelation.from_dict(
            connection,
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            },
        )

    @pytest.fixture
    def empty_data(self, connection: Any) -> Any:
        table = pa.table({"status": pa.array([], type=pa.string()), "value": pa.array([], type=pa.int64())})
        return SqliteRelation.from_arrow(connection, table)

    @pytest.fixture
    def null_data(self, connection: Any) -> Any:
        table = pa.table(
            {
                "status": pa.array(["active", None, "inactive", None], type=pa.string()),
                "value": pa.array([10, 20, 30, 40], type=pa.int64()),
                "score": pa.array([1, None, 3, None], type=pa.int64()),
            }
        )
        return SqliteRelation.from_arrow(connection, table)

    def evaluate_mask(self, mask: Any, data: SqliteRelation) -> list[bool]:
        conn = data.connection
        table_name = data.table_name
        sql = f"SELECT CASE WHEN {mask} THEN 1 ELSE 0 END AS match FROM {quote_ident(table_name)}"  # nosec
        rows = conn.execute(sql).fetchall()
        return [bool(row[0]) for row in rows]

    def apply_mask(self, mask: Any, data: SqliteRelation) -> dict[str, list[Any]]:
        result: dict[str, list[Any]] = data.filter(mask).to_arrow_table().to_pydict()
        return result

    @pytest.mark.skip(reason="SQLite has no decimal storage type; a decimal column cannot be inserted")
    def test_is_in_decimal(self, engine: type[BaseMaskEngine], decimal_sample_data: Any) -> None: ...

    @pytest.mark.skip(reason="SQLite has no decimal storage type; a decimal column cannot be inserted")
    def test_is_in_decimal_unrepresentable_values_match_nothing(
        self, engine: type[BaseMaskEngine], decimal_sample_data: Any
    ) -> None: ...

    @pytest.mark.skip(reason="SQLite has no decimal storage type; a decimal column cannot be inserted")
    def test_decimal_comparison_null_row_is_false(
        self, engine: type[BaseMaskEngine], decimal_sample_data: Any
    ) -> None: ...
