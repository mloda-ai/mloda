from typing import Any

import pytest

from mloda_plugins.compute_framework.base_implementations.sql.sql_base_filter_mask_engine import (
    SqlBaseFilterMaskEngine,
)
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_filter_mask_engine import (
    SqliteFilterMaskEngine,
)
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation
from tests.test_plugins.compute_framework.base_implementations.sql_filter_mask_engine_test_mixin import (
    SqlFilterMaskEngineTestMixin,
)


class TestSqliteSqlFilterMaskEngine(SqlFilterMaskEngineTestMixin):
    @pytest.fixture
    def engine(self) -> type[SqlBaseFilterMaskEngine]:
        return SqliteFilterMaskEngine

    @pytest.fixture
    def sample_data(self, connection: Any) -> Any:
        return SqliteRelation.from_dict(
            connection,
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            },
        )

    def evaluate_mask(self, mask: Any, data: SqliteRelation) -> list[bool]:
        conn = data.connection
        table_name = data.table_name
        sql = f"SELECT CASE WHEN {mask} THEN 1 ELSE 0 END AS match FROM {quote_ident(table_name)}"  # nosec B608
        rows = conn.execute(sql).fetchall()
        return [bool(row[0]) for row in rows]
