from typing import Any

import pyarrow as pa
import pytest

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    duckdb = None  # type: ignore[assignment]
    DUCKDB_AVAILABLE = False

if DUCKDB_AVAILABLE:
    from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_mask_engine import (
        DuckDBMaskEngine,
    )
    from mloda_plugins.compute_framework.base_implementations.sql.sql_base_mask_engine import (
        SqlBaseMaskEngine,
    )
    from tests.test_plugins.compute_framework.base_implementations.sql_mask_engine_test_mixin import (
        SqlMaskEngineTestMixin,
    )

    @pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="duckdb not installed")
    class TestDuckDBSqlMaskEngine(SqlMaskEngineTestMixin):
        @pytest.fixture
        def engine(self) -> type[SqlBaseMaskEngine]:
            return DuckDBMaskEngine

        @pytest.fixture
        def sample_data(self, connection: Any) -> Any:
            table = pa.table(
                {
                    "status": ["active", "inactive", "active", "inactive"],
                    "value": [10, 20, 30, 40],
                }
            )
            return DuckdbRelation.from_arrow(connection, table)

        def evaluate_mask(self, mask: Any, data: DuckdbRelation) -> list[bool]:
            bool_expr = f"CASE WHEN {mask} THEN 1 ELSE 0 END AS __match__"
            projected = data.select(_raw_sql=bool_expr)
            arrow = projected.to_arrow_table()
            return [bool(arrow.column("__match__")[i].as_py()) for i in range(arrow.num_rows)]
