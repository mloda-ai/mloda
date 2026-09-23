from decimal import Decimal
from typing import Any

import pyarrow as pa
import pytest

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_mask_engine import (
    DuckDBMaskEngine,
)
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from tests.test_plugins.compute_framework.base_implementations.sql_mask_engine_test_mixin import (
    SqlMaskEngineTestMixin,
)

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    duckdb = None  # type: ignore[assignment]
    DUCKDB_AVAILABLE = False


@pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="duckdb not installed")
class TestDuckDBSqlMaskEngine(SqlMaskEngineTestMixin):
    mask_engine_class = DuckDBMaskEngine

    @pytest.fixture
    def sample_data(self, connection: Any) -> Any:
        table = pa.table(
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            }
        )
        return DuckdbRelation.from_arrow(connection, table)

    @pytest.fixture
    def empty_data(self, connection: Any) -> Any:
        table = pa.table({"status": pa.array([], type=pa.string()), "value": pa.array([], type=pa.int64())})
        return DuckdbRelation.from_arrow(connection, table)

    @pytest.fixture
    def null_data(self, connection: Any) -> Any:
        table = pa.table(
            {
                "status": pa.array(["active", None, "inactive", None], type=pa.string()),
                "value": pa.array([10, 20, 30, 40], type=pa.int64()),
                "score": pa.array([1, None, 3, None], type=pa.int64()),
                "ratio": pa.array([1.0, float("nan"), 3.0, None], type=pa.float64()),
            }
        )
        return DuckdbRelation.from_arrow(connection, table)

    @pytest.fixture
    def decimal_sample_data(self, connection: Any) -> Any:
        values = [Decimal("12.34"), Decimal("5.50"), None]
        return DuckdbRelation.from_arrow(connection, pa.table({"d": pa.array(values, type=pa.decimal128(10, 2))}))

    def evaluate_mask(self, mask: Any, data: DuckdbRelation) -> list[bool]:
        bool_expr = f"CASE WHEN {mask} THEN 1 ELSE 0 END AS __match__"
        projected = data.project(bool_expr)
        arrow = projected.to_arrow_table()
        return [bool(arrow.column("__match__")[i].as_py()) for i in range(arrow.num_rows)]

    def is_boolean_mask(self, mask: Any, data: DuckdbRelation) -> bool:
        """Projects the condition and checks its DuckDB type, which works at zero rows."""
        return [str(t) for t in data.project(f"({mask}) AS __m").types] == ["BOOLEAN"]

    def apply_mask(self, mask: Any, data: DuckdbRelation) -> dict[str, list[Any]]:
        result: dict[str, list[Any]] = data.filter(mask).to_arrow_table().to_pydict()
        return result

    def test_greater_equal_lowercase_column_matches_nan_semantics(self, connection: Any) -> None:
        """DuckDB binds identifiers case-insensitively; a lowercase name must still hit NaN handling."""
        table = pa.table({"Ratio": pa.array([1.0, float("nan"), 3.0, None], type=pa.float64())})
        rel = DuckdbRelation.from_arrow(connection, table)
        mask = DuckDBMaskEngine.greater_equal(rel, "ratio", 1.0)
        assert self.evaluate_mask(mask, rel) == [True, False, True, False]
