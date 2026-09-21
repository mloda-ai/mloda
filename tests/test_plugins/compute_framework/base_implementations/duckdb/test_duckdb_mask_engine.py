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

    def test_equal_decimal_value_raises_type_error(self, engine: type[DuckDBMaskEngine], sample_data: Any) -> None:
        with pytest.raises(TypeError):
            engine.equal(sample_data, "value", Decimal("12.34"))

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
