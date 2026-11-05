import re
from typing import Any

import pytest

from mloda._testing.compute_framework.mask_suite import SqlMaskEngineSuite
from mloda._testing.framework_adapter import DuckDBAdapter, FrameworkAdapter
from mloda.provider import BaseMaskEngine
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_mask_engine import (
    DuckDBMaskEngine,
)
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment]


class TestDuckDBSqlMaskEngine(SqlMaskEngineSuite):
    adapter_class = DuckDBAdapter

    def evaluate_mask(self, adapter: FrameworkAdapter, mask: Any, data: DuckdbRelation) -> list[bool]:
        bool_expr = f"CASE WHEN {mask} THEN 1 ELSE 0 END AS __match__"
        projected = data.project(bool_expr)
        arrow = projected.to_arrow_table()
        return [bool(arrow.column("__match__")[i].as_py()) for i in range(arrow.num_rows)]

    def is_boolean_mask(self, adapter: FrameworkAdapter, mask: Any, data: DuckdbRelation) -> bool:
        """Projects the condition and checks its DuckDB type, which works at zero rows."""
        return [str(t) for t in data.project(f"({mask}) AS __m").types] == ["BOOLEAN"]

    def select_rows(self, adapter: FrameworkAdapter, mask: Any, data: DuckdbRelation) -> Any:
        return data.filter(mask)

    def test_greater_equal_lowercase_column_raises_value_error(self, adapter: DuckDBAdapter) -> None:
        table = pa.table({"Ratio": pa.array([1.0, float("nan"), 3.0, None], type=pa.float64())})
        rel = DuckdbRelation.from_arrow(adapter.connection, table)
        with pytest.raises(ValueError, match="ratio"):
            DuckDBMaskEngine.greater_equal(rel, "ratio", 1.0)

    @pytest.mark.parametrize(
        "method,arg",
        [
            ("equal", None),
            ("greater_equal", 1.0),
            ("greater_than", 1.0),
            ("is_in", [None]),
        ],
    )
    def test_data_none_raises_type_error(self, engine: type[BaseMaskEngine], method: str, arg: Any) -> None:
        with pytest.raises(
            TypeError,
            match=re.escape(f"DuckDBMaskEngine.{method} needs the DuckdbRelation being masked as data, got None"),
        ):
            getattr(engine, method)(None, "value", arg)
