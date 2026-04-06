from typing import Any

import pyarrow as pa
import pytest

from mloda.core.filter.filter_mask_engine import BaseFilterMaskEngine
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from tests.test_plugins.compute_framework.base_implementations.filter_mask_engine_test_mixin import (
    FilterMaskEngineTestMixin,
)

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    duckdb = None  # type: ignore[assignment]
    DUCKDB_AVAILABLE = False

if DUCKDB_AVAILABLE:
    from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_filter_mask_engine import (
        DuckDBFilterMaskEngine,
    )


@pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="duckdb not installed")
class TestDuckDBFilterMaskEngine(FilterMaskEngineTestMixin):
    @pytest.fixture
    def engine(self) -> type[BaseFilterMaskEngine]:
        return DuckDBFilterMaskEngine

    @pytest.fixture
    def sample_data(self, connection: Any) -> Any:
        table = pa.table(
            {
                "status": ["active", "inactive", "active", "inactive"],
                "value": [10, 20, 30, 40],
            }
        )
        return DuckdbRelation.from_arrow(connection, table)

    def mask_to_list(self, mask: Any) -> list[bool]:
        return list(mask)
