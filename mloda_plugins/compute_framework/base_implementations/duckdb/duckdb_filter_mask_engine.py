from typing import Any

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_base_filter_mask_engine import (
    SqlBaseFilterMaskEngine,
)


class DuckDBFilterMaskEngine(SqlBaseFilterMaskEngine):
    @classmethod
    def supported_data_type(cls) -> type[Any]:
        return DuckdbRelation
