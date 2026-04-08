from typing import Any

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_base_mask_engine import (
    SqlBaseMaskEngine,
)


class DuckDBMaskEngine(SqlBaseMaskEngine):
    @classmethod
    def supported_data_type(cls) -> type[Any]:
        return DuckdbRelation
