from typing import Any

from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_base_filter_mask_engine import (
    SqlBaseFilterMaskEngine,
)


class SqliteFilterMaskEngine(SqlBaseFilterMaskEngine):
    @classmethod
    def supported_data_type(cls) -> type[Any]:
        return SqliteRelation
