from typing import Any

from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_base_mask_engine import (
    SqlBaseMaskEngine,
)


class SqliteMaskEngine(SqlBaseMaskEngine):
    @classmethod
    def supported_data_type(cls) -> type[Any]:
        return SqliteRelation
