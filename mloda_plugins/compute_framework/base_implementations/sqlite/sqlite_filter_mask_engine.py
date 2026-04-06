from typing import Any

from mloda.core.filter.filter_mask_engine import BaseFilterMaskEngine
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation


class SqliteFilterMaskEngine(BaseFilterMaskEngine):
    @classmethod
    def supported_data_type(cls) -> type[Any]:
        return SqliteRelation

    @classmethod
    def _fetch_column(cls, data: Any, column: str) -> list[Any]:
        sql = f"SELECT {quote_ident(column)} FROM {quote_ident(data.table_name)}"  # nosec B608
        rows = data.connection.execute(sql).fetchall()
        return [row[0] for row in rows]

    @classmethod
    def all_true(cls, data: Any) -> list[Any]:
        return [True] * len(data)

    @classmethod
    def combine(cls, mask1: Any, mask2: Any) -> list[Any]:
        return [a and b for a, b in zip(mask1, mask2)]

    @classmethod
    def equal(cls, data: Any, column: str, value: Any) -> list[Any]:
        return [v == value for v in cls._fetch_column(data, column)]

    @classmethod
    def greater_equal(cls, data: Any, column: str, value: Any) -> list[Any]:
        return [v is not None and v >= value for v in cls._fetch_column(data, column)]

    @classmethod
    def less_equal(cls, data: Any, column: str, value: Any) -> list[Any]:
        return [v is not None and v <= value for v in cls._fetch_column(data, column)]

    @classmethod
    def less_than(cls, data: Any, column: str, value: Any) -> list[Any]:
        return [v is not None and v < value for v in cls._fetch_column(data, column)]

    @classmethod
    def is_in(cls, data: Any, column: str, values: Any) -> list[Any]:
        allowed = set(values) if isinstance(values, (list, tuple)) else {values}
        return [v in allowed for v in cls._fetch_column(data, column)]
