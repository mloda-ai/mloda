from typing import Any

from mloda.core.filter.filter_mask_engine import BaseFilterMaskEngine
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident, quote_value


class SqlBaseFilterMaskEngine(BaseFilterMaskEngine):
    """SQL-native filter mask engine that returns SQL condition strings.

    Instead of fetching data into Python, this engine builds SQL WHERE-clause
    fragments that can be embedded in CASE WHEN expressions or other SQL
    constructs by downstream consumers.
    """

    @classmethod
    def all_true(cls, data: Any) -> str:
        return "1 = 1"

    @classmethod
    def combine(cls, mask1: Any, mask2: Any) -> str:
        return f"({mask1}) AND ({mask2})"

    @classmethod
    def equal(cls, data: Any, column: str, value: Any) -> str:
        return f"{quote_ident(column)} = {quote_value(value)}"

    @classmethod
    def greater_equal(cls, data: Any, column: str, value: Any) -> str:
        return f"{quote_ident(column)} >= {quote_value(value)}"

    @classmethod
    def less_equal(cls, data: Any, column: str, value: Any) -> str:
        return f"{quote_ident(column)} <= {quote_value(value)}"

    @classmethod
    def less_than(cls, data: Any, column: str, value: Any) -> str:
        return f"{quote_ident(column)} < {quote_value(value)}"

    @classmethod
    def is_in(cls, data: Any, column: str, values: Any) -> str:
        value_list = values if isinstance(values, (list, tuple)) else [values]
        quoted = ", ".join(quote_value(v) for v in value_list)
        return f"{quote_ident(column)} IN ({quoted})"
