from abc import abstractmethod
from typing import Any

from mloda.core.abstract_plugins.components.mask.base_mask_engine import BaseMaskEngine
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident, quote_value


class SqlBaseMaskEngine(BaseMaskEngine):
    """SQL-native mask engine that returns SQL condition strings.

    Instead of fetching data into Python, this engine builds SQL WHERE-clause
    fragments that can be embedded in CASE WHEN expressions or other SQL
    constructs by downstream consumers.

    Subclasses must implement supported_data_type() for their specific relation type.
    """

    @classmethod
    @abstractmethod
    def supported_data_type(cls) -> type[Any]: ...

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
    def greater_than(cls, data: Any, column: str, value: Any) -> str:
        return f"{quote_ident(column)} > {quote_value(value)}"

    @classmethod
    def is_in(cls, data: Any, column: str, values: Any) -> str:
        value_list = values if isinstance(values, (list, tuple)) else [values]
        non_null = [v for v in value_list if v is not None]
        has_null = len(non_null) != len(value_list)

        conditions: list[str] = []
        if non_null:
            quoted = ", ".join(quote_value(v) for v in non_null)
            conditions.append(f"{quote_ident(column)} IN ({quoted})")
        if has_null:
            conditions.append(f"{quote_ident(column)} IS NULL")

        if not conditions:
            return "1 = 0"
        if len(conditions) == 1:
            return conditions[0]
        return f"({' OR '.join(conditions)})"
