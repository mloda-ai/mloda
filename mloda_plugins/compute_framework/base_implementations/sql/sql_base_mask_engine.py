from abc import abstractmethod
from typing import Any

from mloda.core.abstract_plugins.components.mask.base_mask_engine import BaseMaskEngine
from mloda.core.abstract_plugins.components.mask.null_or_nan import is_null_or_nan, split_null_or_nan
from mloda.core.abstract_plugins.components.utils import require_value_collection
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import (
    null_or_nan_condition,
    quote_ident,
    quote_value,
)


class SqlBaseMaskEngine(BaseMaskEngine):
    """SQL-native mask engine that returns SQL condition strings.

    Instead of fetching data into Python, this engine builds SQL WHERE-clause
    fragments that can be embedded in CASE WHEN expressions or other SQL
    constructs by downstream consumers.

    Subclasses must implement supported_data_type() for their specific relation type.
    A dialect whose float columns can store NaN overrides _nan_condition().
    """

    @classmethod
    @abstractmethod
    def supported_data_type(cls) -> type[Any]: ...

    @classmethod
    def _nan_condition(cls, data: Any, column: str) -> str | None:
        """Return a SQL condition true when column holds NaN, or None if the dialect cannot."""
        return None

    @classmethod
    def _null_or_nan_condition(cls, data: Any, column: str) -> str:
        return null_or_nan_condition(quote_ident(column), cls._nan_condition(data, column))

    @classmethod
    def all_true(cls, data: Any) -> str:
        return "1 = 1"

    @classmethod
    def combine(cls, mask1: Any, mask2: Any) -> str:
        return f"({mask1}) AND ({mask2})"

    @classmethod
    def equal(cls, data: Any, column: str, value: Any) -> str:
        if is_null_or_nan(value):
            return cls._null_or_nan_condition(data, column)
        return f"{quote_ident(column)} = {quote_value(value)}"

    @classmethod
    def greater_equal(cls, data: Any, column: str, value: Any) -> str:
        cond = f"{quote_ident(column)} >= {quote_value(value)}"
        nan_cond = cls._nan_condition(data, column)
        if nan_cond is not None:
            return f"({cond}) AND NOT {nan_cond}"
        return cond

    @classmethod
    def less_equal(cls, data: Any, column: str, value: Any) -> str:
        return f"{quote_ident(column)} <= {quote_value(value)}"

    @classmethod
    def less_than(cls, data: Any, column: str, value: Any) -> str:
        return f"{quote_ident(column)} < {quote_value(value)}"

    @classmethod
    def greater_than(cls, data: Any, column: str, value: Any) -> str:
        cond = f"{quote_ident(column)} > {quote_value(value)}"
        nan_cond = cls._nan_condition(data, column)
        if nan_cond is not None:
            return f"({cond}) AND NOT {nan_cond}"
        return cond

    @classmethod
    def is_in(cls, data: Any, column: str, values: Any) -> str:
        require_value_collection(values, "is_in values")
        if isinstance(values, (set, frozenset)):
            value_list = sorted(values, key=repr)
        else:
            value_list = list(values)
        present, has_null_or_nan = split_null_or_nan(value_list)
        if not present and not has_null_or_nan:
            return "1 = 0"
        parts = []
        if present:
            quoted = ", ".join(quote_value(v) for v in present)
            parts.append(f"{quote_ident(column)} IN ({quoted})")
        if has_null_or_nan:
            parts.append(cls._null_or_nan_condition(data, column))
        if len(parts) == 1:
            return parts[0]
        return f"({parts[0]} OR {parts[1]})"
