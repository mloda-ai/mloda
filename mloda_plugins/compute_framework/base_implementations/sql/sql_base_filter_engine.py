import threading
from typing import Any, Tuple

from mloda.provider import BaseFilterEngine
from mloda.user import SingleFilter

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident


class SqlBaseFilterEngine(BaseFilterEngine):
    """Shared SQL condition-building logic for SQL-based filter engines.

    Subclasses must implement:
    - _apply_filter(data, condition, params): Apply a SQL condition with parameterized values
    - _build_regex_condition(column_name, value): Build regex SQL for the specific dialect
    """

    _thread_local: threading.local = threading.local()

    @classmethod
    def final_filters(cls) -> bool:
        return True

    @classmethod
    def _apply_filter(cls, data: Any, condition: str, params: Tuple[Any, ...] = ()) -> Any:
        raise NotImplementedError

    @classmethod
    def _build_regex_condition(cls, column_name: str, value: str) -> Tuple[str, Tuple[Any, ...]]:
        raise NotImplementedError

    @classmethod
    def do_range_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        min_parameter, max_parameter, max_operator = cls.get_min_max_operator(filter_feature)

        if min_parameter is None or max_parameter is None:
            raise ValueError(f"Filter parameter {filter_feature.parameter} not supported")

        filter_feature_name = filter_feature.name.name

        if max_operator is True:
            condition = f"{quote_ident(filter_feature_name)} >= ? AND {quote_ident(filter_feature_name)} < ?"
        else:
            condition = f"{quote_ident(filter_feature_name)} >= ? AND {quote_ident(filter_feature_name)} <= ?"

        return cls._apply_filter(data, condition, (min_parameter, max_parameter))

    @classmethod
    def do_min_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name.name
        value = filter_feature.parameter.value

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        condition = f"{quote_ident(column_name)} >= ?"
        return cls._apply_filter(data, condition, (value,))

    @classmethod
    def do_max_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name.name

        has_max = filter_feature.parameter.max_value is not None
        has_value = filter_feature.parameter.value is not None

        if has_max:
            min_parameter, max_parameter, max_operator = cls.get_min_max_operator(filter_feature)

            if min_parameter is not None:
                raise ValueError(
                    f"Filter parameter {filter_feature.parameter} not supported as max filter: {filter_feature.name}"
                )

            if max_parameter is None:
                raise ValueError(
                    f"Filter parameter {filter_feature.parameter} is None although expected: {filter_feature.name}"
                )

            if max_operator is True:
                condition = f"{quote_ident(column_name)} < ?"
            else:
                condition = f"{quote_ident(column_name)} <= ?"
            params: Tuple[Any, ...] = (max_parameter,)
        elif has_value:
            value = filter_feature.parameter.value

            if value is None:
                raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

            condition = f"{quote_ident(column_name)} <= ?"
            params = (value,)
        else:
            raise ValueError(f"No valid filter parameter found in {filter_feature.parameter}")

        return cls._apply_filter(data, condition, params)

    @classmethod
    def do_equal_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name.name
        value = filter_feature.parameter.value

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        condition = f"{quote_ident(column_name)} = ?"
        return cls._apply_filter(data, condition, (value,))

    @classmethod
    def do_regex_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name.name
        value = filter_feature.parameter.value

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        condition, params = cls._build_regex_condition(column_name, value)
        return cls._apply_filter(data, condition, params)

    @classmethod
    def do_categorical_inclusion_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name.name
        values = filter_feature.parameter.values

        if values is None:
            raise ValueError(f"Filter parameter 'values' not found in {filter_feature.parameter}")

        if not values:
            raise ValueError(f"Filter parameter 'values' must not be empty in {filter_feature.parameter}")

        placeholders = ", ".join("?" for _ in values)
        condition = f"{quote_ident(column_name)} IN ({placeholders})"
        return cls._apply_filter(data, condition, tuple(values))
