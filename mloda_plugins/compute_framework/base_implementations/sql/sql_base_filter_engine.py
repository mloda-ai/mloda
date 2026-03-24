from typing import Any, Tuple

from mloda.provider import BaseFilterEngine
from mloda.user import SingleFilter

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident


class SqlBaseFilterEngine(BaseFilterEngine):
    """Shared SQL condition-building logic for SQL-based filter engines.

    Column names are quoted via ``quote_ident``. Filter values use ``?``
    placeholders, passed as params to the relation's ``filter()`` method.
    How those params are bound depends on the backend: PEP 249 native
    parameterization where supported, ``inline_params`` fallback otherwise.

    Subclasses must implement:
    - _build_regex_condition(column_name, value): Build regex SQL for the specific dialect
    """

    @classmethod
    def final_filters(cls) -> bool:
        return True

    @classmethod
    def _apply_filter(cls, data: Any, condition: str, params: tuple[Any, ...] = ()) -> Any:
        return data.filter(condition, params)

    @classmethod
    def _build_regex_condition(cls, column_name: str, value: str) -> tuple[str, tuple[Any, ...]]:
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
            params: tuple[Any, ...] = (max_parameter,)
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
