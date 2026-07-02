import re
from typing import Any, Callable, cast
from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda.provider import BaseFilterEngine
from mloda.user import SingleFilter
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import rows_to_columnar
from mloda_plugins.compute_framework.base_implementations.python_dict import python_dict_type_semantics


class PythonDictFilterEngine(BaseFilterEngine):
    """
    Filter engine for PythonDict framework using a COLUMNAR ``dict[str, list[Any]]``.

    Each filter computes a keep-index list over the target column, then slices every
    column by those indices. A missing column or the schema-less ``{}`` is handled
    without crashing (yields an empty keep set / passes through).
    """

    @classmethod
    def final_filters(cls) -> bool:
        """Filters are applied after the feature calculation."""
        return True

    @staticmethod
    def _to_columnar(data: Any) -> dict[str, list[Any]]:
        """Normalize a row-wise ``list[dict]`` to columnar. Columnar dict input passes through."""
        if isinstance(data, list):
            return rows_to_columnar(data)
        return cast(dict[str, list[Any]], data)

    @classmethod
    def _apply_keep(cls, data: Any, column_name: str, predicate: Callable[[Any], bool]) -> Any:
        data = cls._to_columnar(data)
        column = data.get(column_name, [])
        keep = [i for i, value in enumerate(column) if predicate(value)]
        return {c: [data[c][i] for i in keep] for c in data}

    @classmethod
    def _column_semantics(cls, data: Any, column: str) -> ColumnSemantics:
        return python_dict_type_semantics.column_semantics(data, column)

    @classmethod
    def do_range_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        min_parameter, max_parameter, max_operator = cls.get_min_max_operator(filter_feature)

        if min_parameter is None or max_parameter is None:
            raise ValueError(f"Filter parameter {filter_feature.parameter} not supported")

        column_name = filter_feature.name

        if max_operator is True:
            return cls._apply_keep(data, column_name, lambda v: v is not None and min_parameter <= v < max_parameter)
        return cls._apply_keep(data, column_name, lambda v: v is not None and min_parameter <= v <= max_parameter)

    @classmethod
    def do_min_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        value = filter_feature.parameter.value

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        return cls._apply_keep(data, column_name, lambda v: v is not None and v >= value)

    @classmethod
    def _apply_max_exclusive_filter(cls, data: Any, column_name: str, threshold: Any) -> Any:
        return cls._apply_keep(data, column_name, lambda v: v is not None and v < threshold)

    @classmethod
    def _apply_max_inclusive_filter(cls, data: Any, column_name: str, threshold: Any) -> Any:
        return cls._apply_keep(data, column_name, lambda v: v is not None and v <= threshold)

    @classmethod
    def do_equal_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        value = filter_feature.parameter.value

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        return cls._apply_keep(data, column_name, lambda v: v == value)

    @classmethod
    def do_regex_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        value = filter_feature.parameter.value

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        compiled_pattern = re.compile(value)

        return cls._apply_keep(data, column_name, lambda v: v is not None and bool(compiled_pattern.match(str(v))))

    @classmethod
    def do_categorical_inclusion_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        values = filter_feature.parameter.values

        if values is None:
            raise ValueError(f"Filter parameter 'values' not found in {filter_feature.parameter}")

        allowed_set = set(values) if isinstance(values, (list, tuple)) else {values}

        return cls._apply_keep(data, column_name, lambda v: v in allowed_set)
