from typing import Any
from mloda.provider import BaseFilterEngine
from mloda.user import SingleFilter


class PandasFilterEngine(BaseFilterEngine):
    @classmethod
    def final_filters(cls) -> bool:
        """Filters are applied after the feature calculation."""
        return True

    @classmethod
    def do_range_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        min_parameter, max_parameter, max_operator = cls.get_min_max_operator(filter_feature)

        if min_parameter is None or max_parameter is None:
            raise ValueError(f"Filter parameter {filter_feature.parameter} not supported")

        if max_operator is True:
            return data[(data[filter_feature.name] >= min_parameter) & (data[filter_feature.name] < max_parameter)]

        return data[(data[filter_feature.name] >= min_parameter) & (data[filter_feature.name] <= max_parameter)]

    @classmethod
    def do_min_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        value = filter_feature.parameter.value
        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")
        return data[data[filter_feature.name] >= value]

    @classmethod
    def _apply_max_exclusive_filter(cls, data: Any, column_name: str, threshold: Any) -> Any:
        return data[data[column_name] < threshold]

    @classmethod
    def _apply_max_inclusive_filter(cls, data: Any, column_name: str, threshold: Any) -> Any:
        return data[data[column_name] <= threshold]

    @classmethod
    def do_equal_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        value = filter_feature.parameter.value
        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")
        return data[data[filter_feature.name] == value]

    @classmethod
    def do_regex_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        value = filter_feature.parameter.value
        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")
        return data[data[filter_feature.name].astype(str).str.match(value)]

    @classmethod
    def do_categorical_inclusion_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        values = filter_feature.parameter.values
        if values is None:
            raise ValueError(f"Filter parameter 'values' not found in {filter_feature.parameter}")
        return data[data[filter_feature.name].isin(values)]
