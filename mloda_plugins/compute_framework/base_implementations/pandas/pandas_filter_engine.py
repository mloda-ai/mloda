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
    def do_max_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        # Check if this is a complex parameter with max/max_exclusive or a simple one with value
        has_max = filter_feature.parameter.max_value is not None
        has_value = filter_feature.parameter.value is not None

        if has_max:
            # Complex parameter - use get_min_max_operator
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
                return data[data[filter_feature.name] < max_parameter]
            else:
                return data[data[filter_feature.name] <= max_parameter]
        elif has_value:
            # Simple parameter - extract the value
            value = filter_feature.parameter.value
            if value is None:
                raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")
            return data[data[filter_feature.name] <= value]
        else:
            raise ValueError(f"No valid filter parameter found in {filter_feature.parameter}")

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
