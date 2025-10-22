from typing import Any
from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_core.filter.single_filter import SingleFilter


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
        # Handle empty DataFrame edge case
        if len(data) == 0:
            return data

        value = cls.get_parameter_value(filter_feature, "value")
        return data[data[filter_feature.name] >= value]

    @classmethod
    def do_max_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        # Check if this is a complex parameter with max/max_exclusive or a simple one with value
        max_param = cls.get_parameter_value(filter_feature, "max", required=False)
        value_param = cls.get_parameter_value(filter_feature, "value", required=False)

        if max_param is not None:
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

            return (
                data[data[filter_feature.name] < max_parameter]
                if max_operator
                else data[data[filter_feature.name] <= max_parameter]
            )
        elif value_param is not None:
            return data[data[filter_feature.name] <= value_param]
        else:
            raise ValueError(f"No valid filter parameter found in {filter_feature.parameter}")

    @classmethod
    def do_equal_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        value = cls.get_parameter_value(filter_feature, "value")
        return data[data[filter_feature.name] == value]

    @classmethod
    def do_regex_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        value = cls.get_parameter_value(filter_feature, "value")
        return data[data[filter_feature.name].astype(str).str.match(value)]

    @classmethod
    def do_categorical_inclusion_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        values = cls.get_parameter_value(filter_feature, "values")
        return data[data[filter_feature.name].isin(values)]
