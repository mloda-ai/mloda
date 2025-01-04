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
        return data[data[filter_feature.name] >= filter_feature.parameter]

    @classmethod
    def do_max_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        if isinstance(filter_feature.parameter, tuple):
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
        return data[data[filter_feature.name] <= filter_feature.parameter]

    @classmethod
    def do_equal_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        return data[data[filter_feature.name] == filter_feature.parameter]

    @classmethod
    def do_regex_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        return data[data[filter_feature.name].astype(str).str.match(filter_feature.parameter)]

    @classmethod
    def do_categorical_inclusion_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        return data[data[filter_feature.name].isin(filter_feature.parameter)]
