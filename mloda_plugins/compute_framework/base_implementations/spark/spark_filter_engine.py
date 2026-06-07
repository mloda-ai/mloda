from typing import Any
from mloda.provider import BaseFilterEngine
from mloda.user import SingleFilter

try:
    from pyspark.sql import DataFrame
    import pyspark.sql.functions as F
except ImportError:
    DataFrame = None
    F = None


class SparkFilterEngine(BaseFilterEngine):
    @classmethod
    def final_filters(cls) -> bool:
        """Filters are applied after the feature calculation."""
        return True

    @classmethod
    def do_range_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        min_parameter, max_parameter, max_operator = cls.get_min_max_operator(filter_feature)

        if min_parameter is None or max_parameter is None:
            raise ValueError(f"Filter parameter {filter_feature.parameter} not supported")

        column_name = filter_feature.name

        if max_operator is True:
            condition = (F.col(column_name) >= min_parameter) & (F.col(column_name) < max_parameter)
        else:
            condition = (F.col(column_name) >= min_parameter) & (F.col(column_name) <= max_parameter)

        return data.filter(condition)

    @classmethod
    def do_min_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        # Extract the value from the parameter

        value = filter_feature.parameter.value

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        return data.filter(F.col(column_name) >= value)

    @classmethod
    def _apply_max_exclusive_filter(cls, data: Any, column_name: str, threshold: Any) -> Any:
        return data.filter(F.col(column_name) < threshold)

    @classmethod
    def _apply_max_inclusive_filter(cls, data: Any, column_name: str, threshold: Any) -> Any:
        return data.filter(F.col(column_name) <= threshold)

    @classmethod
    def do_equal_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        # Extract the value from the parameter

        value = filter_feature.parameter.value

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        return data.filter(F.col(column_name) == value)

    @classmethod
    def do_regex_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        # Extract the value from the parameter

        value = filter_feature.parameter.value

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        # Use Spark's rlike function for regex filtering
        return data.filter(F.col(column_name).rlike(value))

    @classmethod
    def do_categorical_inclusion_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        # Extract the values from the parameter

        values = filter_feature.parameter.values

        if values is None:
            raise ValueError(f"Filter parameter 'values' not found in {filter_feature.parameter}")

        # Use Spark's isin function for categorical inclusion
        return data.filter(F.col(column_name).isin(values))
