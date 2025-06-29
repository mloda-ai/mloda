from typing import Any
from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_core.filter.single_filter import SingleFilter

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

        column_name = filter_feature.name.name

        if max_operator is True:
            condition = (F.col(column_name) >= min_parameter) & (F.col(column_name) < max_parameter)
        else:
            condition = (F.col(column_name) >= min_parameter) & (F.col(column_name) <= max_parameter)

        return data.filter(condition)

    @classmethod
    def do_min_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name.name

        # Extract the value from the parameter
        value = None
        for param in filter_feature.parameter:
            if param[0] == "value":
                value = param[1]
                break

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        return data.filter(F.col(column_name) >= value)

    @classmethod
    def do_max_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name.name

        # Check if this is a complex parameter with max/max_exclusive or a simple one with value
        has_max = False
        has_value = False

        for param in filter_feature.parameter:
            if param[0] == "max":
                has_max = True
            elif param[0] == "value":
                has_value = True

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
                condition = F.col(column_name) < max_parameter
            else:
                condition = F.col(column_name) <= max_parameter
        elif has_value:
            # Simple parameter - extract the value
            value = None
            for param in filter_feature.parameter:
                if param[0] == "value":
                    value = param[1]
                    break

            if value is None:
                raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

            condition = F.col(column_name) <= value
        else:
            raise ValueError(f"No valid filter parameter found in {filter_feature.parameter}")

        return data.filter(condition)

    @classmethod
    def do_equal_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name.name

        # Extract the value from the parameter
        value = None
        for param in filter_feature.parameter:
            if param[0] == "value":
                value = param[1]
                break

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        return data.filter(F.col(column_name) == value)

    @classmethod
    def do_regex_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name.name

        # Extract the value from the parameter
        value = None
        for param in filter_feature.parameter:
            if param[0] == "value":
                value = param[1]
                break

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        # Use Spark's rlike function for regex filtering
        return data.filter(F.col(column_name).rlike(value))

    @classmethod
    def do_categorical_inclusion_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name.name

        # Extract the values from the parameter
        values = None
        for param in filter_feature.parameter:
            if param[0] == "values":
                values = param[1]
                break

        if values is None:
            raise ValueError(f"Filter parameter 'values' not found in {filter_feature.parameter}")

        # Use Spark's isin function for categorical inclusion
        return data.filter(F.col(column_name).isin(values))
