import re
from typing import Any
from mloda.provider import BaseFilterEngine
from mloda.user import SingleFilter


class PythonDictFilterEngine(BaseFilterEngine):
    """
    Filter engine for PythonDict framework using List[Dict[str, Any]] data structure.

    Implements filtering operations using list comprehensions and Python built-in functions.
    """

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
            # Exclusive max
            return [
                row
                for row in data
                if row.get(column_name) is not None and min_parameter <= row.get(column_name) < max_parameter
            ]
        else:
            # Inclusive max
            return [
                row
                for row in data
                if row.get(column_name) is not None and min_parameter <= row.get(column_name) <= max_parameter
            ]

    @classmethod
    def do_min_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        # Extract the value from the parameter

        value = filter_feature.parameter.value

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        return [row for row in data if row.get(column_name) is not None and row.get(column_name) >= value]

    @classmethod
    def _apply_max_exclusive_filter(cls, data: Any, column_name: str, threshold: Any) -> Any:
        return [row for row in data if row.get(column_name) is not None and row.get(column_name) < threshold]

    @classmethod
    def _apply_max_inclusive_filter(cls, data: Any, column_name: str, threshold: Any) -> Any:
        return [row for row in data if row.get(column_name) is not None and row.get(column_name) <= threshold]

    @classmethod
    def do_equal_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        # Extract the value from the parameter

        value = filter_feature.parameter.value

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        return [row for row in data if row.get(column_name) == value]

    @classmethod
    def do_regex_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        # Extract the value from the parameter

        value = filter_feature.parameter.value

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        # Compile regex pattern for efficiency
        compiled_pattern = re.compile(value)

        return [
            row
            for row in data
            if row.get(column_name) is not None and compiled_pattern.match(str(row.get(column_name)))
        ]

    @classmethod
    def do_categorical_inclusion_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        # Extract the values from the parameter

        values = filter_feature.parameter.values

        if values is None:
            raise ValueError(f"Filter parameter 'values' not found in {filter_feature.parameter}")

        # Convert to set for faster lookup
        allowed_set = set(values) if isinstance(values, (list, tuple)) else {values}

        return [row for row in data if row.get(column_name) in allowed_set]
