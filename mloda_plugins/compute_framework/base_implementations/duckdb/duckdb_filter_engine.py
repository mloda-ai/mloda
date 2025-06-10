from typing import Any
from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_core.filter.single_filter import SingleFilter

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore


class DuckDBFilterEngine(BaseFilterEngine):
    @classmethod
    def final_filters(cls) -> bool:
        """Filters are applied after the feature calculation."""
        return True

    @classmethod
    def do_range_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        min_parameter, max_parameter, max_operator = cls.get_min_max_operator(filter_feature)

        if min_parameter is None or max_parameter is None:
            raise ValueError(f"Filter parameter {filter_feature.parameter} not supported")

        filter_feature_name = filter_feature.name.name

        if max_operator is True:
            condition = f'"{filter_feature_name}" >= {min_parameter} AND "{filter_feature_name}" < {max_parameter}'
        else:
            condition = f'"{filter_feature_name}" >= {min_parameter} AND "{filter_feature_name}" <= {max_parameter}'

        # Use DuckDB relation's filter method to stay lazy
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

        condition = f'"{column_name}" >= {value}'

        return data.filter(condition)

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
                condition = f'"{column_name}" < {max_parameter}'
            else:
                condition = f'"{column_name}" <= {max_parameter}'
        elif has_value:
            # Simple parameter - extract the value
            value = None
            for param in filter_feature.parameter:
                if param[0] == "value":
                    value = param[1]
                    break

            if value is None:
                raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

            condition = f'"{column_name}" <= {value}'
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

        # Handle string values with proper quoting
        if isinstance(value, str):
            condition = f"\"{column_name}\" = '{value}'"
        else:
            condition = f'"{column_name}" = {value}'

        return data.filter(condition)

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

        # Use DuckDB's regexp_matches function for regex filtering
        condition = f"regexp_matches(\"{column_name}\", '{value}')"

        return data.filter(condition)

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

        # Build IN clause with proper quoting for strings
        if values and isinstance(values[0], str):
            values_str = ", ".join(f"'{v}'" for v in values)
        else:
            values_str = ", ".join(str(v) for v in values)

        condition = f'"{column_name}" IN ({values_str})'

        return data.filter(condition)
