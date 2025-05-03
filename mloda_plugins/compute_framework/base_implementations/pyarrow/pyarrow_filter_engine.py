from typing import Any
import pyarrow as pa
import pyarrow.compute as pc

from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_core.filter.single_filter import SingleFilter


class PyArrowFilterEngine(BaseFilterEngine):
    @classmethod
    def final_filters(cls) -> bool:
        """Filters are applied after the feature calculation."""
        return True

    @classmethod
    def do_range_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        min_parameter, max_parameter, max_operator = cls.get_min_max_operator(filter_feature)

        if min_parameter is None or max_parameter is None:
            raise ValueError(f"Filter parameter {filter_feature.parameter} not supported")

        # Get the string name from the FeatureName object
        column_name = str(filter_feature.name)

        # Create boolean masks using PyArrow compute
        min_mask = pc.greater_equal(data[column_name], min_parameter)

        if max_operator is True:
            max_mask = pc.less(data[column_name], max_parameter)
        else:
            max_mask = pc.less_equal(data[column_name], max_parameter)

        # Combine masks and filter the table
        mask = pc.and_(min_mask, max_mask)
        return data.filter(mask)

    @classmethod
    def do_min_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        # Get the string name from the FeatureName object
        column_name = str(filter_feature.name)

        # Extract the value from the parameter
        value = None
        for param in filter_feature.parameter:
            if param[0] == "value":
                value = param[1]
                break

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        # Create boolean mask for min filter
        mask = pc.greater_equal(data[column_name], value)
        return data.filter(mask)

    @classmethod
    def do_max_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        # Get the string name from the FeatureName object
        column_name = str(filter_feature.name)

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
                mask = pc.less(data[column_name], max_parameter)
            else:
                mask = pc.less_equal(data[column_name], max_parameter)

            return data.filter(mask)
        elif has_value:
            # Simple parameter - extract the value
            value = None
            for param in filter_feature.parameter:
                if param[0] == "value":
                    value = param[1]
                    break

            if value is None:
                raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

            # Simple max filter
            mask = pc.less_equal(data[column_name], value)
            return data.filter(mask)
        else:
            raise ValueError(f"No valid filter parameter found in {filter_feature.parameter}")

    @classmethod
    def do_equal_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        # Get the string name from the FeatureName object
        column_name = str(filter_feature.name)

        # Extract the value from the parameter
        value = None
        for param in filter_feature.parameter:
            if param[0] == "value":
                value = param[1]
                break

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        # Create boolean mask for equal filter
        mask = pc.equal(data[column_name], value)
        return data.filter(mask)

    @classmethod
    def do_regex_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        # Get the string name from the FeatureName object
        column_name = str(filter_feature.name)

        # Extract the value from the parameter
        value = None
        for param in filter_feature.parameter:
            if param[0] == "value":
                value = param[1]
                break

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        # Convert column to string type for regex matching
        column = data[column_name]
        # Apply regex filter directly
        mask = pc.match_substring_regex(column, value)
        return data.filter(mask)

    @classmethod
    def do_categorical_inclusion_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        # Get the string name from the FeatureName object
        column_name = str(filter_feature.name)

        # Extract the values from the parameter
        values = None
        for param in filter_feature.parameter:
            if param[0] == "values":
                values = param[1]
                break

        if values is None:
            raise ValueError(f"Filter parameter 'values' not found in {filter_feature.parameter}")

        # Create PyArrow array from the values
        values_array = pa.array(values)
        # Apply is_in filter
        mask = pc.is_in(data[column_name], values_array)
        return data.filter(mask)
