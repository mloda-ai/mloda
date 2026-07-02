from typing import Any
import pyarrow as pa
import pyarrow.compute as pc

from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda.provider import BaseFilterEngine
from mloda.user import SingleFilter
from mloda_plugins.compute_framework.base_implementations.pyarrow import pyarrow_type_semantics


class PyArrowFilterEngine(BaseFilterEngine):
    @classmethod
    def final_filters(cls) -> bool:
        """Filters are applied after the feature calculation."""
        return True

    @classmethod
    def _column_semantics(cls, data: Any, column: str) -> ColumnSemantics:
        return pyarrow_type_semantics.column_semantics(data, column)

    @classmethod
    def do_range_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        min_parameter, max_parameter, max_operator = cls.get_min_max_operator(filter_feature)

        if min_parameter is None or max_parameter is None:
            raise ValueError(f"Filter parameter {filter_feature.parameter} not supported")

        column_name = filter_feature.name

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
        column_name = filter_feature.name

        # Extract the value from the parameter
        value = filter_feature.parameter.value

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        # Create boolean mask for min filter
        mask = pc.greater_equal(data[column_name], value)
        return data.filter(mask)

    @classmethod
    def _apply_max_exclusive_filter(cls, data: Any, column_name: str, threshold: Any) -> Any:
        mask = pc.less(data[column_name], threshold)
        return data.filter(mask)

    @classmethod
    def _apply_max_inclusive_filter(cls, data: Any, column_name: str, threshold: Any) -> Any:
        mask = pc.less_equal(data[column_name], threshold)
        return data.filter(mask)

    @classmethod
    def do_equal_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        # Extract the value from the parameter
        value = filter_feature.parameter.value

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        # Create boolean mask for equal filter
        mask = pc.equal(data[column_name], value)
        return data.filter(mask)

    @classmethod
    def do_regex_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        # Extract the value from the parameter
        value = filter_feature.parameter.value

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        # Convert column to string type for regex matching
        column = data[column_name]
        # Apply regex filter directly
        mask = pc.match_substring_regex(column, value)
        return data.filter(mask)

    @classmethod
    def do_categorical_inclusion_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        # Extract the values from the parameter
        values = filter_feature.parameter.values

        if values is None:
            raise ValueError(f"Filter parameter 'values' not found in {filter_feature.parameter}")

        # Create PyArrow array from the values
        values_array = pa.array(values)
        # Apply is_in filter
        mask = pc.is_in(data[column_name], values_array)
        return data.filter(mask)
