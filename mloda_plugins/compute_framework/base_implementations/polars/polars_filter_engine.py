from typing import Any
from mloda.core.abstract_plugins.components.contract.comparison_contract import ColumnSemantics
from mloda.core.abstract_plugins.components.mask.null_or_nan import split_null_or_nan
from mloda.provider import BaseFilterEngine
from mloda.user import SingleFilter
from mloda_plugins.compute_framework.base_implementations.polars import polars_type_semantics

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


class PolarsFilterEngine(BaseFilterEngine):
    provides_column_semantics = True

    @classmethod
    def final_filters(cls) -> bool:
        """Filters are applied after the feature calculation."""
        return True

    @classmethod
    def _column_semantics(cls, data: Any, column: str) -> ColumnSemantics:
        return polars_type_semantics.column_semantics(data, column)

    @classmethod
    def do_range_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        min_parameter, max_parameter, max_operator = cls.get_min_max_operator(filter_feature)

        if min_parameter is None or max_parameter is None:
            raise ValueError(f"Filter parameter {filter_feature.parameter} not supported")

        filter_feature_name = filter_feature.name

        if max_operator is True:
            return data.filter(
                (pl.col(filter_feature_name) >= min_parameter) & (pl.col(filter_feature_name) < max_parameter)
            )

        return data.filter(
            (pl.col(filter_feature_name) >= min_parameter) & (pl.col(filter_feature_name) <= max_parameter)
        )

    @classmethod
    def do_min_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        # Extract the value from the parameter
        value = filter_feature.parameter.value

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        dtype = data.collect_schema().get(column_name)
        return data.filter(polars_type_semantics.nan_as_null(pl.col(column_name), dtype) >= value)

    @classmethod
    def _apply_max_exclusive_filter(cls, data: Any, column_name: str, threshold: Any) -> Any:
        return data.filter(pl.col(column_name) < threshold)

    @classmethod
    def _apply_max_inclusive_filter(cls, data: Any, column_name: str, threshold: Any) -> Any:
        return data.filter(pl.col(column_name) <= threshold)

    @classmethod
    def do_equal_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        # Extract the value from the parameter
        value = filter_feature.parameter.value

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        return data.filter(pl.col(column_name) == value)

    @classmethod
    def do_regex_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        # Extract the value from the parameter
        value = filter_feature.parameter.value

        if value is None:
            raise ValueError(f"Filter parameter 'value' not found in {filter_feature.parameter}")

        return data.filter(pl.col(column_name).cast(pl.Utf8).str.contains(value))

    @classmethod
    def do_categorical_inclusion_filter(cls, data: Any, filter_feature: SingleFilter) -> Any:
        column_name = filter_feature.name

        # Extract the values from the parameter
        values = filter_feature.parameter.values

        if values is None:
            raise ValueError(f"Filter parameter 'values' not found in {filter_feature.parameter}")

        present, has_null_or_nan = split_null_or_nan(values)
        dtype = data.collect_schema().get(column_name)
        mask = pl.col(column_name).is_in(polars_type_semantics.is_in_values(present, dtype))
        if has_null_or_nan:
            mask = mask | polars_type_semantics.null_or_nan(pl.col(column_name), dtype)
        return data.filter(mask)
