"""
Polars Lazy implementation for aggregated feature groups.
"""

from __future__ import annotations

from typing import Any, Set, Type, Union

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork

from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataframe
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore


class PolarsLazyAggregatedFeatureGroup(AggregatedFeatureGroup):
    """
    Polars Lazy implementation of aggregated feature group.

    This implementation leverages Polars' lazy evaluation capabilities to optimize
    aggregation operations through query planning and deferred execution.
    """

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        """Specify that this feature group works with Polars Lazy DataFrames."""
        return {PolarsLazyDataframe}

    @classmethod
    def _check_source_feature_exists(cls, data: Any, feature_name: str) -> None:
        """Check if the feature exists in the LazyFrame schema."""
        if hasattr(data, "collect_schema"):
            schema_names = set(data.collect_schema().names())
            if feature_name not in schema_names:
                raise ValueError(f"Source feature '{feature_name}' not found in data")
        else:
            raise ValueError("Data does not have a collect_schema method, cannot check feature existence.")

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """Add the result to the LazyFrame using with_columns."""
        # The result is already a Polars expression, so we can use it directly
        return data.with_columns(result.alias(feature_name))

    @classmethod
    def _perform_aggregation(cls, data: Any, aggregation_type: str, mloda_source_feature: str) -> Any:
        """
        Perform the aggregation using Polars lazy expressions.

        Args:
            data: The Polars LazyFrame
            aggregation_type: The type of aggregation to perform
            mloda_source_feature: The name of the source feature to aggregate

        Returns:
            A Polars expression representing the aggregation
        """
        if pl is None:
            raise ImportError("Polars is not installed. To be able to use this framework, please install polars.")

        # Get the column to aggregate
        column = pl.col(mloda_source_feature)

        # Return the aggregation expression based on type
        if aggregation_type == "sum":
            return column.sum()
        elif aggregation_type == "min":
            return column.min()
        elif aggregation_type == "max":
            return column.max()
        elif aggregation_type in ["avg", "mean"]:
            return column.mean()
        elif aggregation_type == "count":
            return column.count()
        elif aggregation_type == "std":
            return column.std()
        elif aggregation_type == "var":
            return column.var()
        elif aggregation_type == "median":
            return column.median()
        else:
            raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
