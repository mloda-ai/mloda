"""
Polars Lazy implementation for aggregated feature groups.
"""

from __future__ import annotations

from typing import Any, List, Set, Type, Union

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork

from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


class PolarsLazyAggregatedFeatureGroup(AggregatedFeatureGroup):
    """
    Polars Lazy implementation of aggregated feature group.

    This implementation leverages Polars' lazy evaluation capabilities to optimize
    aggregation operations through query planning and deferred execution.
    """

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        """Specify that this feature group works with Polars Lazy DataFrames."""
        return {PolarsLazyDataFrame}

    @classmethod
    def _get_available_columns(cls, data: Any) -> Set[str]:
        """Get the set of available column names from the LazyFrame schema."""
        if hasattr(data, "collect_schema"):
            return set(data.collect_schema().names())
        else:
            raise ValueError("Data does not have a collect_schema method, cannot get available columns.")

    @classmethod
    def _check_source_features_exist(cls, data: Any, feature_names: List[str]) -> None:
        """
        Check if the resolved features exist in the LazyFrame schema.

        Args:
            data: The Polars LazyFrame
            feature_names: List of resolved feature names (may contain ~N suffixes)

        Raises:
            ValueError: If none of the resolved features exist in the data
        """
        if hasattr(data, "collect_schema"):
            schema_names = set(data.collect_schema().names())
            missing_features = [name for name in feature_names if name not in schema_names]
            if len(missing_features) == len(feature_names):
                raise ValueError(
                    f"None of the source features {feature_names} found in data. "
                    f"Available columns: {list(schema_names)}"
                )
        else:
            raise ValueError("Data does not have a collect_schema method, cannot check feature existence.")

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """Add the result to the LazyFrame using with_columns."""
        # The result is already a Polars expression, so we can use it directly
        return data.with_columns(result.alias(feature_name))

    @classmethod
    def _perform_aggregation(cls, data: Any, aggregation_type: str, in_features: List[str]) -> Any:
        """
        Perform the aggregation using Polars lazy expressions.

        Supports both single-column and multi-column aggregation:
        - Single column: aggregates values within the column
        - Multi-column: aggregates across columns using horizontal operations

        Args:
            data: The Polars LazyFrame
            aggregation_type: The type of aggregation to perform
            in_features: List of source feature names (may be single or multiple columns)

        Returns:
            A Polars expression representing the aggregation
        """
        if pl is None:
            raise ImportError("Polars is not installed. To be able to use this framework, please install polars.")

        # For multi-column features, use horizontal aggregation (across columns)
        # For single-column features, use vertical aggregation (within column)
        if len(in_features) > 1:
            # Multi-column: aggregate across columns horizontally
            columns = [pl.col(name) for name in in_features]
            if aggregation_type == "sum":
                return pl.sum_horizontal(*columns)
            elif aggregation_type == "min":
                return pl.min_horizontal(*columns)
            elif aggregation_type == "max":
                return pl.max_horizontal(*columns)
            elif aggregation_type in ["avg", "mean"]:
                return pl.mean_horizontal(*columns)
            elif aggregation_type == "count":
                # Count non-null values across columns
                return pl.sum_horizontal(*[col.is_not_null().cast(pl.Int64) for col in columns])
            elif aggregation_type == "std":
                # Polars doesn't have horizontal std, compute manually
                mean_val = pl.mean_horizontal(*columns)
                variance = pl.mean_horizontal(*[(col - mean_val).pow(2) for col in columns])
                return variance.sqrt()
            elif aggregation_type == "var":
                # Polars doesn't have horizontal var, compute manually
                mean_val = pl.mean_horizontal(*columns)
                return pl.mean_horizontal(*[(col - mean_val).pow(2) for col in columns])
            elif aggregation_type == "median":
                # Polars doesn't have horizontal median, use concat and median
                raise ValueError("Median aggregation across multiple columns is not supported in Polars")
            else:
                raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
        else:
            # Single column: vertical aggregation
            column = pl.col(in_features[0])
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
