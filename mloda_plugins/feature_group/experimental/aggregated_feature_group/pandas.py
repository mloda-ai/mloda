"""
Pandas implementation for aggregated feature groups.
"""

from __future__ import annotations

from typing import Any

from mloda.provider import FeatureSet, Implementation
from mloda.user.pandas import PandasDataFrame
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup


class PandasAggregation(Implementation):
    feature_group = AggregatedFeatureGroup
    compute_framework = PandasDataFrame
    divergent = {
        **{op: f"skips NaN in {op}" for op in ("sum", "avg", "mean", "std", "var")},
        "count": "does not count NaN",
    }

    @classmethod
    def calculate(cls, group: type[AggregatedFeatureGroup], data: Any, features: FeatureSet, framework: Any) -> Any:
        return group.aggregate(
            data,
            features,
            perform=cls._perform_aggregation,
            add_column=cls._add_result_to_data,
            columns=cls.compute_framework.extract_column_names,
        )

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """Add the result to the DataFrame."""
        data[feature_name] = result
        return data

    @classmethod
    def _perform_aggregation(cls, data: Any, aggregation_type: str, in_features: list[str]) -> Any:
        """
        Perform the aggregation using Pandas.

        Supports both single-column and multi-column aggregation:
        - Single column: aggregates values within the column (returns scalar)
        - Multi-column: aggregates across columns row-wise (returns Series)

        Args:
            data: The Pandas DataFrame
            aggregation_type: The type of aggregation to perform
            in_features: List of source feature names (may be single or multiple columns)

        Returns:
            The result of the aggregation (Series for multi-column, scalar for single-column)
        """
        # Select the columns to aggregate
        # Note: data[[col]] returns DataFrame with 1 column, data[col] returns Series
        if len(in_features) == 1:
            # Single column: extract as Series for simpler aggregation
            selected_data = data[in_features[0]]
        else:
            # Multiple columns: keep as DataFrame
            selected_data = data[in_features]

        # For multi-column features, aggregate across columns (axis=1)
        # For single-column features, aggregate within the Series
        if len(in_features) > 1:
            # Multi-column: selected_data is DataFrame, aggregate across columns
            if aggregation_type == "sum":
                return selected_data.sum(axis=1)
            elif aggregation_type == "min":
                return selected_data.min(axis=1)
            elif aggregation_type == "max":
                return selected_data.max(axis=1)
            elif aggregation_type in ["avg", "mean"]:
                return selected_data.mean(axis=1)
            elif aggregation_type == "count":
                return selected_data.count(axis=1)
            elif aggregation_type == "std":
                return selected_data.std(axis=1)
            elif aggregation_type == "var":
                return selected_data.var(axis=1)
            elif aggregation_type == "median":
                return selected_data.median(axis=1)
            else:
                raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
        else:
            # Single column: selected_data is Series, aggregate within column
            if aggregation_type == "sum":
                return selected_data.sum()
            elif aggregation_type == "min":
                return selected_data.min()
            elif aggregation_type == "max":
                return selected_data.max()
            elif aggregation_type in ["avg", "mean"]:
                return selected_data.mean()
            elif aggregation_type == "count":
                return selected_data.count()
            elif aggregation_type == "std":
                return selected_data.std()
            elif aggregation_type == "var":
                return selected_data.var()
            elif aggregation_type == "median":
                return selected_data.median()
            else:
                raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
