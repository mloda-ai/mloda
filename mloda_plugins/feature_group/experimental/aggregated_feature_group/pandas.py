"""
Pandas implementation for aggregated feature groups.
"""

from __future__ import annotations

from typing import Any, Set, Type, Union

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup


class PandasAggregatedFeatureGroup(AggregatedFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        """Specify that this feature group works with Pandas."""
        return {PandasDataframe}

    @classmethod
    def _check_source_feature_exists(cls, data: Any, feature_name: str) -> None:
        """Check if the feature exists in the DataFrame."""
        if feature_name not in data.columns:
            raise ValueError(f"Source feature '{feature_name}' not found in data")

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """Add the result to the DataFrame."""
        data[feature_name] = result
        return data

    @classmethod
    def _perform_aggregation(cls, data: Any, aggregation_type: str, mloda_source_feature: str) -> Any:
        """
        Perform the aggregation using Pandas.

        Args:
            data: The Pandas DataFrame
            aggregation_type: The type of aggregation to perform
            mloda_source_feature: The name of the source feature to aggregate

        Returns:
            The result of the aggregation
        """
        if aggregation_type == "sum":
            return data[mloda_source_feature].sum()
        elif aggregation_type == "min":
            return data[mloda_source_feature].min()
        elif aggregation_type == "max":
            return data[mloda_source_feature].max()
        elif aggregation_type in ["avg", "mean"]:
            return data[mloda_source_feature].mean()
        elif aggregation_type == "count":
            return data[mloda_source_feature].count()
        elif aggregation_type == "std":
            return data[mloda_source_feature].std()
        elif aggregation_type == "var":
            return data[mloda_source_feature].var()
        elif aggregation_type == "median":
            return data[mloda_source_feature].median()
        else:
            raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
