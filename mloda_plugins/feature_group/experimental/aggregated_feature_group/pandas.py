"""
Pandas implementation for aggregated feature groups.
"""

from __future__ import annotations

from typing import Any, Set, Type, Union

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import BaseAggregatedFeatureGroup


class PandasAggregatedFeatureGroup(BaseAggregatedFeatureGroup):
    """
    Pandas implementation of aggregated feature group.

    Supports multiple aggregation types in a single class.
    """

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        """Specify that this feature group works with Pandas."""
        return {PandasDataframe}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Perform aggregations using Pandas.

        Processes all requested features, determining the aggregation type
        and source feature from each feature name.

        Adds the aggregated results directly to the input DataFrame.
        """
        # Process each requested feature
        for feature_name in features.get_all_names():
            aggregation_type = cls.get_aggregation_type(feature_name)
            source_feature = cls.get_source_feature(feature_name)

            if source_feature not in data.columns:
                raise ValueError(f"Source feature '{source_feature}' not found in data")

            if aggregation_type not in cls.AGGREGATION_TYPES:
                raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

            # Apply the appropriate aggregation function and add result to the DataFrame
            data[feature_name] = cls._perform_aggregation(data, aggregation_type, source_feature)

        # Return the modified DataFrame
        return data

    @classmethod
    def _perform_aggregation(cls, data: Any, aggregation_type: str, source_feature: str) -> Any:
        """
        Perform the aggregation using Pandas.

        Args:
            data: The Pandas DataFrame
            aggregation_type: The type of aggregation to perform
            source_feature: The name of the source feature to aggregate

        Returns:
            The result of the aggregation
        """
        if aggregation_type == "sum":
            return data[source_feature].sum()
        elif aggregation_type == "min":
            return data[source_feature].min()
        elif aggregation_type == "max":
            return data[source_feature].max()
        elif aggregation_type in ["avg", "mean"]:
            return data[source_feature].mean()
        elif aggregation_type == "count":
            return data[source_feature].count()
        elif aggregation_type == "std":
            return data[source_feature].std()
        elif aggregation_type == "var":
            return data[source_feature].var()
        elif aggregation_type == "median":
            return data[source_feature].median()
        else:
            raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
