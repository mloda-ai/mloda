"""
Base implementation for aggregated feature groups.
"""

from __future__ import annotations

from typing import Any, Optional, Set, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options


class BaseAggregatedFeatureGroup(AbstractFeatureGroup):
    """
    Base class for all aggregated feature groups.

    Follows naming convention: {aggregation_type}_aggr_{source_feature_name}
    Examples: sum_aggr_sales, avg_aggr_temperature, max_aggr_price
    """

    # Define supported aggregation types
    AGGREGATION_TYPES = {
        "sum": "Sum of values",
        "min": "Minimum value",
        "max": "Maximum value",
        "avg": "Average (mean) of values",
        "mean": "Average (mean) of values",
        "count": "Count of non-null values",
        "std": "Standard deviation of values",
        "var": "Variance of values",
        "median": "Median value",
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source feature from the aggregated feature name."""
        source_feature = self.get_source_feature(feature_name.name)
        return {Feature(source_feature)}

    @classmethod
    def get_aggregation_type(cls, feature_name: str) -> str:
        """Extract the aggregation type from the feature name."""
        parts = feature_name.split("_aggr_")
        if len(parts) != 2 or not parts[0]:
            raise ValueError(f"Invalid aggregated feature name format: {feature_name}")
        if not parts[1]:
            raise ValueError(f"Invalid aggregated feature name format: {feature_name}")
        return parts[0]

    @classmethod
    def get_source_feature(cls, feature_name: str) -> str:
        """Extract the source feature name from the aggregated feature name."""
        parts = feature_name.split("_aggr_")
        if len(parts) != 2 or not parts[1]:
            raise ValueError(f"Invalid aggregated feature name format: {feature_name}")
        if not parts[0]:
            raise ValueError(f"Invalid aggregated feature name format: {feature_name}")
        return parts[1]

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        """Check if feature name matches the expected pattern and aggregation type."""
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name

        try:
            agg_type = cls.get_aggregation_type(feature_name)
            return cls._supports_aggregation_type(agg_type)
        except ValueError:
            return False

    @classmethod
    def _supports_aggregation_type(cls, aggregation_type: str) -> bool:
        """Check if this feature group supports the given aggregation type."""
        return aggregation_type in cls.AGGREGATION_TYPES

    @classmethod
    def _perform_aggregation(cls, data: Any, aggregation_type: str, source_feature: str) -> Any:
        """
        Method to perform the aggregation. Should be implemented by subclasses.

        Args:
            data: The input data
            aggregation_type: The type of aggregation to perform
            source_feature: The name of the source feature to aggregate

        Returns:
            The result of the aggregation
        """
        raise NotImplementedError(f"_perform_aggregation not implemented in {cls.__name__}")
