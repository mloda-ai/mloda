"""
Base implementation for aggregated feature groups.
"""

from __future__ import annotations

from typing import Any, Optional, Set, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.feature_chain_parser import FeatureChainParser


class AggregatedFeatureGroup(AbstractFeatureGroup):
    """
    Base class for all aggregated feature groups.

    ## Feature Naming Convention

    Aggregated features follow this naming pattern:
    `{aggregation_type}_aggr__{mloda_source_feature}`

    The source feature (mloda_source_feature) is extracted from the feature name and used
    as input for the aggregation operation. Note the double underscore before the source feature.

    Examples:
    - `sum_aggr__sales`: Sum of sales values
    - `avg_aggr__temperature`: Average of temperature values
    - `max_aggr__price`: Maximum price value
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
        mloda_source_feature = self.mloda_source_feature(feature_name.name)
        return {Feature(mloda_source_feature)}

    # Define the prefix pattern for this feature group
    PREFIX_PATTERN = r"^([\w]+)_aggr__"

    @classmethod
    def get_aggregation_type(cls, feature_name: str) -> str:
        """Extract the aggregation type from the feature name."""
        prefix_part = FeatureChainParser.get_prefix_part(feature_name, cls.PREFIX_PATTERN)
        if prefix_part is None:
            raise ValueError(f"Invalid aggregated feature name format: {feature_name}")
        return prefix_part

    @classmethod
    def mloda_source_feature(cls, feature_name: str) -> str:
        """Extract the source feature name from the aggregated feature name."""
        return FeatureChainParser.extract_source_feature(feature_name, cls.PREFIX_PATTERN)

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
            # First validate that this is a valid feature name for this feature group
            if not FeatureChainParser.validate_feature_name(feature_name, cls.PREFIX_PATTERN):
                return False

            # Then check if the aggregation type is supported
            agg_type = cls.get_aggregation_type(feature_name)
            return cls._supports_aggregation_type(agg_type)
        except ValueError:
            return False

    @classmethod
    def _supports_aggregation_type(cls, aggregation_type: str) -> bool:
        """Check if this feature group supports the given aggregation type."""
        return aggregation_type in cls.AGGREGATION_TYPES

    @classmethod
    def _perform_aggregation(cls, data: Any, aggregation_type: str, mloda_source_feature: str) -> Any:
        """
        Method to perform the aggregation. Should be implemented by subclasses.

        Args:
            data: The input data
            aggregation_type: The type of aggregation to perform
            mloda_source_feature: The name of the source feature to aggregate

        Returns:
            The result of the aggregation
        """
        raise NotImplementedError(f"_perform_aggregation not implemented in {cls.__name__}")
