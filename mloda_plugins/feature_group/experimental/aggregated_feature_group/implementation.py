"""
Abstract helper for the aggregation implementations: the shared per-feature loop.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from mloda.provider import COLUMN_DISCOVERY_HOOKS, FeatureSet, Implementation
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup


class AggregationImplementation(Implementation):
    """Names no pair; a subclass sets feature_group, compute_framework and the column hooks."""

    REQUIRED_COLUMNWISE_HOOKS = COLUMN_DISCOVERY_HOOKS

    @classmethod
    def calculate(cls, group: type[AggregatedFeatureGroup], data: Any, features: FeatureSet) -> Any:
        """Aggregate every requested feature and add the result to the data."""
        for feature in features.get_sorted_features():
            aggregation_type, source_feature_name = group._extract_aggr_and_source_feature(feature)

            # A multi-column source such as "onehot_encoded__product" resolves to its "~N" columns.
            available_columns = cls._get_available_columns(data)
            resolved_columns = group.resolve_multi_column_feature(source_feature_name, available_columns)

            cls._check_source_features_exist(data, resolved_columns)

            if aggregation_type not in group.AGGREGATION_TYPES:
                raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

            result = cls._perform_aggregation(data, aggregation_type, resolved_columns)

            data = cls._add_result_to_data(data, feature.name, result)

        return data

    @classmethod
    @abstractmethod
    def _get_available_columns(cls, data: Any) -> set[str]:
        """The column names of the data."""

    @classmethod
    @abstractmethod
    def _check_source_features_exist(cls, data: Any, feature_names: list[str]) -> None:
        """Raise when none of the resolved source columns exist."""

    @classmethod
    @abstractmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """Add the aggregation result as a column."""

    @classmethod
    @abstractmethod
    def _perform_aggregation(cls, data: Any, aggregation_type: str, in_features: list[str]) -> Any:
        """Aggregate one column vertically, or several columns row-wise."""
