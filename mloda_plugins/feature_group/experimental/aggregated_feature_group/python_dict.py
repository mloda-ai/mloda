"""PythonDict implementation for aggregated feature groups."""

from __future__ import annotations

import statistics
from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda.user.python_dict import PythonDictFramework, row_count
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import is_missing
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup


class PythonDictAggregation(Implementation[AggregatedFeatureGroup]):
    """Aggregation over columnar dicts using the standard library."""

    feature_group = AggregatedFeatureGroup
    compute_framework = PythonDictFramework

    @classmethod
    def calculate(
        cls, group: type[AggregatedFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.aggregate(
            data,
            features,
            perform=cls._perform_aggregation,
            add_column=cls._add_result_to_data,
            columns=cls.compute_framework.extract_column_names,
        )

    @classmethod
    def _add_result_to_data(cls, data: dict[str, list[Any]], feature_name: str, result: Any) -> dict[str, list[Any]]:
        """Add a list as the column, or broadcast a scalar to every row."""
        data[feature_name] = list(result) if isinstance(result, list) else [result] * row_count(data)
        return data

    @classmethod
    def _perform_aggregation(cls, data: dict[str, list[Any]], aggregation_type: str, in_features: list[str]) -> Any:
        """Aggregate one column to a scalar, or several columns row-wise to a list."""
        if len(in_features) == 1:
            return cls._reduce(aggregation_type, data[in_features[0]])
        columns = [data[name] for name in in_features]
        return [cls._reduce(aggregation_type, list(cells)) for cells in zip(*columns)]

    @staticmethod
    def _reduce(aggregation_type: str, cells: list[Any]) -> Any:
        values = [cell for cell in cells if not is_missing(cell)]
        if aggregation_type == "count":
            return len(values)
        if aggregation_type not in ("sum", "min", "max", "avg", "mean", "median", "std", "var"):
            raise ValueError(f"Unsupported aggregation type: {aggregation_type}")
        if not values:
            return 0 if aggregation_type == "sum" else None
        if aggregation_type == "sum":
            return sum(values)
        if aggregation_type == "min":
            return min(values)
        if aggregation_type == "max":
            return max(values)
        if aggregation_type == "median":
            return statistics.median(values)
        if aggregation_type in ("avg", "mean"):
            return statistics.fmean(values)
        if len(values) < 2:
            return None
        return statistics.stdev(values) if aggregation_type == "std" else statistics.variance(values)
