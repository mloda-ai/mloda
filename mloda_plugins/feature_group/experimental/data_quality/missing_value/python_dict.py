"""
PythonDict implementation for missing value imputation feature groups.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Type, Union

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork

from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup


class PythonDictMissingValueFeatureGroup(MissingValueFeatureGroup):
    """
    PythonDict implementation for missing value imputation feature groups.

    This implementation uses pure Python operations on List[Dict[str, Any]] data structures
    to perform missing value imputation without external dependencies.
    """

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PythonDictFramework}

    @classmethod
    def _check_source_feature_exists(cls, data: List[Dict[str, Any]], feature_name: str) -> None:
        """Check if the feature exists in the data."""
        if not data:
            raise ValueError("Data cannot be empty")

        # Check if feature exists in any row
        feature_exists = any(feature_name in row for row in data)
        if not feature_exists:
            raise ValueError(f"Source feature '{feature_name}' not found in data")

    @classmethod
    def _add_result_to_data(
        cls, data: List[Dict[str, Any]], feature_name: str, result: List[Any]
    ) -> List[Dict[str, Any]]:
        """Add the result to the data."""
        if len(result) != len(data):
            raise ValueError(f"Result length {len(result)} does not match data length {len(data)}")

        for i, row in enumerate(data):
            row[feature_name] = result[i]

        return data

    @classmethod
    def _perform_imputation(
        cls,
        data: List[Dict[str, Any]],
        imputation_method: str,
        mloda_source_feature: str,
        constant_value: Optional[Any] = None,
        group_by_features: Optional[List[str]] = None,
    ) -> List[Any]:
        """
        Perform the imputation using pure Python operations.

        Args:
            data: The List[Dict] data structure
            imputation_method: The type of imputation to perform
            mloda_source_feature: The name of the source feature to impute
            constant_value: The constant value to use for imputation (if method is 'constant')
            group_by_features: Optional list of features to group by before imputation

        Returns:
            The result of the imputation as a list of values
        """
        # Extract the source feature values
        source_values = [row.get(mloda_source_feature) for row in data]

        # If there are no missing values, return the original values
        if not any(value is None for value in source_values):
            return source_values

        # If group_by_features is provided, perform grouped imputation
        if group_by_features:
            return cls._perform_grouped_imputation(
                data, imputation_method, mloda_source_feature, constant_value, group_by_features
            )

        # Perform non-grouped imputation
        if imputation_method == "mean":
            return cls._impute_mean(source_values)
        elif imputation_method == "median":
            return cls._impute_median(source_values)
        elif imputation_method == "mode":
            return cls._impute_mode(source_values)
        elif imputation_method == "constant":
            return cls._impute_constant(source_values, constant_value)
        elif imputation_method == "ffill":
            return cls._impute_ffill(source_values)
        elif imputation_method == "bfill":
            return cls._impute_bfill(source_values)
        else:
            raise ValueError(f"Unsupported imputation method: {imputation_method}")

    @classmethod
    def _perform_grouped_imputation(
        cls,
        data: List[Dict[str, Any]],
        imputation_method: str,
        mloda_source_feature: str,
        constant_value: Optional[Any],
        group_by_features: List[str],
    ) -> List[Any]:
        """
        Perform imputation within groups.

        Args:
            data: The List[Dict] data structure
            imputation_method: The type of imputation to perform
            mloda_source_feature: The name of the source feature to impute
            constant_value: The constant value to use for imputation (if method is 'constant')
            group_by_features: List of features to group by before imputation

        Returns:
            The result of the grouped imputation as a list of values
        """
        # Create groups based on group_by_features
        groups: Dict[tuple[Any, ...], List[int]] = {}
        for i, row in enumerate(data):
            group_key = tuple(row.get(feature) for feature in group_by_features)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(i)

        # Initialize result with original values
        result = [row.get(mloda_source_feature) for row in data]

        if imputation_method == "constant":
            # Constant imputation is the same regardless of groups
            return cls._impute_constant(result, constant_value)

        # Calculate overall statistics for fallback
        non_null_values = [val for val in result if val is not None]
        overall_mean = statistics.mean(non_null_values) if non_null_values else None
        overall_median = statistics.median(non_null_values) if non_null_values else None
        overall_mode = None
        if non_null_values:
            mode_counter = Counter(non_null_values)
            overall_mode = mode_counter.most_common(1)[0][0] if mode_counter else None

        # Process each group
        for group_indices in groups.values():
            group_values = [result[i] for i in group_indices]
            group_non_null = [val for val in group_values if val is not None]

            if imputation_method == "mean":
                if group_non_null:
                    group_mean = statistics.mean(group_non_null)
                    for i in group_indices:
                        if result[i] is None:
                            result[i] = group_mean
                else:
                    # Fall back to overall mean
                    for i in group_indices:
                        if result[i] is None:
                            result[i] = overall_mean

            elif imputation_method == "median":
                if group_non_null:
                    group_median = statistics.median(group_non_null)
                    for i in group_indices:
                        if result[i] is None:
                            result[i] = group_median
                else:
                    # Fall back to overall median
                    for i in group_indices:
                        if result[i] is None:
                            result[i] = overall_median

            elif imputation_method == "mode":
                if group_non_null:
                    mode_counter = Counter(group_non_null)
                    group_mode = mode_counter.most_common(1)[0][0] if mode_counter else overall_mode
                    for i in group_indices:
                        if result[i] is None:
                            result[i] = group_mode
                else:
                    # Fall back to overall mode
                    for i in group_indices:
                        if result[i] is None:
                            result[i] = overall_mode

            elif imputation_method == "ffill":
                # Forward fill within group
                last_valid = None
                for i in group_indices:
                    if result[i] is not None:
                        last_valid = result[i]
                    elif last_valid is not None:
                        result[i] = last_valid

            elif imputation_method == "bfill":
                # Backward fill within group
                # First pass: find next valid values
                next_valid = None
                for i in reversed(group_indices):
                    if result[i] is not None:
                        next_valid = result[i]
                    elif next_valid is not None:
                        result[i] = next_valid

        return result

    @classmethod
    def _impute_mean(cls, values: List[Any]) -> List[Any]:
        """Impute missing values with the mean."""
        non_null_values = [val for val in values if val is not None]
        if not non_null_values:
            return values  # No non-null values to calculate mean

        mean_value = statistics.mean(non_null_values)
        return [mean_value if val is None else val for val in values]

    @classmethod
    def _impute_median(cls, values: List[Any]) -> List[Any]:
        """Impute missing values with the median."""
        non_null_values = [val for val in values if val is not None]
        if not non_null_values:
            return values  # No non-null values to calculate median

        median_value = statistics.median(non_null_values)
        return [median_value if val is None else val for val in values]

    @classmethod
    def _impute_mode(cls, values: List[Any]) -> List[Any]:
        """Impute missing values with the mode (most frequent value)."""
        non_null_values = [val for val in values if val is not None]
        if not non_null_values:
            return values  # No non-null values to calculate mode

        # Count frequencies
        counter = Counter(non_null_values)
        mode_value = counter.most_common(1)[0][0]
        return [mode_value if val is None else val for val in values]

    @classmethod
    def _impute_constant(cls, values: List[Any], constant_value: Any) -> List[Any]:
        """Impute missing values with a constant value."""
        return [constant_value if val is None else val for val in values]

    @classmethod
    def _impute_ffill(cls, values: List[Any]) -> List[Any]:
        """Impute missing values with forward fill."""
        result = values.copy()
        last_valid = None

        for i, val in enumerate(result):
            if val is not None:
                last_valid = val
            elif last_valid is not None:
                result[i] = last_valid

        return result

    @classmethod
    def _impute_bfill(cls, values: List[Any]) -> List[Any]:
        """Impute missing values with backward fill."""
        result = values.copy()
        next_valid = None

        # Iterate backwards
        for i in range(len(result) - 1, -1, -1):
            if result[i] is not None:
                next_valid = result[i]
            elif next_valid is not None:
                result[i] = next_valid

        return result
