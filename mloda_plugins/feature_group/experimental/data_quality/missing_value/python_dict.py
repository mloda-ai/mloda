"""PythonDict missing value imputation."""

from __future__ import annotations

import numbers
import statistics
from collections import Counter
from decimal import Decimal
from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation

from mloda.user.python_dict import PythonDictFramework, row_count
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup

_PLAIN_TYPES = frozenset({str, int, bool, bytes})
_NAT_TYPES = frozenset({("pandas", "NaTType"), ("numpy", "datetime64"), ("numpy", "timedelta64")})


class PythonDictMissingValueImputation(Implementation[MissingValueFeatureGroup]):
    """PythonDict missing value imputation."""

    feature_group = MissingValueFeatureGroup
    compute_framework = PythonDictFramework

    @classmethod
    def calculate(
        cls, group: type[MissingValueFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        if not data:
            raise ValueError("Data cannot be empty")
        return group.impute(
            data,
            features,
            perform=cls._perform_imputation,
            add_column=cls._add_result_to_data,
            columns=cls.compute_framework.extract_column_names,
        )

    @classmethod
    def _add_result_to_data(
        cls, data: dict[str, list[Any]], feature_name: str, result: list[Any]
    ) -> dict[str, list[Any]]:
        """Add the result to the data."""
        if len(result) != row_count(data):
            raise ValueError(f"Result length {len(result)} does not match data length {row_count(data)}")

        data[feature_name] = list(result)

        return data

    @classmethod
    def _nan_to_none(cls, data: dict[str, list[Any]], columns: list[str]) -> dict[str, list[Any]]:
        """Copy data with NaN-like cells replaced by None in the given columns."""
        normalized = dict(data)
        for name in columns:
            if name in data:
                normalized[name] = [None if cls._is_missing(v) else v for v in data[name]]
        return normalized

    @staticmethod
    def _is_missing(v: Any) -> bool:
        """Type-based check that never calls a user-defined comparison."""
        t = type(v)
        if v is None:
            return True
        if t in _PLAIN_TYPES:
            return False
        if t is float:
            return bool(v != v)
        module = t.__module__.split(".")[0]
        if module == "pandas" and t.__name__ == "NAType":
            return True
        if (module, t.__name__) in _NAT_TYPES:
            return bool(v != v)
        if isinstance(v, Decimal):
            return v.is_nan()
        return module in ("builtins", "numpy") and isinstance(v, numbers.Complex) and bool(v != v)

    @classmethod
    def _mode_of(cls, values: list[Any]) -> Any:
        """Smallest value among the most frequent ones."""
        counter = Counter(values)
        top = max(counter.values())
        return min((v for v, c in counter.items() if c == top), key=cls._mode_key)

    @staticmethod
    def _mode_key(v: Any) -> tuple[bool, str, Any]:
        real = isinstance(v, numbers.Real)
        return (not real, "" if real else type(v).__name__, v)

    @classmethod
    def _perform_imputation(
        cls,
        data: dict[str, list[Any]],
        imputation_method: str,
        in_features: list[str],
        constant_value: Any | None = None,
        group_by_features: list[str] | None = None,
    ) -> list[Any]:
        """
        Perform the imputation using pure Python operations.

        Supports both single-column and multi-column imputation:
        - Single column: [feature_name] - imputes values within the column
        - Multi-column: [feature~0, feature~1, ...] - imputes across columns

        Args:
            data: The List[Dict] data structure
            imputation_method: The type of imputation to perform
            in_features: List of resolved source feature names to impute
            constant_value: The constant value to use for imputation (if method is 'constant')
            group_by_features: Optional list of features to group by before imputation

        Returns:
            The result of the imputation as a list of values
        """
        data = cls._nan_to_none(data, in_features)
        # Handle single column case (backward compatibility)
        if len(in_features) == 1:
            source_feature = in_features[0]
            # Extract the source feature values
            source_values = list(data[source_feature])

            # If there are no missing values, return the original values
            if not any(value is None for value in source_values):
                return source_values

            # If group_by_features is provided, perform grouped imputation
            if group_by_features:
                return cls._perform_grouped_imputation(
                    data, imputation_method, source_feature, constant_value, group_by_features
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
        else:
            # Multi-column case: impute across columns (row-wise)
            # For multi-column features, we compute imputation values across columns for each row
            result = []
            for i in range(row_count(data)):
                row_values = [data[col][i] for col in in_features]

                if imputation_method == "mean":
                    non_null = [v for v in row_values if v is not None]
                    result.append(statistics.mean(non_null) if non_null else None)
                elif imputation_method == "median":
                    non_null = [v for v in row_values if v is not None]
                    result.append(statistics.median(non_null) if non_null else None)
                elif imputation_method == "mode":
                    non_null = [v for v in row_values if v is not None]
                    if non_null:
                        result.append(cls._mode_of(non_null))
                    else:
                        result.append(None)
                elif imputation_method == "constant":
                    result.append(constant_value)
                elif imputation_method == "sum":
                    non_null = [v for v in row_values if v is not None]
                    result.append(sum(non_null) if non_null else None)
                elif imputation_method == "min":
                    non_null = [v for v in row_values if v is not None]
                    result.append(min(non_null) if non_null else None)
                elif imputation_method == "max":
                    non_null = [v for v in row_values if v is not None]
                    result.append(max(non_null) if non_null else None)
                else:
                    raise ValueError(f"Unsupported imputation method for multi-column: {imputation_method}")

            return result

    @classmethod
    def _perform_grouped_imputation(
        cls,
        data: dict[str, list[Any]],
        imputation_method: str,
        in_features: str,  # Note: grouped imputation only supports single column
        constant_value: Any | None,
        group_by_features: list[str],
    ) -> list[Any]:
        """
        Perform imputation within groups.

        Args:
            data: The List[Dict] data structure
            imputation_method: The type of imputation to perform
            in_features: The name of the source feature to impute
            constant_value: The constant value to use for imputation (if method is 'constant')
            group_by_features: List of features to group by before imputation

        Returns:
            The result of the grouped imputation as a list of values
        """
        # Create groups based on group_by_features
        groups: dict[tuple[Any, ...], list[int]] = {}
        keyless: list[int] = []
        for i in range(row_count(data)):
            group_key = tuple(data[feature][i] for feature in group_by_features)
            if any(cls._is_missing(v) for v in group_key):
                keyless.append(i)
                continue
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(i)

        # Initialize result with original values
        result = list(data[in_features])

        if imputation_method == "constant":
            # Constant imputation is the same regardless of groups
            return cls._impute_constant(result, constant_value)

        # Calculate overall statistics for fallback
        non_null_values = [val for val in result if val is not None]
        overall = None
        if non_null_values:
            if imputation_method == "mean":
                overall = statistics.mean(non_null_values)
            elif imputation_method == "median":
                overall = statistics.median(non_null_values)
            elif imputation_method == "mode":
                overall = cls._mode_of(non_null_values)

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
                            result[i] = overall

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
                            result[i] = overall

            elif imputation_method == "mode":
                if group_non_null:
                    group_mode = cls._mode_of(group_non_null)
                    for i in group_indices:
                        if result[i] is None:
                            result[i] = group_mode
                else:
                    # Fall back to overall mode
                    for i in group_indices:
                        if result[i] is None:
                            result[i] = overall

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

        if imputation_method in ("mean", "median", "mode"):
            for i in keyless:
                if result[i] is None:
                    result[i] = overall

        return result

    @classmethod
    def _impute_mean(cls, values: list[Any]) -> list[Any]:
        """Impute missing values with the mean."""
        non_null_values = [val for val in values if val is not None]
        if not non_null_values:
            return values  # No non-null values to calculate mean

        mean_value = statistics.mean(non_null_values)
        return [mean_value if val is None else val for val in values]

    @classmethod
    def _impute_median(cls, values: list[Any]) -> list[Any]:
        """Impute missing values with the median."""
        non_null_values = [val for val in values if val is not None]
        if not non_null_values:
            return values  # No non-null values to calculate median

        median_value = statistics.median(non_null_values)
        return [median_value if val is None else val for val in values]

    @classmethod
    def _impute_mode(cls, values: list[Any]) -> list[Any]:
        """Impute missing values with the mode (most frequent value)."""
        non_null_values = [val for val in values if val is not None]
        if not non_null_values:
            return values  # No non-null values to calculate mode

        # Count frequencies
        mode_value = cls._mode_of(non_null_values)
        return [mode_value if val is None else val for val in values]

    @classmethod
    def _impute_constant(cls, values: list[Any], constant_value: Any) -> list[Any]:
        """Impute missing values with a constant value."""
        return [constant_value if val is None else val for val in values]

    @classmethod
    def _impute_ffill(cls, values: list[Any]) -> list[Any]:
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
    def _impute_bfill(cls, values: list[Any]) -> list[Any]:
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
