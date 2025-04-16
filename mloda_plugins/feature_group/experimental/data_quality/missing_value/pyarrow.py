"""
PyArrow implementation for missing value imputation feature groups.
"""

from __future__ import annotations

from typing import Any, List, Optional, Set, Type, Union

import pyarrow as pa
import pyarrow.compute as pc

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork

from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup


class PyArrowMissingValueFeatureGroup(MissingValueFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PyarrowTable}

    @classmethod
    def _check_source_feature_exists(cls, data: pa.Table, feature_name: str) -> None:
        """Check if the feature exists in the Table."""
        if feature_name not in data.schema.names:
            raise ValueError(f"Source feature '{feature_name}' not found in data")

    @classmethod
    def _add_result_to_data(cls, data: pa.Table, feature_name: str, result: Any) -> pa.Table:
        """Add the result to the Table."""
        return data.append_column(feature_name, result)

    @classmethod
    def _perform_imputation(
        cls,
        data: pa.Table,
        imputation_method: str,
        mloda_source_feature: str,
        constant_value: Optional[Any] = None,
        group_by_features: Optional[List[str]] = None,
    ) -> pa.Array:
        """
        Perform the imputation using PyArrow compute functions.

        Args:
            data: The PyArrow Table
            imputation_method: The type of imputation to perform
            mloda_source_feature: The name of the source feature to impute
            constant_value: The constant value to use for imputation (if method is 'constant')
            group_by_features: Optional list of features to group by before imputation

        Returns:
            The result of the imputation as a PyArrow Array
        """
        # Get the source column
        source_column = data.column(mloda_source_feature)

        # If there are no missing values, return the original column
        if pc.count(pc.is_null(source_column)).as_py() == 0:
            return source_column

        # If group_by_features is provided, perform grouped imputation
        if group_by_features:
            return cls._perform_grouped_imputation(
                data, imputation_method, mloda_source_feature, constant_value, group_by_features
            )

        # Perform non-grouped imputation
        if imputation_method == "mean":
            fill_value = pc.mean(source_column).as_py()
            return pc.fill_null(source_column, fill_value)
        elif imputation_method == "median":
            # PyArrow doesn't have a direct median function
            # We can approximate it using quantile with q=0.5
            result = pc.quantile(source_column, q=0.5)
            fill_value = result[0].as_py() if len(result) > 0 else None
            return pc.fill_null(source_column, fill_value)
        elif imputation_method == "mode":
            # PyArrow doesn't have a direct mode function
            # We need to compute the mode manually
            value_counts = pc.value_counts(source_column)
            if len(value_counts) > 0:
                # Find the index with the maximum count
                counts = value_counts.field("counts")
                max_count = pc.max(counts).as_py()

                # Find all indices where count equals max_count
                max_indices = []
                for i in range(len(counts)):
                    if counts[i].as_py() == max_count:
                        max_indices.append(i)

                # Use the first index with maximum count
                if max_indices:
                    mode_value = value_counts.field("values")[max_indices[0]].as_py()
                    return pc.fill_null(source_column, mode_value)

            return source_column
        elif imputation_method == "constant":
            return pc.fill_null(source_column, constant_value)
        elif imputation_method == "ffill":
            # Forward fill implementation
            return cls._perform_fill_direction(source_column, "forward")
        elif imputation_method == "bfill":
            # Backward fill implementation
            return cls._perform_fill_direction(source_column, "backward")
        else:
            raise ValueError(f"Unsupported imputation method: {imputation_method}")

    @classmethod
    def _perform_grouped_imputation(
        cls,
        data: pa.Table,
        imputation_method: str,
        mloda_source_feature: str,
        constant_value: Optional[Any],
        group_by_features: List[str],
    ) -> pa.Array:
        """
        Perform imputation within groups.

        Args:
            data: The PyArrow Table
            imputation_method: The type of imputation to perform
            source_feature: The name of the source feature to impute
            constant_value: The constant value to use for imputation (if method is 'constant')
            group_by_features: List of features to group by before imputation

        Returns:
            The result of the grouped imputation as a PyArrow Array
        """
        # Get the source column
        source_column = data.column(mloda_source_feature)

        if imputation_method == "constant":
            # Constant imputation is the same regardless of groups
            return pc.fill_null(source_column, constant_value)

        # Calculate the overall imputation value to use as fallback
        overall_value = None
        if imputation_method == "mean":
            overall_value = pc.mean(source_column).as_py()
        elif imputation_method == "median":
            result = pc.quantile(source_column, q=0.5)
            overall_value = result[0].as_py() if len(result) > 0 else None
        elif imputation_method == "mode":
            value_counts = pc.value_counts(source_column)
            if len(value_counts) > 0:
                # Find the index with the maximum count
                counts = value_counts.field("counts")
                max_count = pc.max(counts).as_py()

                # Find all indices where count equals max_count
                max_indices = []
                for i in range(len(counts)):
                    if counts[i].as_py() == max_count:
                        max_indices.append(i)

                # Use the first index with maximum count
                if max_indices:
                    overall_value = value_counts.field("values")[max_indices[0]].as_py()

        # Create a list to store the results
        results = []

        # For each row, determine the group and apply the appropriate imputation
        for i in range(data.num_rows):
            # Get the value for this row
            value = source_column[i].as_py()

            # If the value is not null, keep it as is
            if value is not None:
                results.append(value)
                continue

            # Get the group key for this row
            group_key = tuple(data[group_feature][i].as_py() for group_feature in group_by_features)

            # Create a mask for this group
            group_masks = []
            for j, group_feature in enumerate(group_by_features):
                group_masks.append(pc.equal(data[group_feature], pa.scalar(group_key[j])))

            group_mask = group_masks[0]
            for mask in group_masks[1:]:
                group_mask = pc.and_(group_mask, mask)

            # Get the values for this group
            group_data = pc.filter(source_column, group_mask)

            # Calculate the imputation value for this group
            group_value = None
            if imputation_method == "mean":
                group_value = pc.mean(group_data).as_py()
            elif imputation_method == "median":
                result = pc.quantile(group_data, q=0.5)
                group_value = result[0].as_py() if len(result) > 0 else None
            elif imputation_method == "mode":
                value_counts = pc.value_counts(group_data)
                if len(value_counts) > 0:
                    # Find the index with the maximum count
                    counts = value_counts.field("counts")
                    max_count = pc.max(counts).as_py()

                    # Find all indices where count equals max_count
                    max_indices = []
                    for i in range(len(counts)):
                        if counts[i].as_py() == max_count:
                            max_indices.append(i)

                    # Use the first index with maximum count
                    if max_indices:
                        group_value = value_counts.field("values")[max_indices[0]].as_py()
            elif imputation_method == "ffill":
                # For ffill, we need to find the last non-null value before this row in the group
                valid_indices = pc.indices_nonzero(pc.is_valid(group_data))
                if len(valid_indices) > 0:
                    # Find the largest valid index that is less than the current index
                    valid_indices_before = [idx for idx in valid_indices.to_pylist() if idx < i]
                    if valid_indices_before:
                        last_valid_idx = max(valid_indices_before)
                        group_value = group_data[last_valid_idx].as_py()
            elif imputation_method == "bfill":
                # For bfill, we need to find the first non-null value after this row in the group
                valid_indices = pc.indices_nonzero(pc.is_valid(group_data))
                if len(valid_indices) > 0:
                    # Find the smallest valid index that is greater than the current index
                    valid_indices_after = [idx for idx in valid_indices.to_pylist() if idx > i]
                    if valid_indices_after:
                        next_valid_idx = min(valid_indices_after)
                        group_value = group_data[next_valid_idx].as_py()

            # If the group imputation value is None, fall back to the overall value
            if group_value is None:
                results.append(overall_value)
            else:
                results.append(group_value)

        # Convert the results to a PyArrow array
        return pa.array(results)

    @classmethod
    def _perform_fill_direction(cls, column: pa.Array, direction: str) -> pa.Array:
        """
        Perform forward or backward fill on a column.

        Args:
            column: The PyArrow Array to fill
            direction: The direction to fill ('forward' or 'backward')

        Returns:
            The filled PyArrow Array
        """
        # Convert to Python list for easier manipulation
        values = column.to_pylist()

        if direction == "forward":
            # Forward fill
            last_valid = None
            for i in range(len(values)):
                if values[i] is not None:
                    last_valid = values[i]
                elif last_valid is not None:
                    values[i] = last_valid
        elif direction == "backward":
            # Backward fill
            last_valid = None
            for i in range(len(values) - 1, -1, -1):
                if values[i] is not None:
                    last_valid = values[i]
                elif last_valid is not None:
                    values[i] = last_valid

        # Convert back to PyArrow array
        return pa.array(values)
