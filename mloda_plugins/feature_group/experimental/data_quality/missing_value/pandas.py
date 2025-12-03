"""
Pandas implementation for missing value imputation feature groups.
"""

from __future__ import annotations

from typing import Any, List, Optional, Set, Type, Union


from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import MissingValueFeatureGroup

try:
    import pandas as pd
except ImportError:
    pd = None


class PandasMissingValueFeatureGroup(MissingValueFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PandasDataFrame}

    @classmethod
    def _get_available_columns(cls, data: pd.DataFrame) -> Set[str]:
        """Get the set of available column names from the DataFrame."""
        return set(data.columns)

    @classmethod
    def _check_source_features_exist(cls, data: pd.DataFrame, feature_names: List[str]) -> None:
        """Check if the resolved source features exist in the DataFrame."""
        missing_features = [f for f in feature_names if f not in data.columns]
        if missing_features:
            raise ValueError(f"Source features not found in data: {missing_features}")

    @classmethod
    def _add_result_to_data(cls, data: pd.DataFrame, feature_name: str, result: Any) -> pd.DataFrame:
        """Add the result to the DataFrame."""
        data[feature_name] = result
        return data

    @classmethod
    def _perform_imputation(
        cls,
        data: pd.DataFrame,
        imputation_method: str,
        in_features: List[str],
        constant_value: Optional[Any] = None,
        group_by_features: Optional[List[str]] = None,
    ) -> pd.Series:
        """
        Perform the imputation using Pandas.

        Supports both single-column and multi-column imputation:
        - Single column: [feature_name] - imputes values within the column
        - Multi-column: [feature~0, feature~1, ...] - imputes across columns

        Args:
            data: The Pandas DataFrame
            imputation_method: The type of imputation to perform
            in_features: List of resolved source feature names to impute
            constant_value: The constant value to use for imputation (if method is 'constant')
            group_by_features: Optional list of features to group by before imputation

        Returns:
            The result of the imputation as a Pandas Series
        """
        # Handle single column case (backward compatibility)
        if len(in_features) == 1:
            source_feature = in_features[0]
            # Create a copy of the source feature to avoid modifying the original
            result = data[source_feature].copy()

            # If there are no missing values, return the original series
            if not result.isna().any():
                return result

            # If group_by_features is provided, perform grouped imputation
            if group_by_features:
                return cls._perform_grouped_imputation(
                    data, imputation_method, source_feature, constant_value, group_by_features
                )

            # Perform non-grouped imputation
            if imputation_method == "mean":
                return result.fillna(result.mean())
            elif imputation_method == "median":
                return result.fillna(result.median())
            elif imputation_method == "mode":
                # Get the most frequent value (first mode if multiple)
                mode_value = result.mode().iloc[0] if not result.mode().empty else None
                return result.fillna(mode_value)
            elif imputation_method == "constant":
                return result.fillna(constant_value)
            elif imputation_method == "ffill":
                return result.ffill()
            elif imputation_method == "bfill":
                return result.bfill()
            else:
                raise ValueError(f"Unsupported imputation method: {imputation_method}")
        else:
            # Multi-column case: impute across columns (row-wise)
            # For multi-column features, we compute imputation values across columns for each row
            # and return a single Series with one value per row
            df_subset = data[in_features]

            if imputation_method == "mean":
                return df_subset.mean(axis=1)
            elif imputation_method == "median":
                return df_subset.median(axis=1)
            elif imputation_method == "mode":
                # Get the most frequent value for each row
                return (
                    df_subset.mode(axis=1).iloc[:, 0]
                    if not df_subset.mode(axis=1).empty
                    else pd.Series([None] * len(df_subset))
                )
            elif imputation_method == "constant":
                # For constant, return the constant value for all rows
                return pd.Series([constant_value] * len(df_subset), index=df_subset.index)
            elif imputation_method == "sum":
                return df_subset.sum(axis=1)
            elif imputation_method == "min":
                return df_subset.min(axis=1)
            elif imputation_method == "max":
                return df_subset.max(axis=1)
            else:
                raise ValueError(f"Unsupported imputation method for multi-column: {imputation_method}")

    @classmethod
    def _perform_grouped_imputation(
        cls,
        data: pd.DataFrame,
        imputation_method: str,
        in_features: str,
        constant_value: Optional[Any],
        group_by_features: List[str],
    ) -> pd.Series:
        """
        Perform imputation within groups.

        Args:
            data: The Pandas DataFrame
            imputation_method: The type of imputation to perform
            source_feature: The name of the source feature to impute
            constant_value: The constant value to use for imputation (if method is 'constant')
            group_by_features: List of features to group by before imputation

        Returns:
            The result of the grouped imputation as a Pandas Series
        """
        # Create a copy of the source feature to avoid modifying the original
        result = data[in_features].copy()

        if imputation_method == "constant":
            # Constant imputation is the same regardless of groups
            return result.fillna(constant_value)

        # Group the data
        grouped = data.groupby(group_by_features)

        if imputation_method == "mean":
            # Calculate mean for each group
            group_means = grouped[in_features].transform("mean")
            # For groups with all NaN values, group_means will have NaN
            # Fall back to the overall mean for those groups
            overall_mean = data[in_features].mean()
            return result.fillna(group_means).fillna(overall_mean)
        elif imputation_method == "median":
            # Calculate median for each group
            group_medians = grouped[in_features].transform("median")
            # For groups with all NaN values, group_medians will have NaN
            # Fall back to the overall median for those groups
            overall_median = data[in_features].median()
            return result.fillna(group_medians).fillna(overall_median)
        elif imputation_method == "mode":
            # Mode is more complex - we need to find the mode for each group
            # and apply it to the missing values in that group
            for name, group in grouped:
                # Get indices for this group
                group_indices = group.index
                # Get the mode for this group
                group_mode = group[in_features].mode()
                if not group_mode.empty:
                    mode_value = group_mode.iloc[0]
                    # Apply the mode to missing values in this group
                    result.loc[group_indices] = result.loc[group_indices].fillna(mode_value)
            return result
        elif imputation_method == "ffill":
            # Forward fill within groups
            return grouped[in_features].transform(lambda x: x.ffill())
        elif imputation_method == "bfill":
            # Backward fill within groups
            return grouped[in_features].transform(lambda x: x.bfill())
        else:
            raise ValueError(f"Unsupported imputation method: {imputation_method}")
