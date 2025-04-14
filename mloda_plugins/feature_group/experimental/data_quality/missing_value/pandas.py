"""
Pandas implementation for missing value imputation feature groups.
"""

from __future__ import annotations

from typing import Any, List, Optional, Set, Type, Union

import pandas as pd

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.data_quality.missing_value.base import BaseMissingValueFeatureGroup


class PandasMissingValueFeatureGroup(BaseMissingValueFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PandasDataframe}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Perform missing value imputation using Pandas.

        Processes all requested features, determining the imputation method
        and source feature from each feature name.

        Adds the imputed results directly to the input DataFrame.
        """
        # Get constant value and group by features from options if available
        constant_value = None
        group_by_features = None

        if features.options:
            constant_value = features.options.get("constant_value")
            group_by_features = features.options.get("group_by_features")

        # Process each requested feature
        for feature_name in features.get_all_names():
            imputation_method = cls.get_imputation_method(feature_name)
            source_feature = cls.get_source_feature(feature_name)

            if source_feature not in data.columns:
                raise ValueError(f"Source feature '{source_feature}' not found in data")

            # Validate group by features if provided
            if group_by_features:
                for group_feature in group_by_features:
                    if group_feature not in data.columns:
                        raise ValueError(f"Group by feature '{group_feature}' not found in data")

            # Validate constant value is provided for constant imputation
            if imputation_method == "constant" and constant_value is None:
                raise ValueError("Constant value must be provided for constant imputation method")

            # Apply the appropriate imputation function and add result to the DataFrame
            data[feature_name] = cls._perform_imputation(
                data, imputation_method, source_feature, constant_value, group_by_features
            )

        # Return the modified DataFrame
        return data

    @classmethod
    def _perform_imputation(
        cls,
        data: pd.DataFrame,
        imputation_method: str,
        source_feature: str,
        constant_value: Optional[Any] = None,
        group_by_features: Optional[List[str]] = None,
    ) -> pd.Series:
        """
        Perform the imputation using Pandas.

        Args:
            data: The Pandas DataFrame
            imputation_method: The type of imputation to perform
            source_feature: The name of the source feature to impute
            constant_value: The constant value to use for imputation (if method is 'constant')
            group_by_features: Optional list of features to group by before imputation

        Returns:
            The result of the imputation as a Pandas Series
        """
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
            return result.fillna(method="ffill")
        elif imputation_method == "bfill":
            return result.fillna(method="bfill")
        else:
            raise ValueError(f"Unsupported imputation method: {imputation_method}")

    @classmethod
    def _perform_grouped_imputation(
        cls,
        data: pd.DataFrame,
        imputation_method: str,
        source_feature: str,
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
        result = data[source_feature].copy()

        if imputation_method == "constant":
            # Constant imputation is the same regardless of groups
            return result.fillna(constant_value)

        # Group the data
        grouped = data.groupby(group_by_features)

        if imputation_method == "mean":
            # Calculate mean for each group
            group_means = grouped[source_feature].transform("mean")
            # For groups with all NaN values, group_means will have NaN
            # Fall back to the overall mean for those groups
            overall_mean = data[source_feature].mean()
            return result.fillna(group_means).fillna(overall_mean)
        elif imputation_method == "median":
            # Calculate median for each group
            group_medians = grouped[source_feature].transform("median")
            # For groups with all NaN values, group_medians will have NaN
            # Fall back to the overall median for those groups
            overall_median = data[source_feature].median()
            return result.fillna(group_medians).fillna(overall_median)
        elif imputation_method == "mode":
            # Mode is more complex - we need to find the mode for each group
            # and apply it to the missing values in that group
            for name, group in grouped:
                # Get indices for this group
                group_indices = group.index
                # Get the mode for this group
                group_mode = group[source_feature].mode()
                if not group_mode.empty:
                    mode_value = group_mode.iloc[0]
                    # Apply the mode to missing values in this group
                    result.loc[group_indices] = result.loc[group_indices].fillna(mode_value)
            return result
        elif imputation_method == "ffill":
            # Forward fill within groups
            return grouped[source_feature].transform(lambda x: x.fillna(method="ffill"))
        elif imputation_method == "bfill":
            # Backward fill within groups
            return grouped[source_feature].transform(lambda x: x.fillna(method="bfill"))
        else:
            raise ValueError(f"Unsupported imputation method: {imputation_method}")
