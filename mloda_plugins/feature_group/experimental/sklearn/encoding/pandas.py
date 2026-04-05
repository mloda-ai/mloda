"""
Pandas implementation for scikit-learn encoding feature groups.
"""

from __future__ import annotations

from typing import Any, List, Set, Type

from mloda.provider import ComputeFramework

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.sklearn.encoding.base import EncodingFeatureGroup


class PandasEncodingFeatureGroup(EncodingFeatureGroup):
    """
    Pandas implementation for scikit-learn encoding feature groups.

    This implementation works with pandas DataFrames and provides seamless
    integration between mloda's pandas compute framework and scikit-learn encoders.
    """

    @classmethod
    def compute_framework_rule(cls) -> Set[Type[ComputeFramework]]:
        """Specify that this feature group works with Pandas."""
        return {PandasDataFrame}

    @classmethod
    def _check_source_features_exist(cls, data: Any, feature_names: List[str]) -> None:
        """Check if the features exist in the DataFrame."""
        for feature_name in feature_names:
            if feature_name not in data.columns:
                raise ValueError(f"Source feature '{feature_name}' not found in data")

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """Add the result to the DataFrame."""
        import re

        # Convert sparse matrix to dense if needed
        if hasattr(result, "toarray"):
            result = result.toarray()

        # Handle different result shapes
        if hasattr(result, "shape") and len(result.shape) == 2:
            if result.shape[1] > 1:
                # Multi-column result (e.g., from OneHotEncoder)
                # Check if a specific column is requested via ~N suffix
                column_match = re.match(r"^(.+)~(\d+)$", feature_name)
                if column_match:
                    requested_column_index = int(column_match.group(2))
                    if requested_column_index < result.shape[1]:
                        data[feature_name] = result[:, requested_column_index]
                    else:
                        raise ValueError(
                            f"Requested column index {requested_column_index} is out of range. "
                            f"Available columns: 0-{result.shape[1] - 1}"
                        )
                else:
                    # Full multi-column encoding - create all columns with ~ separator
                    named_columns = cls.apply_naming_convention(result, feature_name)
                    for col_name, col_data in named_columns.items():
                        data[col_name] = col_data
            else:
                # Single column 2D result - flatten to 1D
                data[feature_name] = result.flatten()
        elif hasattr(result, "shape") and len(result.shape) == 1:
            # 1D result
            data[feature_name] = result
        else:
            # Scalar or other result type
            data[feature_name] = result

        return data

    @classmethod
    def _extract_training_data(cls, data: Any, source_feature: str) -> Any:
        """
        Extract training data for the specified feature from pandas DataFrame.

        Args:
            data: The pandas DataFrame
            source_feature: Name of the source feature

        Returns:
            Training data for sklearn encoder
        """
        # Extract the specified column
        feature_data = data[source_feature]

        # Handle missing values by dropping rows with NaN
        # This is a simple strategy - more sophisticated handling could be added
        feature_data = feature_data.dropna()

        # For categorical encoders, we need to handle the data format properly
        # LabelEncoder expects 1D array, OneHotEncoder and OrdinalEncoder expect 2D array
        feature_values = feature_data.to_numpy()

        # Return 1D array - the base class will handle reshaping based on encoder type
        return feature_values

    @classmethod
    def _apply_encoder(cls, data: Any, source_feature: str, fitted_encoder: Any) -> Any:
        """
        Apply the fitted encoder to the pandas DataFrame.

        Args:
            data: The pandas DataFrame
            source_feature: Name of the source feature
            fitted_encoder: The fitted sklearn encoder

        Returns:
            Encoded data
        """
        # Extract the specified column
        feature_data = data[source_feature]

        # Handle missing values - for prediction, we need to handle them differently
        # than during training. For categorical data, we'll use the string "unknown"
        feature_data = feature_data.fillna("unknown")

        # Convert to appropriate format for sklearn
        feature_values = feature_data.to_numpy()

        # Handle different encoder types
        encoder_class_name = fitted_encoder.__class__.__name__

        if encoder_class_name == "LabelEncoder":
            # LabelEncoder expects 1D array
            if len(feature_values.shape) > 1:
                feature_values = feature_values.flatten()
            result = fitted_encoder.transform(feature_values)
        else:
            # OneHotEncoder and OrdinalEncoder expect 2D array
            if len(feature_values.shape) == 1:
                feature_values = feature_values.reshape(-1, 1)
            result = fitted_encoder.transform(feature_values)

        return result
