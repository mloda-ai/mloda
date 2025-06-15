"""
Pandas implementation for scikit-learn encoding feature groups.
"""

from __future__ import annotations

from typing import Any, Set, Type, Union

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.sklearn.encoding.base import EncodingFeatureGroup


class PandasEncodingFeatureGroup(EncodingFeatureGroup):
    """
    Pandas implementation for scikit-learn encoding feature groups.

    This implementation works with pandas DataFrames and provides seamless
    integration between mloda's pandas compute framework and scikit-learn encoders.
    """

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        """Specify that this feature group works with Pandas."""
        return {PandasDataframe}

    @classmethod
    def _check_source_feature_exists(cls, data: Any, feature_name: str) -> None:
        """Check if the feature exists in the DataFrame."""
        if feature_name not in data.columns:
            raise ValueError(f"Source feature '{feature_name}' not found in data")

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any, encoder_type: str) -> Any:
        """Add the result to the DataFrame."""
        import re

        # Handle different result types from sklearn encoders
        if encoder_type == "onehot":
            # Check if this is a request for a specific OneHot column (e.g., onehot_encoded__category~1)
            onehot_column_match = re.match(r"^onehot_encoded__(.+)~(\d+)$", feature_name)

            # OneHotEncoder returns a sparse matrix or dense array with multiple columns
            if hasattr(result, "toarray"):
                # Convert sparse matrix to dense
                result = result.toarray()

            if hasattr(result, "shape") and len(result.shape) == 2 and result.shape[1] > 1:
                if onehot_column_match:
                    # Specific column requested - only add that column
                    requested_column_index = int(onehot_column_match.group(2))
                    if requested_column_index < result.shape[1]:
                        data[feature_name] = result[:, requested_column_index]
                    else:
                        raise ValueError(
                            f"Requested OneHot column index {requested_column_index} is out of range. Available columns: 0-{result.shape[1] - 1}"
                        )
                else:
                    # Full OneHot encoding requested - create all columns with ~ separator
                    for i in range(result.shape[1]):
                        column_name = f"{feature_name}~{i}"
                        data[column_name] = result[:, i]
            else:
                # Single column or unexpected format
                data[feature_name] = result.flatten() if hasattr(result, "flatten") else result
        else:
            # LabelEncoder and OrdinalEncoder return single column results
            if hasattr(result, "shape") and len(result.shape) == 2:
                # 2D result - flatten to 1D
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
        feature_values = feature_data.values

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
        feature_values = feature_data.values

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
