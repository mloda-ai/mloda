"""
Pandas implementation for scikit-learn encoding feature groups.
"""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet, Implementation

from mloda.user.pandas import PandasDataFrame
from mloda_plugins.feature_group.experimental.sklearn.encoding.base import EncodingFeatureGroup


class PandasEncoding(Implementation[EncodingFeatureGroup]):
    """
    Pandas implementation for scikit-learn encoding feature groups.

    This implementation works with pandas DataFrames and provides seamless
    integration between mloda's pandas compute framework and scikit-learn encoders.
    """

    feature_group = EncodingFeatureGroup
    compute_framework = PandasDataFrame

    @classmethod
    def calculate(
        cls, group: type[EncodingFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.encode(
            data,
            features,
            training_data=cls._extract_training_data,
            apply=cls._apply_encoder,
            add_column=cls._add_result_to_data,
        )

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """Add the result to the DataFrame."""
        # Convert sparse matrix to dense if needed
        if hasattr(result, "toarray"):
            result = result.toarray()

        # Handle different result shapes
        if hasattr(result, "shape") and len(result.shape) == 2:
            if result.shape[1] > 1:
                # Multi-column result (e.g., from OneHotEncoder)
                named_columns = FeatureGroup.apply_naming_convention(result, feature_name)
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

        # Match _apply_encoder's transform-time policy so the encoder never sees an unseen label.
        # Normalize the whole column to a uniform string dtype so a numeric column with NaN
        # does not mix floats and "unknown" strings, which sklearn's fit validation rejects.
        feature_data = feature_data.where(feature_data.notna(), "unknown").astype(str)

        # For categorical encoders, we need to handle the data format properly
        # LabelEncoder expects 1D array, OneHotEncoder and OrdinalEncoder expect 2D array
        feature_values = feature_data.to_numpy()

        # Return 1D array - the group handles reshaping based on encoder type
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
        # than during training. For categorical data, we'll use the string "unknown".
        # Must stay policy-identical with _extract_training_data.
        feature_data = feature_data.where(feature_data.notna(), "unknown").astype(str)

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
