"""
Pandas implementation for scikit-learn scaling feature groups.
"""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework

from mloda.user.pandas import PandasDataFrame
from mloda_plugins.feature_group.columnwise_hooks import SklearnPandasColumnwiseHooks
from mloda_plugins.feature_group.experimental.sklearn.scaling.base import ScalingFeatureGroup


class PandasScalingFeatureGroup(SklearnPandasColumnwiseHooks, ScalingFeatureGroup):
    """
    Pandas implementation for scikit-learn scaling feature groups.

    This implementation works with pandas DataFrames and provides seamless
    integration between mloda's pandas compute framework and scikit-learn scalers.
    """

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        """Specify that this feature group works with Pandas."""
        return {PandasDataFrame}

    @classmethod
    def _extract_training_data(cls, data: Any, source_feature: str) -> Any:
        """
        Extract training data for the specified feature from pandas DataFrame.

        Args:
            data: The pandas DataFrame
            source_feature: Name of the source feature

        Returns:
            Training data as numpy array for sklearn
        """
        # Extract the specified column
        feature_data = data[[source_feature]]

        # Handle missing values by dropping rows with NaN
        # This is a simple strategy - more sophisticated handling could be added
        feature_data = feature_data.dropna()

        # Convert to numpy array for sklearn
        return feature_data.values

    @classmethod
    def _apply_scaler(cls, data: Any, source_feature: str, fitted_scaler: Any) -> Any:
        """
        Apply the fitted scaler to the pandas DataFrame.

        Args:
            data: The pandas DataFrame
            source_feature: Name of the source feature
            fitted_scaler: The fitted sklearn scaler

        Returns:
            Scaled data as numpy array
        """
        # Extract the specified column
        feature_data = data[[source_feature]]

        # Handle missing values - for prediction, we need to handle them differently
        # than during training. Here we'll use simple forward fill and backward fill
        feature_data = feature_data.ffill().bfill()

        # If there are still NaN values, fill with 0 (this is a simple strategy)
        feature_data = feature_data.fillna(0)

        # Convert to numpy array and apply scaler
        X = feature_data.values
        result = fitted_scaler.transform(X)

        return result
