"""
Pandas implementation for scikit-learn pipeline feature groups.
"""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework

from mloda.user.pandas import PandasDataFrame
from mloda_plugins.feature_group.columnwise_hooks import SklearnPandasColumnwiseHooks
from mloda_plugins.feature_group.experimental.sklearn.pipeline.base import SklearnPipelineFeatureGroup


class PandasSklearnPipelineFeatureGroup(SklearnPandasColumnwiseHooks, SklearnPipelineFeatureGroup):
    """
    Pandas implementation for scikit-learn pipeline feature groups.

    This implementation works with pandas DataFrames and provides seamless
    integration between mloda's pandas compute framework and scikit-learn pipelines.
    """

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        """Specify that this feature group works with Pandas."""
        return {PandasDataFrame}

    @classmethod
    def _extract_training_data(cls, data: Any, source_features: list[Any]) -> Any:
        """
        Extract training data for the specified features from pandas DataFrame.

        Args:
            data: The pandas DataFrame
            source_features: List of source feature names

        Returns:
            Training data as numpy array for sklearn
        """
        # Extract the specified columns
        feature_data = data[source_features]

        # Handle missing values by dropping rows with NaN
        # This is a simple strategy - more sophisticated handling could be added
        feature_data = feature_data.dropna()

        # Convert to numpy array for sklearn
        return feature_data.values

    @classmethod
    def _apply_pipeline(cls, data: Any, source_features: list[Any], fitted_pipeline: Any) -> Any:
        """
        Apply the fitted pipeline to the pandas DataFrame.

        Args:
            data: The pandas DataFrame
            source_features: List of source feature names
            fitted_pipeline: The fitted sklearn pipeline

        Returns:
            Transformed data as numpy array
        """
        # Extract the specified columns
        feature_data = data[source_features]

        # Handle missing values - for prediction, we need to handle them differently
        # than during training. Here we'll use simple forward fill and backward fill
        feature_data = feature_data.ffill().bfill()

        # If there are still NaN values, fill with 0 (this is a simple strategy)
        feature_data = feature_data.fillna(0)

        # Convert to numpy array and apply pipeline
        X = feature_data.values
        result = fitted_pipeline.transform(X)

        return result
