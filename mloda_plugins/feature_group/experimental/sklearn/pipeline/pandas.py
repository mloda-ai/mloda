"""
Pandas implementation for scikit-learn pipeline feature groups.
"""

from __future__ import annotations

import numpy as np
from typing import Any, List, Set, Type, Union

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.sklearn.pipeline.base import SklearnPipelineFeatureGroup


class PandasSklearnPipelineFeatureGroup(SklearnPipelineFeatureGroup):
    """
    Pandas implementation for scikit-learn pipeline feature groups.

    This implementation works with pandas DataFrames and provides seamless
    integration between mloda's pandas compute framework and scikit-learn pipelines.
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
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """Add the result to the DataFrame."""
        # Handle different result types from sklearn pipelines
        if hasattr(result, "shape") and len(result.shape) == 2:
            # Multi-dimensional result (e.g., from PCA, multiple features)
            if result.shape[1] == 1:
                # Single column result
                data[feature_name] = result.flatten()
            else:
                # Multiple columns - use naming convention with ~ separator
                for i in range(result.shape[1]):
                    column_name = f"{feature_name}~{i}"
                    data[column_name] = result[:, i]
        elif hasattr(result, "shape") and len(result.shape) == 1:
            # Single dimensional result
            data[feature_name] = result
        else:
            # Scalar or other result type
            data[feature_name] = result

        return data

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
