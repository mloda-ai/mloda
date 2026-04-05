"""
Pandas implementation for scikit-learn pipeline feature groups.
"""

from __future__ import annotations

from typing import Any, List

from mloda.provider import ComputeFramework

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.sklearn.pipeline.base import SklearnPipelineFeatureGroup


class PandasSklearnPipelineFeatureGroup(SklearnPipelineFeatureGroup):
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
    def _check_source_features_exist(cls, data: Any, feature_names: List[str]) -> None:
        """Check if the features exist in the DataFrame."""
        missing_features = [f for f in feature_names if f not in data.columns]
        if missing_features:
            raise ValueError(
                f"Source features not found in data: {missing_features}. Available columns: {list(data.columns)}"
            )

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
                named_columns = cls.apply_naming_convention(result, feature_name)
                for col_name, col_data in named_columns.items():
                    data[col_name] = col_data
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
