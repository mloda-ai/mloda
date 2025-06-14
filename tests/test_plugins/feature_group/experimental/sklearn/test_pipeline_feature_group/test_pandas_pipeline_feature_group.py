"""
Tests for PandasSklearnPipelineFeatureGroup.
"""

from typing import Any
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from mloda_plugins.feature_group.experimental.sklearn.pipeline.pandas import PandasSklearnPipelineFeatureGroup
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe


class TestPandasSklearnPipelineFeatureGroup:
    """Test cases for PandasSklearnPipelineFeatureGroup."""

    def test_compute_framework_rule(self) -> None:
        """Test that the feature group specifies pandas framework."""
        rule = PandasSklearnPipelineFeatureGroup.compute_framework_rule()
        assert PandasDataframe in rule  # type: ignore

    def test_check_source_feature_exists_valid(self) -> None:
        """Test checking for existing source features."""
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

        # Should not raise exception for existing features
        PandasSklearnPipelineFeatureGroup._check_source_feature_exists(df, "feature1")
        PandasSklearnPipelineFeatureGroup._check_source_feature_exists(df, "feature2")

    def test_check_source_feature_exists_invalid(self) -> None:
        """Test checking for non-existing source features."""
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

        with pytest.raises(ValueError, match="Source feature 'nonexistent' not found in data"):
            PandasSklearnPipelineFeatureGroup._check_source_feature_exists(df, "nonexistent")

    def test_add_result_to_data_1d_array(self) -> None:
        """Test adding 1D array result to DataFrame."""
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

        result = np.array([10, 20, 30])
        updated_df = PandasSklearnPipelineFeatureGroup._add_result_to_data(df, "new_feature", result)

        assert "new_feature" in updated_df.columns
        assert list(updated_df["new_feature"]) == [10, 20, 30]

    def test_add_result_to_data_2d_array_single_column(self) -> None:
        """Test adding 2D array with single column to DataFrame."""
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

        result = np.array([[10], [20], [30]])
        updated_df = PandasSklearnPipelineFeatureGroup._add_result_to_data(df, "new_feature", result)

        assert "new_feature" in updated_df.columns
        assert list(updated_df["new_feature"]) == [10, 20, 30]

    def test_add_result_to_data_2d_array_multiple_columns(self) -> None:
        """Test adding 2D array with multiple columns to DataFrame."""
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

        result = np.array([[10, 11], [20, 21], [30, 31]])
        updated_df = PandasSklearnPipelineFeatureGroup._add_result_to_data(df, "new_feature", result)

        assert "new_feature~0" in updated_df.columns
        assert "new_feature~1" in updated_df.columns
        assert list(updated_df["new_feature~0"]) == [10, 20, 30]
        assert list(updated_df["new_feature~1"]) == [11, 21, 31]

    def test_add_result_to_data_scalar(self) -> None:
        """Test adding scalar result to DataFrame."""
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

        result = 42
        updated_df = PandasSklearnPipelineFeatureGroup._add_result_to_data(df, "new_feature", result)

        assert "new_feature" in updated_df.columns
        assert updated_df["new_feature"].iloc[0] == 42

    def test_extract_training_data_single_feature(self) -> None:
        """Test extracting training data for single feature."""
        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0], "feature2": [5.0, 6.0, 7.0, 8.0]})

        result = PandasSklearnPipelineFeatureGroup._extract_training_data(df, ["feature1"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 1)
        assert list(result.flatten()) == [1.0, 2.0, 3.0, 4.0]

    def test_extract_training_data_multiple_features(self) -> None:
        """Test extracting training data for multiple features."""
        df = pd.DataFrame(
            {"feature1": [1.0, 2.0, 3.0, 4.0], "feature2": [5.0, 6.0, 7.0, 8.0], "feature3": [9.0, 10.0, 11.0, 12.0]}
        )

        result = PandasSklearnPipelineFeatureGroup._extract_training_data(df, ["feature1", "feature2"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 2)
        assert list(result[:, 0]) == [1.0, 2.0, 3.0, 4.0]
        assert list(result[:, 1]) == [5.0, 6.0, 7.0, 8.0]

    def test_extract_training_data_with_nan(self) -> None:
        """Test extracting training data with NaN values."""
        df = pd.DataFrame({"feature1": [1.0, 2.0, np.nan, 4.0], "feature2": [5.0, np.nan, 7.0, 8.0]})

        result = PandasSklearnPipelineFeatureGroup._extract_training_data(df, ["feature1", "feature2"])

        assert isinstance(result, np.ndarray)
        # Should drop rows with NaN, leaving only row with index 0 and 3
        assert result.shape == (2, 2)
        assert list(result[:, 0]) == [1.0, 4.0]
        assert list(result[:, 1]) == [5.0, 8.0]

    def test_apply_pipeline_single_feature(self) -> None:
        """Test applying pipeline to single feature."""
        # Skip test if sklearn not available
        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            pytest.skip("scikit-learn not available")

        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0], "feature2": [5.0, 6.0, 7.0, 8.0]})

        # Create and fit a simple scaler
        scaler = StandardScaler()
        scaler.fit([[1.0], [2.0], [3.0], [4.0]])

        result = PandasSklearnPipelineFeatureGroup._apply_pipeline(df, ["feature1"], scaler)

        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 1)
        # StandardScaler should center around 0
        assert abs(result.mean()) < 1e-10

    def test_apply_pipeline_multiple_features(self) -> None:
        """Test applying pipeline to multiple features."""
        # Skip test if sklearn not available
        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            pytest.skip("scikit-learn not available")

        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0], "feature2": [5.0, 6.0, 7.0, 8.0]})

        # Create and fit a scaler
        scaler = StandardScaler()
        scaler.fit([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]])

        result = PandasSklearnPipelineFeatureGroup._apply_pipeline(df, ["feature1", "feature2"], scaler)

        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 2)

    def test_apply_pipeline_with_nan(self) -> None:
        """Test applying pipeline with NaN values."""
        # Skip test if sklearn not available
        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            pytest.skip("scikit-learn not available")

        df = pd.DataFrame({"feature1": [1.0, 2.0, np.nan, 4.0], "feature2": [5.0, np.nan, 7.0, 8.0]})

        # Create and fit a scaler
        scaler = StandardScaler()
        scaler.fit([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]])

        result = PandasSklearnPipelineFeatureGroup._apply_pipeline(df, ["feature1", "feature2"], scaler)

        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 2)
        # Should handle NaN values by filling them
        assert not np.isnan(result).any()

    @patch("mloda_plugins.feature_group.experimental.sklearn.sklearn_artifact.SklearnArtifact.custom_loader")
    @patch("mloda_plugins.feature_group.experimental.sklearn.sklearn_artifact.SklearnArtifact.custom_saver")
    def test_calculate_feature_end_to_end(self, mock_saver: Any, mock_loader: Any) -> None:
        """Test end-to-end feature calculation."""
        # Skip test if sklearn not available
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
        except ImportError:
            pytest.skip("scikit-learn not available")

        # Mock no existing artifact
        mock_loader.return_value = None

        # Create test data
        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0], "feature2": [5.0, 6.0, 7.0, 8.0]})

        # Create feature set with artifact saving enabled
        features = FeatureSet()
        features.add(Feature("sklearn_pipeline_scaling__feature1"))

        # Set up artifact lifecycle flags to trigger saving
        features.artifact_to_save = "sklearn_pipeline_scaling__feature1"  # This triggers artifact saving
        features.artifact_to_load = None  # No existing artifact to load

        # Calculate feature
        result_df = PandasSklearnPipelineFeatureGroup.calculate_feature(df, features)

        # Verify result
        assert "sklearn_pipeline_scaling__feature1" in result_df.columns

        # Verify that save_artifact was set (this is what triggers the actual saving)
        assert hasattr(features, "save_artifact")
        assert features.save_artifact is not None

        # Verify the artifact data contains expected keys (now in multiple artifact format)
        assert isinstance(features.save_artifact, dict)
        artifact_key = "sklearn_pipeline_scaling__feature1"
        assert artifact_key in features.save_artifact

        artifact_data = features.save_artifact[artifact_key]
        assert "fitted_transformer" in artifact_data
        assert "feature_names" in artifact_data
        assert "pipeline_name" in artifact_data
        assert "training_timestamp" in artifact_data

    def test_calculate_feature_missing_source_feature(self) -> None:
        """Test calculate_feature with missing source feature."""
        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0]})

        features = FeatureSet()
        features.add(Feature("sklearn_pipeline_scaling__nonexistent"))

        with pytest.raises(ValueError, match="Source feature 'nonexistent' not found in data"):
            PandasSklearnPipelineFeatureGroup.calculate_feature(df, features)
