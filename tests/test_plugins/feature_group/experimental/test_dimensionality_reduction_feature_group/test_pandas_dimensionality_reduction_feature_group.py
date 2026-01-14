"""
Tests for the PandasDimensionalityReductionFeatureGroup class.
"""

import pytest
import pandas as pd
import numpy as np

from mloda.user import Feature
from mloda.provider import FeatureSet
from mloda.user import Options
from mloda_plugins.feature_group.experimental.dimensionality_reduction.pandas import (
    PandasDimensionalityReductionFeatureGroup,
)


class TestPandasDimensionalityReductionFeatureGroup:
    """Tests for the PandasDimensionalityReductionFeatureGroup class."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        # Create a DataFrame with multiple features and 100 samples
        np.random.seed(42)

        # Create data with some structure for dimensionality reduction
        n_samples = 100
        n_features = 10

        # Create a DataFrame with random data
        data = np.random.randn(n_samples, n_features)

        # Add some structure (correlations between features)
        data[:, 1] = data[:, 0] * 0.8 + np.random.randn(n_samples) * 0.2
        data[:, 2] = data[:, 0] * 0.6 + np.random.randn(n_samples) * 0.4

        # Create column names
        columns = [f"feature{i}" for i in range(n_features)]

        # Add a categorical column for LDA
        categories = ["A", "B", "C"]
        category_column = np.random.choice(categories, size=n_samples)

        # Create a DataFrame
        df = pd.DataFrame(data, columns=columns)
        df["category"] = category_column

        return df

    def test_check_source_feature_exists(self, sample_data: pd.DataFrame) -> None:
        """Test the _check_source_feature_exists method."""
        # Valid feature
        PandasDimensionalityReductionFeatureGroup._check_source_feature_exists(sample_data, "feature0")

        # Invalid feature
        with pytest.raises(ValueError):
            PandasDimensionalityReductionFeatureGroup._check_source_feature_exists(sample_data, "invalid_feature")

    def test_add_result_to_data(self, sample_data: pd.DataFrame) -> None:
        """Test the _add_result_to_data method."""
        # Create a result array (2D reduction of 10 samples)
        result = np.random.randn(100, 2)

        # Add the result to the data
        updated_data = PandasDimensionalityReductionFeatureGroup._add_result_to_data(
            sample_data, "feature0,feature1__pca_2d", result
        )

        # Check that the result was added
        assert "feature0,feature1__pca_2d~dim1" in updated_data.columns
        assert "feature0,feature1__pca_2d~dim2" in updated_data.columns
        assert len(updated_data["feature0,feature1__pca_2d~dim1"]) == len(sample_data)
        assert (updated_data["feature0,feature1__pca_2d~dim1"].values == result[:, 0]).all()
        assert (updated_data["feature0,feature1__pca_2d~dim2"].values == result[:, 1]).all()

    def test_perform_pca_reduction(self, sample_data: pd.DataFrame) -> None:
        """Test the _perform_pca_reduction method."""
        # Extract features
        X = sample_data[["feature0", "feature1", "feature2"]].values

        # Standardize the features
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform PCA reduction
        result = PandasDimensionalityReductionFeatureGroup._perform_pca_reduction(X_scaled, 2)

        # Check that the result has the expected shape
        assert result.shape == (100, 2)

    def test_perform_tsne_reduction(self, sample_data: pd.DataFrame) -> None:
        """Test the _perform_tsne_reduction method."""
        # Extract features (use minimal samples for t-SNE - this is just a unit test)
        # Use only 10 samples to ensure fast execution even in parallel
        X = sample_data[["feature0", "feature1", "feature2"]].iloc[:10].values

        # Standardize the features
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform t-SNE reduction with minimal iterations for unit test
        result = PandasDimensionalityReductionFeatureGroup._perform_tsne_reduction(
            X_scaled, 2, max_iter=250, n_iter_without_progress=30, method="barnes_hut"
        )

        # Check that the result has the expected shape
        assert result.shape == (10, 2)

    def test_perform_ica_reduction(self, sample_data: pd.DataFrame) -> None:
        """Test the _perform_ica_reduction method."""
        # Extract features
        X = sample_data[["feature0", "feature1", "feature2"]].values

        # Standardize the features
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform ICA reduction
        result = PandasDimensionalityReductionFeatureGroup._perform_ica_reduction(X_scaled, 2)

        # Check that the result has the expected shape
        assert result.shape == (100, 2)

    def test_perform_lda_reduction(self, sample_data: pd.DataFrame) -> None:
        """Test the _perform_lda_reduction method."""
        # Extract features
        X = sample_data[["feature0", "feature1", "feature2"]].values

        # Standardize the features
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform LDA reduction
        result = PandasDimensionalityReductionFeatureGroup._perform_lda_reduction(X_scaled, 2, sample_data)

        # Check that the result has the expected shape (LDA can produce at most n_classes-1 components)
        assert result.shape[0] == 100
        assert result.shape[1] <= 2

    def test_perform_isomap_reduction(self, sample_data: pd.DataFrame) -> None:
        """Test the _perform_isomap_reduction method."""
        # Extract features (use fewer samples for Isomap to speed up the test)
        X = sample_data[["feature0", "feature1", "feature2"]].iloc[:20].values

        # Standardize the features
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform Isomap reduction
        result = PandasDimensionalityReductionFeatureGroup._perform_isomap_reduction(X_scaled, 2)

        # Check that the result has the expected shape
        assert result.shape == (20, 2)

    def test_calculate_feature_pca(self, sample_data: pd.DataFrame) -> None:
        """Test the calculate_feature method with PCA."""
        # Create a feature set
        feature_set = FeatureSet()
        feature_set.add(Feature("feature0,feature1,feature2__pca_2d"))

        # Calculate the feature
        result = PandasDimensionalityReductionFeatureGroup.calculate_feature(sample_data, feature_set)

        # Check that the result has the expected columns
        assert "feature0,feature1,feature2__pca_2d~dim1" in result.columns
        assert "feature0,feature1,feature2__pca_2d~dim2" in result.columns

    # def test_calculate_feature_tsne(self, sample_data: pd.DataFrame) -> None:
    #    """Test the calculate_feature method with t-SNE."""
    # Create a feature set (use a small subset for t-SNE to speed up the test)
    #    small_sample = sample_data.iloc[:20].copy()
    #    feature_set = FeatureSet()
    #    feature_set.add(Feature("feature0,feature1,feature2__tsne_2d"))

    # Calculate the feature
    #    result = PandasDimensionalityReductionFeatureGroup.calculate_feature(small_sample, feature_set)

    #    # Check that the result has the expected columns
    #    assert "feature0,feature1,feature2__tsne_2d~dim1" in result.columns
    #    assert "feature0,feature1,feature2__tsne_2d~dim2" in result.columns

    def test_calculate_feature_multiple(self, sample_data: pd.DataFrame) -> None:
        """Test the calculate_feature method with multiple dimensionality reduction features."""
        # Create a feature set (use a small subset to speed up the test)
        small_sample = sample_data.iloc[:20].copy()
        feature_set = FeatureSet()
        feature_set.add(Feature("feature0,feature1,feature2__pca_2d"))
        feature_set.add(Feature("feature0,feature1,feature2__ica_2d"))

        # Calculate the features
        result = PandasDimensionalityReductionFeatureGroup.calculate_feature(small_sample, feature_set)

        # Check that the result has the expected columns
        assert "feature0,feature1,feature2__pca_2d~dim1" in result.columns
        assert "feature0,feature1,feature2__pca_2d~dim2" in result.columns
        assert "feature0,feature1,feature2__ica_2d~dim1" in result.columns
        assert "feature0,feature1,feature2__ica_2d~dim2" in result.columns

    def test_invalid_dimension(self, sample_data: pd.DataFrame) -> None:
        """Test with an invalid dimension (too large)."""
        # Create a feature set with a dimension that's too large
        feature_set = FeatureSet()
        feature_set.add(Feature("feature0,feature1,feature2__pca_20d"))  # Only 3 features, but asking for 20 dimensions

        # Calculate the feature (should raise an error)
        with pytest.raises(ValueError, match="Target dimension .* must be less than the number of source features"):
            PandasDimensionalityReductionFeatureGroup.calculate_feature(sample_data, feature_set)

    def test_missing_source_feature(self, sample_data: pd.DataFrame) -> None:
        """Test with a missing source feature."""
        # Create a feature set with a missing source feature
        feature_set = FeatureSet()
        feature_set.add(Feature("missing_feature__pca_2d"))

        # Calculate the feature (should raise an error)
        with pytest.raises(ValueError, match="Feature 'missing_feature' not found in the data"):
            PandasDimensionalityReductionFeatureGroup.calculate_feature(sample_data, feature_set)

    def test_unsupported_algorithm(self, sample_data: pd.DataFrame) -> None:
        """Test with an unsupported algorithm."""
        # This test should pass because the base class validation should catch this before
        # it gets to the pandas implementation
        feature_set = FeatureSet()
        feature_set.add(Feature("feature0,feature1__unsupported_2d"))

        # The match_feature_group_criteria method should return False for this feature
        assert not PandasDimensionalityReductionFeatureGroup.match_feature_group_criteria(
            "feature0,feature1__unsupported_2d", Options()
        )

    def test_tsne_with_custom_parameters(self, sample_data: pd.DataFrame) -> None:
        """Test t-SNE with custom configurable parameters."""
        from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import (
            DimensionalityReductionFeatureGroup,
        )

        # Create a feature set with custom t-SNE parameters
        # Use only 10 samples - this is just a unit test to verify parameters work
        small_sample = sample_data.iloc[:10].copy()
        feature = Feature(
            "feature0,feature1,feature2__tsne_2d",
            Options(
                {
                    DimensionalityReductionFeatureGroup.TSNE_MAX_ITER: 250,  # Minimal for unit test
                    DimensionalityReductionFeatureGroup.TSNE_N_ITER_WITHOUT_PROGRESS: 30,  # Minimal for unit test
                    DimensionalityReductionFeatureGroup.TSNE_METHOD: "barnes_hut",
                }
            ),
        )
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Calculate the feature
        result = PandasDimensionalityReductionFeatureGroup.calculate_feature(small_sample, feature_set)

        # Check that the result has the expected columns
        assert "feature0,feature1,feature2__tsne_2d~dim1" in result.columns
        assert "feature0,feature1,feature2__tsne_2d~dim2" in result.columns

    def test_pca_with_custom_svd_solver(self, sample_data: pd.DataFrame) -> None:
        """Test PCA with custom SVD solver parameter."""
        from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import (
            DimensionalityReductionFeatureGroup,
        )

        # Create a feature set with custom PCA parameters
        feature = Feature(
            "feature0,feature1,feature2__pca_2d",
            Options({DimensionalityReductionFeatureGroup.PCA_SVD_SOLVER: "full"}),
        )
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Calculate the feature
        result = PandasDimensionalityReductionFeatureGroup.calculate_feature(sample_data, feature_set)

        # Check that the result has the expected columns
        assert "feature0,feature1,feature2__pca_2d~dim1" in result.columns
        assert "feature0,feature1,feature2__pca_2d~dim2" in result.columns

    def test_ica_with_custom_max_iter(self, sample_data: pd.DataFrame) -> None:
        """Test ICA with custom max_iter parameter."""
        from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import (
            DimensionalityReductionFeatureGroup,
        )

        # Create a feature set with custom ICA parameters
        feature = Feature(
            "feature0,feature1,feature2__ica_2d",
            Options({DimensionalityReductionFeatureGroup.ICA_MAX_ITER: 300}),
        )
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Calculate the feature
        result = PandasDimensionalityReductionFeatureGroup.calculate_feature(sample_data, feature_set)

        # Check that the result has the expected columns
        assert "feature0,feature1,feature2__ica_2d~dim1" in result.columns
        assert "feature0,feature1,feature2__ica_2d~dim2" in result.columns

    def test_isomap_with_custom_n_neighbors(self, sample_data: pd.DataFrame) -> None:
        """Test Isomap with custom n_neighbors parameter."""
        from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import (
            DimensionalityReductionFeatureGroup,
        )

        # Create a feature set with custom Isomap parameters (use small sample)
        small_sample = sample_data.iloc[:20].copy()
        feature = Feature(
            "feature0,feature1,feature2__isomap_2d",
            Options({DimensionalityReductionFeatureGroup.ISOMAP_N_NEIGHBORS: 3}),
        )
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Calculate the feature
        result = PandasDimensionalityReductionFeatureGroup.calculate_feature(small_sample, feature_set)

        # Check that the result has the expected columns
        assert "feature0,feature1,feature2__isomap_2d~dim1" in result.columns
        assert "feature0,feature1,feature2__isomap_2d~dim2" in result.columns
