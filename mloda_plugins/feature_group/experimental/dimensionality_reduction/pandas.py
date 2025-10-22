"""
Pandas implementation for dimensionality reduction feature groups.
"""

from __future__ import annotations

from typing import Any, List, cast


try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None  # type: ignore

# Check if required packages are available
SKLEARN_AVAILABLE = True
try:
    from sklearn.decomposition import PCA, FastICA
    from sklearn.manifold import TSNE, Isomap
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
except ImportError:
    SKLEARN_AVAILABLE = False


from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import DimensionalityReductionFeatureGroup


class PandasDimensionalityReductionFeatureGroup(DimensionalityReductionFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFrameWork]]:
        """Define the compute framework for this feature group."""
        return {PandasDataframe}

    @classmethod
    def _check_source_feature_exists(cls, data: pd.DataFrame, feature_name: str) -> None:
        """
        Check if the source feature exists in the DataFrame.

        Args:
            data: The pandas DataFrame
            feature_name: The name of the feature to check

        Raises:
            ValueError: If the feature does not exist in the DataFrame
        """
        if feature_name not in data.columns:
            raise ValueError(f"Feature '{feature_name}' not found in the data")

    @classmethod
    def _add_result_to_data(cls, data: pd.DataFrame, feature_name: str, result: np.ndarray) -> pd.DataFrame:  # type: ignore
        """
        Add the dimensionality reduction result to the DataFrame using the multiple result columns pattern.

        Instead of storing the entire result array in a single column, this method creates multiple
        columns following the naming convention `feature_name~dim{i+1}` for each dimension.

        Args:
            data: The pandas DataFrame
            feature_name: The name of the feature to add
            result: The dimensionality reduction result (reduced features)

        Returns:
            The updated DataFrame with the dimensionality reduction result added as multiple columns
        """
        # Add individual dimension columns using the multiple result columns pattern
        algorithm, dimension = cls.parse_reduction_prefix(feature_name)
        named_columns = cls.apply_naming_convention(result, feature_name, suffix_generator=lambda i: f"dim{i + 1}")
        for col_name, col_data in named_columns.items():
            data[col_name] = col_data

        return data

    @classmethod
    def _perform_reduction(
        cls,
        data: Any,
        algorithm: str,
        dimension: int,
        source_features: List[str],
        options: Any,
    ) -> np.ndarray:  # type: ignore
        """
        Perform dimensionality reduction on the specified features.

        Args:
            data: The pandas DataFrame
            algorithm: The dimensionality reduction algorithm to use
            dimension: The target dimension for the reduction
            source_features: The list of source features to use for dimensionality reduction
            options: Options containing algorithm-specific parameters

        Returns:
            A numpy array containing the reduced features
        """
        # Import the base class to access option keys
        from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import (
            DimensionalityReductionFeatureGroup,
        )

        # Cast data to pandas DataFrame
        df = cast(pd.DataFrame, data)

        # Extract the features to use for dimensionality reduction
        X = df[source_features].copy()

        # Handle missing values (replace with mean)
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].mean())

        # Convert to numpy array
        X_array = X.values

        # Check if scikit-learn is available
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for dimensionality reduction. "
                "Please install it with 'pip install scikit-learn'."
            )

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array)

        # Check if dimension is valid
        if dimension >= X_scaled.shape[1]:
            raise ValueError(
                f"Target dimension ({dimension}) must be less than the number of source features ({X_scaled.shape[1]})"
            )

        # Perform dimensionality reduction based on the algorithm
        if algorithm == "pca":
            svd_solver = options.get(DimensionalityReductionFeatureGroup.PCA_SVD_SOLVER)
            if svd_solver is None:
                svd_solver = DimensionalityReductionFeatureGroup.PROPERTY_MAPPING[
                    DimensionalityReductionFeatureGroup.PCA_SVD_SOLVER
                ]["default"]
            return cls._perform_pca_reduction(X_scaled, dimension, svd_solver)
        elif algorithm == "tsne":
            max_iter_val = options.get(DimensionalityReductionFeatureGroup.TSNE_MAX_ITER)
            if max_iter_val is None:
                max_iter_val = DimensionalityReductionFeatureGroup.PROPERTY_MAPPING[
                    DimensionalityReductionFeatureGroup.TSNE_MAX_ITER
                ]["default"]
            max_iter = int(max_iter_val)

            n_iter_without_progress_val = options.get(DimensionalityReductionFeatureGroup.TSNE_N_ITER_WITHOUT_PROGRESS)
            if n_iter_without_progress_val is None:
                n_iter_without_progress_val = DimensionalityReductionFeatureGroup.PROPERTY_MAPPING[
                    DimensionalityReductionFeatureGroup.TSNE_N_ITER_WITHOUT_PROGRESS
                ]["default"]
            n_iter_without_progress = int(n_iter_without_progress_val)

            method = options.get(DimensionalityReductionFeatureGroup.TSNE_METHOD)
            if method is None:
                method = DimensionalityReductionFeatureGroup.PROPERTY_MAPPING[
                    DimensionalityReductionFeatureGroup.TSNE_METHOD
                ]["default"]
            return cls._perform_tsne_reduction(X_scaled, dimension, max_iter, n_iter_without_progress, method)
        elif algorithm == "ica":
            max_iter_val = options.get(DimensionalityReductionFeatureGroup.ICA_MAX_ITER)
            if max_iter_val is None:
                max_iter_val = DimensionalityReductionFeatureGroup.PROPERTY_MAPPING[
                    DimensionalityReductionFeatureGroup.ICA_MAX_ITER
                ]["default"]
            max_iter = int(max_iter_val)
            return cls._perform_ica_reduction(X_scaled, dimension, max_iter)
        elif algorithm == "lda":
            return cls._perform_lda_reduction(X_scaled, dimension, df)
        elif algorithm == "isomap":
            n_neighbors_val = options.get(DimensionalityReductionFeatureGroup.ISOMAP_N_NEIGHBORS)
            if n_neighbors_val is None:
                n_neighbors_val = DimensionalityReductionFeatureGroup.PROPERTY_MAPPING[
                    DimensionalityReductionFeatureGroup.ISOMAP_N_NEIGHBORS
                ]["default"]
            n_neighbors = int(n_neighbors_val)
            return cls._perform_isomap_reduction(X_scaled, dimension, n_neighbors)
        else:
            raise ValueError(f"Unsupported dimensionality reduction algorithm: {algorithm}")

    @classmethod
    def _perform_pca_reduction(cls, X: np.ndarray, dimension: int, svd_solver: str = "auto") -> np.ndarray:  # type: ignore
        """
        Perform Principal Component Analysis (PCA).

        Args:
            X: The feature matrix
            dimension: The target dimension
            svd_solver: SVD solver algorithm ('auto', 'full', 'arpack', 'randomized')

        Returns:
            A numpy array containing the reduced features
        """
        # Check if scikit-learn is available
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for PCA dimensionality reduction")

        # Perform PCA
        pca = PCA(n_components=dimension, random_state=42, svd_solver=svd_solver)
        return pca.fit_transform(X)  # type: ignore

    @classmethod
    def _perform_tsne_reduction(
        cls,
        X: np.ndarray,  # type: ignore
        dimension: int,
        max_iter: int = 250,
        n_iter_without_progress: int = 50,
        method: str = "barnes_hut",
    ) -> np.ndarray:  # type: ignore
        """
        Perform t-Distributed Stochastic Neighbor Embedding (t-SNE).

        Args:
            X: The feature matrix
            dimension: The target dimension
            max_iter: Maximum number of iterations for optimization
            n_iter_without_progress: Maximum iterations without progress before early stopping
            method: Computation method ('barnes_hut' or 'exact')

        Returns:
            A numpy array containing the reduced features
        """
        # Check if scikit-learn is available
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for t-SNE dimensionality reduction")

        # Adjust perplexity based on sample size
        # t-SNE requires perplexity < n_samples
        n_samples = X.shape[0]
        perplexity = min(30, n_samples // 3)  # Default is 30, but we need to ensure it's less than n_samples

        # For very small datasets (< 20 samples), use exact method as it's faster
        # and more reliable for unit tests
        if n_samples < 20:
            actual_method = "exact"
            # Use minimal iterations for small datasets in unit tests
            actual_max_iter = min(max_iter, 250)
        else:
            actual_method = method
            actual_max_iter = max_iter

        # Perform t-SNE with configurable parameters
        tsne = TSNE(
            n_components=dimension,
            random_state=42,
            perplexity=perplexity,
            max_iter=actual_max_iter,
            n_iter_without_progress=n_iter_without_progress,
            method=actual_method,
        )
        return tsne.fit_transform(X)  # type: ignore

    @classmethod
    def _perform_ica_reduction(cls, X: np.ndarray, dimension: int, max_iter: int = 200) -> np.ndarray:  # type: ignore
        """
        Perform Independent Component Analysis (ICA).

        Args:
            X: The feature matrix
            dimension: The target dimension
            max_iter: Maximum number of iterations

        Returns:
            A numpy array containing the reduced features
        """
        # Check if scikit-learn is available
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for ICA dimensionality reduction")

        # Perform ICA
        ica = FastICA(n_components=dimension, random_state=42, max_iter=max_iter)
        return ica.fit_transform(X)  # type: ignore

    @classmethod
    def _perform_lda_reduction(cls, X: np.ndarray, dimension: int, df: pd.DataFrame) -> np.ndarray:  # type: ignore
        """
        Perform Linear Discriminant Analysis (LDA).

        Args:
            X: The feature matrix
            dimension: The target dimension
            df: The original DataFrame (needed for target variable)

        Returns:
            A numpy array containing the reduced features
        """
        # Check if scikit-learn is available
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for LDA dimensionality reduction")

        # LDA requires a target variable
        # We'll try to find a categorical column in the DataFrame
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns

        if len(categorical_columns) == 0:
            raise ValueError("LDA requires a categorical target variable, but none was found in the data")

        # Use the first categorical column as the target
        target_column = categorical_columns[0]
        y = df[target_column].values

        # Perform LDA
        lda = LinearDiscriminantAnalysis(n_components=dimension)
        return lda.fit_transform(X, y)  # type: ignore

    @classmethod
    def _perform_isomap_reduction(cls, X: np.ndarray, dimension: int, n_neighbors: int = 5) -> np.ndarray:  # type: ignore
        """
        Perform Isometric Mapping (Isomap).

        Args:
            X: The feature matrix
            dimension: The target dimension
            n_neighbors: Number of neighbors to consider for each point

        Returns:
            A numpy array containing the reduced features
        """
        # Check if scikit-learn is available
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Isomap dimensionality reduction")

        # Perform Isomap
        isomap = Isomap(n_components=dimension, n_neighbors=n_neighbors)
        return isomap.fit_transform(X)  # type: ignore
