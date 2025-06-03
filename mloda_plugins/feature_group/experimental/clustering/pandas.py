"""
Pandas implementation for clustering feature groups.
"""

from __future__ import annotations

from typing import Any, List, Union, cast


try:
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
except ImportError:
    KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering = None, None, None, None
    StandardScaler = None
    silhouette_score = None


try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None  # type: ignore


from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.clustering.base import ClusteringFeatureGroup


class PandasClusteringFeatureGroup(ClusteringFeatureGroup):
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
        Add the clustering result to the DataFrame.

        Args:
            data: The pandas DataFrame
            feature_name: The name of the feature to add
            result: The clustering result (cluster assignments)

        Returns:
            The updated DataFrame with the clustering result added
        """
        data[feature_name] = result
        return data

    @classmethod
    def _perform_clustering(
        cls,
        data: Any,
        algorithm: str,
        k_value: Union[int, str],
        source_features: List[str],
    ) -> np.ndarray:  # type: ignore
        """
        Perform clustering on the specified features.

        Args:
            data: The pandas DataFrame
            algorithm: The clustering algorithm to use
            k_value: The number of clusters (or 'auto' for algorithms that determine this automatically)
            source_features: The list of source features to use for clustering

        Returns:
            A numpy array containing the cluster assignments
        """
        # Cast data to pandas DataFrame
        df = cast(pd.DataFrame, data)

        # Extract the features to use for clustering
        X = df[source_features].copy()

        # Handle missing values (replace with mean)
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].mean())

        # Convert to numpy array
        X_array = X.values

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array)

        # Perform clustering based on the algorithm
        if algorithm == "kmeans":
            return cls._perform_kmeans_clustering(X_scaled, k_value)
        elif algorithm == "hierarchical" or algorithm == "agglomerative":
            return cls._perform_hierarchical_clustering(X_scaled, k_value)
        elif algorithm == "dbscan":
            return cls._perform_dbscan_clustering(X_scaled, k_value)
        elif algorithm == "spectral":
            return cls._perform_spectral_clustering(X_scaled, k_value)
        elif algorithm == "affinity":
            return cls._perform_affinity_clustering(X_scaled, k_value)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    @classmethod
    def _perform_kmeans_clustering(cls, X: np.ndarray, k_value: Union[int, str]) -> np.ndarray:  # type: ignore
        """
        Perform K-means clustering.

        Args:
            X: The feature matrix
            k_value: The number of clusters or 'auto'

        Returns:
            A numpy array containing the cluster assignments
        """
        if k_value == "auto":
            # Determine optimal k using silhouette score
            k_value = cls._find_optimal_k(X, algorithm="kmeans")

        # Ensure k_value is an integer
        k = int(k_value)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        return kmeans.fit_predict(X)  # type: ignore

    @classmethod
    def _perform_hierarchical_clustering(cls, X: np.ndarray, k_value: Union[int, str]) -> np.ndarray:  # type: ignore
        """
        Perform hierarchical clustering.

        Args:
            X: The feature matrix
            k_value: The number of clusters or 'auto'

        Returns:
            A numpy array containing the cluster assignments
        """
        if k_value == "auto":
            # Determine optimal k using silhouette score
            k_value = cls._find_optimal_k(X, algorithm="hierarchical")

        # Ensure k_value is an integer
        k = int(k_value)

        # Perform hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=k)
        return hierarchical.fit_predict(X)  # type: ignore

    @classmethod
    def _perform_dbscan_clustering(cls, X: np.ndarray, k_value: Union[int, str]) -> np.ndarray:  # type: ignore
        """
        Perform DBSCAN clustering.

        Args:
            X: The feature matrix
            k_value: The number of clusters or 'auto'

        Returns:
            A numpy array containing the cluster assignments
        """
        # DBSCAN doesn't require a specific number of clusters
        # If k_value is not 'auto', we use it to determine the eps parameter
        if k_value == "auto":
            # Use default parameters
            dbscan = DBSCAN(eps=0.5, min_samples=5)
        else:
            # Use k_value to adjust the eps parameter
            # Smaller k_value means larger eps (fewer clusters)
            k = int(k_value)
            eps = 1.0 / k
            dbscan = DBSCAN(eps=eps, min_samples=5)

        return dbscan.fit_predict(X)  # type: ignore

    @classmethod
    def _perform_spectral_clustering(cls, X: np.ndarray, k_value: Union[int, str]) -> np.ndarray:  # type: ignore
        """
        Perform spectral clustering.

        Args:
            X: The feature matrix
            k_value: The number of clusters or 'auto'

        Returns:
            A numpy array containing the cluster assignments
        """
        if k_value == "auto":
            # Determine optimal k using silhouette score
            k_value = cls._find_optimal_k(X, algorithm="spectral")

        # Ensure k_value is an integer
        k = int(k_value)

        # Perform spectral clustering
        spectral = SpectralClustering(n_clusters=k, assign_labels="discretize", random_state=42)
        return spectral.fit_predict(X)  # type: ignore

    @classmethod
    def _perform_affinity_clustering(cls, X: np.ndarray, k_value: Union[int, str]) -> np.ndarray:  # type: ignore
        """
        Perform affinity propagation clustering.

        Args:
            X: The feature matrix
            k_value: The number of clusters or 'auto'

        Returns:
            A numpy array containing the cluster assignments
        """
        # Affinity propagation doesn't require a specific number of clusters
        # and automatically determines the optimal number
        # We'll use scikit-learn's AffinityPropagation
        from sklearn.cluster import AffinityPropagation

        # If k_value is not 'auto', we use it to adjust the damping parameter
        if k_value == "auto":
            # Use default parameters
            affinity = AffinityPropagation(random_state=42)
        else:
            # Use k_value to adjust the damping parameter
            # Higher damping tends to produce fewer clusters
            k = int(k_value)
            damping = 0.5 + 0.3 * (1.0 / k)  # Adjust damping based on k
            damping = min(0.99, max(0.5, damping))  # Keep damping between 0.5 and 0.99
            affinity = AffinityPropagation(damping=damping, random_state=42)

        return affinity.fit_predict(X)  # type: ignore

    @classmethod
    def _find_optimal_k(cls, X: np.ndarray, algorithm: str, max_k: int = 10) -> int:  # type: ignore
        """
        Find the optimal number of clusters using silhouette score.

        Args:
            X: The feature matrix
            algorithm: The clustering algorithm to use
            max_k: The maximum number of clusters to consider

        Returns:
            The optimal number of clusters
        """
        # Start from 2 clusters (silhouette score requires at least 2 clusters)
        k_range = range(2, min(max_k + 1, len(X)))

        # If we have too few samples, return 2
        if len(k_range) == 0:
            return 2

        best_score = -1
        best_k = 2

        for k in k_range:
            # Perform clustering with the current k
            if algorithm == "kmeans":
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
            elif algorithm == "hierarchical":
                hierarchical = AgglomerativeClustering(n_clusters=k)
                labels = hierarchical.fit_predict(X)
            elif algorithm == "spectral":
                spectral = SpectralClustering(n_clusters=k, assign_labels="discretize", random_state=42)
                labels = spectral.fit_predict(X)
            else:
                # Default to K-means
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)

            # Calculate silhouette score
            # Skip if there's only one cluster or if all points are in the same cluster
            unique_labels = np.unique(labels)
            if len(unique_labels) <= 1 or len(unique_labels) >= len(X):
                continue

            score = silhouette_score(X, labels)

            # Update best_k if we found a better score
            if score > best_score:
                best_score = score
                best_k = k

        return best_k
