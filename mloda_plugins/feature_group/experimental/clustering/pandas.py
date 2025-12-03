"""
Pandas implementation for clustering feature groups.
"""

from __future__ import annotations

from typing import Any, List, Set, TYPE_CHECKING, Union, cast

if TYPE_CHECKING:
    from numpy.typing import NDArray

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
    np = None  # type: ignore[assignment]


from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.clustering.base import ClusteringFeatureGroup


class PandasClusteringFeatureGroup(ClusteringFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFrameWork]]:
        """Define the compute framework for this feature group."""
        return {PandasDataFrame}

    @classmethod
    def _get_available_columns(cls, data: pd.DataFrame) -> Set[str]:
        """Get the set of available column names from the DataFrame."""
        return set(data.columns)

    @classmethod
    def _check_source_features_exist(cls, data: pd.DataFrame, feature_names: List[str]) -> None:
        """
        Check if the resolved features exist in the DataFrame.

        Args:
            data: The Pandas DataFrame
            feature_names: List of resolved feature names (may contain ~N suffixes)

        Raises:
            ValueError: If none of the resolved features exist in the data
        """
        missing_features = [name for name in feature_names if name not in data.columns]
        if len(missing_features) == len(feature_names):
            raise ValueError(
                f"None of the source features {feature_names} found in data. Available columns: {list(data.columns)}"
            )

    @classmethod
    def _add_result_to_data(cls, data: "pd.DataFrame", feature_name: str, result: "NDArray[Any]") -> "pd.DataFrame":
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
    ) -> "NDArray[Any]":
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
        df = cast("pd.DataFrame", data)

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
    def _perform_kmeans_clustering(cls, X: "NDArray[Any]", k_value: Union[int, str]) -> "NDArray[Any]":
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
        return cast("NDArray[Any]", kmeans.fit_predict(X))

    @classmethod
    def _perform_hierarchical_clustering(cls, X: "NDArray[Any]", k_value: Union[int, str]) -> "NDArray[Any]":
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
        return cast("NDArray[Any]", hierarchical.fit_predict(X))

    @classmethod
    def _perform_dbscan_clustering(cls, X: "NDArray[Any]", k_value: Union[int, str]) -> "NDArray[Any]":
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

        return cast("NDArray[Any]", dbscan.fit_predict(X))

    @classmethod
    def _perform_spectral_clustering(cls, X: "NDArray[Any]", k_value: Union[int, str]) -> "NDArray[Any]":
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
        return cast("NDArray[Any]", spectral.fit_predict(X))

    @classmethod
    def _perform_affinity_clustering(cls, X: "NDArray[Any]", k_value: Union[int, str]) -> "NDArray[Any]":
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

        return cast("NDArray[Any]", affinity.fit_predict(X))

    @classmethod
    def _find_optimal_k(cls, X: "NDArray[Any]", algorithm: str, max_k: int = 10) -> int:
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

    @classmethod
    def _perform_clustering_with_probabilities(
        cls,
        data: Any,
        algorithm: str,
        k_value: Union[int, str],
        source_features: List[str],
    ) -> tuple["NDArray[Any]", "NDArray[Any]"]:
        """
        Perform clustering and return both labels and probabilities/distances.

        For each algorithm, we compute cluster probabilities or distances:
        - KMeans: Uses transform() to get distances to cluster centers, then converts to probabilities
        - Hierarchical/Agglomerative: Computes distances to cluster centroids
        - DBSCAN: Returns binary membership (1 for assigned cluster, 0 for others)
        - Spectral: Computes distances to cluster centroids
        - Affinity: Returns binary membership based on exemplar assignment

        Args:
            data: The pandas DataFrame
            algorithm: The clustering algorithm to use
            k_value: The number of clusters (or 'auto' for algorithms that determine this automatically)
            source_features: The list of source features to use for clustering

        Returns:
            A tuple of (cluster_labels, probabilities) where probabilities[i, j] is the
            probability/distance of sample i belonging to cluster j
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
            return cls._perform_kmeans_clustering_with_probabilities(X_scaled, k_value)
        elif algorithm == "hierarchical" or algorithm == "agglomerative":
            return cls._perform_hierarchical_clustering_with_probabilities(X_scaled, k_value)
        elif algorithm == "dbscan":
            return cls._perform_dbscan_clustering_with_probabilities(X_scaled, k_value)
        elif algorithm == "spectral":
            return cls._perform_spectral_clustering_with_probabilities(X_scaled, k_value)
        elif algorithm == "affinity":
            return cls._perform_affinity_clustering_with_probabilities(X_scaled, k_value)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    @classmethod
    def _perform_kmeans_clustering_with_probabilities(
        cls,
        X: "NDArray[Any]",
        k_value: Union[int, str],
    ) -> tuple["NDArray[Any]", "NDArray[Any]"]:
        """
        Perform K-means clustering and return probabilities based on distances to centroids.
        """
        if k_value == "auto":
            k_value = cls._find_optimal_k(X, algorithm="kmeans")

        k = int(k_value)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Get distances to cluster centers
        distances = kmeans.transform(X)

        # Convert distances to probabilities using softmax-like transformation
        # Smaller distance = higher probability
        # Use negative distances and apply softmax
        neg_distances = -distances
        # Normalize using softmax to get probabilities
        exp_neg_dist = np.exp(neg_distances - np.max(neg_distances, axis=1, keepdims=True))
        probabilities = exp_neg_dist / np.sum(exp_neg_dist, axis=1, keepdims=True)

        return labels, probabilities

    @classmethod
    def _perform_hierarchical_clustering_with_probabilities(
        cls,
        X: "NDArray[Any]",
        k_value: Union[int, str],
    ) -> tuple["NDArray[Any]", "NDArray[Any]"]:
        """
        Perform hierarchical clustering and compute distances to cluster centroids.
        """
        if k_value == "auto":
            k_value = cls._find_optimal_k(X, algorithm="hierarchical")

        k = int(k_value)
        hierarchical = AgglomerativeClustering(n_clusters=k)
        labels = hierarchical.fit_predict(X)

        # Compute cluster centroids
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Compute distances from each point to each centroid
        from scipy.spatial.distance import cdist

        distances = cdist(X, centroids, metric="euclidean")

        # Convert distances to probabilities
        neg_distances = -distances
        exp_neg_dist = np.exp(neg_distances - np.max(neg_distances, axis=1, keepdims=True))
        probabilities = exp_neg_dist / np.sum(exp_neg_dist, axis=1, keepdims=True)

        return labels, probabilities

    @classmethod
    def _perform_dbscan_clustering_with_probabilities(
        cls,
        X: "NDArray[Any]",
        k_value: Union[int, str],
    ) -> tuple["NDArray[Any]", "NDArray[Any]"]:
        """
        Perform DBSCAN clustering. For probabilities, returns binary membership.
        """
        # DBSCAN doesn't provide probabilities, so we use binary membership
        if k_value == "auto":
            dbscan = DBSCAN(eps=0.5, min_samples=5)
        else:
            k = int(k_value)
            eps = 1.0 / k
            dbscan = DBSCAN(eps=eps, min_samples=5)

        labels = dbscan.fit_predict(X)

        # Get unique cluster labels (excluding noise label -1)
        unique_labels = np.unique(labels[labels >= 0])
        n_clusters = len(unique_labels)

        # Create binary membership matrix
        probabilities = np.zeros((len(X), max(n_clusters, 1)))
        for i, label in enumerate(labels):
            if label >= 0:
                # Find the index of this cluster in the unique labels
                cluster_idx = np.where(unique_labels == label)[0][0]
                probabilities[i, cluster_idx] = 1.0

        return labels, probabilities

    @classmethod
    def _perform_spectral_clustering_with_probabilities(
        cls,
        X: "NDArray[Any]",
        k_value: Union[int, str],
    ) -> tuple["NDArray[Any]", "NDArray[Any]"]:
        """
        Perform spectral clustering and compute distances to cluster centroids.
        """
        if k_value == "auto":
            k_value = cls._find_optimal_k(X, algorithm="spectral")

        k = int(k_value)
        spectral = SpectralClustering(n_clusters=k, assign_labels="discretize", random_state=42)
        labels = spectral.fit_predict(X)

        # Compute cluster centroids
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Compute distances from each point to each centroid
        from scipy.spatial.distance import cdist

        distances = cdist(X, centroids, metric="euclidean")

        # Convert distances to probabilities
        neg_distances = -distances
        exp_neg_dist = np.exp(neg_distances - np.max(neg_distances, axis=1, keepdims=True))
        probabilities = exp_neg_dist / np.sum(exp_neg_dist, axis=1, keepdims=True)

        return labels, probabilities

    @classmethod
    def _perform_affinity_clustering_with_probabilities(
        cls,
        X: "NDArray[Any]",
        k_value: Union[int, str],
    ) -> tuple["NDArray[Any]", "NDArray[Any]"]:
        """
        Perform affinity propagation clustering. Returns binary membership based on exemplar assignment.
        """
        from sklearn.cluster import AffinityPropagation

        if k_value == "auto":
            affinity = AffinityPropagation(random_state=42)
        else:
            k = int(k_value)
            damping = 0.5 + 0.3 * (1.0 / k)
            damping = min(0.99, max(0.5, damping))
            affinity = AffinityPropagation(damping=damping, random_state=42)

        labels = affinity.fit_predict(X)

        # Get the number of clusters
        n_clusters = len(np.unique(labels))

        # Create binary membership matrix
        probabilities = np.zeros((len(X), n_clusters))
        for i, label in enumerate(labels):
            probabilities[i, label] = 1.0

        return labels, probabilities
