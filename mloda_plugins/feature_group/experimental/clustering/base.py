"""
Base implementation for clustering feature groups.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from mloda.provider import OperationFeatureGroup
from mloda.user import Feature
from mloda.provider import (
    FeatureChainParser,
    FeatureChainParserMixin,
)
from mloda.provider import FeatureSet
from mloda.provider import DefaultOptionKeys
from mloda.provider import PropertySpec, is_positive_int


def _is_bool(value: Any) -> bool:
    """Accept only a real bool, so 1 or "true" is rejected rather than silently coerced."""
    return isinstance(value, bool)


class ClusteringFeatureGroup(FeatureChainParserMixin, OperationFeatureGroup):
    """
    Base class for all clustering feature groups.

    Clustering feature groups group similar data points using various clustering algorithms.
    They allow you to identify patterns and structures in your data by grouping similar
    observations together.

    ## Feature Naming Convention

    Clustering features follow this naming pattern:
    `{in_features}__cluster_{algorithm}_{k_value}`

    The source features come first, followed by the clustering operation.
    Note the double underscore separating the source features from the operation.

    Examples:
    - `customer_behavior__cluster_kmeans_5`: K-means clustering with 5 clusters on customer behavior data
    - `transaction_patterns__cluster_hierarchical_3`: Hierarchical clustering with 3 clusters on transaction patterns
    - `sensor_readings__cluster_dbscan_auto`: DBSCAN clustering with automatic cluster detection on sensor readings

    ## Configuration-Based Creation

    ClusteringFeatureGroup supports configuration-based creation using the new Options
    group/context architecture. This allows features to be created from options rather
    than explicit feature names.

    To create a clustering feature using configuration:

    ```python
    feature = Feature(
        name="placeholder",  # Placeholder name, will be replaced
        options=Options(
            context={
                ClusteringFeatureGroup.ALGORITHM: "kmeans",
                ClusteringFeatureGroup.K_VALUE: 5,
                DefaultOptionKeys.in_features: "customer_behavior",
            }
        )
    )

    # The Engine will automatically parse this into a feature with name "customer_behavior__cluster_kmeans_5"
    ```

    ## Parameter Classification

    ### Context Parameters (Default)
    These parameters don't affect Feature Group resolution/splitting:
    - `algorithm`: The clustering algorithm to use
    - `k_value`: The number of clusters or 'auto' for automatic determination
    - `in_features`: The source features to use for clustering

    ### Group Parameters
    Currently none for ClusteringFeatureGroup. Parameters that affect Feature Group
    resolution/splitting would be placed here.

    ## Supported Clustering Algorithms

    - `kmeans`: K-means clustering
    - `hierarchical`: Hierarchical clustering
    - `dbscan`: Density-Based Spatial Clustering of Applications with Noise
    - `spectral`: Spectral clustering
    - `agglomerative`: Agglomerative clustering
    - `affinity`: Affinity propagation

    ## Requirements
    - The input data must contain the source features to be used for clustering
    - For algorithms that require a specific number of clusters (like k-means), the k_value must be provided
    - For algorithms that don't require a specific number of clusters (like DBSCAN), use 'auto' as the k_value
    """

    # Option keys for clustering configuration
    ALGORITHM = "algorithm"
    K_VALUE = "k_value"
    OUTPUT_PROBABILITIES = "output_probabilities"

    # Define supported clustering algorithms
    CLUSTERING_ALGORITHMS = {
        "kmeans": "K-means clustering",
        "hierarchical": "Hierarchical clustering",
        "dbscan": "Density-Based Spatial Clustering of Applications with Noise",
        "spectral": "Spectral clustering",
        "agglomerative": "Agglomerative clustering",
        "affinity": "Affinity propagation",
    }

    # Define the prefix pattern for this feature group
    PREFIX_PATTERN = r".*__cluster_(?P<algorithm>[\w]+)_(?P<k_value>[\w]+)$"

    # In-feature configuration for FeatureChainParserMixin
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = None  # Unlimited in_features allowed

    # Property mapping for configuration-based feature creation
    PROPERTY_MAPPING = {
        ALGORITHM: PropertySpec(
            "Clustering algorithm to use",
            allowed_values=CLUSTERING_ALGORITHMS,
            context=True,
            strict_validation=True,
        ),
        K_VALUE: PropertySpec(
            "Number of clusters or 'auto' for automatic determination",
            context=True,
            strict_validation=True,
            element_validator=lambda value: value == "auto" or is_positive_int(value),
        ),
        DefaultOptionKeys.in_features: PropertySpec(
            "Source features to use for clustering",
            context=True,
            strict_validation=False,
        ),
        # Both hooks share _is_bool. element_validator produces the rejection message and checks the
        # declared default at construction; it runs on BOTH match paths, so it alone enforces the
        # value space. match_guard additionally judges the raw, un-unpacked value.
        OUTPUT_PROBABILITIES: PropertySpec(
            "Whether to output cluster probabilities/distances as separate columns using ~N suffix pattern",
            context=True,
            strict_validation=True,
            default=False,
            element_validator=_is_bool,
            match_guard=_is_bool,
        ),
    }

    @classmethod
    def parse_clustering_prefix(cls, feature_name: str) -> tuple[str, str]:
        """
        Parse the clustering suffix into its components.

        Args:
            feature_name: The feature name to parse

        Returns:
            A tuple containing (algorithm, k_value)

        Raises:
            ValueError: If the suffix doesn't match the expected pattern
        """
        parsed = FeatureChainParser.parse_name(feature_name, cls._get_prefix_patterns())
        if not parsed.matched:
            raise ValueError(
                f"Invalid clustering feature name format: {feature_name}. "
                f"Expected format: {{in_features}}__cluster_{{algorithm}}_{{k_value}}"
            )

        captures = cast(dict[str, str], parsed.named_captures)
        algorithm = captures[cls.ALGORITHM]
        k_value = captures[cls.K_VALUE]

        # Validate algorithm
        if algorithm not in cls.CLUSTERING_ALGORITHMS:
            raise ValueError(
                f"Unsupported clustering algorithm: {algorithm}. "
                f"Supported algorithms: {', '.join(cls.CLUSTERING_ALGORITHMS.keys())}"
            )

        # Validate k_value
        if k_value != "auto" and not k_value.isdigit():
            raise ValueError(f"Invalid k_value: {k_value}. Must be a positive integer or 'auto'.")

        if k_value != "auto" and int(k_value) <= 0:
            raise ValueError("k_value must be positive")

        return algorithm, k_value

    @classmethod
    def get_k_value(cls, feature_name: str) -> int | str:
        """
        Extract the k_value from the feature name.

        Returns:
            An integer k_value or the string 'auto'
        """
        k_value = cls.parse_clustering_prefix(feature_name)[1]
        return k_value if k_value == "auto" else int(k_value)

    @classmethod
    def _extract_clustering_params(cls, feature: Feature) -> tuple[str | None, int | str | None]:
        """
        Extract algorithm and k_value from a feature.

        Each value comes from the feature name when it owns the match, otherwise from options.

        Args:
            feature: The feature to extract parameters from

        Returns:
            Tuple of (algorithm, k_value) or (None, None) if extraction fails

        Raises:
            ValueError: If string-based parsing fails due to invalid format
        """
        algorithm = cls._resolve_operation(feature, cls.ALGORITHM)
        k_value_raw = cls._resolve_operation(feature, cls.K_VALUE)

        if k_value_raw is None:
            return algorithm, None

        k_value = "auto" if k_value_raw == "auto" else int(k_value_raw)
        return algorithm, k_value

    @classmethod
    def _extract_algorithm_k_value_and_source_features(cls, feature: Feature) -> tuple[str, int | str, list[str]]:
        """
        Extract algorithm, k_value, and source features from a feature.

        Tries string-based approach first, falls back to configuration-based.

        Args:
            feature: The feature to extract parameters from

        Returns:
            Tuple of (algorithm, k_value, source_features_list)

        Raises:
            ValueError: If parameters cannot be extracted
        """
        source_features = cls._extract_validated_source_features(feature)
        algorithm, k_value = cls._extract_clustering_params(feature)

        if algorithm is None or k_value is None:
            raise ValueError(f"Could not extract algorithm and k_value from: {feature.name}")

        return algorithm, k_value, source_features

    @classmethod
    def cluster(
        cls,
        data: Any,
        features: FeatureSet,
        *,
        perform: Callable[[Any, str, int | str, list[str]], Any],
        perform_with_probabilities: Callable[[Any, str, int | str, list[str]], tuple[Any, Any]],
        add_column: Callable[[Any, str, Any], Any],
        columns: Callable[[Any], set[str]],
    ) -> Any:
        """Shared loop: resolve each feature's sources, cluster, add labels (and probability columns)."""
        for feature in features.get_sorted_features():
            algorithm, k_value, source_features = cls._extract_algorithm_k_value_and_source_features(feature)

            available = columns(data)
            resolved: list[str] = []
            for source_feature in source_features:
                resolved.extend(cls.resolve_multi_column_feature(source_feature, available))

            if not any(name in available for name in resolved):
                listing = sorted(available, key=str)
                raise ValueError(f"None of the source features {resolved} found in data. Available columns: {listing}")

            if feature.options.get(cls.OUTPUT_PROBABILITIES):
                result, probabilities = perform_with_probabilities(data, algorithm, k_value, resolved)
                data = add_column(data, feature.name, result)
                for cluster_idx in range(probabilities.shape[1]):
                    data = add_column(data, f"{feature.name}~{cluster_idx}", probabilities[:, cluster_idx])
            else:
                result = perform(data, algorithm, k_value, resolved)
                data = add_column(data, feature.name, result)

        return data
