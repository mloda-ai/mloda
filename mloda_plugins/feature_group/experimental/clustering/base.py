"""
Base implementation for clustering feature groups.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, Optional, Set, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class ClusteringFeatureGroup(AbstractFeatureGroup):
    # Option keys for clustering configuration
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
    PREFIX_PATTERN = r".*__cluster_([\w]+)_([\w]+)$"

    # Property mapping for configuration-based feature creation
    PROPERTY_MAPPING = {
        ALGORITHM: {
            **CLUSTERING_ALGORITHMS,  # All supported algorithms as valid values
            DefaultOptionKeys.mloda_context: True,  # Mark as context parameter
            DefaultOptionKeys.mloda_strict_validation: True,  # Enable strict validation
        },
        K_VALUE: {
            "explanation": "Number of clusters or 'auto' for automatic determination",
            DefaultOptionKeys.mloda_context: True,  # Mark as context parameter
            DefaultOptionKeys.mloda_strict_validation: True,  # Enable strict validation
            DefaultOptionKeys.mloda_validation_function: lambda value: value == "auto"
            or (isinstance(value, (int, str)) and str(value).isdigit() and int(value) > 0),
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source features to use for clustering",
            DefaultOptionKeys.mloda_context: True,  # Mark as context parameter
            DefaultOptionKeys.mloda_strict_validation: False,  # Flexible validation
        },
        OUTPUT_PROBABILITIES: {
            "explanation": "Whether to output cluster probabilities/distances as separate columns using ~N suffix pattern",
            DefaultOptionKeys.mloda_context: True,  # Mark as context parameter
            DefaultOptionKeys.mloda_strict_validation: False,  # Flexible validation
            DefaultOptionKeys.mloda_default: False,  # Default is False (don't output probabilities)
            DefaultOptionKeys.mloda_validation_function: lambda value: isinstance(value, bool),
        },
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source features from either string parsing or configuration-based options."""

        # string based
        source_features_str: str | None = None
        _, source_features_str = FeatureChainParser.parse_feature_name(feature_name, [self.PREFIX_PATTERN])

        if source_features_str is not None:
            # Handle multiple source features (ampersand-separated)
            source_features = set()
            for feature in source_features_str.split("&"):
                source_features.add(Feature(feature.strip()))
            return source_features

        # configuration based
        source_features_frozen = options.get_in_features()
        if len(source_features_frozen) < 1:
            raise ValueError(f"Feature '{feature_name}' requires at least one source feature, but none were provided.")
        return set(source_features_frozen)

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
        # Extract the suffix part (everything after the double underscore)
        suffix_start = feature_name.find("__")
        if suffix_start == -1:
            raise ValueError(
                f"Invalid clustering feature name format: {feature_name}. Missing double underscore separator."
            )

        suffix = feature_name[suffix_start + 2 :]

        # Parse the suffix components
        parts = suffix.split("_")
        if len(parts) != 3 or parts[0] != "cluster":
            raise ValueError(
                f"Invalid clustering feature name format: {feature_name}. "
                f"Expected format: {{in_features}}__cluster_{{algorithm}}_{{k_value}}"
            )

        algorithm, k_value = parts[1], parts[2]

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
    def get_k_value(cls, feature_name: str) -> Union[int, str]:
        """
        Extract the k_value from the feature name.

        Returns:
            An integer k_value or the string 'auto'
        """
        k_value = cls.parse_clustering_prefix(feature_name)[1]
        return k_value if k_value == "auto" else int(k_value)

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        """Check if feature name matches the expected pattern for clustering features."""

        # Use the unified parser with property mapping for full configuration support
        result = FeatureChainParser.match_configuration_feature_chain_parser(
            feature_name,
            options,
            property_mapping=cls.PROPERTY_MAPPING,
            prefix_patterns=[cls.PREFIX_PATTERN],
        )

        # If it matches and it's a string-based feature, validate with our custom logic
        if result:
            feature_name_str = feature_name.name if isinstance(feature_name, FeatureName) else feature_name

            # Check if this is a string-based feature (contains the pattern)
            if FeatureChainParser.is_chained_feature(feature_name_str):
                try:
                    # Use existing validation logic that validates algorithm and k_value
                    cls.parse_clustering_prefix(feature_name_str)
                except ValueError:
                    # If validation fails, this feature doesn't match
                    return False
        return result

    @classmethod
    def _extract_algorithm_k_value_and_source_features(cls, feature: Feature) -> tuple[str, Union[int, str], list[str]]:
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
        algorithm = None
        k_value: str | int | None = None
        source_features = None

        # string based
        algorithm_str, source_features_str = FeatureChainParser.parse_feature_name(feature.name, [cls.PREFIX_PATTERN])
        if algorithm_str is not None and source_features_str is not None:
            # Parse the algorithm and k_value from the prefix
            algorithm, k_value_str = cls.parse_clustering_prefix(feature.get_name())

            # Convert k_value to appropriate type
            if k_value_str == "auto":
                k_value = "auto"
            else:
                k_value = int(k_value_str)

            # Parse source features (ampersand-separated)
            source_features = [feature.strip() for feature in source_features_str.split("&")]

            return algorithm, k_value, source_features

        # configuration based
        source_features_frozen = feature.options.get_in_features()
        source_features = [source_feature.get_name() for source_feature in source_features_frozen]

        algorithm = feature.options.get(cls.ALGORITHM)
        k_value_raw = feature.options.get(cls.K_VALUE)

        # Convert k_value to appropriate type
        if k_value_raw == "auto":
            k_value = "auto"
        else:
            k_value = int(k_value_raw)

        if algorithm is None or k_value is None or not source_features:
            raise ValueError(f"Could not extract algorithm, k_value, and source features from: {feature.name}")

        return algorithm, k_value, source_features

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Perform clustering operations.

        Processes all requested features, determining the clustering algorithm,
        k_value, and source features from either string parsing or configuration-based options.

        Supports multi-column features by using resolve_multi_column_feature() to
        automatically discover columns matching the pattern feature_name~N.

        Optionally outputs cluster probabilities/distances as separate columns using
        the ~N suffix pattern when output_probabilities=True.

        Adds the clustering results directly to the input data structure.
        """
        # Process each requested feature
        for feature in features.features:
            algorithm, k_value, source_features = cls._extract_algorithm_k_value_and_source_features(feature)

            # Resolve multi-column features automatically
            # If source_features contains "onehot_encoded__product", this discovers
            # ["onehot_encoded__product~0", "onehot_encoded__product~1", ...]
            available_columns = cls._get_available_columns(data)
            resolved_features = []
            for source_feature in source_features:
                resolved_columns = cls.resolve_multi_column_feature(source_feature, available_columns)
                resolved_features.extend(resolved_columns)

            # Check that resolved features exist
            cls._check_source_features_exist(data, resolved_features)

            # Check if we should output probabilities
            output_probabilities = (
                feature.options.get(cls.OUTPUT_PROBABILITIES)
                if feature.options.get(cls.OUTPUT_PROBABILITIES) is not None
                else False
            )

            # Perform clustering
            if output_probabilities:
                # Get both cluster labels and probabilities
                result, probabilities = cls._perform_clustering_with_probabilities(
                    data, algorithm, k_value, resolved_features
                )

                # Add the cluster labels
                data = cls._add_result_to_data(data, feature.get_name(), result)

                # Add probability columns using ~N suffix pattern
                for cluster_idx in range(probabilities.shape[1]):
                    prob_column_name = f"{feature.get_name()}~{cluster_idx}"
                    data = cls._add_result_to_data(data, prob_column_name, probabilities[:, cluster_idx])
            else:
                # Original behavior: only output cluster labels
                result = cls._perform_clustering(data, algorithm, k_value, resolved_features)
                data = cls._add_result_to_data(data, feature.get_name(), result)

        return data

    @classmethod
    @abstractmethod
    def _get_available_columns(cls, data: Any) -> Set[str]:
        """
        Get the set of available column names from the data.

        Args:
            data: The input data

        Returns:
            Set of column names available in the data
        """
        ...

    @classmethod
    @abstractmethod
    def _check_source_features_exist(cls, data: Any, feature_names: List[str]) -> None:
        """
        Check if the resolved source features exist in the data.

        Args:
            data: The input data
            feature_names: List of resolved feature names (may contain ~N suffixes)

        Raises:
            ValueError: If none of the features exist in the data
        """
        ...

    @classmethod
    @abstractmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """
        Add the result to the data.

        Args:
            data: The input data
            feature_name: The name of the feature to add
            result: The result to add

        Returns:
            The updated data
        """
        ...

    @classmethod
    @abstractmethod
    def _perform_clustering(
        cls,
        data: Any,
        algorithm: str,
        k_value: Union[int, str],
        source_features: list[str],
    ) -> Any:
        """
        Method to perform the clustering. Should be implemented by subclasses.

        Args:
            data: The input data
            algorithm: The clustering algorithm to use
            k_value: The number of clusters (or 'auto' for algorithms that determine this automatically)
            source_features: The list of source features to use for clustering

        Returns:
            The result of the clustering (typically cluster assignments)
        """
        ...

    @classmethod
    @abstractmethod
    def _perform_clustering_with_probabilities(
        cls,
        data: Any,
        algorithm: str,
        k_value: Union[int, str],
        source_features: list[str],
    ) -> tuple[Any, Any]:
        """
        Method to perform clustering and return both labels and probabilities/distances.

        Args:
            data: The input data
            algorithm: The clustering algorithm to use
            k_value: The number of clusters (or 'auto' for algorithms that determine this automatically)
            source_features: The list of source features to use for clustering

        Returns:
            A tuple of (cluster_labels, probabilities) where:
            - cluster_labels: Array of cluster assignments for each sample
            - probabilities: 2D array where probabilities[i, j] is the probability/distance
                           of sample i belonging to cluster j
        """
        ...
