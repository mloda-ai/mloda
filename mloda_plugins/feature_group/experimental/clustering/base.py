"""
Base implementation for clustering feature groups.
"""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda_core.abstract_plugins.components.feature_chainer.feature_chainer_parser_configuration import (
    FeatureChainParserConfiguration,
    create_configurable_parser,
)
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class ClusteringFeatureGroup(AbstractFeatureGroup):
    # Option keys for clustering configuration
    ALGORITHM = "algorithm"
    K_VALUE = "k_value"
    """
    Base class for all clustering feature groups.

    Clustering feature groups group similar data points using various clustering algorithms.
    They allow you to identify patterns and structures in your data by grouping similar
    observations together.

    ## Feature Naming Convention

    Clustering features follow this naming pattern:
    `cluster_{algorithm}_{k_value}__{mloda_source_features}`

    The source features (mloda_source_features) are extracted from the feature name and used
    as input for the clustering algorithm. Note the double underscore before the source features.

    Examples:
    - `cluster_kmeans_5__customer_behavior`: K-means clustering with 5 clusters on customer behavior data
    - `cluster_hierarchical_3__transaction_patterns`: Hierarchical clustering with 3 clusters on transaction patterns
    - `cluster_dbscan_auto__sensor_readings`: DBSCAN clustering with automatic cluster detection on sensor readings

    ## Configuration-Based Creation

    ClusteringFeatureGroup supports configuration-based creation through the
    FeatureChainParserConfiguration mechanism. This allows features to be created
    from options rather than explicit feature names.

    To create a clustering feature using configuration:

    ```python
    feature = Feature(
        "PlaceHolder",  # Placeholder name, will be replaced
        Options({
            ClusteringFeatureGroup.ALGORITHM: "kmeans",
            ClusteringFeatureGroup.K_VALUE: 5,
            DefaultOptionKeys.mloda_source_feature: "customer_behavior"
        })
    )

    # The Engine will automatically parse this into a feature with name "cluster_kmeans_5__customer_behavior"
    ```

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
    PREFIX_PATTERN = r"^cluster_([\w]+)_([\w]+)__"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source features from the clustering feature name."""
        source_features_str = FeatureChainParser.extract_source_feature(feature_name.name, self.PREFIX_PATTERN)

        # Handle multiple source features (comma-separated)
        source_features = set()
        for feature in source_features_str.split(","):
            source_features.add(Feature(feature.strip()))

        return source_features

    @classmethod
    def parse_clustering_prefix(cls, feature_name: str) -> tuple[str, str]:
        """
        Parse the clustering prefix into its components.

        Args:
            feature_name: The feature name to parse

        Returns:
            A tuple containing (algorithm, k_value)

        Raises:
            ValueError: If the prefix doesn't match the expected pattern
        """
        # Extract the prefix part (everything before the double underscore)
        prefix_end = feature_name.find("__")
        if prefix_end == -1:
            raise ValueError(
                f"Invalid clustering feature name format: {feature_name}. Missing double underscore separator."
            )

        prefix = feature_name[:prefix_end]

        # Parse the prefix components
        parts = prefix.split("_")
        if len(parts) != 3 or parts[0] != "cluster":
            raise ValueError(
                f"Invalid clustering feature name format: {feature_name}. "
                f"Expected format: cluster_{{algorithm}}_{{k_value}}__{{mloda_source_features}}"
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
    def get_algorithm(cls, feature_name: str) -> str:
        """Extract the clustering algorithm from the feature name."""
        return cls.parse_clustering_prefix(feature_name)[0]

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
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name

        try:
            # First validate that this is a valid feature name for this feature group
            if not FeatureChainParser.validate_feature_name(feature_name, cls.PREFIX_PATTERN):
                return False

            # Then validate the clustering components
            cls.parse_clustering_prefix(feature_name)
            return True
        except ValueError:
            return False

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Perform clustering operations.

        Processes all requested features, determining the clustering algorithm,
        k_value, and source features from each feature name.

        Adds the clustering results directly to the input data structure.
        """
        # Process each requested feature
        for feature_name in features.get_all_names():
            algorithm, k_value = cls.parse_clustering_prefix(feature_name)
            source_features_str = FeatureChainParser.extract_source_feature(feature_name, cls.PREFIX_PATTERN)

            # Parse source features (comma-separated)
            source_features = [feature.strip() for feature in source_features_str.split(",")]

            # Check if all source features exist
            for source_feature in source_features:
                cls._check_source_feature_exists(data, source_feature)

            # Convert k_value to int if it's not 'auto'
            if k_value == "auto":
                k_value_parsed: Union[int, str] = "auto"
            else:
                k_value_parsed = int(k_value)

            # Perform clustering
            result = cls._perform_clustering(data, algorithm, k_value_parsed, source_features)

            # Add the result to the data
            data = cls._add_result_to_data(data, feature_name, result)

        return data

    @classmethod
    def _check_source_feature_exists(cls, data: Any, feature_name: str) -> None:
        """
        Check if the source feature exists in the data.

        Args:
            data: The input data
            feature_name: The name of the feature to check

        Raises:
            ValueError: If the feature does not exist in the data
        """
        raise NotImplementedError(f"_check_source_feature_exists not implemented in {cls.__name__}")

    @classmethod
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
        raise NotImplementedError(f"_add_result_to_data not implemented in {cls.__name__}")

    @classmethod
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
        raise NotImplementedError(f"_perform_clustering not implemented in {cls.__name__}")

    @classmethod
    def configurable_feature_chain_parser(cls) -> Optional[Type[FeatureChainParserConfiguration]]:
        """
        Returns the FeatureChainParserConfiguration class for this feature group.

        This method allows the Engine to automatically create features with the correct
        naming convention based on configuration options, rather than requiring explicit
        feature names.

        Returns:
            A configured FeatureChainParserConfiguration class
        """

        # Define validation functions within the method scope
        def raise_unsupported_algorithm(algorithm: str) -> bool:
            """Raise an error for unsupported clustering algorithm."""
            raise ValueError(
                f"Unsupported clustering algorithm: {algorithm}. "
                f"Supported algorithms: {list(cls.CLUSTERING_ALGORITHMS.keys())}"
            )

        def validate_k_value(k_value: Union[int, str]) -> bool:
            """Validate the k_value."""
            if k_value == "auto":
                return True

            try:
                k_value_int = int(k_value) if isinstance(k_value, str) else k_value
                if k_value_int <= 0:
                    raise ValueError("k_value must be positive")
                return True
            except (ValueError, TypeError):
                raise ValueError(f"Invalid k_value: {k_value}. Must be a positive integer or 'auto'.")

        # Create and return the configured parser
        return create_configurable_parser(
            parse_keys=[
                cls.ALGORITHM,
                cls.K_VALUE,
                DefaultOptionKeys.mloda_source_feature,
            ],
            feature_name_template="cluster_{algorithm}_{k_value}__{mloda_source_feature}",
            validation_rules={
                cls.ALGORITHM: lambda x: x in cls.CLUSTERING_ALGORITHMS or raise_unsupported_algorithm(x),
                cls.K_VALUE: validate_k_value,
            },
        )
