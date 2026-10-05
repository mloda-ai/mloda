"""
Base implementation for dimensionality reduction feature groups.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, cast

from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.provider import (
    FeatureChainParser,
    FeatureChainParserMixin,
)
from mloda.provider import COLUMNWISE_HOOKS
from mloda.provider import FeatureSet
from mloda.user import Options
from mloda.provider import DefaultOptionKeys
from mloda.provider import PropertySpec, is_positive_int


class DimensionalityReductionFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """
    Base class for all dimensionality reduction feature groups.

    Dimensionality reduction feature groups reduce the dimensionality of feature spaces
    using various techniques like PCA, t-SNE, Isomap, etc. They support both string-based
    feature creation and configuration-based creation with proper group/context parameter separation.

    ## Supported Dimensionality Reduction Algorithms

    - `pca`: Principal Component Analysis
    - `tsne`: t-Distributed Stochastic Neighbor Embedding
    - `ica`: Independent Component Analysis
    - `lda`: Linear Discriminant Analysis
    - `isomap`: Isometric Mapping

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features follow the naming pattern: `{in_features}__{algorithm}_{dimension}d`

    Examples:
    ```python
    features = [
        "customer_metrics__pca_2d",      # PCA reduction to 2 dimensions
        "product_features__tsne_3d",     # t-SNE reduction to 3 dimensions
        "sensor_readings__isomap_10d"    # Isomap reduction to 10 dimensions
    ]
    ```

    ### 2. Configuration-Based Creation

    Uses Options with proper group/context parameter separation:

    ```python
    feature = Feature(
        name="placeholder",  # Placeholder name, will be replaced
        options=Options(
            context={
                DimensionalityReductionFeatureGroup.ALGORITHM: "pca",
                DimensionalityReductionFeatureGroup.DIMENSION: 2,
                DefaultOptionKeys.in_features: "customer_metrics",
            }
        )
    )
    ```

    ## Result Columns

    The dimensionality reduction results are stored using the multiple result columns pattern.
    For each dimension in the reduced space, a column is created with the naming convention:
    `{feature_name}~dim{i+1}`

    ## Parameter Classification

    ### Context Parameters (Default)
    These parameters don't affect Feature Group resolution/splitting:
    - `algorithm`: The dimensionality reduction algorithm to use
    - `dimension`: Target dimension for the reduction
    - `in_features`: Source features to reduce

    ### Group Parameters
    Currently none for DimensionalityReductionFeatureGroup. Parameters that affect Feature Group
    resolution/splitting would be placed here.

    ## Requirements
    - The input data must contain the source features to be used for dimensionality reduction
    - The dimension parameter must be a positive integer less than the number of source features
    """

    # Option keys for dimensionality reduction configuration
    ALGORITHM = "algorithm"
    DIMENSION = "dimension"

    # Algorithm-specific option keys
    TSNE_MAX_ITER = "tsne_max_iter"
    TSNE_N_ITER_WITHOUT_PROGRESS = "tsne_n_iter_without_progress"
    TSNE_METHOD = "tsne_method"
    PCA_SVD_SOLVER = "pca_svd_solver"
    ICA_MAX_ITER = "ica_max_iter"
    ISOMAP_N_NEIGHBORS = "isomap_n_neighbors"

    # Define supported dimensionality reduction algorithms
    REDUCTION_ALGORITHMS = {
        "pca": "Principal Component Analysis",
        "tsne": "t-Distributed Stochastic Neighbor Embedding",
        "ica": "Independent Component Analysis",
        "lda": "Linear Discriminant Analysis",
        "isomap": "Isometric Mapping",
    }

    # Define the prefix pattern for this feature group
    PREFIX_PATTERN = r".*__(?P<algorithm>[\w]+)_(?P<dimension>\d+)d$"

    # In-feature configuration for FeatureChainParserMixin
    IN_FEATURE_SEPARATOR = ","
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = None

    # Hooks calculate_feature calls: _check_source_features_exist, _add_result_to_data.
    REQUIRED_COLUMNWISE_HOOKS = COLUMNWISE_HOOKS

    PROPERTY_MAPPING = {
        ALGORITHM: PropertySpec(
            "Dimensionality reduction algorithm to use",
            allowed_values=REDUCTION_ALGORITHMS,
            context=True,
            strict_validation=True,
        ),
        DIMENSION: PropertySpec(
            "Target dimension for the reduction (positive integer)",
            context=True,
            strict_validation=True,
            element_validator=is_positive_int,
        ),
        DefaultOptionKeys.in_features: PropertySpec(
            "Source features to use for dimensionality reduction",
            context=True,
            strict_validation=False,
        ),
        # The algorithm-specific numeric keys below share is_positive_int across both hooks.
        # element_validator produces the rejection message and checks the declared default at
        # construction; it runs on BOTH match paths, so it alone enforces the value space. match_guard
        # additionally judges the raw, un-unpacked value.
        # t-SNE specific parameters
        TSNE_MAX_ITER: PropertySpec(
            "Maximum number of iterations for t-SNE optimization",
            context=True,
            strict_validation=True,
            default=250,
            element_validator=is_positive_int,
            match_guard=is_positive_int,
        ),
        TSNE_N_ITER_WITHOUT_PROGRESS: PropertySpec(
            "Maximum iterations without progress before early stopping (t-SNE)",
            context=True,
            strict_validation=True,
            default=50,
            element_validator=is_positive_int,
            match_guard=is_positive_int,
        ),
        TSNE_METHOD: PropertySpec(
            "t-SNE computation method",
            allowed_values={
                "barnes_hut": "Barnes-Hut approximation (faster, O(n log n))",
                "exact": "Exact method (slower, O(n^2))",
            },
            context=True,
            strict_validation=True,
            default="barnes_hut",
        ),
        # PCA specific parameters
        PCA_SVD_SOLVER: PropertySpec(
            "SVD solver algorithm for PCA",
            allowed_values={
                "auto": "Automatically choose solver based on data shape",
                "full": "Full SVD using LAPACK",
                "arpack": "Truncated SVD using ARPACK",
                "randomized": "Randomized SVD",
            },
            context=True,
            strict_validation=True,
            default="auto",
        ),
        # ICA specific parameters
        ICA_MAX_ITER: PropertySpec(
            "Maximum number of iterations for ICA",
            context=True,
            strict_validation=True,
            default=200,
            element_validator=is_positive_int,
            match_guard=is_positive_int,
        ),
        # Isomap specific parameters
        ISOMAP_N_NEIGHBORS: PropertySpec(
            "Number of neighbors for Isomap",
            context=True,
            strict_validation=True,
            default=5,
            element_validator=is_positive_int,
            match_guard=is_positive_int,
        ),
    }

    @classmethod
    def parse_reduction_suffix(cls, feature_name: str) -> tuple[str, int]:
        """
        Parse the dimensionality reduction suffix into its components.

        Args:
            feature_name: The feature name to parse

        Returns:
            A tuple containing (algorithm, dimension)

        Raises:
            ValueError: If the suffix doesn't match the expected pattern
        """
        parsed = FeatureChainParser.parse_name(feature_name, cls._get_prefix_patterns())
        if not parsed.matched:
            raise ValueError(
                f"Invalid dimensionality reduction feature name format: {feature_name}. "
                f"Expected format: {{in_features}}__{{algorithm}}_{{dimension}}d"
            )

        captures = cast(dict[str, str], parsed.named_captures)
        algorithm = captures[cls.ALGORITHM]
        dimension_str = captures[cls.DIMENSION]

        # Validate algorithm
        if algorithm not in cls.REDUCTION_ALGORITHMS:
            raise ValueError(
                f"Unsupported dimensionality reduction algorithm: {algorithm}. "
                f"Supported algorithms: {', '.join(cls.REDUCTION_ALGORITHMS.keys())}"
            )

        # Validate dimension
        dimension = int(dimension_str)
        if dimension <= 0:
            raise ValueError(f"Invalid dimension: {dimension_str}. Must be a positive integer.")
        return algorithm, dimension

    @classmethod
    def _extract_algorithm_dimension_and_source_features(cls, feature: Feature) -> tuple[str, int, list[str], Options]:
        """
        Extract algorithm, dimension, source features, and algorithm-specific options from a feature.

        Tries string-based parsing first, falls back to configuration-based approach.

        Args:
            feature: The feature to extract parameters from

        Returns:
            Tuple of (algorithm, dimension, source_features_list, algorithm_options)

        Raises:
            ValueError: If parameters cannot be extracted
        """
        source_features = cls._extract_source_features(feature)
        algorithm, dimension, algo_options = cls._extract_dim_reduction_params(feature)
        if algorithm is None or dimension is None:
            raise ValueError(f"Could not extract algorithm and dimension from: {feature.name}")
        return algorithm, dimension, source_features, algo_options

    @classmethod
    def _extract_dim_reduction_params(cls, feature: Feature) -> tuple[str | None, int | None, Options]:
        """
        Extract dimensionality reduction algorithm, dimension, and options from a feature.

        Tries string-based parsing first, falls back to configuration-based approach.

        Args:
            feature: The feature to extract parameters from

        Returns:
            Tuple of (algorithm, dimension, algorithm_options)
        """
        algorithm = cls._resolve_operation(feature, cls.ALGORITHM)
        dimension_raw = cls._resolve_operation(feature, cls.DIMENSION)

        if algorithm is None or dimension_raw is None:
            return None, None, feature.options

        # Validate algorithm
        if algorithm not in cls.REDUCTION_ALGORITHMS:
            raise ValueError(
                f"Unsupported dimensionality reduction algorithm: {algorithm}. "
                f"Supported algorithms: {', '.join(cls.REDUCTION_ALGORITHMS.keys())}"
            )

        # Validate and convert dimension
        dimension = int(dimension_raw)
        if dimension <= 0:
            raise ValueError(f"Invalid dimension: {dimension}. Must be a positive integer.")

        return algorithm, dimension, feature.options

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Perform dimensionality reduction operations.

        Processes all requested features, determining the dimensionality reduction algorithm,
        dimension, and source features from either string parsing or configuration-based options.

        Adds the dimensionality reduction results directly to the input data structure.
        """

        # Process each requested feature
        for feature in features.get_sorted_features():
            algorithm, dimension, source_features, options = cls._extract_algorithm_dimension_and_source_features(
                feature
            )

            # Check if all source features exist
            cls._check_source_features_exist(data, source_features)

            # Perform dimensionality reduction
            result = cls._perform_reduction(data, algorithm, dimension, source_features, options)

            # Add the result to the data
            data = cls._add_result_to_data(data, feature.name, result)
        return data

    @classmethod
    @abstractmethod
    def _perform_reduction(
        cls,
        data: Any,
        algorithm: str,
        dimension: int,
        source_features: list[str],
        options: Options,
    ) -> Any:
        """
        Method to perform the dimensionality reduction. Should be implemented by subclasses.

        Args:
            data: The input data
            algorithm: The dimensionality reduction algorithm to use
            dimension: The target dimension for the reduction
            source_features: The list of source features to use for dimensionality reduction
            options: Options containing algorithm-specific parameters

        Returns:
            The result of the dimensionality reduction (typically the reduced features)
        """
        ...
