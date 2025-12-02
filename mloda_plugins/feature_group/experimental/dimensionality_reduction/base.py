"""
Base implementation for dimensionality reduction feature groups.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional, Set, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class DimensionalityReductionFeatureGroup(AbstractFeatureGroup):
    """
    Base class for all dimensionality reduction feature groups.

    Dimensionality reduction feature groups reduce the dimensionality of feature spaces
    using various techniques like PCA, t-SNE, UMAP, etc. They support both string-based
    feature creation and configuration-based creation with proper group/context parameter separation.

    ## Supported Dimensionality Reduction Algorithms

    - `pca`: Principal Component Analysis
    - `tsne`: t-Distributed Stochastic Neighbor Embedding
    - `umap`: Uniform Manifold Approximation and Projection
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
        "sensor_readings__umap_10d"      # UMAP reduction to 10 dimensions
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
        "umap": "Uniform Manifold Approximation and Projection",
        "ica": "Independent Component Analysis",
        "lda": "Linear Discriminant Analysis",
        "isomap": "Isometric Mapping",
    }

    # Define the prefix pattern for this feature group
    PREFIX_PATTERN = r".*__([\w]+)_(\d+)d$"

    PROPERTY_MAPPING = {
        ALGORITHM: {
            **REDUCTION_ALGORITHMS,
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: True,
        },
        DIMENSION: {
            "explanation": "Target dimension for the reduction (positive integer)",
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: True,
            DefaultOptionKeys.mloda_validation_function: lambda value: isinstance(value, (int, str))
            and str(value).isdigit()
            and int(value) > 0,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source features to use for dimensionality reduction",
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: False,
        },
        # t-SNE specific parameters
        TSNE_MAX_ITER: {
            "explanation": "Maximum number of iterations for t-SNE optimization",
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: False,
            "default": 250,
            DefaultOptionKeys.mloda_validation_function: lambda value: isinstance(value, (int, str))
            and str(value).isdigit()
            and int(value) > 0,
        },
        TSNE_N_ITER_WITHOUT_PROGRESS: {
            "explanation": "Maximum iterations without progress before early stopping (t-SNE)",
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: False,
            "default": 50,
            DefaultOptionKeys.mloda_validation_function: lambda value: isinstance(value, (int, str))
            and str(value).isdigit()
            and int(value) > 0,
        },
        TSNE_METHOD: {
            "barnes_hut": "Barnes-Hut approximation (faster, O(n log n))",
            "exact": "Exact method (slower, O(n^2))",
            "explanation": "t-SNE computation method",
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: False,
            "default": "barnes_hut",
        },
        # PCA specific parameters
        PCA_SVD_SOLVER: {
            "auto": "Automatically choose solver based on data shape",
            "full": "Full SVD using LAPACK",
            "arpack": "Truncated SVD using ARPACK",
            "randomized": "Randomized SVD",
            "explanation": "SVD solver algorithm for PCA",
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: False,
            "default": "auto",
        },
        # ICA specific parameters
        ICA_MAX_ITER: {
            "explanation": "Maximum number of iterations for ICA",
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: False,
            "default": 200,
            DefaultOptionKeys.mloda_validation_function: lambda value: isinstance(value, (int, str))
            and str(value).isdigit()
            and int(value) > 0,
        },
        # Isomap specific parameters
        ISOMAP_N_NEIGHBORS: {
            "explanation": "Number of neighbors for Isomap",
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: False,
            "default": 5,
            DefaultOptionKeys.mloda_validation_function: lambda value: isinstance(value, (int, str))
            and str(value).isdigit()
            and int(value) > 0,
        },
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source feature from either configuration-based options or string parsing."""

        source_feature: str | None = None

        # Try string-based parsing first
        _, source_feature = FeatureChainParser.parse_feature_name(feature_name, [self.PREFIX_PATTERN])
        if source_feature is not None:
            # Handle multiple source features (comma-separated)
            source_features = set()
            for feature in source_feature.split(","):
                source_features.add(Feature(feature.strip()))
            return source_features

        # Fall back to configuration-based approach
        source_featurez = options.get_in_features()
        if len(source_featurez) != 1:
            raise ValueError(
                f"Expected exactly one source feature, but found {len(source_featurez)}: {source_featurez}"
            )
        return set(source_featurez)

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
        # Extract the suffix part (everything after the last double underscore)
        suffix_start = feature_name.rfind("__")
        if suffix_start == -1:
            raise ValueError(
                f"Invalid dimensionality reduction feature name format: {feature_name}. Missing double underscore separator."
            )

        suffix = feature_name[suffix_start + 2 :]  # Skip the "__"

        # Parse the suffix components
        parts = suffix.split("_")
        if len(parts) != 2 or not parts[1].endswith("d"):
            raise ValueError(
                f"Invalid dimensionality reduction feature name format: {feature_name}. "
                f"Expected format: {{in_features}}__{{algorithm}}_{{dimension}}d"
            )

        algorithm = parts[0]
        dimension_str = parts[1][:-1]  # Remove the 'd' suffix

        # Validate algorithm
        if algorithm not in cls.REDUCTION_ALGORITHMS:
            raise ValueError(
                f"Unsupported dimensionality reduction algorithm: {algorithm}. "
                f"Supported algorithms: {', '.join(cls.REDUCTION_ALGORITHMS.keys())}"
            )

        # Validate dimension
        try:
            dimension = int(dimension_str)
            if dimension <= 0:
                raise ValueError(f"Invalid dimension: {dimension}. Must be a positive integer.")
            return algorithm, dimension
        except ValueError:
            raise ValueError(f"Invalid dimension: {dimension_str}. Must be a positive integer.")

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        """Check if feature name matches the expected pattern for dimensionality reduction features."""

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
                    # Use existing validation logic that validates algorithm and dimension
                    cls.parse_reduction_suffix(feature_name_str)
                except ValueError:
                    # If validation fails, this feature doesn't match
                    return False
        return result

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
        algorithm = None
        dimension = None
        source_features = None

        # Try string-based parsing first
        feature_name_str = feature.name.name if hasattr(feature.name, "name") else str(feature.name)

        if FeatureChainParser.is_chained_feature(feature_name_str):
            algorithm, dimension = cls.parse_reduction_suffix(feature_name_str)
            source_features_str = FeatureChainParser.extract_source_feature(feature_name_str, cls.PREFIX_PATTERN)
            source_features = [feature.strip() for feature in source_features_str.split(",")]
            # For string-based features, still extract algorithm-specific options from feature.options
            return algorithm, dimension, source_features, feature.options

        # Fall back to configuration-based approach
        source_features_set = feature.options.get_in_features()
        source_feature = next(iter(source_features_set))
        source_features = [source_feature.get_name()]

        algorithm = feature.options.get(cls.ALGORITHM)
        dimension = feature.options.get(cls.DIMENSION)

        if algorithm is None or dimension is None:
            raise ValueError(f"Could not extract algorithm and dimension from: {feature.name}")

        # Validate algorithm
        if algorithm not in cls.REDUCTION_ALGORITHMS:
            raise ValueError(
                f"Unsupported dimensionality reduction algorithm: {algorithm}. "
                f"Supported algorithms: {', '.join(cls.REDUCTION_ALGORITHMS.keys())}"
            )

        # Validate and convert dimension

        dimension = int(dimension)
        if dimension <= 0:
            raise ValueError(f"Invalid dimension: {dimension}. Must be a positive integer.")

        return algorithm, dimension, source_features, feature.options

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Perform dimensionality reduction operations.

        Processes all requested features, determining the dimensionality reduction algorithm,
        dimension, and source features from either string parsing or configuration-based options.

        Adds the dimensionality reduction results directly to the input data structure.
        """

        # Process each requested feature
        for feature in features.features:
            algorithm, dimension, source_features, options = cls._extract_algorithm_dimension_and_source_features(
                feature
            )

            # Check if all source features exist
            for source_feature in source_features:
                cls._check_source_feature_exists(data, source_feature)

            # Perform dimensionality reduction
            result = cls._perform_reduction(data, algorithm, dimension, source_features, options)

            # Add the result to the data
            data = cls._add_result_to_data(data, feature.get_name(), result)
        return data

    @classmethod
    @abstractmethod
    def _check_source_feature_exists(cls, data: Any, feature_name: str) -> None:
        """
        Check if the source feature exists in the data.

        Args:
            data: The input data
            feature_name: The name of the feature to check

        Raises:
            ValueError: If the feature does not exist in the data
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
