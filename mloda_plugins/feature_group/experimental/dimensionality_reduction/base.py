"""
Base implementation for dimensionality reduction feature groups.
"""

from __future__ import annotations

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

    Features follow the naming pattern: `{algorithm}_{dimension}d__{mloda_source_features}`

    Examples:
    ```python
    features = [
        "pca_2d__customer_metrics",      # PCA reduction to 2 dimensions
        "tsne_3d__product_features",     # t-SNE reduction to 3 dimensions
        "umap_10d__sensor_readings"      # UMAP reduction to 10 dimensions
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
                DefaultOptionKeys.mloda_source_feature: "customer_metrics",
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
    - `mloda_source_feature`: Source features to reduce

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
    PATTERN = "__"
    PREFIX_PATTERN = r"^([\w]+)_(\d+)d__"

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
        DefaultOptionKeys.mloda_source_feature: {
            "explanation": "Source features to use for dimensionality reduction",
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: False,
        },
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source feature from either configuration-based options or string parsing."""

        source_feature: str | None = None

        # Try string-based parsing first
        _, source_feature = FeatureChainParser.parse_feature_name(feature_name, self.PATTERN, [self.PREFIX_PATTERN])
        if source_feature is not None:
            # Handle multiple source features (comma-separated)
            source_features = set()
            for feature in source_feature.split(","):
                source_features.add(Feature(feature.strip()))
            return source_features

        # Fall back to configuration-based approach
        source_featurez = options.get_source_features()
        if len(source_featurez) != 1:
            raise ValueError(
                f"Expected exactly one source feature, but found {len(source_featurez)}: {source_featurez}"
            )
        return set(source_featurez)

    @classmethod
    def parse_reduction_prefix(cls, feature_name: str) -> tuple[str, int]:
        """
        Parse the dimensionality reduction prefix into its components.

        Args:
            feature_name: The feature name to parse

        Returns:
            A tuple containing (algorithm, dimension)

        Raises:
            ValueError: If the prefix doesn't match the expected pattern
        """
        # Extract the prefix part (everything before the double underscore)
        prefix_end = feature_name.find("__")
        if prefix_end == -1:
            raise ValueError(
                f"Invalid dimensionality reduction feature name format: {feature_name}. Missing double underscore separator."
            )

        prefix = feature_name[:prefix_end]

        # Parse the prefix components
        parts = prefix.split("_")
        if len(parts) != 2 or not parts[1].endswith("d"):
            raise ValueError(
                f"Invalid dimensionality reduction feature name format: {feature_name}. "
                f"Expected format: {{algorithm}}_{{dimension}}d__{{mloda_source_features}}"
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
            pattern=cls.PATTERN,
            prefix_patterns=[cls.PREFIX_PATTERN],
        )

        # If it matches and it's a string-based feature, validate with our custom logic
        if result:
            feature_name_str = feature_name.name if isinstance(feature_name, FeatureName) else feature_name

            # Check if this is a string-based feature (contains the pattern)
            if cls.PATTERN in feature_name_str:
                try:
                    # Use existing validation logic that validates algorithm and dimension
                    cls.parse_reduction_prefix(feature_name_str)
                except ValueError:
                    # If validation fails, this feature doesn't match
                    return False
        return result

    @classmethod
    def _extract_algorithm_dimension_and_source_features(cls, feature: Feature) -> tuple[str, int, list[str]]:
        """
        Extract algorithm, dimension, and source features from a feature.

        Tries string-based parsing first, falls back to configuration-based approach.

        Args:
            feature: The feature to extract parameters from

        Returns:
            Tuple of (algorithm, dimension, source_features_list)

        Raises:
            ValueError: If parameters cannot be extracted
        """
        algorithm = None
        dimension = None
        source_features = None

        # Try string-based parsing first
        feature_name_str = feature.name.name if hasattr(feature.name, "name") else str(feature.name)

        if cls.PATTERN in feature_name_str:
            algorithm, dimension = cls.parse_reduction_prefix(feature_name_str)
            source_features_str = FeatureChainParser.extract_source_feature(feature_name_str, cls.PREFIX_PATTERN)
            source_features = [feature.strip() for feature in source_features_str.split(",")]
            return algorithm, dimension, source_features

        # Fall back to configuration-based approach
        source_features_set = feature.options.get_source_features()
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

        return algorithm, dimension, source_features

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
            algorithm, dimension, source_features = cls._extract_algorithm_dimension_and_source_features(feature)

            # Check if all source features exist
            for source_feature in source_features:
                cls._check_source_feature_exists(data, source_feature)

            # Perform dimensionality reduction
            result = cls._perform_reduction(data, algorithm, dimension, source_features)

            # Add the result to the data
            data = cls._add_result_to_data(data, feature.get_name(), result)
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
    def _perform_reduction(
        cls,
        data: Any,
        algorithm: str,
        dimension: int,
        source_features: list[str],
    ) -> Any:
        """
        Method to perform the dimensionality reduction. Should be implemented by subclasses.

        Args:
            data: The input data
            algorithm: The dimensionality reduction algorithm to use
            dimension: The target dimension for the reduction
            source_features: The list of source features to use for dimensionality reduction

        Returns:
            The result of the dimensionality reduction (typically the reduced features)
        """
        raise NotImplementedError(f"_perform_reduction not implemented in {cls.__name__}")
