"""
Base implementation for dimensionality reduction feature groups.
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


class DimensionalityReductionFeatureGroup(AbstractFeatureGroup):
    # Option keys for dimensionality reduction configuration
    ALGORITHM = "algorithm"
    DIMENSION = "dimension"
    """
    Base class for all dimensionality reduction feature groups.

    Dimensionality reduction feature groups reduce the dimensionality of feature spaces
    using various techniques like PCA, t-SNE, UMAP, etc. They allow you to transform
    high-dimensional data into a lower-dimensional representation while preserving
    important structures and relationships.

    ## Feature Naming Convention

    Dimensionality reduction features follow this naming pattern:
    `{algorithm}_{dimension}d__{mloda_source_features}`

    The source features (mloda_source_features) are extracted from the feature name and used
    as input for the dimensionality reduction algorithm. Note the double underscore before 
    the source features.

    Examples:
    - `pca_2d__customer_metrics`: PCA reduction to 2 dimensions of customer metrics
    - `tsne_3d__product_features`: t-SNE reduction to 3 dimensions of product features
    - `umap_10d__sensor_readings`: UMAP reduction to 10 dimensions of sensor readings

    ## Result Columns

    The dimensionality reduction results are stored using the multiple result columns pattern.
    For each dimension in the reduced space, a column is created with the naming convention:
    `{feature_name}~dim{i+1}`

    For example, a PCA reduction to 2 dimensions with the feature name `pca_2d__customer_metrics`
    will create the following columns:
    - `pca_2d__customer_metrics~dim1`: First principal component
    - `pca_2d__customer_metrics~dim2`: Second principal component

    This allows for easy access to individual dimensions and leverages the multiple result
    columns pattern for efficient data handling.

    ## Configuration-Based Creation

    DimensionalityReductionFeatureGroup supports configuration-based creation through the
    FeatureChainParserConfiguration mechanism. This allows features to be created
    from options rather than explicit feature names.

    To create a dimensionality reduction feature using configuration:

    ```python
    feature = Feature(
        "PlaceHolder",  # Placeholder name, will be replaced
        Options({
            DimensionalityReductionFeatureGroup.ALGORITHM: "pca",
            DimensionalityReductionFeatureGroup.DIMENSION: 2,
            DefaultOptionKeys.mloda_source_feature: "customer_metrics"
        })
    )

    # The Engine will automatically parse this into a feature with name "pca_2d__customer_metrics"
    ```

    ## Supported Dimensionality Reduction Algorithms

    - `pca`: Principal Component Analysis
    - `tsne`: t-Distributed Stochastic Neighbor Embedding
    - `umap`: Uniform Manifold Approximation and Projection
    - `ica`: Independent Component Analysis
    - `lda`: Linear Discriminant Analysis
    - `isomap`: Isometric Mapping

    ## Requirements
    - The input data must contain the source features to be used for dimensionality reduction
    - The dimension parameter must be a positive integer less than the number of source features
    """

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
    PREFIX_PATTERN = r"^([\w]+)_(\d+)d__"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract source features from the dimensionality reduction feature name."""
        source_features_str = FeatureChainParser.extract_source_feature(feature_name.name, self.PREFIX_PATTERN)

        # Handle multiple source features (comma-separated)
        source_features = set()
        for feature in source_features_str.split(","):
            source_features.add(Feature(feature.strip()))

        return source_features

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
    def get_algorithm(cls, feature_name: str) -> str:
        """Extract the dimensionality reduction algorithm from the feature name."""
        return cls.parse_reduction_prefix(feature_name)[0]

    @classmethod
    def get_dimension(cls, feature_name: str) -> int:
        """Extract the dimension from the feature name."""
        return cls.parse_reduction_prefix(feature_name)[1]

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        """Check if feature name matches the expected pattern for dimensionality reduction features."""
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name

        try:
            # First validate that this is a valid feature name for this feature group
            if not FeatureChainParser.validate_feature_name(feature_name, cls.PREFIX_PATTERN):
                return False

            # Then validate the dimensionality reduction components
            cls.parse_reduction_prefix(feature_name)
            return True
        except ValueError:
            return False

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Perform dimensionality reduction operations.

        Processes all requested features, determining the dimensionality reduction algorithm,
        dimension, and source features from each feature name.

        Adds the dimensionality reduction results directly to the input data structure.
        """
        # Process each requested feature
        for feature_name in features.get_all_names():
            algorithm, dimension = cls.parse_reduction_prefix(feature_name)
            source_features_str = FeatureChainParser.extract_source_feature(feature_name, cls.PREFIX_PATTERN)

            # Parse source features (comma-separated)
            source_features = [feature.strip() for feature in source_features_str.split(",")]

            # Check if all source features exist
            for source_feature in source_features:
                cls._check_source_feature_exists(data, source_feature)

            # Perform dimensionality reduction
            result = cls._perform_reduction(data, algorithm, dimension, source_features)

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
            """Raise an error for unsupported dimensionality reduction algorithm."""
            raise ValueError(
                f"Unsupported dimensionality reduction algorithm: {algorithm}. "
                f"Supported algorithms: {list(cls.REDUCTION_ALGORITHMS.keys())}"
            )

        def validate_dimension(dimension: Any) -> bool:
            """Validate the dimension."""
            try:
                dimension_int = int(dimension) if isinstance(dimension, str) else dimension
                if dimension_int <= 0:
                    raise ValueError("Dimension must be positive")
                return True
            except (ValueError, TypeError):
                raise ValueError(f"Invalid dimension: {dimension}. Must be a positive integer.")

        # Create and return the configured parser
        return create_configurable_parser(
            parse_keys=[
                cls.ALGORITHM,
                cls.DIMENSION,
                DefaultOptionKeys.mloda_source_feature,
            ],
            feature_name_template="{algorithm}_{dimension}d__{mloda_source_feature}",
            validation_rules={
                cls.ALGORITHM: lambda x: x in cls.REDUCTION_ALGORITHMS or raise_unsupported_algorithm(x),
                cls.DIMENSION: validate_dimension,
            },
        )
