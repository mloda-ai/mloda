"""
Base implementation for node centrality feature groups.
"""

from __future__ import annotations

from typing import Any, Optional

from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.provider import FeatureChainParser
from mloda.provider import (
    FeatureChainParserMixin,
)
from mloda.provider import FeatureSet
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class NodeCentralityFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    # Option keys for centrality configuration
    CENTRALITY_TYPE = "centrality_type"
    GRAPH_TYPE = "graph_type"
    WEIGHT_COLUMN = "weight_column"
    """
    Base class for all node centrality feature groups.

    Node centrality feature groups calculate various centrality metrics for nodes in a graph.
    These metrics help identify important nodes in network data based on different definitions
    of importance.

    ## Feature Naming Convention

    Node centrality features follow this naming pattern:
    `{node_feature}__{centrality_type}_centrality`

    The node feature comes first, followed by the centrality operation.
    Note the double underscore separating the source feature from the operation.

    Examples:
    - `user__degree_centrality`: Degree centrality for user nodes
    - `product__betweenness_centrality`: Betweenness centrality for product nodes
    - `website__pagerank_centrality`: PageRank centrality for website nodes

    ## Configuration-Based Creation

    NodeCentralityFeatureGroup supports configuration-based. This allows features to be created
    from options rather than explicit feature names.

    To create a centrality feature using configuration:

    ```python
    feature = Feature(
        "PlaceHolder",  # Placeholder name, will be replaced
        Options({
            NodeCentralityFeatureGroup.CENTRALITY_TYPE: "degree",
            DefaultOptionKeys.in_features: "user"
        })
    )

    # The Engine will automatically parse this into a feature with name "user__degree_centrality"
    ```

    ### Important Note on Multiple Features

    When creating multiple features with different source features or options, each feature
    will be processed separately and may result in multiple DataFrames in the output. For example:

    ```python
    # These features will likely be processed into separate DataFrames
    degree_undirected = Feature(
        "placeholder",
        Options({
            NodeCentralityFeatureGroup.CENTRALITY_TYPE: "degree",
            DefaultOptionKeys.in_features: "source",
            NodeCentralityFeatureGroup.GRAPH_TYPE: "undirected",
        })
    )

    degree_directed = Feature(
        "placeholder",
        Options({
            NodeCentralityFeatureGroup.CENTRALITY_TYPE: "degree",
            DefaultOptionKeys.in_features: "target",  # Different source feature
            NodeCentralityFeatureGroup.GRAPH_TYPE: "directed",
        })
    )
    ```

    In this case, the result will contain two separate DataFrames: one with "source__degree_centrality"
    and another with "target__degree_centrality". This behavior occurs because features with different
    source features are processed by different feature groups.

    ## Supported Centrality Types

    - `degree`: Measures the number of connections a node has
    - `betweenness`: Measures how often a node lies on the shortest path between other nodes
    - `closeness`: Measures how close a node is to all other nodes
    - `eigenvector`: Measures the influence of a node in a network
    - `pagerank`: A variant of eigenvector centrality used by Google

    ## Graph Types

    - `directed`: A graph where edges have direction
    - `undirected`: A graph where edges have no direction (default)

    ## Requirements
    - The input data must contain edge information (source and target columns)
    - For weighted centrality calculations, a weight column can be specified
    
    ## Important Implementation Note
    
    When using configuration-based creation with different options,
    each feature will be processed separately and may result in multiple DataFrames in 
    the output. This is because feature groups are currently split by different options.
    """

    # Define supported centrality types
    CENTRALITY_TYPES = {
        "degree": "Measures the number of connections a node has",
        "betweenness": "Measures how often a node lies on the shortest path between other nodes",
        "closeness": "Measures how close a node is to all other nodes",
        "eigenvector": "Measures the influence of a node in a network",
        "pagerank": "A variant of eigenvector centrality used by Google",
    }

    # Define supported graph types
    GRAPH_TYPES = {
        "directed": "A graph where edges have direction",
        "undirected": "A graph where edges have no direction",
    }

    # Define the suffix pattern for this feature group (L→R format: source__operation)
    PREFIX_PATTERN = r".*__([\w]+)_centrality$"

    # In-feature configuration for FeatureChainParserMixin
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    # Property mapping for configuration-based feature creation
    PROPERTY_MAPPING = {
        # Context parameters (don't affect Feature Group resolution)
        CENTRALITY_TYPE: {
            **CENTRALITY_TYPES,  # All supported centrality types as valid options
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        GRAPH_TYPE: {
            **GRAPH_TYPES,  # All supported graph types as valid options
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.default: "undirected",
        },
        WEIGHT_COLUMN: {
            "explanation": "Column name for edge weights (optional)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: None,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature representing the nodes for centrality calculation",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def parse_centrality_prefix(cls, feature_name: str) -> str:
        """
        Parse the centrality suffix to extract the centrality type.

        Args:
            feature_name: The feature name to parse (format: {source}__{centrality_type}_centrality)

        Returns:
            The centrality type

        Raises:
            ValueError: If the suffix doesn't match the expected pattern
        """
        # Extract the suffix part (everything after the LAST double underscore for L→R format)
        suffix_start = feature_name.rfind("__")
        if suffix_start == -1:
            raise ValueError(
                f"Invalid centrality feature name format: {feature_name}. Missing double underscore separator."
            )

        suffix = feature_name[suffix_start + 2 :]

        # Parse the suffix components
        parts = suffix.split("_")
        if len(parts) != 2 or parts[1] != "centrality":
            raise ValueError(
                f"Invalid centrality feature name format: {feature_name}. "
                f"Expected format: {{source}}__{{centrality_type}}_centrality"
            )

        centrality_type = parts[0]

        # Validate centrality type
        if centrality_type not in cls.CENTRALITY_TYPES:
            raise ValueError(
                f"Unsupported centrality type: {centrality_type}. "
                f"Supported types: {', '.join(cls.CENTRALITY_TYPES.keys())}"
            )

        return centrality_type

    @classmethod
    def get_centrality_type(cls, feature_name: str) -> str:
        """Extract the centrality type from the feature name."""
        return cls.parse_centrality_prefix(feature_name)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Calculate centrality metrics for nodes in a graph.

        Processes all requested features, determining the centrality type
        and source features from each feature name.

        Adds the centrality results directly to the input data structure.
        """
        # Process each feature
        for feature in features.features:
            centrality_type, source_feature_str = cls._extract_centrality_and_source_feature(feature)

            # Common parameter extraction (works for both approaches)
            graph_type = feature.options.get(cls.GRAPH_TYPE) or "undirected"
            weight_column = feature.options.get(cls.WEIGHT_COLUMN)

            # Validate graph type
            if graph_type not in cls.GRAPH_TYPES:
                raise ValueError(
                    f"Unsupported graph type: {graph_type}. Supported types: {', '.join(cls.GRAPH_TYPES.keys())}"
                )

            # Check if source feature exists
            cls._check_source_feature_exists(data, source_feature_str)

            # Calculate centrality
            result = cls._calculate_centrality(data, centrality_type, source_feature_str, graph_type, weight_column)

            # Add the result to the data
            data = cls._add_result_to_data(data, feature.get_name(), result)

        return data

    @classmethod
    def _extract_centrality_and_source_feature(cls, feature: Feature) -> tuple[str, str]:
        """
        Extract centrality type and source feature name from a feature.

        Tries string-based parsing first, falls back to configuration-based approach.

        Args:
            feature: The feature to extract parameters from

        Returns:
            Tuple of (centrality_type, source_feature_name)

        Raises:
            ValueError: If parameters cannot be extracted
        """
        source_features = cls._extract_source_features(feature)
        centrality_type = cls._extract_centrality_type(feature)
        if centrality_type is None:
            raise ValueError(f"Could not extract centrality type from: {feature.name}")
        return centrality_type, source_features[0]

    @classmethod
    def _extract_centrality_type(cls, feature: Feature) -> Optional[str]:
        """
        Extract centrality type from a feature.

        Tries string-based parsing first, falls back to configuration-based approach.

        Args:
            feature: The feature to extract centrality type from

        Returns:
            The centrality type, or None if not found

        Raises:
            ValueError: If centrality type cannot be extracted
        """
        # Try string-based parsing first
        suffix_part, source_feature_name = FeatureChainParser.parse_feature_name(feature.name, [cls.PREFIX_PATTERN])
        if source_feature_name is not None and suffix_part is not None:
            # The suffix_part is already the centrality type (extracted by regex group)
            if suffix_part in cls.CENTRALITY_TYPES:
                return suffix_part

        # Fall back to configuration-based approach
        centrality_type = feature.options.get(cls.CENTRALITY_TYPE)
        return str(centrality_type) if centrality_type is not None else None

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
    def _calculate_centrality(
        cls,
        data: Any,
        centrality_type: str,
        node_feature: str,
        graph_type: str = "undirected",
        weight_column: Optional[str] = None,
    ) -> Any:
        """
        Method to calculate the centrality. Should be implemented by subclasses.

        Args:
            data: The input data
            centrality_type: The type of centrality to calculate
            node_feature: The feature representing the nodes
            graph_type: The type of graph (directed or undirected)
            weight_column: The column to use for edge weights (optional)

        Returns:
            The result of the centrality calculation
        """
        raise NotImplementedError(f"_calculate_centrality not implemented in {cls.__name__}")
