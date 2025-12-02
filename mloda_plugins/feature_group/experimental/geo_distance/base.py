"""
Base implementation for geo distance feature groups.
"""

from __future__ import annotations

from typing import Any, Optional, Set, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class GeoDistanceFeatureGroup(AbstractFeatureGroup):
    """
    Base class for all geo distance feature groups.

    The GeoDistanceFeatureGroup calculates distances between geographic points,
    such as haversine (great-circle), euclidean, or manhattan distances. Supports both
    string-based feature creation and configuration-based creation with proper
    group/context parameter separation.

    ## Feature Creation Methods

    ### 1. String-Based Creation

    Features follow the naming pattern: `{point1_feature}&{point2_feature}__{distance_type}_distance`

    Examples:
    ```python
    features = [
        "customer_location&store_location__haversine_distance",  # Great-circle distance
        "origin&destination__euclidean_distance",               # Straight-line distance
        "pickup&dropoff__manhattan_distance"                    # Manhattan distance
    ]
    ```

    ### 2. Configuration-Based Creation

    Uses Options with proper group/context parameter separation:

    ```python
    feature = Feature(
        name="placeholder",
        options=Options(
            context={
                GeoDistanceFeatureGroup.DISTANCE_TYPE: "haversine",
                DefaultOptionKeys.in_features: ["customer_location", "store_location"],
            }
        )
    )
    ```

    ## Parameter Classification

    ### Context Parameters (Default)
    These parameters don't affect Feature Group resolution/splitting:
    - `distance_type`: The type of distance calculation (haversine, euclidean, manhattan)
    - `in_features`: The source features (list of exactly 2 point features)

    ### Group Parameters
    Currently none for GeoDistanceFeatureGroup. Parameters that affect Feature Group
    resolution/splitting would be placed here.

    ## Supported Distance Types

    - `haversine`: Great-circle distance on a sphere (for lat/lon coordinates)
    - `euclidean`: Straight-line distance between two points
    - `manhattan`: Sum of absolute differences between coordinates

    ## Requirements
    - Exactly 2 source features (point features) are required
    - Point features should contain coordinate data (tuples, lists, or separate x/y columns)
    - For haversine distance, coordinates should be in (latitude, longitude) format
    - For euclidean/manhattan distance, coordinates should be in (x, y) format
    """

    # Option keys for distance type
    DISTANCE_TYPE = "distance_type"

    # Define supported distance types
    DISTANCE_TYPES = {
        "haversine": "Great-circle distance on a sphere (for lat/lon coordinates)",
        "euclidean": "Straight-line distance between two points",
        "manhattan": "Sum of absolute differences between coordinates",
    }

    # Define the prefix pattern for this feature group
    PREFIX_PATTERN = r".*__([\w]+)_distance$"

    # Property mapping for configuration-based features with group/context separation
    PROPERTY_MAPPING = {
        DISTANCE_TYPE: {
            **DISTANCE_TYPES,
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: True,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source features (exactly 2 point features required)",
            DefaultOptionKeys.mloda_context: True,
            DefaultOptionKeys.mloda_strict_validation: True,
            DefaultOptionKeys.mloda_validation_function: lambda x: (
                # Accept individual strings (when parser iterates over list elements)
                isinstance(x, str)
                or
                # Accept collections with exactly 2 elements (when validating the whole list)
                (isinstance(x, (list, tuple, frozenset, set)) and len(x) == 2)
            ),
        },
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract point features from either configuration-based options or string parsing."""

        # Try string-based parsing first
        try:
            # For L->R: "point1&point2__distance_type_distance"
            # We need to extract everything before the last "__"
            feature_name_str = feature_name.name if hasattr(feature_name, "name") else str(feature_name)
            parts = feature_name_str.rsplit("__", 1)
            if len(parts) == 2:
                # parts[0] contains "point1&point2", parts[1] contains "distance_type_distance"
                point_parts = parts[0].split("&", 1)
                if len(point_parts) == 2:
                    return {Feature(point_parts[0]), Feature(point_parts[1])}
        except (ValueError, AttributeError):
            pass

        # Fall back to configuration-based approach
        source_features = options.get_in_features()
        if len(source_features) != 2:
            raise ValueError(
                f"Expected exactly 2 source features for geo distance, got {len(source_features)}: {source_features}"
            )
        return set(source_features)

    @classmethod
    def get_distance_type(cls, feature_name: str) -> str:
        """Extract the distance type from the feature name."""
        distance_type, _ = FeatureChainParser.parse_feature_name(feature_name, [cls.PREFIX_PATTERN])
        if distance_type is None:
            raise ValueError(f"Invalid geo distance feature name format: {feature_name}")

        # Remove the "_distance" suffix to get just the distance type
        distance_type = distance_type.replace("_distance", "").strip("_")
        if distance_type not in cls.DISTANCE_TYPES:
            raise ValueError(
                f"Unsupported distance type: {distance_type}. Supported types: {', '.join(cls.DISTANCE_TYPES.keys())}"
            )

        return distance_type

    @classmethod
    def get_point_features(cls, feature_name: str) -> tuple[str, str]:
        """Extract the two point features from the feature name."""
        # For L->R: "point1&point2__distance_type_distance"
        # Split from right to remove the distance suffix
        parts = feature_name.rsplit("__", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid geo distance feature name format: {feature_name}. Missing double underscore separator."
            )

        # Now split the remaining part to get the two points using & separator
        point_parts = parts[0].split("&", 1)
        if len(point_parts) != 2:
            raise ValueError(
                f"Invalid geo distance feature name format: {feature_name}. Expected two point features separated by ampersand (&)."
            )

        return point_parts[0], point_parts[1]

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        """Check if feature name matches the expected pattern for geo distance features."""

        # Use the unified parser with property mapping for full configuration support
        return FeatureChainParser.match_configuration_feature_chain_parser(
            feature_name,
            options,
            property_mapping=cls.PROPERTY_MAPPING,
            prefix_patterns=[cls.PREFIX_PATTERN],
        )

    @classmethod
    def _supports_distance_type(cls, distance_type: str) -> bool:
        """Check if this feature group supports the given distance type."""
        return distance_type in cls.DISTANCE_TYPES

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Calculate distances between point features.

        Processes all requested features, determining the distance type
        and point features from either string parsing or configuration-based options.

        Adds the calculated distances directly to the input data structure.
        """
        # Process each requested feature
        for feature in features.features:
            distance_type, point1_feature, point2_feature = cls._extract_geo_distance_parameters(feature)

            cls._check_point_features_exist(data, point1_feature, point2_feature)

            if distance_type not in cls.DISTANCE_TYPES:
                raise ValueError(f"Unsupported distance type: {distance_type}")

            result = cls._calculate_distance(data, distance_type, point1_feature, point2_feature)

            data = cls._add_result_to_data(data, feature.get_name(), result)

        return data

    @classmethod
    def _extract_geo_distance_parameters(cls, feature: Feature) -> tuple[str, str, str]:
        """
        Extract geo distance parameters from a feature.

        Tries string-based parsing first, falls back to configuration-based approach.

        Args:
            feature: The feature to extract parameters from

        Returns:
            Tuple of (distance_type, point1_feature, point2_feature)

        Raises:
            ValueError: If parameters cannot be extracted
        """
        # Try string-based parsing first
        feature_name_str = feature.name.name if hasattr(feature.name, "name") else str(feature.name)

        if FeatureChainParser.is_chained_feature(feature_name_str):
            distance_type = cls.get_distance_type(feature_name_str)
            point1_feature, point2_feature = cls.get_point_features(feature_name_str)
            return distance_type, point1_feature, point2_feature

        # Fall back to configuration-based approach
        source_features = feature.options.get_in_features()
        if len(source_features) != 2:
            raise ValueError(
                f"Expected exactly 2 source features for geo distance, got {len(source_features)}: {source_features}"
            )

        source_feature_names = [sf.get_name() for sf in source_features]
        point1_feature, point2_feature = source_feature_names

        distance_type = feature.options.get(cls.DISTANCE_TYPE)
        if distance_type is None:
            raise ValueError(f"Could not extract distance_type from feature: {feature.name}")

        # Validate distance type
        if distance_type not in cls.DISTANCE_TYPES:
            raise ValueError(
                f"Unsupported distance type: {distance_type}. Supported types: {', '.join(cls.DISTANCE_TYPES.keys())}"
            )

        return distance_type, point1_feature, point2_feature

    @classmethod
    def _check_point_features_exist(cls, data: Any, point1_feature: str, point2_feature: str) -> None:
        """
        Check if the point features exist in the data.

        Args:
            data: The input data
            point1_feature: The name of the first point feature
            point2_feature: The name of the second point feature

        Raises:
            ValueError: If either feature does not exist in the data
        """
        raise NotImplementedError(f"_check_point_features_exist not implemented in {cls.__name__}")

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
    def _calculate_distance(cls, data: Any, distance_type: str, point1_feature: str, point2_feature: str) -> Any:
        """
        Method to calculate the distance. Should be implemented by subclasses.

        Args:
            data: The input data
            distance_type: The type of distance to calculate
            point1_feature: The name of the first point feature
            point2_feature: The name of the second point feature

        Returns:
            The calculated distance
        """
        raise NotImplementedError(f"_calculate_distance not implemented in {cls.__name__}")
