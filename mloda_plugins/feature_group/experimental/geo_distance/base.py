"""
Base implementation for geo distance feature groups.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.provider import FeatureSet
from mloda.provider import (
    FeatureChainParserMixin,
)
from mloda.provider import COLUMNWISE_HOOKS
from mloda.provider import DefaultOptionKeys
from mloda.provider import PropertySpec


class GeoDistanceFeatureGroup(FeatureChainParserMixin, FeatureGroup):
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
    PREFIX_PATTERN = r".*__(?P<distance_type>[\w]+)_distance$"

    # In-feature configuration for FeatureChainParserMixin
    # Geo distance requires exactly 2 point features
    MIN_IN_FEATURES = 2
    MAX_IN_FEATURES = 2

    # Hooks calculate_feature calls: _check_source_features_exist, _add_result_to_data.
    REQUIRED_COLUMNWISE_HOOKS = COLUMNWISE_HOOKS

    # Property mapping for configuration-based features with group/context separation
    PROPERTY_MAPPING = {
        DISTANCE_TYPE: PropertySpec(
            "Type of distance calculation to perform",
            allowed_values=DISTANCE_TYPES,
            context=True,
            strict_validation=True,
        ),
        DefaultOptionKeys.in_features: PropertySpec(
            "Source features (exactly 2 point features required)",
            context=True,
            strict_validation=True,
            # Core unpacks the sequence, so this judges ONE point feature name.
            # The arity (exactly 2) is enforced by MIN_IN_FEATURES / MAX_IN_FEATURES.
            element_validator=lambda point: isinstance(point, str),
        ),
    }

    @classmethod
    def get_distance_type(cls, feature_name: str) -> str:
        """Extract the distance type from the feature name."""
        distance_type = cls.resolve_feature_name(feature_name).value_for(cls.DISTANCE_TYPE)
        if distance_type is None:
            raise ValueError(f"Invalid geo distance feature name format: {feature_name}")

        if distance_type not in cls.DISTANCE_TYPES:
            raise ValueError(
                f"Unsupported distance type: {distance_type}. Supported types: {', '.join(cls.DISTANCE_TYPES.keys())}"
            )

        return distance_type

    @classmethod
    def get_point_features(cls, feature_name: str) -> tuple[str, str]:
        """Extract the two point features from the feature name."""
        resolution = cls.resolve_feature_name(feature_name)
        if not resolution.parsed.matched or not resolution.sources:
            raise ValueError(f"Invalid geo distance feature name format: {feature_name}")

        reason = cls.source_features_reason(feature_name, resolution.sources)
        if reason is not None:
            raise ValueError(reason)
        return resolution.sources[0], resolution.sources[1]

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Calculate distances between point features.

        Processes all requested features, determining the distance type
        and point features from either string parsing or configuration-based options.

        Adds the calculated distances directly to the input data structure.
        """
        # Process each requested feature
        for feature in features.get_sorted_features():
            distance_type, point1_feature, point2_feature = cls._extract_geo_distance_parameters(feature)

            cls._check_source_features_exist(data, [point1_feature, point2_feature])

            if distance_type not in cls.DISTANCE_TYPES:
                raise ValueError(f"Unsupported distance type: {distance_type}")

            result = cls._calculate_distance(data, distance_type, point1_feature, point2_feature)

            data = cls._add_result_to_data(data, feature.name, result)

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
        # Use the mixin method to extract source features
        source_features = cls._extract_source_features(feature)
        cls.validate_in_feature_count(feature.name, len(source_features))

        distance_type = cls._extract_distance_unit(feature)

        if distance_type is None:
            raise ValueError(f"Could not extract geo distance parameters from: {feature.name}")

        return distance_type, source_features[0], source_features[1]

    @classmethod
    def _extract_distance_unit(cls, feature: Feature) -> str | None:
        """Extract the distance type from the prefix-gated name or the options, or None."""
        distance_type = cls._resolve_operation(feature, cls.DISTANCE_TYPE)
        if distance_type is None:
            return None

        # Validate distance type
        if distance_type not in cls.DISTANCE_TYPES:
            raise ValueError(
                f"Unsupported distance type: {distance_type}. Supported types: {', '.join(cls.DISTANCE_TYPES.keys())}"
            )

        return str(distance_type)

    @classmethod
    @abstractmethod
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
        ...
