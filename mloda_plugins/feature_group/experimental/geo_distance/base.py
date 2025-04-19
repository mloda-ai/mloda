"""
Base implementation for geo distance feature groups.
"""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_chainer.feature_chainer_parser_configuration import (
    FeatureChainParserConfiguration,
    create_configurable_parser,
)
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser


class GeoDistanceFeatureGroup(AbstractFeatureGroup):
    """
    Base class for all geo distance feature groups.

    The GeoDistanceFeatureGroup calculates distances between geographic points,
    such as haversine (great-circle), euclidean, or manhattan distances. It extracts
    the two point features from the feature name and calculates the distance between them.

    ## Feature Naming Convention

    Geo distance features follow this naming pattern:
    `{distance_type}_distance__{point1_feature}__{point2_feature}`

    The point features are extracted from the feature name and used as inputs for the
    distance calculation. Note the double underscore before each point feature.

    Examples:
    - `haversine_distance__customer_location__store_location`: Great-circle distance between customer and store
    - `euclidean_distance__origin__destination`: Straight-line distance between origin and destination
    - `manhattan_distance__pickup__dropoff`: Manhattan distance between pickup and dropoff points

    ## Configuration-Based Creation

    GeoDistanceFeatureGroup supports configuration-based creation through the
    FeatureChainParserConfiguration mechanism. This allows features to be created
    from options rather than explicit feature names.
    """

    # Option keys for distance type and point features
    DISTANCE_TYPE = "distance_type"
    POINT1_FEATURE = "point1_feature"
    POINT2_FEATURE = "point2_feature"

    # Define supported distance types
    DISTANCE_TYPES = {
        "haversine": "Great-circle distance on a sphere (for lat/lon coordinates)",
        "euclidean": "Straight-line distance between two points",
        "manhattan": "Sum of absolute differences between coordinates",
    }

    # Define the prefix pattern for this feature group
    PREFIX_PATTERN = r"^([\w]+)_distance__"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """Extract point features from the geo distance feature name."""

        # Extract the source feature part (everything after the prefix)
        source_part = FeatureChainParser.extract_source_feature(feature_name.name, self.PREFIX_PATTERN)

        # Split the source part by double underscore to get the two point features
        parts = source_part.split("__", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid geo distance feature name format: {feature_name.name}. Expected two point features separated by double underscore."
            )

        point1_feature, point2_feature = parts

        return {Feature(point1_feature), Feature(point2_feature)}

    @classmethod
    def get_distance_type(cls, feature_name: str) -> str:
        """Extract the distance type from the feature name."""
        prefix_part = FeatureChainParser.get_prefix_part(feature_name, cls.PREFIX_PATTERN)
        if prefix_part is None:
            raise ValueError(f"Invalid geo distance feature name format: {feature_name}")
        return prefix_part

    @classmethod
    def get_point_features(cls, feature_name: str) -> tuple[str, str]:
        """Extract the two point features from the feature name."""
        # Extract the source feature part (everything after the prefix)
        source_part = FeatureChainParser.extract_source_feature(feature_name, cls.PREFIX_PATTERN)

        # Split the source part by double underscore to get the two point features
        parts = source_part.split("__", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid geo distance feature name format: {feature_name}. Expected two point features separated by double underscore."
            )

        return parts[0], parts[1]

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[Any] = None,
    ) -> bool:
        """Check if feature name matches the expected pattern and distance type."""
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name

        try:
            # First validate that this is a valid feature name for this feature group
            if not FeatureChainParser.validate_feature_name(feature_name, cls.PREFIX_PATTERN):
                return False

            # Then check if the distance type is supported
            distance_type = cls.get_distance_type(feature_name)
            if not cls._supports_distance_type(distance_type):
                return False

            # Finally check if the feature name has two point features
            try:
                cls.get_point_features(feature_name)
                return True
            except ValueError:
                return False

        except ValueError:
            return False

    @classmethod
    def _supports_distance_type(cls, distance_type: str) -> bool:
        """Check if this feature group supports the given distance type."""
        return distance_type in cls.DISTANCE_TYPES

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        Calculate distances between point features.

        Processes all requested features, determining the distance type
        and point features from each feature name.

        Adds the calculated distances directly to the input data structure.
        """
        # Process each requested feature
        for feature_name in features.get_all_names():
            distance_type = cls.get_distance_type(feature_name)
            point1_feature, point2_feature = cls.get_point_features(feature_name)

            cls._check_point_features_exist(data, point1_feature, point2_feature)

            if distance_type not in cls.DISTANCE_TYPES:
                raise ValueError(f"Unsupported distance type: {distance_type}")

            result = cls._calculate_distance(data, distance_type, point1_feature, point2_feature)

            data = cls._add_result_to_data(data, feature_name, result)

        return data

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

        # Define validation function within the method scope
        def raise_unsupported_distance_type(distance_type: str) -> bool:
            """Raise an error for unsupported distance type."""
            raise ValueError(
                f"Unsupported distance type: {distance_type}. Supported types: {list(cls.DISTANCE_TYPES.keys())}"
            )

        # Create and return the configured parser
        return create_configurable_parser(
            parse_keys=[
                cls.DISTANCE_TYPE,
                cls.POINT1_FEATURE,
                cls.POINT2_FEATURE,
            ],
            feature_name_template="{distance_type}_distance__{point1_feature}__{point2_feature}",
            validation_rules={
                cls.DISTANCE_TYPE: lambda x: x in cls.DISTANCE_TYPES or raise_unsupported_distance_type(x),
            },
        )
