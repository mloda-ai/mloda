"""
Pandas implementation for geo distance feature groups.
"""

from __future__ import annotations

import math
from typing import Any, Set, Type, Union

import numpy as np

from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork

from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.geo_distance.base import GeoDistanceFeatureGroup


class PandasGeoDistanceFeatureGroup(GeoDistanceFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        """Specify that this feature group works with Pandas."""
        return {PandasDataframe}

    @classmethod
    def _check_point_features_exist(cls, data: Any, point1_feature: str, point2_feature: str) -> None:
        """Check if the point features exist in the DataFrame."""
        if point1_feature not in data.columns:
            raise ValueError(f"Point feature '{point1_feature}' not found in data")
        if point2_feature not in data.columns:
            raise ValueError(f"Point feature '{point2_feature}' not found in data")

    @classmethod
    def _add_result_to_data(cls, data: Any, feature_name: str, result: Any) -> Any:
        """Add the result to the DataFrame."""
        data[feature_name] = result
        return data

    @classmethod
    def _calculate_distance(cls, data: Any, distance_type: str, point1_feature: str, point2_feature: str) -> Any:
        """
        Calculate the distance between two point features using the specified distance type.

        Args:
            data: The Pandas DataFrame
            distance_type: The type of distance to calculate
            point1_feature: The name of the first point feature
            point2_feature: The name of the second point feature

        Returns:
            The calculated distance as a Pandas Series
        """
        if distance_type == "haversine":
            return cls._calculate_haversine_distance(data, point1_feature, point2_feature)
        elif distance_type == "euclidean":
            return cls._calculate_euclidean_distance(data, point1_feature, point2_feature)
        elif distance_type == "manhattan":
            return cls._calculate_manhattan_distance(data, point1_feature, point2_feature)
        else:
            raise ValueError(f"Unsupported distance type: {distance_type}")

    @classmethod
    def _calculate_haversine_distance(cls, data: Any, point1_feature: str, point2_feature: str) -> Any:
        """
        Calculate the haversine (great-circle) distance between two points.

        The haversine formula determines the great-circle distance between two points
        on a sphere given their longitudes and latitudes.

        Args:
            data: The Pandas DataFrame
            point1_feature: The name of the first point feature (lat, lon tuple/list)
            point2_feature: The name of the second point feature (lat, lon tuple/list)

        Returns:
            The haversine distance in kilometers as a Pandas Series
        """
        # Earth radius in kilometers
        R = 6371.0

        # Extract coordinates
        # Assuming each point feature contains (latitude, longitude) as a tuple or list
        lat1 = data[point1_feature].apply(lambda p: p[0])
        lon1 = data[point1_feature].apply(lambda p: p[1])
        lat2 = data[point2_feature].apply(lambda p: p[0])
        lon2 = data[point2_feature].apply(lambda p: p[1])

        # Convert degrees to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        # Haversine formula
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        # Calculate distance
        distance = R * c

        return distance

    @classmethod
    def _calculate_euclidean_distance(cls, data: Any, point1_feature: str, point2_feature: str) -> Any:
        """
        Calculate the Euclidean (straight-line) distance between two points.

        Args:
            data: The Pandas DataFrame
            point1_feature: The name of the first point feature (x, y coordinates)
            point2_feature: The name of the second point feature (x, y coordinates)

        Returns:
            The Euclidean distance as a Pandas Series
        """
        # Extract coordinates
        # Assuming each point feature contains (x, y) as a tuple or list
        x1 = data[point1_feature].apply(lambda p: p[0])
        y1 = data[point1_feature].apply(lambda p: p[1])
        x2 = data[point2_feature].apply(lambda p: p[0])
        y2 = data[point2_feature].apply(lambda p: p[1])

        # Calculate Euclidean distance
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        return distance

    @classmethod
    def _calculate_manhattan_distance(cls, data: Any, point1_feature: str, point2_feature: str) -> Any:
        """
        Calculate the Manhattan (taxicab) distance between two points.

        The Manhattan distance is the sum of the absolute differences of their Cartesian coordinates.

        Args:
            data: The Pandas DataFrame
            point1_feature: The name of the first point feature (x, y coordinates)
            point2_feature: The name of the second point feature (x, y coordinates)

        Returns:
            The Manhattan distance as a Pandas Series
        """
        # Extract coordinates
        # Assuming each point feature contains (x, y) as a tuple or list
        x1 = data[point1_feature].apply(lambda p: p[0])
        y1 = data[point1_feature].apply(lambda p: p[1])
        x2 = data[point2_feature].apply(lambda p: p[0])
        y2 = data[point2_feature].apply(lambda p: p[1])

        # Calculate Manhattan distance
        distance = np.abs(x2 - x1) + np.abs(y2 - y1)

        return distance
