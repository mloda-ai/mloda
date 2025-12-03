"""
Tests for the GeoDistanceFeatureGroup.
"""

import pandas as pd
import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options

from mloda_plugins.feature_group.experimental.geo_distance.base import GeoDistanceFeatureGroup
from mloda_plugins.feature_group.experimental.geo_distance.pandas import PandasGeoDistanceFeatureGroup


class TestGeoDistanceFeatureGroup:
    """Test cases for the GeoDistanceFeatureGroup."""

    def test_feature_name_parsing(self) -> None:
        """Test parsing of feature names."""
        # Test valid feature names
        assert GeoDistanceFeatureGroup.get_distance_type("point1&point2__haversine_distance") == "haversine"

        point1, point2 = GeoDistanceFeatureGroup.get_point_features("point1&point2__haversine_distance")
        assert point1 == "point1"
        assert point2 == "point2"

        # Test invalid feature names
        with pytest.raises(ValueError):
            GeoDistanceFeatureGroup.get_distance_type("invalid_feature_name")

        with pytest.raises(ValueError):
            GeoDistanceFeatureGroup.get_point_features("point1__haversine_distance")

    def test_match_feature_group_criteria(self) -> None:
        """Test matching of feature names to feature group criteria."""
        # Test valid feature names
        assert GeoDistanceFeatureGroup.match_feature_group_criteria("point1&point2__haversine_distance", Options())
        assert GeoDistanceFeatureGroup.match_feature_group_criteria("point1&point2__euclidean_distance", Options())
        assert GeoDistanceFeatureGroup.match_feature_group_criteria("point1&point2__manhattan_distance", Options())

        assert not GeoDistanceFeatureGroup.match_feature_group_criteria("point1&point2__haversine_invalid", Options())

    def test_input_features(self) -> None:
        """Test extraction of input features."""
        feature_group = GeoDistanceFeatureGroup()
        feature_name = FeatureName("point1&point2__haversine_distance")

        input_features = feature_group.input_features(Options(), feature_name)
        # Raise an exception if input_features is None
        if input_features is None:
            raise AssertionError("input_features should not be None")

        assert len(input_features) == 2
        assert Feature("point1") in input_features
        assert Feature("point2") in input_features


class TestPandasGeoDistanceFeatureGroup:
    """Test cases for the PandasGeoDistanceFeatureGroup."""

    def setup_method(self) -> None:
        """Set up test data."""
        # Create a test DataFrame with point features
        self.df = pd.DataFrame(
            {
                # San Francisco coordinates
                "sf": [(37.7749, -122.4194) for _ in range(3)],
                # New York coordinates
                "nyc": [(40.7128, -74.0060) for _ in range(3)],
                # Los Angeles coordinates
                "la": [(34.0522, -118.2437) for _ in range(3)],
                # Points for Euclidean and Manhattan distance tests
                "point1": [(0, 0), (1, 1), (2, 2)],
                "point2": [(3, 4), (5, 5), (6, 8)],
            }
        )

    def test_check_point_features_exist(self) -> None:
        """Test checking if point features exist."""
        # Test with existing features
        PandasGeoDistanceFeatureGroup._check_point_features_exist(self.df, "sf", "nyc")

        # Test with non-existing features
        with pytest.raises(ValueError):
            PandasGeoDistanceFeatureGroup._check_point_features_exist(self.df, "sf", "invalid")

        with pytest.raises(ValueError):
            PandasGeoDistanceFeatureGroup._check_point_features_exist(self.df, "invalid", "nyc")

    def test_haversine_distance(self) -> None:
        """Test calculation of haversine distance."""
        # Calculate haversine distance between San Francisco and New York
        distance = PandasGeoDistanceFeatureGroup._calculate_haversine_distance(self.df, "sf", "nyc")

        # Expected distance is approximately 4130 km
        assert distance[0] == pytest.approx(4130, abs=100)

        # Calculate haversine distance between San Francisco and Los Angeles
        distance = PandasGeoDistanceFeatureGroup._calculate_haversine_distance(self.df, "sf", "la")

        # Expected distance is approximately 560 km
        assert distance[0] == pytest.approx(560, abs=50)

    def test_euclidean_distance(self) -> None:
        """Test calculation of euclidean distance."""
        # Calculate euclidean distance between point1 and point2
        distance = PandasGeoDistanceFeatureGroup._calculate_euclidean_distance(self.df, "point1", "point2")

        # Expected distances: sqrt(3^2 + 4^2) = 5, sqrt(4^2 + 4^2) = 5.66, sqrt(4^2 + 6^2) = 7.21
        assert distance[0] == pytest.approx(5.0, abs=0.01)
        assert distance[1] == pytest.approx(5.66, abs=0.01)
        assert distance[2] == pytest.approx(7.21, abs=0.01)

    def test_manhattan_distance(self) -> None:
        """Test calculation of manhattan distance."""
        # Calculate manhattan distance between point1 and point2
        distance = PandasGeoDistanceFeatureGroup._calculate_manhattan_distance(self.df, "point1", "point2")

        # Expected distances: |3-0| + |4-0| = 7, |5-1| + |5-1| = 8, |6-2| + |8-2| = 10
        assert distance[0] == 7
        assert distance[1] == 8
        assert distance[2] == 10

    def test_calculate_feature(self) -> None:
        """Test calculation of features."""
        # Create a feature set with haversine distance feature
        feature_set = FeatureSet()
        feature_set.add(Feature("sf&nyc__haversine_distance"))

        # Calculate the feature
        result_df = PandasGeoDistanceFeatureGroup.calculate_feature(self.df.copy(), feature_set)

        # Check if the feature was added to the DataFrame
        assert "sf&nyc__haversine_distance" in result_df.columns

        # Check if the distance is approximately correct
        assert result_df["sf&nyc__haversine_distance"][0] == pytest.approx(4130, abs=100)

        # Test with multiple features
        feature_set = FeatureSet()
        feature_set.add(Feature("sf&nyc__haversine_distance"))
        feature_set.add(Feature("point1&point2__euclidean_distance"))
        feature_set.add(Feature("point1&point2__manhattan_distance"))

        # Calculate the features
        result_df = PandasGeoDistanceFeatureGroup.calculate_feature(self.df.copy(), feature_set)

        # Check if all features were added to the DataFrame
        assert "sf&nyc__haversine_distance" in result_df.columns
        assert "point1&point2__euclidean_distance" in result_df.columns
        assert "point1&point2__manhattan_distance" in result_df.columns

        # Check if the distances are approximately correct
        assert result_df["sf&nyc__haversine_distance"][0] == pytest.approx(4130, abs=100)
        assert result_df["point1&point2__euclidean_distance"][0] == pytest.approx(5.0, abs=0.01)
        assert result_df["point1&point2__manhattan_distance"][0] == 7
