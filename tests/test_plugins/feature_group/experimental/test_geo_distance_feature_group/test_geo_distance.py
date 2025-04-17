"""
Tests for the GeoDistanceFeatureGroup.
"""

import unittest
import pandas as pd
import numpy as np
import math

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options

from mloda_plugins.feature_group.experimental.geo_distance.base import GeoDistanceFeatureGroup
from mloda_plugins.feature_group.experimental.geo_distance.pandas import PandasGeoDistanceFeatureGroup


class TestGeoDistanceFeatureGroup(unittest.TestCase):
    """Test cases for the GeoDistanceFeatureGroup."""

    def test_feature_name_parsing(self) -> None:
        """Test parsing of feature names."""
        # Test valid feature names
        self.assertEqual(GeoDistanceFeatureGroup.get_distance_type("haversine_distance__point1__point2"), "haversine")

        point1, point2 = GeoDistanceFeatureGroup.get_point_features("haversine_distance__point1__point2")
        self.assertEqual(point1, "point1")
        self.assertEqual(point2, "point2")

        # Test invalid feature names
        with self.assertRaises(ValueError):
            GeoDistanceFeatureGroup.get_distance_type("invalid_feature_name")

        with self.assertRaises(ValueError):
            GeoDistanceFeatureGroup.get_point_features("haversine_distance__point1")

    def test_match_feature_group_criteria(self) -> None:
        """Test matching of feature names to feature group criteria."""
        # Test valid feature names
        self.assertTrue(
            GeoDistanceFeatureGroup.match_feature_group_criteria("haversine_distance__point1__point2", Options())
        )
        self.assertTrue(
            GeoDistanceFeatureGroup.match_feature_group_criteria("euclidean_distance__point1__point2", Options())
        )
        self.assertTrue(
            GeoDistanceFeatureGroup.match_feature_group_criteria("manhattan_distance__point1__point2", Options())
        )

        # Test invalid feature names
        self.assertFalse(
            GeoDistanceFeatureGroup.match_feature_group_criteria("invalid_distance__point1__point2", Options())
        )
        self.assertFalse(GeoDistanceFeatureGroup.match_feature_group_criteria("haversine_distance__point1", Options()))
        self.assertFalse(
            GeoDistanceFeatureGroup.match_feature_group_criteria("haversine_invalid__point1__point2", Options())
        )

    def test_input_features(self) -> None:
        """Test extraction of input features."""
        feature_group = GeoDistanceFeatureGroup()
        feature_name = FeatureName("haversine_distance__point1__point2")

        input_features = feature_group.input_features(Options(), feature_name)
        # Raise an exception if input_features is None
        if input_features is None:
            raise AssertionError("input_features should not be None")

        self.assertEqual(len(input_features), 2)
        self.assertTrue(Feature("point1") in input_features)
        self.assertTrue(Feature("point2") in input_features)

    def test_feature_chain_parser_configuration(self) -> None:
        """Test the feature chain parser configuration."""
        parser_config_class = GeoDistanceFeatureGroup.configurable_feature_chain_parser()

        # Raise an exception if parser_config_class is None
        if parser_config_class is None:
            raise AssertionError("parser_config_class should not be None")

        # Test parsing keys
        parse_keys = parser_config_class.parse_keys()
        self.assertEqual(len(parse_keys), 3)
        self.assertTrue(GeoDistanceFeatureGroup.DISTANCE_TYPE in parse_keys)
        self.assertTrue(GeoDistanceFeatureGroup.POINT1_FEATURE in parse_keys)
        self.assertTrue(GeoDistanceFeatureGroup.POINT2_FEATURE in parse_keys)

        # Test parsing from options
        options = Options(
            {
                GeoDistanceFeatureGroup.DISTANCE_TYPE: "haversine",
                GeoDistanceFeatureGroup.POINT1_FEATURE: "point1",
                GeoDistanceFeatureGroup.POINT2_FEATURE: "point2",
            }
        )

        feature_name = parser_config_class.parse_from_options(options)
        self.assertEqual(feature_name, "haversine_distance__point1__point2")

        # Test with missing options
        options = Options(
            {GeoDistanceFeatureGroup.DISTANCE_TYPE: "haversine", GeoDistanceFeatureGroup.POINT1_FEATURE: "point1"}
        )

        feature_name = parser_config_class.parse_from_options(options)
        self.assertIsNone(feature_name)

        # Test with invalid distance type
        options = Options(
            {
                GeoDistanceFeatureGroup.DISTANCE_TYPE: "invalid",
                GeoDistanceFeatureGroup.POINT1_FEATURE: "point1",
                GeoDistanceFeatureGroup.POINT2_FEATURE: "point2",
            }
        )

        with self.assertRaises(ValueError):
            parser_config_class.parse_from_options(options)


class TestPandasGeoDistanceFeatureGroup(unittest.TestCase):
    """Test cases for the PandasGeoDistanceFeatureGroup."""

    def setUp(self) -> None:
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
        with self.assertRaises(ValueError):
            PandasGeoDistanceFeatureGroup._check_point_features_exist(self.df, "sf", "invalid")

        with self.assertRaises(ValueError):
            PandasGeoDistanceFeatureGroup._check_point_features_exist(self.df, "invalid", "nyc")

    def test_haversine_distance(self) -> None:
        """Test calculation of haversine distance."""
        # Calculate haversine distance between San Francisco and New York
        distance = PandasGeoDistanceFeatureGroup._calculate_haversine_distance(self.df, "sf", "nyc")

        # Expected distance is approximately 4130 km
        self.assertAlmostEqual(distance[0], 4130, delta=100)

        # Calculate haversine distance between San Francisco and Los Angeles
        distance = PandasGeoDistanceFeatureGroup._calculate_haversine_distance(self.df, "sf", "la")

        # Expected distance is approximately 560 km
        self.assertAlmostEqual(distance[0], 560, delta=50)

    def test_euclidean_distance(self) -> None:
        """Test calculation of euclidean distance."""
        # Calculate euclidean distance between point1 and point2
        distance = PandasGeoDistanceFeatureGroup._calculate_euclidean_distance(self.df, "point1", "point2")

        # Expected distances: sqrt(3^2 + 4^2) = 5, sqrt(4^2 + 4^2) = 5.66, sqrt(4^2 + 6^2) = 7.21
        self.assertAlmostEqual(distance[0], 5.0, delta=0.01)
        self.assertAlmostEqual(distance[1], 5.66, delta=0.01)
        self.assertAlmostEqual(distance[2], 7.21, delta=0.01)

    def test_manhattan_distance(self) -> None:
        """Test calculation of manhattan distance."""
        # Calculate manhattan distance between point1 and point2
        distance = PandasGeoDistanceFeatureGroup._calculate_manhattan_distance(self.df, "point1", "point2")

        # Expected distances: |3-0| + |4-0| = 7, |5-1| + |5-1| = 8, |6-2| + |8-2| = 10
        self.assertEqual(distance[0], 7)
        self.assertEqual(distance[1], 8)
        self.assertEqual(distance[2], 10)

    def test_calculate_feature(self) -> None:
        """Test calculation of features."""
        # Create a feature set with haversine distance feature
        feature_set = FeatureSet()
        feature_set.add(Feature("haversine_distance__sf__nyc"))

        # Calculate the feature
        result_df = PandasGeoDistanceFeatureGroup.calculate_feature(self.df.copy(), feature_set)

        # Check if the feature was added to the DataFrame
        self.assertTrue("haversine_distance__sf__nyc" in result_df.columns)

        # Check if the distance is approximately correct
        self.assertAlmostEqual(result_df["haversine_distance__sf__nyc"][0], 4130, delta=100)

        # Test with multiple features
        feature_set = FeatureSet()
        feature_set.add(Feature("haversine_distance__sf__nyc"))
        feature_set.add(Feature("euclidean_distance__point1__point2"))
        feature_set.add(Feature("manhattan_distance__point1__point2"))

        # Calculate the features
        result_df = PandasGeoDistanceFeatureGroup.calculate_feature(self.df.copy(), feature_set)

        # Check if all features were added to the DataFrame
        self.assertTrue("haversine_distance__sf__nyc" in result_df.columns)
        self.assertTrue("euclidean_distance__point1__point2" in result_df.columns)
        self.assertTrue("manhattan_distance__point1__point2" in result_df.columns)

        # Check if the distances are approximately correct
        self.assertAlmostEqual(result_df["haversine_distance__sf__nyc"][0], 4130, delta=100)
        self.assertAlmostEqual(result_df["euclidean_distance__point1__point2"][0], 5.0, delta=0.01)
        self.assertEqual(result_df["manhattan_distance__point1__point2"][0], 7)
