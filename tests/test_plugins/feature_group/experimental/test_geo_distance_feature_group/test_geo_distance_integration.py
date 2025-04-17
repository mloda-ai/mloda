"""
Integration tests for geo distance feature groups.
"""

from typing import Any, Dict, List
import pandas as pd

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.geo_distance.base import GeoDistanceFeatureGroup
from mloda_plugins.feature_group.experimental.geo_distance.pandas import PandasGeoDistanceFeatureGroup

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


# List of geo distance features to test
GEO_DISTANCE_FEATURES: List[Feature | str] = [
    "haversine_distance__sf__nyc",  # Haversine distance between San Francisco and New York
    "euclidean_distance__point1__point2",  # Euclidean distance between point1 and point2
    "manhattan_distance__point1__point2",  # Manhattan distance between point1 and point2
]


class GeoDistanceTestDataCreator(ATestDataCreator):
    """Base class for geo distance test data creators."""

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
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


class PandasGeoDistanceTestDataCreator(GeoDistanceTestDataCreator):
    """Pandas implementation of the geo distance test data creator."""

    compute_framework = PandasDataframe


def validate_geo_distance_features(df: pd.DataFrame, expected_features: List[Feature | str]) -> None:
    """
    Validate geo distance features in a Pandas DataFrame.

    Args:
        df: DataFrame containing geo distance features
        expected_features: List of expected feature names

    Raises:
        AssertionError: If validation fails
    """
    # Verify all expected features exist
    for feature in expected_features:
        # Get the feature name if it's a Feature object, otherwise use it directly
        feature_name = feature.name if isinstance(feature, Feature) else feature
        assert feature_name in df.columns, f"Expected feature '{feature_name}' not found"

    # Validate specific distance calculations for the first row
    if "haversine_distance__sf__nyc" in df.columns:
        # Expected distance is approximately 4130 km
        assert abs(df["haversine_distance__sf__nyc"].iloc[0] - 4130) < 100, (
            "Haversine distance calculation is incorrect"
        )

    if "euclidean_distance__point1__point2" in df.columns:
        # Expected distance is 5.0 for the first row
        assert abs(df["euclidean_distance__point1__point2"].iloc[0] - 5.0) < 0.1, (
            "Euclidean distance calculation is incorrect"
        )

    if "manhattan_distance__point1__point2" in df.columns:
        # Expected distance is 7 for the first row
        assert df["manhattan_distance__point1__point2"].iloc[0] == 7, "Manhattan distance calculation is incorrect"


class TestGeoDistancePandasIntegration:
    """Integration tests for the geo distance feature group using Pandas."""

    def test_geo_distance_with_data_creator(self) -> None:
        """Test geo distance features with mlodaAPI using DataCreator."""

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {PandasGeoDistanceTestDataCreator, PandasGeoDistanceFeatureGroup}
        )

        # Run the API with multiple geo distance features
        result = mlodaAPI.run_all(
            [
                "sf",  # Source data - San Francisco coordinates
                "nyc",  # Source data - New York coordinates
                "point1",  # Source data - Point 1
                "point2",  # Source data - Point 2
                "haversine_distance__sf__nyc",  # Haversine distance between SF and NYC
                "euclidean_distance__point1__point2",  # Euclidean distance between point1 and point2
                "manhattan_distance__point1__point2",  # Manhattan distance between point1 and point2
            ],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) == 2  # Two DataFrames: one for source data, one for geo distance features

        # Find the DataFrame with the geo distance features
        distance_df = None
        for df in result:
            if "haversine_distance__sf__nyc" in df.columns:
                distance_df = df
                break

        assert distance_df is not None, "DataFrame with geo distance features not found"

        # Validate the geo distance features
        validate_geo_distance_features(distance_df, GEO_DISTANCE_FEATURES)

    def test_geo_distance_with_configuration(self) -> None:
        """Test geo distance features with mlodaAPI using configuration-based feature creation."""

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {PandasGeoDistanceTestDataCreator, PandasGeoDistanceFeatureGroup}
        )

        # Get the parser configuration class
        parser = GeoDistanceFeatureGroup.configurable_feature_chain_parser()
        if parser is None:
            raise ValueError("Feature chain parser is not available.")

        # Create features using configuration
        haversine_config = Feature(
            "x",  # Temporary name, will be replaced
            Options(
                {
                    GeoDistanceFeatureGroup.DISTANCE_TYPE: "haversine",
                    GeoDistanceFeatureGroup.POINT1_FEATURE: "sf",
                    GeoDistanceFeatureGroup.POINT2_FEATURE: "nyc",
                }
            ),
        )

        euclidean_config = Feature(
            "x",  # Temporary name, will be replaced
            Options(
                {
                    GeoDistanceFeatureGroup.DISTANCE_TYPE: "euclidean",
                    GeoDistanceFeatureGroup.POINT1_FEATURE: "point1",
                    GeoDistanceFeatureGroup.POINT2_FEATURE: "point2",
                }
            ),
        )

        manhattan_config = Feature(
            "x",  # Temporary name, will be replaced
            Options(
                {
                    GeoDistanceFeatureGroup.DISTANCE_TYPE: "manhattan",
                    GeoDistanceFeatureGroup.POINT1_FEATURE: "point1",
                    GeoDistanceFeatureGroup.POINT2_FEATURE: "point2",
                }
            ),
        )

        # Parse the features using the parser
        haversine_feature = parser.create_feature_without_options(haversine_config)
        euclidean_feature = parser.create_feature_without_options(euclidean_config)
        manhattan_feature = parser.create_feature_without_options(manhattan_config)

        if haversine_feature is None or euclidean_feature is None or manhattan_feature is None:
            raise ValueError("Failed to create features using the parser.")

        # Run the API with pre-parsed features
        result = mlodaAPI.run_all(
            [
                "sf",  # Source data - San Francisco coordinates
                "nyc",  # Source data - New York coordinates
                "point1",  # Source data - Point 1
                "point2",  # Source data - Point 2
                haversine_feature,  # Haversine distance between SF and NYC
                euclidean_feature,  # Euclidean distance between point1 and point2
                manhattan_feature,  # Manhattan distance between point1 and point2
            ],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) == 2  # Two DataFrames: one for source data, one for geo distance features

        # Find the DataFrame with the geo distance features
        distance_df = None
        for df in result:
            if "haversine_distance__sf__nyc" in df.columns:
                distance_df = df
                break

        assert distance_df is not None, "DataFrame with geo distance features not found"

        # Validate the geo distance features
        validate_geo_distance_features(distance_df, GEO_DISTANCE_FEATURES)

        # Now test with mloda parsing the features directly
        result2 = mlodaAPI.run_all(
            [
                "sf",  # Source data - San Francisco coordinates
                "nyc",  # Source data - New York coordinates
                "point1",  # Source data - Point 1
                "point2",  # Source data - Point 2
                haversine_config,  # Haversine distance between SF and NYC (unparsed)
                euclidean_config,  # Euclidean distance between point1 and point2 (unparsed)
                manhattan_config,  # Manhattan distance between point1 and point2 (unparsed)
            ],
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result2) == 2  # Two DataFrames: one for source data, one for geo distance features

        # Find the DataFrame with the geo distance features
        distance_df = None
        for df in result2:
            if "haversine_distance__sf__nyc" in df.columns:
                distance_df = df
                break

        assert distance_df is not None, "DataFrame with geo distance features not found"

        # Validate the geo distance features
        validate_geo_distance_features(distance_df, GEO_DISTANCE_FEATURES)
