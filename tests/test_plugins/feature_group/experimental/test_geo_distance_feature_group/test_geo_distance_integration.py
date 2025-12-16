"""
Integration tests for geo distance feature groups.
"""

from typing import Any, Dict, FrozenSet, List
import pandas as pd

import mloda
from mloda import Feature
from mloda import Options
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.geo_distance.base import GeoDistanceFeatureGroup
from mloda_plugins.feature_group.experimental.geo_distance.pandas import PandasGeoDistanceFeatureGroup

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


# List of geo distance features to test
GEO_DISTANCE_FEATURES: List[Feature | str] = [
    "sf&nyc__haversine_distance",  # Haversine distance between San Francisco and New York
    "point1&point2__euclidean_distance",  # Euclidean distance between point1 and point2
    "point1&point2__manhattan_distance",  # Manhattan distance between point1 and point2
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

    compute_framework = PandasDataFrame


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
    if "sf&nyc__haversine_distance" in df.columns:
        # Expected distance is approximately 4130 km
        assert abs(df["sf&nyc__haversine_distance"].iloc[0] - 4130) < 100, "Haversine distance calculation is incorrect"

    if "point1&point2__euclidean_distance" in df.columns:
        # Expected distance is 5.0 for the first row
        assert abs(df["point1&point2__euclidean_distance"].iloc[0] - 5.0) < 0.1, (
            "Euclidean distance calculation is incorrect"
        )

    if "point1&point2__manhattan_distance" in df.columns:
        # Expected distance is 7 for the first row
        assert df["point1&point2__manhattan_distance"].iloc[0] == 7, "Manhattan distance calculation is incorrect"


class TestGeoDistancePandasIntegration:
    """Integration tests for the geo distance feature group using Pandas."""

    def test_geo_distance_with_data_creator(self) -> None:
        """Test geo distance features with API using DataCreator."""

        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PandasGeoDistanceTestDataCreator, PandasGeoDistanceFeatureGroup}
        )

        # Run the API with multiple geo distance features
        result = mloda.run_all(
            [
                "sf",  # Source data - San Francisco coordinates
                "nyc",  # Source data - New York coordinates
                "point1",  # Source data - Point 1
                "point2",  # Source data - Point 2
                "sf&nyc__haversine_distance",  # Haversine distance between SF and NYC
                "point1&point2__euclidean_distance",  # Euclidean distance between point1 and point2
                "point1&point2__manhattan_distance",  # Manhattan distance between point1 and point2
            ],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) == 2  # Two DataFrames: one for source data, one for geo distance features

        # Find the DataFrame with the geo distance features
        distance_df = None
        for df in result:
            if "sf&nyc__haversine_distance" in df.columns:
                distance_df = df
                break

        assert distance_df is not None, "DataFrame with geo distance features not found"

        # Validate the geo distance features
        validate_geo_distance_features(distance_df, GEO_DISTANCE_FEATURES)

    def test_geo_distance_with_configuration(self) -> None:
        """Test geo distance features with API using configuration-based feature creation."""

        # Enable the necessary feature groups
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PandasGeoDistanceTestDataCreator, PandasGeoDistanceFeatureGroup}
        )

        # Create features using configuration-based approach
        haversine_config = Feature(
            "haversine_geo_distance",
            Options(
                context={
                    GeoDistanceFeatureGroup.DISTANCE_TYPE: "haversine",
                    DefaultOptionKeys.in_features: frozenset(["sf", "nyc"]),
                }
            ),
        )

        euclidean_config = Feature(
            "euclidean_geo_distance",
            Options(
                context={
                    GeoDistanceFeatureGroup.DISTANCE_TYPE: "euclidean",
                    DefaultOptionKeys.in_features: frozenset(["point1", "point2"]),
                }
            ),
        )

        manhattan_config = Feature(
            "manhattan_geo_distance",
            Options(
                context={
                    GeoDistanceFeatureGroup.DISTANCE_TYPE: "manhattan",
                    DefaultOptionKeys.in_features: frozenset(["point1", "point2"]),
                }
            ),
        )

        # Run the API with configuration-based features
        result = mloda.run_all(
            [
                "sf",  # Source data - San Francisco coordinates
                "nyc",  # Source data - New York coordinates
                "point1",  # Source data - Point 1
                "point2",  # Source data - Point 2
                haversine_config,  # Haversine distance between SF and NYC
                euclidean_config,  # Euclidean distance between point1 and point2
                manhattan_config,  # Manhattan distance between point1 and point2
            ],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) == 2  # Two DataFrames: one for source data, one for geo distance features

        # Find the DataFrame with the geo distance features
        distance_df = None
        for df in result:
            if "haversine_geo_distance" in df.columns:
                distance_df = df
                break

        assert distance_df is not None, "DataFrame with geo distance features not found"

        # Validate that the configuration-based features were created
        assert "haversine_geo_distance" in distance_df.columns
        assert "euclidean_geo_distance" in distance_df.columns
        assert "manhattan_geo_distance" in distance_df.columns

        # Validate specific distance calculations for the first row
        # Expected haversine distance is approximately 4130 km
        assert abs(distance_df["haversine_geo_distance"].iloc[0] - 4130) < 100, (
            "Haversine distance calculation is incorrect"
        )

        # Expected euclidean distance is 5.0 for the first row
        assert abs(distance_df["euclidean_geo_distance"].iloc[0] - 5.0) < 0.1, (
            "Euclidean distance calculation is incorrect"
        )

        # Expected manhattan distance is 7 for the first row
        assert distance_df["manhattan_geo_distance"].iloc[0] == 7, "Manhattan distance calculation is incorrect"
