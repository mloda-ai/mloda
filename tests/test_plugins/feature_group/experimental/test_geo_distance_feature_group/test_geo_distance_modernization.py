"""
Tests for the modernized GeoDistanceFeatureGroup with both string-based and configuration-based features.

This test verifies that the modernization successfully supports both approaches:
1. String-based feature creation (legacy)
2. Configuration-based feature creation (modern)
"""

from typing import Any, Dict, List, Union

import pytest

from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_collection import Features
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.api.request import mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe
from mloda_plugins.feature_group.experimental.geo_distance.base import GeoDistanceFeatureGroup
from mloda_plugins.feature_group.experimental.geo_distance.pandas import PandasGeoDistanceFeatureGroup
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


class GeoDistanceModernizationTestDataCreator(ATestDataCreator):
    """Test data creator for geo distance modernization tests."""

    compute_framework = PandasDataframe

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {
            # San Francisco coordinates (lat, lon)
            "sf_location": [(37.7749, -122.4194), (37.7849, -122.4094), (37.7649, -122.4294)],
            # New York coordinates (lat, lon)
            "nyc_location": [(40.7128, -74.0060), (40.7228, -74.0160), (40.7028, -73.9960)],
            # Los Angeles coordinates (lat, lon)
            "la_location": [(34.0522, -118.2437), (34.0622, -118.2537), (34.0422, -118.2337)],
            # Points for Euclidean and Manhattan distance tests (x, y)
            "point_a": [(0, 0), (1, 1), (2, 2)],
            "point_b": [(3, 4), (5, 5), (6, 8)],
            # Additional test points
            "origin": [(0, 0), (10, 10), (20, 20)],
            "destination": [(5, 5), (15, 15), (25, 25)],
        }


class TestGeoDistanceModernization:
    """Test the modernized GeoDistanceFeatureGroup with both string-based and configuration-based features."""

    def test_string_based_feature_creation(self) -> None:
        """Test that string-based feature creation still works (legacy approach)."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {GeoDistanceModernizationTestDataCreator, PandasGeoDistanceFeatureGroup}
        )

        # Create string-based features
        features: Features | List[str | Feature] = [
            Feature("haversine_distance__sf_location__nyc_location"),
            Feature("euclidean_distance__point_a__point_b"),
            Feature("manhattan_distance__origin__destination"),
        ]

        # Run the API
        api = mlodaAPI(
            features,
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        api._batch_run()
        results = api.get_result()

        # Verify that all features were created successfully
        assert len(results) == 1
        result_df = results[0]

        assert "haversine_distance__sf_location__nyc_location" in result_df.columns
        assert "euclidean_distance__point_a__point_b" in result_df.columns
        assert "manhattan_distance__origin__destination" in result_df.columns
        assert len(result_df) > 0  # Should have data

        # Verify reasonable distance values
        # Haversine distance between SF and NYC should be around 4130 km
        haversine_distances = result_df["haversine_distance__sf_location__nyc_location"]
        assert all(distance > 4000 and distance < 4300 for distance in haversine_distances)

    def test_configuration_based_feature_creation(self) -> None:
        """Test that configuration-based feature creation works (modern approach)."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {GeoDistanceModernizationTestDataCreator, PandasGeoDistanceFeatureGroup}
        )

        # Create configuration-based features
        features: Features | List[str | Feature] = [
            Feature(
                name="geo_distance_haversine",
                options=Options(
                    context={
                        GeoDistanceFeatureGroup.DISTANCE_TYPE: "haversine",
                        DefaultOptionKeys.mloda_source_feature: frozenset(["sf_location", "la_location"]),
                    }
                ),
            ),
            Feature(
                name="geo_distance_euclidean",
                options=Options(
                    context={
                        GeoDistanceFeatureGroup.DISTANCE_TYPE: "euclidean",
                        DefaultOptionKeys.mloda_source_feature: frozenset(["point_a", "point_b"]),
                    }
                ),
            ),
            Feature(
                name="geo_distance_manhattan",
                options=Options(
                    context={
                        GeoDistanceFeatureGroup.DISTANCE_TYPE: "manhattan",
                        DefaultOptionKeys.mloda_source_feature: frozenset(["origin", "destination"]),
                    }
                ),
            ),
        ]

        # Run the API
        api = mlodaAPI(
            features,
            compute_frameworks={PandasDataframe},
            plugin_collector=plugin_collector,
        )

        api._batch_run()
        results = api.get_result()

        # Verify that all features were created successfully
        assert len(results) == 1
        result_df = results[0]

        assert "geo_distance_haversine" in result_df.columns
        assert "geo_distance_euclidean" in result_df.columns
        assert "geo_distance_manhattan" in result_df.columns
        assert len(result_df) > 0  # Should have data

        # Verify reasonable distance values
        # Haversine distance between SF and LA should be around 560 km
        haversine_distances = result_df["geo_distance_haversine"]
        assert all(distance > 500 and distance < 650 for distance in haversine_distances)

    def test_both_approaches_produce_equivalent_results(self) -> None:
        """Test that both string-based and configuration-based approaches produce equivalent functionality."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {GeoDistanceModernizationTestDataCreator, PandasGeoDistanceFeatureGroup}
        )

        # Create string-based feature
        string_feature = Feature("euclidean_distance__point_a__point_b")

        # Create configuration-based feature with same parameters
        config_feature = Feature(
            name="config_euclidean",
            options=Options(
                context={
                    GeoDistanceFeatureGroup.DISTANCE_TYPE: "euclidean",
                    DefaultOptionKeys.mloda_source_feature: frozenset(["point_a", "point_b"]),
                }
            ),
        )

        # Run both approaches
        api1 = mlodaAPI([string_feature], {PandasDataframe}, plugin_collector=plugin_collector)
        api1._batch_run()
        results1 = api1.get_result()

        api2 = mlodaAPI([config_feature], {PandasDataframe}, plugin_collector=plugin_collector)
        api2._batch_run()
        results2 = api2.get_result()

        # Both should produce results with their respective feature names
        assert "euclidean_distance__point_a__point_b" in results1[0].columns
        assert "config_euclidean" in results2[0].columns

        # Results should have the same structure and values (same calculation)
        assert len(results1[0]) == len(results2[0])

        # The calculated distances should be identical
        string_distances = results1[0]["euclidean_distance__point_a__point_b"].values
        config_distances = results2[0]["config_euclidean"].values

        # Allow for small floating point differences
        assert all(abs(s - c) < 1e-10 for s, c in zip(string_distances, config_distances))

    def test_parameter_validation_in_configuration_based_features(self) -> None:
        """Test that parameter validation works correctly for configuration-based features."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {GeoDistanceModernizationTestDataCreator, PandasGeoDistanceFeatureGroup}
        )

        # Test invalid distance type
        with pytest.raises(Exception):  # Should fail during validation
            feature = Feature(
                name="placeholder",
                options=Options(
                    context={
                        GeoDistanceFeatureGroup.DISTANCE_TYPE: "invalid_distance_type",
                        DefaultOptionKeys.mloda_source_feature: frozenset(["point_a", "point_b"]),
                    }
                ),
            )
            api = mlodaAPI([feature], {PandasDataframe}, plugin_collector=plugin_collector)
            api._batch_run()

        # Test invalid number of source features (only 1 instead of 2)
        with pytest.raises(Exception):  # Should fail during validation
            feature = Feature(
                name="placeholder",
                options=Options(
                    context={
                        GeoDistanceFeatureGroup.DISTANCE_TYPE: "euclidean",
                        DefaultOptionKeys.mloda_source_feature: frozenset(["point_a"]),  # Only 1 feature
                    }
                ),
            )
            api = mlodaAPI([feature], {PandasDataframe}, plugin_collector=plugin_collector)
            api._batch_run()

        # Test invalid number of source features (3 instead of 2)
        with pytest.raises(Exception):  # Should fail during validation
            feature = Feature(
                name="placeholder",
                options=Options(
                    context={
                        GeoDistanceFeatureGroup.DISTANCE_TYPE: "euclidean",
                        DefaultOptionKeys.mloda_source_feature: frozenset(
                            ["point_a", "point_b", "origin"]
                        ),  # 3 features
                    }
                ),
            )
            api = mlodaAPI([feature], {PandasDataframe}, plugin_collector=plugin_collector)
            api._batch_run()

    def test_multiple_distance_types_configuration_based(self) -> None:
        """Test multiple distance types using configuration-based approach."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {GeoDistanceModernizationTestDataCreator, PandasGeoDistanceFeatureGroup}
        )

        distance_types = ["haversine", "euclidean", "manhattan"]
        features: Features | List[str | Feature] = []

        for distance_type in distance_types:
            # Use appropriate point features for each distance type
            if distance_type == "haversine":
                source_features = frozenset(["sf_location", "nyc_location"])
            else:
                source_features = frozenset(["point_a", "point_b"])

            feature: Feature = Feature(
                name=f"distance_{distance_type}",
                options=Options(
                    context={
                        GeoDistanceFeatureGroup.DISTANCE_TYPE: distance_type,
                        DefaultOptionKeys.mloda_source_feature: source_features,
                    }
                ),
            )
            if isinstance(features, Features):
                raise TypeError(
                    "Expected features to be a list, but got Features instance. "
                    "Please convert to a list of Feature or string."
                )
            features.append(feature)

        # Run the API with multiple features
        api = mlodaAPI(features, {PandasDataframe}, plugin_collector=plugin_collector)
        api._batch_run()
        results = api.get_result()

        # Verify all features were created
        assert len(results) == 1
        for distance_type in distance_types:
            feature_name = f"distance_{distance_type}"
            assert feature_name in results[0].columns

    def test_context_parameters_dont_affect_feature_group_resolution(self) -> None:
        """Test that context parameters don't affect Feature Group resolution/splitting."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {GeoDistanceModernizationTestDataCreator, PandasGeoDistanceFeatureGroup}
        )

        # Create two features with different context parameters (distance types)
        feature1 = Feature(
            name="distance_haversine",
            options=Options(
                context={
                    GeoDistanceFeatureGroup.DISTANCE_TYPE: "haversine",
                    DefaultOptionKeys.mloda_source_feature: frozenset(["sf_location", "nyc_location"]),
                }
            ),
        )

        feature2 = Feature(
            name="distance_euclidean",
            options=Options(
                context={
                    GeoDistanceFeatureGroup.DISTANCE_TYPE: "euclidean",  # Different distance type (context parameter)
                    DefaultOptionKeys.mloda_source_feature: frozenset(
                        [
                            "point_a",
                            "point_b",
                        ]
                    ),  # Different source features (context parameter)
                }
            ),
        )

        # Run the API with both features
        api = mlodaAPI([feature1, feature2], {PandasDataframe}, plugin_collector=plugin_collector)
        api._batch_run()
        results = api.get_result()

        # Both features should be processed together (same Feature Group resolution)
        assert len(results) == 1
        assert "distance_haversine" in results[0].columns
        assert "distance_euclidean" in results[0].columns

    def test_match_feature_group_criteria_with_property_mapping(self) -> None:
        """Test that match_feature_group_criteria works with the new PROPERTY_MAPPING approach."""
        # Test string-based feature matching
        string_feature_name = "haversine_distance__sf_location__nyc_location"
        string_options = Options({})

        assert GeoDistanceFeatureGroup.match_feature_group_criteria(string_feature_name, string_options) is True

        # Test configuration-based feature matching
        config_options = Options(
            context={
                GeoDistanceFeatureGroup.DISTANCE_TYPE: "euclidean",
                DefaultOptionKeys.mloda_source_feature: frozenset(["point_a", "point_b"]),
            }
        )

        assert GeoDistanceFeatureGroup.match_feature_group_criteria("geo_distance_feature", config_options) is True

        # Test invalid configuration-based feature (should return False or raise exception during validation)
        invalid_config_options = Options(
            context={
                GeoDistanceFeatureGroup.DISTANCE_TYPE: "invalid_distance_type",
                DefaultOptionKeys.mloda_source_feature: frozenset(["point_a", "point_b"]),
            }
        )

        # This should return False due to validation failure
        try:
            result = GeoDistanceFeatureGroup.match_feature_group_criteria(
                "geo_distance_feature", invalid_config_options
            )
            assert result is False
        except ValueError:
            # Validation failure is also acceptable behavior
            pass

        # Test non-matching string-based feature
        assert (
            GeoDistanceFeatureGroup.match_feature_group_criteria("not_a_geo_distance_feature", string_options) is False
        )

    def test_dual_approach_in_single_run(self) -> None:
        """Test that both string-based and configuration-based features can be used together in a single run."""
        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups(
            {GeoDistanceModernizationTestDataCreator, PandasGeoDistanceFeatureGroup}
        )

        # Mix string-based and configuration-based features
        features: Features | List[str | Feature] = [
            # String-based features
            Feature("haversine_distance__sf_location__la_location"),
            Feature("euclidean_distance__point_a__point_b"),
            # Configuration-based features
            Feature(
                name="config_manhattan",
                options=Options(
                    context={
                        GeoDistanceFeatureGroup.DISTANCE_TYPE: "manhattan",
                        DefaultOptionKeys.mloda_source_feature: frozenset(["origin", "destination"]),
                    }
                ),
            ),
            Feature(
                name="config_haversine",
                options=Options(
                    context={
                        GeoDistanceFeatureGroup.DISTANCE_TYPE: "haversine",
                        DefaultOptionKeys.mloda_source_feature: frozenset(["nyc_location", "la_location"]),
                    }
                ),
            ),
        ]

        # Run the API with mixed features
        api = mlodaAPI(features, {PandasDataframe}, plugin_collector=plugin_collector)
        api._batch_run()
        results = api.get_result()

        # All features should be processed successfully
        assert len(results) == 1
        result_df = results[0]

        # Check that all features are present
        assert "haversine_distance__sf_location__la_location" in result_df.columns
        assert "euclidean_distance__point_a__point_b" in result_df.columns
        assert "config_manhattan" in result_df.columns
        assert "config_haversine" in result_df.columns

        # Verify data exists
        assert len(result_df) > 0

    def test_input_features_dual_approach(self) -> None:
        """Test that input_features method works correctly for both approaches."""
        feature_group = GeoDistanceFeatureGroup()

        # Test string-based approach
        from mloda_core.abstract_plugins.components.feature_name import FeatureName

        string_feature_name = FeatureName("haversine_distance__sf_location__nyc_location")
        string_options = Options({})

        input_features = feature_group.input_features(string_options, string_feature_name)
        assert input_features is not None
        assert len(input_features) == 2
        feature_names = {f.get_name() for f in input_features}
        assert "sf_location" in feature_names
        assert "nyc_location" in feature_names

        # Test configuration-based approach
        config_feature_name = FeatureName("some_feat")
        config_options = Options(
            context={
                DefaultOptionKeys.mloda_source_feature: frozenset(["point_a", "point_b"]),
            }
        )

        input_features = feature_group.input_features(config_options, config_feature_name)
        assert input_features is not None
        assert len(input_features) == 2
        feature_names = {f.get_name() for f in input_features}
        assert "point_a" in feature_names
        assert "point_b" in feature_names
