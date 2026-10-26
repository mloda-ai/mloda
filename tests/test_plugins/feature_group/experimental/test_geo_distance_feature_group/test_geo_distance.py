"""
Tests for the GeoDistanceFeatureGroup.
"""

from typing import Any

import pandas as pd
import pytest

from tests.test_plugins.feature_group.experimental.core_input_check import assert_core_rejects_missing_input
from mloda.core.abstract_plugins.components.match_rejection import MatchRejection

from mloda.user import Feature
from mloda.user import FeatureName
from mloda.provider import DefaultOptionKeys, FeatureSet
from mloda.user import Options

from mloda_plugins.feature_group.experimental.geo_distance.base import GeoDistanceFeatureGroup
from mloda_plugins.feature_group.experimental.geo_distance.pandas import PandasGeoDistance


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

    def test_get_point_features_rejects_extra_ampersand_and_non_matching_name(self) -> None:
        name = "a&b&c__haversine_distance"
        with pytest.raises(ValueError) as exc_info:
            GeoDistanceFeatureGroup.get_point_features(name)
        assert str(exc_info.value) == GeoDistanceFeatureGroup.in_feature_count_reason(name, 3)

        with pytest.raises(ValueError):
            GeoDistanceFeatureGroup.get_point_features("a&b__custom")

    def test_match_feature_group_criteria(self) -> None:
        """Test matching of feature names to feature group criteria."""
        # Test valid feature names
        assert GeoDistanceFeatureGroup.match_feature_group_criteria("point1&point2__haversine_distance", Options())
        assert GeoDistanceFeatureGroup.match_feature_group_criteria("point1&point2__euclidean_distance", Options())
        assert GeoDistanceFeatureGroup.match_feature_group_criteria("point1&point2__manhattan_distance", Options())

        assert not GeoDistanceFeatureGroup.match_feature_group_criteria("point1&point2__haversine_invalid", Options())

    @pytest.mark.parametrize("in_features", [["p1", "p2"], ["b", "a"]])
    def test_declared_in_features_contradicting_the_name_aborts(self, in_features: list[str]) -> None:
        options = Options(context={DefaultOptionKeys.in_features: in_features})

        with pytest.raises(ValueError) as exc_info:
            GeoDistanceFeatureGroup.match_feature_group_criteria("a&b__haversine_distance", options)

        assert "in_features" in str(exc_info.value)

    def test_declared_in_features_agreeing_with_the_name_matches(self) -> None:
        options = Options(context={DefaultOptionKeys.in_features: ["a", "b"]})

        assert GeoDistanceFeatureGroup.match_feature_group_criteria("a&b__haversine_distance", options) is True

    @pytest.mark.parametrize("name", ["&b__haversine_distance", "a&__haversine_distance"])
    def test_empty_operand_name_does_not_match(self, name: str) -> None:
        assert not GeoDistanceFeatureGroup.match_feature_group_criteria(name, Options())

    def test_empty_operand_in_features_config_does_not_match(self) -> None:
        options = Options(
            context={
                GeoDistanceFeatureGroup.DISTANCE_TYPE: "haversine",
                DefaultOptionKeys.in_features: ["", "b"],
            }
        )
        assert not GeoDistanceFeatureGroup.match_feature_group_criteria("x", options)

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

    def test_input_features_name_path_wrong_count_uses_standard_message(self) -> None:
        name = "a&b&c__haversine_distance"

        with pytest.raises(ValueError) as exc_info:
            GeoDistanceFeatureGroup().input_features(Options(), FeatureName(name))

        assert str(exc_info.value) == GeoDistanceFeatureGroup.in_feature_count_reason(name, 3)

    @pytest.mark.parametrize("name", ["x", "a__b", "my&name__custom"], ids=["plain", "dunder", "amp_dunder"])
    @pytest.mark.parametrize("operands", [["a", "b"], ["b", "a"]], ids=["a_b", "b_a"])
    def test_extract_geo_distance_parameters_keeps_declared_order(self, operands: list[str], name: str) -> None:
        options = Options(
            context={
                GeoDistanceFeatureGroup.DISTANCE_TYPE: "euclidean",
                DefaultOptionKeys.in_features: operands,
            }
        )

        result = GeoDistanceFeatureGroup._extract_geo_distance_parameters(Feature(name, options=options))

        assert result == ("euclidean", operands[0], operands[1])

    @pytest.mark.parametrize(
        "feature,count",
        [
            (Feature("a&b&c__haversine_distance"), 3),
            (
                Feature(
                    "x",
                    options=Options(
                        context={
                            GeoDistanceFeatureGroup.DISTANCE_TYPE: "euclidean",
                            DefaultOptionKeys.in_features: ["a"],
                        }
                    ),
                ),
                1,
            ),
            (
                Feature(
                    "x",
                    options=Options(
                        context={
                            GeoDistanceFeatureGroup.DISTANCE_TYPE: "euclidean",
                            DefaultOptionKeys.in_features: ["a", "b", "c"],
                        }
                    ),
                ),
                3,
            ),
        ],
        ids=["name_3", "config_1", "config_3"],
    )
    def test_extract_geo_distance_parameters_wrong_count_uses_standard_message(
        self, feature: Feature, count: int
    ) -> None:
        with pytest.raises(ValueError) as exc_info:
            GeoDistanceFeatureGroup._extract_geo_distance_parameters(feature)

        assert str(exc_info.value) == GeoDistanceFeatureGroup.in_feature_count_reason(feature.name, count)

    def test_extract_geo_distance_parameters_rejects_unsupported_name_path_distance(self) -> None:
        with pytest.raises(ValueError, match="Unsupported distance type"):
            GeoDistanceFeatureGroup._extract_geo_distance_parameters(Feature("a&b__foo_distance"))

    def test_doubled_distance_suffix_is_a_recorded_non_match(self, rejection_window: dict[str, MatchRejection]) -> None:
        feature = Feature(
            "a&b__haversine_distance_distance",
            options=Options(context={GeoDistanceFeatureGroup.DISTANCE_TYPE: "haversine"}),
        )
        assert GeoDistanceFeatureGroup.match_feature_group_criteria(feature.name, feature.options) is False
        reasons = [r.reason for r in rejection_window.values()]
        assert len(reasons) == 1
        assert "haversine_distance" in reasons[0]
        with pytest.raises(ValueError, match="Unsupported distance type"):
            GeoDistanceFeatureGroup._extract_geo_distance_parameters(feature)

    def test_name_bound_distance_type_is_validated_despite_a_valid_option(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        options = Options(
            context={
                GeoDistanceFeatureGroup.DISTANCE_TYPE: "euclidean",
                DefaultOptionKeys.in_features: ["p1", "p2"],
            }
        )
        assert GeoDistanceFeatureGroup.match_feature_group_criteria("my&name__custom_distance", options) is False
        assert len(rejection_window) == 1
        assert "custom" in next(iter(rejection_window.values())).reason

    def test_valid_name_still_resolves_the_distance_type(self) -> None:
        assert GeoDistanceFeatureGroup.get_distance_type("p1&p2__haversine_distance") == "haversine"


class TestPandasGeoDistance:
    """Test cases for PandasGeoDistance."""

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

    @pytest.mark.parametrize(
        ("name", "missing", "available"),
        [
            ("sf&invalid__haversine_distance", ("invalid",), ("la", "nyc", "point1", "point2", "sf")),
            ("zz&yy__haversine_distance", ("yy", "zz"), ("la", "nyc", "point1", "point2", "sf")),
        ],
        ids=["one_missing", "both_missing"],
    )
    def test_core_rejects_missing_input_and_lists_sorted_columns(
        self, name: str, missing: tuple[str, ...], available: tuple[str, ...]
    ) -> None:
        assert_core_rejects_missing_input(PandasGeoDistance, self.df.copy(), name, missing, available)

    def test_haversine_distance(self) -> None:
        """Test calculation of haversine distance."""
        # Calculate haversine distance between San Francisco and New York
        distance = PandasGeoDistance._calculate_haversine_distance(self.df, "sf", "nyc")

        # Expected distance is approximately 4130 km
        assert distance[0] == pytest.approx(4130, abs=100)

        # Calculate haversine distance between San Francisco and Los Angeles
        distance = PandasGeoDistance._calculate_haversine_distance(self.df, "sf", "la")

        # Expected distance is approximately 560 km
        assert distance[0] == pytest.approx(560, abs=50)

    def test_euclidean_distance(self) -> None:
        """Test calculation of euclidean distance."""
        # Calculate euclidean distance between point1 and point2
        distance = PandasGeoDistance._calculate_euclidean_distance(self.df, "point1", "point2")

        # Expected distances: sqrt(3^2 + 4^2) = 5, sqrt(4^2 + 4^2) = 5.66, sqrt(4^2 + 6^2) = 7.21
        assert distance[0] == pytest.approx(5.0, abs=0.01)
        assert distance[1] == pytest.approx(5.66, abs=0.01)
        assert distance[2] == pytest.approx(7.21, abs=0.01)

    def test_manhattan_distance(self) -> None:
        """Test calculation of manhattan distance."""
        # Calculate manhattan distance between point1 and point2
        distance = PandasGeoDistance._calculate_manhattan_distance(self.df, "point1", "point2")

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
        result_df = PandasGeoDistance.calculate(GeoDistanceFeatureGroup, self.df.copy(), feature_set, None)

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
        result_df = PandasGeoDistance.calculate(GeoDistanceFeatureGroup, self.df.copy(), feature_set, None)

        # Check if all features were added to the DataFrame
        assert "sf&nyc__haversine_distance" in result_df.columns
        assert "point1&point2__euclidean_distance" in result_df.columns
        assert "point1&point2__manhattan_distance" in result_df.columns

        # Check if the distances are approximately correct
        assert result_df["sf&nyc__haversine_distance"][0] == pytest.approx(4130, abs=100)
        assert result_df["point1&point2__euclidean_distance"][0] == pytest.approx(5.0, abs=0.01)
        assert result_df["point1&point2__manhattan_distance"][0] == 7

    def test_missing_point_returns_nan(self) -> None:
        """A missing point produces NaN for that row; other rows are unaffected."""
        df = pd.DataFrame(
            {
                "a": [(52.5, 13.4), None, (48.1, 11.6), float("nan"), pd.NA],
                "b": [(48.1, 11.6), (48.1, 11.6), None, (48.1, 11.6), (48.1, 11.6)],
            }
        )

        haversine = PandasGeoDistance._calculate_haversine_distance(df, "a", "b")
        assert not pd.isna(haversine.iloc[0])
        assert pd.isna(haversine.iloc[1])
        assert pd.isna(haversine.iloc[2])
        assert pd.isna(haversine.iloc[3])
        assert pd.isna(haversine.iloc[4])

        euclidean = PandasGeoDistance._calculate_euclidean_distance(df, "a", "b")
        assert not pd.isna(euclidean.iloc[0])
        assert pd.isna(euclidean.iloc[1])
        assert pd.isna(euclidean.iloc[2])
        assert pd.isna(euclidean.iloc[3])
        assert pd.isna(euclidean.iloc[4])

        manhattan = PandasGeoDistance._calculate_manhattan_distance(df, "a", "b")
        assert not pd.isna(manhattan.iloc[0])
        assert pd.isna(manhattan.iloc[1])
        assert pd.isna(manhattan.iloc[2])
        assert pd.isna(manhattan.iloc[3])
        assert pd.isna(manhattan.iloc[4])

    @pytest.mark.parametrize("distance_type", ["haversine", "euclidean", "manhattan"])
    @pytest.mark.parametrize("malformed", [(52.5,), 5, "12"])
    def test_malformed_point_raises(self, distance_type: str, malformed: Any) -> None:
        """A malformed point raises instead of silently becoming NaN."""
        df = pd.DataFrame({"a": [(52.5, 13.4), malformed], "b": [(48.1, 11.6), (48.1, 11.6)]})

        with pytest.raises((IndexError, TypeError, ValueError)):
            PandasGeoDistance._calculate_distance(df, distance_type, "a", "b")
