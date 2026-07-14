"""geo_distance's strict in_features spec is enforced on the string-named path too (issue #732).

``in_features`` declares strict_validation=True with an ``element_validator`` (each point must be a
str), and the string-named path now validates the option values that are present. A non-str point
is a non-match with an actionable reason, never an exception.
"""

from __future__ import annotations

from mloda.provider import DefaultOptionKeys
from mloda.user import Options
from mloda_plugins.feature_group.experimental.geo_distance.base import GeoDistanceFeatureGroup


GEO_FEATURE = "customer_location&store_location__haversine_distance"


class TestInFeaturesValidatedOnNamePath:
    def test_non_str_point_is_a_non_match(self) -> None:
        """The element_validator rejects a non-str point on the string-named path."""
        options = Options(context={DefaultOptionKeys.in_features: [1, 2]})

        assert GeoDistanceFeatureGroup.match_feature_group_criteria(GEO_FEATURE, options) is False

    def test_rejection_reason_names_key_and_value(self) -> None:
        options = Options(context={DefaultOptionKeys.in_features: [1, 2]})

        reason = GeoDistanceFeatureGroup._strict_validation_rejection_reason(GEO_FEATURE, options)

        assert reason is not None
        assert str(DefaultOptionKeys.in_features) in reason
        assert "1" in reason

    def test_str_points_still_match(self) -> None:
        """Guard against over-rejecting: valid points on a name-matched feature still match."""
        options = Options(context={DefaultOptionKeys.in_features: ["customer_location", "store_location"]})

        assert GeoDistanceFeatureGroup.match_feature_group_criteria(GEO_FEATURE, options) is True

    def test_name_match_without_options_still_matches(self) -> None:
        """The name carries the points and the distance type: no options at all still matches."""
        assert GeoDistanceFeatureGroup.match_feature_group_criteria(GEO_FEATURE, Options()) is True


class TestInFeaturesValidatedOnConfigPath:
    def test_non_str_point_is_a_non_match(self) -> None:
        options = Options(
            context={
                GeoDistanceFeatureGroup.DISTANCE_TYPE: "haversine",
                DefaultOptionKeys.in_features: [1, 2],
            }
        )

        assert GeoDistanceFeatureGroup.match_feature_group_criteria("placeholder", options) is False

    def test_str_points_still_match(self) -> None:
        options = Options(
            context={
                GeoDistanceFeatureGroup.DISTANCE_TYPE: "haversine",
                DefaultOptionKeys.in_features: ["customer_location", "store_location"],
            }
        )

        assert GeoDistanceFeatureGroup.match_feature_group_criteria("placeholder", options) is True
