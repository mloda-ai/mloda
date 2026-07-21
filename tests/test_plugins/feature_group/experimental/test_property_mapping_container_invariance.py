"""Plugin-level fallout of uniform sequence unpacking (issue #600).

Core used to hand a tuple option to the ``element_validator`` as the STRING
``"('a', 'b')"`` while handing a list option its real elements. Plugins that accept a
sequence option had to defend against both shapes, so the container the user typed
leaked into what the feature group matched.

Once elements arrive properly, an ``element_validator`` is a per-ELEMENT predicate and
nothing else. These tests pin that contract at the plugin surface: the same operations
match whether the caller writes a list, a tuple or a frozenset.
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.provider import DefaultOptionKeys
from mloda.user import Options
from mloda_plugins.feature_group.experimental.geo_distance.base import GeoDistanceFeatureGroup
from mloda_plugins.feature_group.experimental.text_cleaning.base import TextCleaningFeatureGroup

VALID_OPERATIONS: list[tuple[str, Any]] = [
    ("list", ["normalize", "remove_stopwords"]),
    ("tuple", ("normalize", "remove_stopwords")),
    ("frozenset", frozenset({"normalize", "remove_stopwords"})),
]

INVALID_OPERATIONS: list[tuple[str, Any]] = [
    ("list", ["normalize", "bogus_operation"]),
    ("tuple", ("normalize", "bogus_operation")),
    ("frozenset", frozenset({"normalize", "bogus_operation"})),
]


def _text_cleaning_options(operations: Any) -> Options:
    return Options(
        context={
            TextCleaningFeatureGroup.CLEANING_OPERATIONS: operations,
            DefaultOptionKeys.in_features: "review",
        }
    )


class TestTextCleaningOperationsAreValidatedPerElement:
    """Each cleaning operation is validated individually against ``SUPPORTED_OPERATIONS``: the same
    operations match whether the caller writes a list, a tuple or a frozenset (issue #600)."""

    @pytest.mark.parametrize(("label", "operations"), VALID_OPERATIONS, ids=[label for label, _ in VALID_OPERATIONS])
    def test_supported_operations_match_in_any_container(self, label: str, operations: Any) -> None:
        """The same supported operations match whether written as a list, tuple or frozenset."""
        assert TextCleaningFeatureGroup.match_feature_group_criteria("placeholder", _text_cleaning_options(operations))

    @pytest.mark.parametrize(
        ("label", "operations"), INVALID_OPERATIONS, ids=[label for label, _ in INVALID_OPERATIONS]
    )
    def test_unsupported_operation_rejects_in_any_container(self, label: str, operations: Any) -> None:
        """One unsupported operation rejects the feature group, whatever the container."""
        assert not TextCleaningFeatureGroup.match_feature_group_criteria(
            "placeholder", _text_cleaning_options(operations)
        )

    def test_every_supported_operation_is_accepted_on_its_own(self) -> None:
        """Each declared operation passes the per-element check in a single-element sequence."""
        for operation in TextCleaningFeatureGroup.SUPPORTED_OPERATIONS:
            assert TextCleaningFeatureGroup.match_feature_group_criteria(
                "placeholder", _text_cleaning_options([operation])
            ), f"{operation} must be accepted"


class TestGeoDistanceInFeaturesContainerInvariance:
    """``in_features`` matches identically whichever sequence container carries the points.

    The plugin's ``element_validator`` carries a dead branch that accepts "collections
    with exactly 2 elements (when validating the whole list)", a leftover from the days
    when a container could reach it whole. With uniform unpacking it only ever sees
    individual point names, and a tuple must behave exactly like a list.
    """

    @pytest.mark.parametrize(
        ("label", "in_features"),
        [
            ("list", ["point1", "point2"]),
            ("tuple", ("point1", "point2")),
            ("frozenset", frozenset({"point1", "point2"})),
            ("set", {"point1", "point2"}),
        ],
        ids=["list", "tuple", "frozenset", "set"],
    )
    def test_two_points_match_in_any_container(self, label: str, in_features: Any) -> None:
        """Two point features match whether written as a list, tuple, set or frozenset."""
        options = Options(
            context={
                GeoDistanceFeatureGroup.DISTANCE_TYPE: "haversine",
                DefaultOptionKeys.in_features: in_features,
            }
        )

        assert GeoDistanceFeatureGroup.match_feature_group_criteria("placeholder", options) is True
