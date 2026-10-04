"""ClaimRoute, NamePolicy and SourceMatch value objects, and their public exports."""

import dataclasses

import pytest

import mloda.provider as provider
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy, SourceMatch
from mloda.core.abstract_plugins.components.input_data.format_feature_group import FormatFeatureGroup


def test_name_policy_members() -> None:
    assert {p.name for p in NamePolicy} == {"CHECKED", "DECLARED", "OPEN"}


def test_claim_route_defaults_and_frozen() -> None:
    route = ClaimRoute("csv", NamePolicy.CHECKED, True)
    assert route.required_options == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        route.searched = False  # type: ignore[misc]


def test_claim_route_is_hashable_and_compares_by_value() -> None:
    assert ClaimRoute("a", NamePolicy.OPEN, False, ("x",)) == ClaimRoute("a", NamePolicy.OPEN, False, ("x",))
    assert ClaimRoute("a", NamePolicy.OPEN, False) != ClaimRoute("a", NamePolicy.OPEN, False, ("x",))
    assert len({ClaimRoute("a", NamePolicy.OPEN, False), ClaimRoute("a", NamePolicy.OPEN, False)}) == 1


def test_source_match_equality_and_hash_use_source_only() -> None:
    one = SourceMatch(source="h:t", access={"detail": 1})
    two = SourceMatch(source="h:t", access={"detail": 2})
    assert one == two
    assert hash(one) == hash(two)
    assert one != SourceMatch(source="h:other", access={"detail": 1})
    assert len({one, two}) == 1


def test_source_match_repr_excludes_access() -> None:
    match = SourceMatch(source="h:t", access={"detail": "abc-value-xyz"})
    assert "abc-value-xyz" not in repr(match)
    assert "h:t" in repr(match)


def test_source_match_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        SourceMatch(source="a", access=None).source = "b"  # type: ignore[misc]


def test_provider_exports_are_the_core_objects() -> None:
    assert provider.ClaimRoute is ClaimRoute
    assert provider.NamePolicy is NamePolicy
    assert provider.SourceMatch is SourceMatch
    assert provider.FormatFeatureGroup is FormatFeatureGroup
    for name in ("ClaimRoute", "NamePolicy", "SourceMatch", "FormatFeatureGroup"):
        assert name in provider.__all__
