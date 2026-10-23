"""ClaimRoute, NamePolicy and SourceMatch value objects, and their public exports."""

import importlib
import importlib.util
from typing import Any

import pytest

import mloda.provider as provider
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy, SourceMatch


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


def test_provider_exports_are_the_core_objects() -> None:
    assert provider.ClaimRoute is ClaimRoute
    assert provider.NamePolicy is NamePolicy
    assert provider.SourceMatch is SourceMatch
    assert provider.QUERY_OPTION == "query_text"
    assert provider.HANDLE_OPTION == "data_access_handle"
    for name in ("ClaimRoute", "NamePolicy", "SourceMatch", "SourceFeatureGroup", "QUERY_OPTION", "HANDLE_OPTION"):
        assert name in provider.__all__


def test_source_feature_group_replaces_format_feature_group() -> None:
    base = "mloda.core.abstract_plugins.components.input_data."
    assert importlib.util.find_spec(base + "source_feature_group") is not None
    assert importlib.util.find_spec(base + "format_feature_group") is None
    module = importlib.import_module(base + "source_feature_group")
    assert provider.SourceFeatureGroup is module.SourceFeatureGroup
    assert not hasattr(provider, "FormatFeatureGroup")
    assert "FormatFeatureGroup" not in provider.__all__


@pytest.mark.parametrize(
    "kwargs,field_name",
    [
        ({"source_kind": ""}, "source_kind"),
        ({"source_kind": 5}, "source_kind"),
        ({"names": "open"}, "names"),
        ({"searched": "yes"}, "searched"),
        ({"searched": 1}, "searched"),
        ({"required_options": ["q"]}, "required_options"),
        ({"required_options": (1,)}, "required_options"),
    ],
)
def test_claim_route_rejects_a_badly_typed_field(kwargs: dict[str, Any], field_name: str) -> None:
    fields: dict[str, Any] = {"source_kind": "x", "names": NamePolicy.OPEN, "searched": False}
    fields.update(kwargs)
    with pytest.raises(TypeError, match=field_name):
        ClaimRoute(**fields)


def test_provider_exports_read_file_fg() -> None:
    from mloda.core.abstract_plugins.components.input_data.read_file_fg import ReadFileFG

    assert provider.ReadFileFG is ReadFileFG
    assert "ReadFileFG" in provider.__all__


def test_provider_exports_implementation() -> None:
    from mloda.core.abstract_plugins.implementation import Implementation

    assert provider.Implementation is Implementation
    assert "Implementation" in provider.__all__
