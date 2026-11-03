"""source_applicability reports which source suite rules apply from a group's claim routes."""

from typing import Any

from mloda._testing.source_applicability import NO_POINTER, SourceApplicability, source_applicability
from mloda.core.abstract_plugins.components.data_access_collection import CONNECTION_HANDLE_OPTION
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy
from mloda.provider import SourceFeatureGroup


def _make_group(*routes: ClaimRoute) -> type[SourceFeatureGroup]:
    def find_sources(cls: Any, route: Any, feature_name: Any, options: Any, dac: Any = None) -> list[Any]:
        return []

    def load_neutral(cls: Any, match: Any, features: Any) -> Any:
        return None

    return type(
        "ApplicabilityProbeFG",
        (SourceFeatureGroup,),
        {
            "CLAIM_ROUTES": tuple(routes),
            "find_sources": classmethod(find_sources),
            "load_neutral": classmethod(load_neutral),
        },
    )


def _searched() -> ClaimRoute:
    return ClaimRoute("probe", NamePolicy.DECLARED, True)


def _pointed_only() -> ClaimRoute:
    return ClaimRoute("probe", NamePolicy.OPEN, False)


def test_no_pointer_sentinel_is_a_distinct_object() -> None:
    assert NO_POINTER is not None
    assert NO_POINTER is NO_POINTER


def test_a_plain_searched_group_with_a_pointer() -> None:
    result = source_applicability(_make_group(_searched()), True)
    assert result == SourceApplicability(pointer=True, data_access_handle=True, query_route=False, checked_route=False)


def test_has_pointer_false_clears_the_pointer_flag() -> None:
    assert source_applicability(_make_group(_searched()), False).pointer is False


def test_a_connection_handle_route_has_no_data_access_handle() -> None:
    route = ClaimRoute("probe", NamePolicy.DECLARED, True, (CONNECTION_HANDLE_OPTION,))
    assert source_applicability(_make_group(route), False).data_access_handle is False


def test_one_connection_handle_route_among_others_still_disables_the_data_access_handle() -> None:
    route = ClaimRoute("probe", NamePolicy.DECLARED, True, (CONNECTION_HANDLE_OPTION,))
    assert source_applicability(_make_group(_searched(), route), True).data_access_handle is False


def test_a_query_text_route_sets_query_route() -> None:
    route = ClaimRoute("probe", NamePolicy.OPEN, False, ("query_text",))
    assert source_applicability(_make_group(_pointed_only(), route), True).query_route is True


def test_a_group_without_a_query_route() -> None:
    assert source_applicability(_make_group(_pointed_only()), True).query_route is False


def test_a_checked_route_sets_checked_route() -> None:
    route = ClaimRoute("probe", NamePolicy.CHECKED, True)
    assert source_applicability(_make_group(route), True).checked_route is True


def test_declared_and_open_routes_are_not_checked() -> None:
    result = source_applicability(_make_group(_searched(), _pointed_only()), True)
    assert result.checked_route is False
