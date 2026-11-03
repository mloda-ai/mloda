"""Which source suite rules apply to a group, read from its claim routes; importable without pytest or pyarrow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from mloda.core.abstract_plugins.components.data_access_collection import CONNECTION_HANDLE_OPTION
from mloda.core.abstract_plugins.components.input_data.claim_route import NamePolicy
from mloda.core.abstract_plugins.components.input_data.source_feature_group import SourceFeatureGroup

# Sentinel returned by a pointer hook when the group takes no pointer value.
NO_POINTER: Final[Any] = object()


@dataclass(frozen=True)
class SourceApplicability:
    """The route facts that gate the source suite rules."""

    pointer: bool
    data_access_handle: bool
    query_route: bool
    checked_route: bool


def source_applicability(group: type[SourceFeatureGroup], has_pointer: bool) -> SourceApplicability:
    """Applicability of the suite rules for a group; has_pointer says whether it takes a pointer value."""
    routes = group.CLAIM_ROUTES
    return SourceApplicability(
        pointer=has_pointer,
        data_access_handle=any(route.searched for route in routes)
        and not any(CONNECTION_HANDLE_OPTION in route.required_options for route in routes),
        query_route=group.serves_query_text(),
        checked_route=any(route.names is NamePolicy.CHECKED for route in routes),
    )
