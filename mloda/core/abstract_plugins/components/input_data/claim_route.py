"""Value objects a FormatFeatureGroup uses to declare what it claims, plus the match-time scope variable."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any


class NamePolicy(Enum):
    """How a route decides that a feature name belongs to its source."""

    CHECKED = "checked"
    DECLARED = "declared"
    OPEN = "open"


@dataclass(frozen=True)
class ClaimRoute:
    """One way a format group claims features; searched routes also fire when mloda searches on its own."""

    source_kind: str
    names: NamePolicy
    searched: bool
    required_options: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_kind, str) or not self.source_kind:
            raise TypeError(f"ClaimRoute.source_kind must be a non-empty str, got {self.source_kind!r}")
        if not isinstance(self.names, NamePolicy):
            raise TypeError(f"ClaimRoute.names must be a NamePolicy, got {self.names!r}")
        if not isinstance(self.searched, bool):
            raise TypeError(f"ClaimRoute.searched must be a bool, got {self.searched!r}")
        if not isinstance(self.required_options, tuple) or not all(isinstance(o, str) for o in self.required_options):
            raise TypeError(f"ClaimRoute.required_options must be a tuple of str, got {self.required_options!r}")


@dataclass(frozen=True)
class SourceMatch:
    """A matched source: equality and hash use the credential-free source only, never the access."""

    source: str
    access: Any = field(compare=False, hash=False, repr=False, default=None)


if TYPE_CHECKING:
    from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
    from mloda.core.abstract_plugins.components.input_data.format_feature_group import FormatFeatureGroup

    DataAccessReader = type[BaseInputData] | type[FormatFeatureGroup]
else:
    DataAccessReader = type

_FEATURE_GROUP_SCOPE: ContextVar[Any] = ContextVar("mloda_feature_group_scope", default=None)


_CONTAIN_ABORTS: ContextVar[bool] = ContextVar("mloda_contain_match_aborts", default=False)


def aborts_are_contained() -> bool:
    """True inside the global filter probe, where a match abort must decline instead of raise."""
    return _CONTAIN_ABORTS.get()


def current_feature_group_scope() -> Any:
    """The feature_group= scope of the request being probed, or None."""
    return _FEATURE_GROUP_SCOPE.get()


@contextmanager
def feature_group_scope(scope: Any, contain_aborts: bool = False) -> Iterator[None]:
    """Make scope (or none) current for claim matching inside the block; optionally contain match aborts."""
    token = _FEATURE_GROUP_SCOPE.set(scope)
    contain_token = _CONTAIN_ABORTS.set(contain_aborts)
    try:
        yield
    finally:
        _CONTAIN_ABORTS.reset(contain_token)
        _FEATURE_GROUP_SCOPE.reset(token)
