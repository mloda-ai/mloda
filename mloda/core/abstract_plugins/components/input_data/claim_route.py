"""Value objects a FormatFeatureGroup uses to declare what it claims, plus the match-time scope variable."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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


@dataclass(frozen=True)
class SourceMatch:
    """A matched source: equality and hash use the credential-free source only, never the access."""

    source: str
    access: Any = field(compare=False, hash=False, repr=False, default=None)


DataAccessReader = type[Any]
"""The reader class (a BaseInputData or FormatFeatureGroup subclass) in an (owner, access) pair."""

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
