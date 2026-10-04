"""Value objects a FormatFeatureGroup uses to declare what it claims, plus the match-time scope variable."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


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


class DataAccessReader(Protocol):
    """What the matching and plan layers need of the owner in a (owner, access) pair."""

    def data_access_name(self) -> str: ...

    def data_access_identity(self, data_access: Any) -> str: ...


_FEATURE_GROUP_SCOPE: ContextVar[Any] = ContextVar("mloda_feature_group_scope", default=None)


def current_feature_group_scope() -> Any:
    """The feature_group= scope of the request being probed, or None."""
    return _FEATURE_GROUP_SCOPE.get()


@contextmanager
def feature_group_scope(scope: Any) -> Iterator[None]:
    """Make scope (or none) current for claim matching inside the block."""
    token = _FEATURE_GROUP_SCOPE.set(scope)
    try:
        yield
    finally:
        _FEATURE_GROUP_SCOPE.reset(token)
