"""Value objects a SourceFeatureGroup uses to declare what it claims, plus the match-time scope variable."""

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
    """One way a source group claims features; searched routes also fire when mloda searches on its own."""

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
    from mloda.core.abstract_plugins.components.input_data.source_feature_group import SourceFeatureGroup

    DataAccessReader = type[SourceFeatureGroup]
else:
    DataAccessReader = type

_FEATURE_GROUP_SCOPE: ContextVar[Any] = ContextVar("mloda_feature_group_scope", default=None)


_CONTAIN_ABORTS: ContextVar[bool] = ContextVar("mloda_contain_match_aborts", default=False)


def aborts_are_contained() -> bool:
    """True inside the global filter probe, where a match abort must decline instead of raise."""
    return _CONTAIN_ABORTS.get()


_OPEN_CLAIM: ContextVar[list[bool] | None] = ContextVar("mloda_open_claim", default=None)


def note_open_claim() -> None:
    """Record that the candidate being matched claimed through an open route alone; a no-op outside open_claim_scope."""
    flag = _OPEN_CLAIM.get()
    if flag is not None:
        flag[:] = [True]


@contextmanager
def open_claim_scope() -> Iterator[list[bool]]:
    """Collect whether the candidate matched inside the block claimed through an open route alone."""
    flag: list[bool] = []
    token = _OPEN_CLAIM.set(flag)
    try:
        yield flag
    finally:
        _OPEN_CLAIM.reset(token)


_COMPUTED_ELSEWHERE: ContextVar[str | None] = ContextVar("mloda_computed_elsewhere", default=None)


def computed_elsewhere(name: str) -> bool:
    """True when a non-source candidate computes name, the feature being matched."""
    return _COMPUTED_ELSEWHERE.get() == name


@contextmanager
def computed_elsewhere_scope(name: str | None) -> Iterator[None]:
    """Set the name a non-source candidate computes inside the block."""
    token = _COMPUTED_ELSEWHERE.set(name)
    try:
        yield
    finally:
        _COMPUTED_ELSEWHERE.reset(token)


_PIN_SCOPE: ContextVar[tuple[set[str], list[tuple[str, Any]]] | None] = ContextVar("mloda_pin_scope", default=None)


def pin_is_unreported(path: str) -> bool:
    """True unless path was already reported within pin_report_scope."""
    scope = _PIN_SCOPE.get()
    return scope is None or path not in scope[0]


def note_pending_pin(path: str, rejection: Any) -> None:
    """Note the record made for a pinned path; it counts as reported only if it survives as the harvest."""
    scope = _PIN_SCOPE.get()
    if scope is not None:
        scope[1].append((path, rejection))


@contextmanager
def pin_report_scope(reported: set[str]) -> Iterator[list[tuple[str, Any]]]:
    """Share one reported-pins set across the candidates of a resolution; yields this candidate's pending records."""
    pending: list[tuple[str, Any]] = []
    token = _PIN_SCOPE.set((reported, pending))
    try:
        yield pending
    finally:
        _PIN_SCOPE.reset(token)


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
