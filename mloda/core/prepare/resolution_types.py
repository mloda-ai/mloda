"""Captured facts of one resolution pass: it imports neither the renderer nor the matcher, so both can depend on it."""

from dataclasses import dataclass, field
from typing import Literal

from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.abstract_plugins.components.match_rejection import INPUT_DATA_OWNED_STAGE, INPUT_DATA_STAGE
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup


@dataclass(frozen=True)
class CandidateFrameworks:
    """One candidate's own accessible frameworks, split by the match-time capability hook."""

    supported: frozenset[type[ComputeFramework]] = frozenset()
    rejected: frozenset[type[ComputeFramework]] = frozenset()


EliminationStage = Literal[
    "value_rejection",
    "input_data",
    "matcher_error",
    "domain",
    "scope",
    "capability",
    "frameworks_not_enabled",
    "framework_pin",
    "links",
]

# A stage belongs here when its gate's hook cannot receive the feature name, so the outcome is the same for
# every name the candidate declares.
NAME_INDEPENDENT_STAGES: frozenset[EliminationStage] = frozenset({"domain", "scope", "frameworks_not_enabled", "links"})


def rejection_elimination_stage(recorded_stage: str) -> EliminationStage:
    """The elimination stage a recorded rejection's free-form stage hint maps onto."""
    # Only the two input-data hints are engine-known; every other hint is provider text this side never
    # validates, so it falls back. Shared, so the two seams cannot drift into two taxonomies.
    if recorded_stage in (INPUT_DATA_STAGE, INPUT_DATA_OWNED_STAGE):
        return "input_data"
    return "value_rejection"


@dataclass(frozen=True)
class Elimination:
    """Why one near-miss candidate lost: the first gate it failed and that gate's reason."""

    stage: EliminationStage
    reason: str


@dataclass(frozen=True)
class RenderFacts:
    """Facts captured during the decision pass so rendering needs no provider hook.

    The empty instance is the success value: the winning path captures nothing.
    """

    domains: dict[type[FeatureGroup], str] = field(default_factory=dict)
    concrete_frameworks: tuple[str, ...] = ()
    known_names: tuple[str, ...] = ()
    # Class name and prefix of each eliminated near-miss, so a "Did you mean" suggestion that merely echoes
    # a candidate the near-miss block already named can be suppressed.
    eliminated_hints: frozenset[str] = frozenset()
    # Names that no live accessible group declares and that no live group's class-name prefix covers, so no
    # surviving candidate is known to own them.
    dead_only_names: frozenset[str] = frozenset()


@dataclass(frozen=True)
class EvaluationResult:
    """Non-raising result of matching a feature against accessible plugins."""

    identified: FeatureGroupEnvironmentMapping
    criteria_matched: set[type[FeatureGroup]] = field(default_factory=set)
    abstract_matched: set[type[FeatureGroup]] = field(default_factory=set)
    candidate_frameworks: dict[type[FeatureGroup], CandidateFrameworks] = field(default_factory=dict)
    eliminations: dict[type[FeatureGroup], Elimination] = field(default_factory=dict)
    facts: RenderFacts = field(default_factory=RenderFacts)

    @property
    def failure_kind(self) -> Literal["multiple", "abstract_only", "none"] | None:
        # "none" means no winner in the identified mapping, not that nothing matched: an all-rejected
        # concrete group still yields "none" with a non-empty criteria_matched.
        n = len(self.identified)
        if n == 1:
            return None
        if n > 1:
            return "multiple"
        if self.abstract_matched:
            return "abstract_only"
        return "none"


@dataclass(frozen=True)
class ResolutionRecord:
    """One feature's identification during planning: its name, whether it was requested, and its EvaluationResult."""

    feature_name: str
    requested: bool
    result: EvaluationResult


# Cap on records attached to a FeatureResolutionError so the copy stays bounded on huge requests.
PARTIAL_RECORDS_CAP = 1000


@dataclass(frozen=True)
class ResolutionDiagnosis:
    """Non-raising outcome of mlodaAPI.diagnose, a whole-request resolution preflight.

    complete is True iff the whole request resolved; then records equals session.resolution_report() and
    feature_name/failed_result/message are None. On a resolution failure records holds the features resolved
    before the failing one (capped at PARTIAL_RECORDS_CAP on huge requests), feature_name/failed_result carry
    that feature's failed evaluation, and message is its rendered text. On a configuration error records is
    empty and only message is set.
    """

    records: list[ResolutionRecord]
    complete: bool
    feature_name: str | None = None
    failed_result: EvaluationResult | None = None
    message: str | None = None
