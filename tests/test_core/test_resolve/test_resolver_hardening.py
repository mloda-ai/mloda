"""Failing tests for issue #722 Stage 3 hardening (post-review fix round).

Pins review-mandated hardening of the resolver stack (``mloda.core.resolve.resolver``,
``mloda.core.resolve.outcome``, ``mloda.core.api.plugin_docs.resolve_feature``, and the
engine adapter / Engine call path):

1.  ResolutionOutcome payloads are redacted: the raw provider message stays on the
    internal ``PluginFailure.message`` field (it feeds the rendered errors) but never
    leaks into ``to_payload()``.
2.  A raising ``get_domain`` hook is guarded fail-closed into a FAILED outcome.
3.  A raising ``index_columns`` hook is guarded fail-closed into a FAILED outcome.
    Investigated: ``_supports_request_links`` consults ``index_columns()`` BEFORE the
    empty-links short-circuit, so the raise propagates even for a link-less request;
    both the no-links and with-links requests pin the guard.
4.  A raising ``supports_index`` hook is guarded fail-closed into a FAILED outcome.
5.  links=() and links=None are equivalent (TARGET CONTRACT, passes today).
6.  A LINK_INDEX-rejected candidate records NOT_EVALUATED framework evaluations,
    never SUPPORTED (new FrameworkStatus member).
7.  resolve_feature samples framework availability exactly once per call.
8.  resolve_feature never raises when an ``is_available`` probe raises.
9.  A mapping-derived snapshot never consults the strict-mode env var. Investigated:
    ``strict_mode_from_env`` RAISES ValueError on an invalid value, so this pins that
    ``IdentifyFeatureGroupClass`` over a hand-built mapping still works under an
    invalid env value.
10. One Engine builds its mapping-derived environment snapshot at most once, not
    once per requested feature.

RED phase: tests 1, 2, 3, 4, 6, 7, 8, 9, and 10 fail against the current
implementation; test 5 passes and pins the target contract.

Probe names use the unique probe722h_ prefix so process-global scans in other tests
never trip these fixtures. Hooks that cannot be name-gated (get_domain, index_columns,
supports_index, is_available) raise only while ARMED by their test and are disarmed by
fixture teardown: in the RED phase their raise propagates through pytest, which parks
the exception (and the frames referencing the probe class) in sys.last_exc, so a
gc-scoped class would stay alive and poison later tests on the same xdist worker.
"""

from __future__ import annotations

import gc
import json
from typing import Callable, Iterator
from unittest.mock import patch

import pytest

import mloda.core.prepare.identify_feature_group as identify_module
import mloda.core.resolve.resolver as resolver_module
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_collection import Features
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import (
    PluginCollector,
    strict_mode_from_env,
)
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.api.plugin_info import ResolvedFeature
from mloda.core.core.engine import Engine
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.core.resolve.environment import ResolutionEnvironmentSnapshot
from mloda.core.resolve.identity import PluginIdentity
from mloda.core.resolve.outcome import (
    CandidateEvaluation,
    CandidateStatus,
    FrameworkEvaluation,
    FrameworkStatus,
    PluginFailure,
    RejectionReason,
    ResolutionOutcome,
    ResolutionStatus,
)
from mloda.core.resolve.request import ResolutionRequestSnapshot
from mloda.core.resolve.resolver import FeatureGroupResolver, snapshot_from_mapping


CLEAN_FEATURE = "probe722h_clean"
REDACTION_FEATURE = "probe722h_secret"
DOMAIN_FEATURE = "probe722h_domain"
INDEX_BOOM_FEATURE = "probe722h_index_boom"
SUPPORTS_BOOM_FEATURE = "probe722h_supports_boom"
LINKS_EQUAL_FEATURE = "probe722h_links_equal"
NOT_EVALUATED_FEATURE = "probe722h_not_evaluated"
COUNTING_FEATURE = "probe722h_counting"
ENGINE_FEATURE_ONE = "probe722h_engine_one"
ENGINE_FEATURE_TWO = "probe722h_engine_two"

DOMAIN_722H = "probe722h_dom_ops"

# Fake credential-shaped provider message used to prove payload redaction; never a real secret.
_FAKE_CREDENTIAL_MESSAGE_722H = "credential=hush-722h"  # nosec B105
DOMAIN_BOOM_TEXT = "domain boom 722h"
INDEX_BOOM_TEXT = "index boom 722h"
SUPPORTS_BOOM_TEXT = "supports boom 722h"
AVAIL_BOOM_TEXT = "avail boom 722h"
RAISING_AVAIL_CLASS_NAME = "CfwRaisingAvail722H"


# ---------------------------------------------------------------------------
# Mock compute frameworks (unique names, suffix 722H)
# ---------------------------------------------------------------------------


class CfwAlpha722H(ComputeFramework):
    """General-purpose framework for the hand-built resolver mappings."""


class CfwEngine722H(ComputeFramework):
    """Framework carried by the Engine memoization probe."""


# Counts every is_available() call on the counting framework; cleared per test (xdist-safe:
# workers are separate processes, and within one worker the owning test resets it first).
_RESOLVE_AVAILABILITY_CALLS_722H: list[str] = []


class CfwCountingResolve722H(ComputeFramework):
    """Available framework whose availability probe counts its own invocations."""

    @staticmethod
    def is_available() -> bool:
        _RESOLVE_AVAILABILITY_CALLS_722H.append("call")
        return True


# Hook keys armed per test via the arm_hook_722h fixture; disarmed on teardown. While
# disarmed, every hook below degrades to an inert default, so the module-level classes
# stay harmless in every other test of the process (xdist-safe: per-process state).
_ARMED_HOOKS_722H: set[str] = set()


class CfwRaisingAvail722H(ComputeFramework):
    """Framework whose is_available() raises while armed; unavailable otherwise.

    EVERY availability sampling in the process hits this hook, so the raise is armed
    only for the one test pinning it and the class is otherwise a plain unavailable
    framework.
    """

    @staticmethod
    def is_available() -> bool:
        if "is_available" in _ARMED_HOOKS_722H:
            raise RuntimeError(AVAIL_BOOM_TEXT)
        return False


# ---------------------------------------------------------------------------
# Probe feature groups (match ONLY their own probe722h_* names)
# ---------------------------------------------------------------------------


class _HardeningProbeBase722H(FeatureGroup):
    """Shared probe base: never matches anything itself."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class CleanProbe722H(_HardeningProbeBase722H):
    """Single clean candidate: matches its own name on one framework."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722H}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == CLEAN_FEATURE


class SecretCriteriaBoom722H(_HardeningProbeBase722H):
    """Raises a ValueError carrying a credential-shaped message, gated on its name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722H}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        # Gated on the probe name so process-global scans in other tests never trip the raise.
        if str(feature_name) == REDACTION_FEATURE:
            raise ValueError(_FAKE_CREDENTIAL_MESSAGE_722H)
        return False


class DomainRival722H(_HardeningProbeBase722H):
    """Clean rival in the probe domain; must not hide the domain-hook failure."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722H}

    @classmethod
    def get_domain(cls) -> Domain:
        return Domain(DOMAIN_722H)

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == DOMAIN_FEATURE


class IndexedConsistencyProbe722H(_HardeningProbeBase722H):
    """Indexed candidate for the links=() equals links=None consistency pin."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722H}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("probe722h_row_id",))]

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == LINKS_EQUAL_FEATURE


class NotEvaluatedProbe722H(_HardeningProbeBase722H):
    """Declares an index no request link carries: rejected at the LINK_INDEX stage."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722H}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("probe722h_ne_key",))]

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == NOT_EVALUATED_FEATURE


class CountingResolveProbe722H(_HardeningProbeBase722H):
    """Probe declaring only the counting framework, for the single-sampling pin."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwCountingResolve722H}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == COUNTING_FEATURE


class EngineProbe722H(_HardeningProbeBase722H):
    """Root probe matching both engine feature names on one framework."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwEngine722H}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) in (ENGINE_FEATURE_ONE, ENGINE_FEATURE_TWO)


class LinkAnchorLeft722H(_HardeningProbeBase722H):
    """Anchors the left side of the foreign link; never matches a name."""


class LinkAnchorRight722H(_HardeningProbeBase722H):
    """Anchors the right side of the foreign link; never matches a name."""


class ProbeDomainBoom722H(_HardeningProbeBase722H):
    """Probe whose get_domain() raises while armed; the hook cannot be name-gated."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722H}

    @classmethod
    def get_domain(cls) -> Domain:
        if "get_domain" in _ARMED_HOOKS_722H:
            raise RuntimeError(DOMAIN_BOOM_TEXT)
        return Domain.get_default_domain()

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == DOMAIN_FEATURE


class ProbeIndexBoom722H(_HardeningProbeBase722H):
    """Probe whose index_columns() raises while armed; the hook cannot be name-gated."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722H}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        if "index_columns" in _ARMED_HOOKS_722H:
            raise RuntimeError(INDEX_BOOM_TEXT)
        return None

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == INDEX_BOOM_FEATURE


class ProbeSupportsIndexBoom722H(_HardeningProbeBase722H):
    """Probe with valid index_columns but a supports_index() raising while armed."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAlpha722H}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("probe722h_si_key",))]

    @classmethod
    def supports_index(cls, index: Index) -> bool | None:
        if "supports_index" in _ARMED_HOOKS_722H:
            raise RuntimeError(SUPPORTS_BOOM_TEXT)
        return False

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == SUPPORTS_BOOM_FEATURE


@pytest.fixture
def arm_hook_722h() -> Iterator[Callable[[str], None]]:
    """Arm raising hooks for one test; teardown disarms them even when the raise propagated."""
    armed_keys: list[str] = []

    def _arm(key: str) -> None:
        _ARMED_HOOKS_722H.add(key)
        armed_keys.append(key)

    yield _arm
    for key in armed_keys:
        _ARMED_HOOKS_722H.discard(key)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve(
    mapping: FeatureGroupEnvironmentMapping,
    feature: Feature,
    links: set[Link] | None = None,
) -> ResolutionOutcome:
    request = ResolutionRequestSnapshot.from_feature(feature, links=links)
    return FeatureGroupResolver().resolve(request, snapshot_from_mapping(mapping))


def _candidate(outcome: ResolutionOutcome, feature_group: type[FeatureGroup]) -> CandidateEvaluation:
    identity = PluginIdentity.from_class(feature_group)
    matches = [candidate for candidate in outcome.candidates if candidate.identity == identity]
    assert len(matches) == 1, f"expected exactly one evaluation for {identity.render()}, got {len(matches)}"
    return matches[0]


def _framework_evaluation(candidate: CandidateEvaluation, framework: type[ComputeFramework]) -> FrameworkEvaluation:
    matches = [evaluation for evaluation in candidate.frameworks if evaluation.framework is framework]
    assert len(matches) == 1, f"expected exactly one evaluation for {framework.__name__}, got {len(matches)}"
    return matches[0]


def _reasons(candidate: CandidateEvaluation) -> set[RejectionReason]:
    return {rejection.reason for rejection in candidate.rejections}


def _sole_failure(outcome: ResolutionOutcome) -> PluginFailure:
    assert len(outcome.failures) == 1, f"expected exactly one failure, got {len(outcome.failures)}"
    return outcome.failures[0]


def _foreign_link() -> Link:
    return Link.inner(
        JoinSpec(LinkAnchorLeft722H, "probe722h_left_id"),
        JoinSpec(LinkAnchorRight722H, "probe722h_right_id"),
    )


def _off_collector(*probes: type[FeatureGroup]) -> PluginCollector:
    return PluginCollector.enabled_feature_groups(set(probes)).set_strict_mode("off")


# ---------------------------------------------------------------------------
# 1. Outcome payload is redacted
# ---------------------------------------------------------------------------


def test_outcome_payload_redacts_provider_message() -> None:
    """The raw provider text stays on the internal message field; to_payload() never carries it."""
    mapping: FeatureGroupEnvironmentMapping = {SecretCriteriaBoom722H: {CfwAlpha722H}}
    outcome = _resolve(mapping, Feature(REDACTION_FEATURE))

    assert outcome.status is ResolutionStatus.FAILED
    failure = _sole_failure(outcome)
    # Internal field keeps the raw text: it feeds the rendered engine and debug errors.
    assert failure.message == _FAKE_CREDENTIAL_MESSAGE_722H
    assert failure.stage == "match_feature_group_criteria"
    assert failure.category == "ValueError"
    assert failure.plugin == PluginIdentity.from_class(SecretCriteriaBoom722H)

    dumped = json.dumps(outcome.to_payload())
    assert "hush-722h" not in dumped
    assert "SecretCriteriaBoom722H" in dumped
    assert "match_feature_group_criteria" in dumped
    assert "ValueError" in dumped


# ---------------------------------------------------------------------------
# 2. get_domain is guarded
# ---------------------------------------------------------------------------


def test_raising_get_domain_fails_closed_without_propagating(arm_hook_722h: Callable[[str], None]) -> None:
    """A raising get_domain becomes a structured FAILED outcome; a clean rival cannot hide it."""
    arm_hook_722h("get_domain")
    mapping: FeatureGroupEnvironmentMapping = {
        ProbeDomainBoom722H: {CfwAlpha722H},
        DomainRival722H: {CfwAlpha722H},
    }

    outcome = _resolve(mapping, Feature(DOMAIN_FEATURE, domain=DOMAIN_722H))

    assert outcome.status is ResolutionStatus.FAILED
    assert outcome.winner is None

    failing = _candidate(outcome, ProbeDomainBoom722H)
    assert failing.status is CandidateStatus.FAILED
    assert failing.failure is not None
    assert failing.failure.category == "RuntimeError"
    assert DOMAIN_BOOM_TEXT in failing.failure.message
    assert failing.failure.plugin == PluginIdentity.from_class(ProbeDomainBoom722H)

    assert any(
        failure.category == "RuntimeError" and DOMAIN_BOOM_TEXT in failure.message for failure in outcome.failures
    )

    # The clean rival survives its own evaluation but must not hide the failure.
    assert _candidate(outcome, DomainRival722H).status is CandidateStatus.SURVIVOR


# ---------------------------------------------------------------------------
# 3. index_columns is guarded
# ---------------------------------------------------------------------------


def test_raising_index_columns_fails_closed_without_propagating(arm_hook_722h: Callable[[str], None]) -> None:
    """A raising index_columns becomes a structured FAILED outcome, with and without links.

    Investigated for the Red phase: _supports_request_links calls index_columns() BEFORE
    the empty-links short-circuit, so the hook is consulted even for a link-less request.
    Both requests therefore pin the same fail-closed guard.
    """
    arm_hook_722h("index_columns")
    mapping: FeatureGroupEnvironmentMapping = {ProbeIndexBoom722H: {CfwAlpha722H}}

    for links in (None, {_foreign_link()}):
        outcome = _resolve(mapping, Feature(INDEX_BOOM_FEATURE), links=links)

        assert outcome.status is ResolutionStatus.FAILED
        assert outcome.winner is None

        failing = _candidate(outcome, ProbeIndexBoom722H)
        assert failing.status is CandidateStatus.FAILED
        assert failing.failure is not None
        assert failing.failure.category == "RuntimeError"
        assert INDEX_BOOM_TEXT in failing.failure.message

        assert any(
            failure.category == "RuntimeError" and INDEX_BOOM_TEXT in failure.message for failure in outcome.failures
        )


# ---------------------------------------------------------------------------
# 4. supports_index is guarded
# ---------------------------------------------------------------------------


def test_raising_supports_index_fails_closed_without_propagating(arm_hook_722h: Callable[[str], None]) -> None:
    """A raising supports_index on a linked request becomes a structured FAILED outcome."""
    arm_hook_722h("supports_index")
    mapping: FeatureGroupEnvironmentMapping = {ProbeSupportsIndexBoom722H: {CfwAlpha722H}}

    outcome = _resolve(mapping, Feature(SUPPORTS_BOOM_FEATURE), links={_foreign_link()})

    assert outcome.status is ResolutionStatus.FAILED
    assert outcome.winner is None

    failing = _candidate(outcome, ProbeSupportsIndexBoom722H)
    assert failing.status is CandidateStatus.FAILED
    assert failing.failure is not None
    assert failing.failure.category == "RuntimeError"
    assert SUPPORTS_BOOM_TEXT in failing.failure.message
    assert failing.failure.plugin == PluginIdentity.from_class(ProbeSupportsIndexBoom722H)

    assert any(
        failure.category == "RuntimeError" and SUPPORTS_BOOM_TEXT in failure.message for failure in outcome.failures
    )


# ---------------------------------------------------------------------------
# 5. links=() equals links=None (TARGET CONTRACT, passes today)
# ---------------------------------------------------------------------------


def test_empty_link_set_resolves_like_no_links() -> None:
    """TARGET CONTRACT: an explicit empty link set behaves exactly like links=None.

    The old engine rejected indexed feature groups when the request carried an explicit
    empty link set while accepting links=None; the resolver deliberately treats both as
    "no links" (consistency change decided in the Stage 3 review).
    """
    mapping: FeatureGroupEnvironmentMapping = {IndexedConsistencyProbe722H: {CfwAlpha722H}}

    with_none = _resolve(mapping, Feature(LINKS_EQUAL_FEATURE), links=None)
    with_empty = _resolve(mapping, Feature(LINKS_EQUAL_FEATURE), links=set())

    assert with_none.status is with_empty.status
    assert with_none.status is ResolutionStatus.RESOLVED
    assert with_none.winner is not None
    assert with_empty.winner is not None
    assert with_none.winner.feature_group is IndexedConsistencyProbe722H
    assert with_empty.winner.feature_group is IndexedConsistencyProbe722H


# ---------------------------------------------------------------------------
# 6. LINK_INDEX rejection records NOT_EVALUATED frameworks
# ---------------------------------------------------------------------------


def test_link_index_rejected_candidate_frameworks_are_not_evaluated() -> None:
    """Frameworks that were never capability-evaluated must not be recorded as SUPPORTED."""
    mapping: FeatureGroupEnvironmentMapping = {NotEvaluatedProbe722H: {CfwAlpha722H}}
    outcome = _resolve(mapping, Feature(NOT_EVALUATED_FEATURE), links={_foreign_link()})

    assert outcome.status is ResolutionStatus.NOT_FOUND
    candidate = _candidate(outcome, NotEvaluatedProbe722H)
    assert candidate.status is CandidateStatus.REJECTED
    assert RejectionReason.LINK_INDEX in _reasons(candidate)

    # getattr keeps the module mypy-clean until the Green phase adds the member.
    not_evaluated = getattr(FrameworkStatus, "NOT_EVALUATED", None)
    assert not_evaluated is not None, "FrameworkStatus must grow a NOT_EVALUATED member"
    assert not_evaluated.value == "not_evaluated"

    evaluation = _framework_evaluation(candidate, CfwAlpha722H)
    assert evaluation.status is not FrameworkStatus.SUPPORTED
    assert evaluation.status is not_evaluated


# ---------------------------------------------------------------------------
# 7. resolve_feature samples availability exactly once
# ---------------------------------------------------------------------------


def test_resolve_feature_samples_availability_exactly_once() -> None:
    """One resolve_feature call consults a framework's is_available exactly once.

    Today it is 2: once in the plugin_docs pre-sampling (the compute_frameworks argument)
    and once inside the environment factory pipeline.
    """
    collector = _off_collector(CountingResolveProbe722H)
    _RESOLVE_AVAILABILITY_CALLS_722H.clear()

    result = resolve_feature(COUNTING_FEATURE, plugin_collector=collector)

    assert result.error is None  # premise guard: the probe resolves cleanly
    assert result.feature_group is CountingResolveProbe722H
    assert len(_RESOLVE_AVAILABILITY_CALLS_722H) == 1


# ---------------------------------------------------------------------------
# 8. resolve_feature never raises on a raising is_available
# ---------------------------------------------------------------------------


def test_resolve_feature_never_raises_on_raising_is_available(arm_hook_722h: Callable[[str], None]) -> None:
    """A raising availability probe becomes ResolvedFeature.error, never a raise."""
    arm_hook_722h("is_available")
    assert CfwRaisingAvail722H.__name__ == RAISING_AVAIL_CLASS_NAME  # premise guard for the error assert below
    collector = _off_collector(CleanProbe722H)

    result = resolve_feature(CLEAN_FEATURE, plugin_collector=collector)

    assert isinstance(result, ResolvedFeature)
    assert result.feature_group is None
    assert result.error is not None
    # The rendered error must identify the failing availability probe (raw internal text
    # or the framework identity; payload redaction concerns to_payload only).
    assert AVAIL_BOOM_TEXT in result.error or RAISING_AVAIL_CLASS_NAME in result.error


# ---------------------------------------------------------------------------
# 9. Mapping-derived snapshots never consult the strict-mode env
# ---------------------------------------------------------------------------


def test_mapping_snapshot_does_not_consult_strict_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The engine adapter works under an invalid strict-mode env value.

    strict_mode_from_env RAISES on invalid values (premise guard below), so a
    mapping-derived snapshot that consulted the env would explode here. The mapping
    already encodes the survivors of any strict filtering; the snapshot must not
    re-read run configuration from the environment.
    """
    monkeypatch.setenv("MLODA_PLUGIN_REGISTRY_STRICT", "definitely-invalid-722h")

    with pytest.raises(ValueError, match="Invalid strict mode"):
        strict_mode_from_env()

    mapping: FeatureGroupEnvironmentMapping = {CleanProbe722H: {CfwAlpha722H}}
    identifier = IdentifyFeatureGroupClass(
        feature=Feature(CLEAN_FEATURE),
        accessible_plugins=mapping,
        links=None,
    )
    feature_group, compute_frameworks = identifier.get()
    assert feature_group is CleanProbe722H
    assert compute_frameworks == {CfwAlpha722H}
    assert identifier.outcome.status is ResolutionStatus.RESOLVED


# ---------------------------------------------------------------------------
# 10. Engine builds the environment snapshot at most once
# ---------------------------------------------------------------------------


def test_engine_builds_environment_snapshot_at_most_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """One Engine derives its snapshot from the mapping at most once, not once per feature.

    Observable: snapshot_from_mapping invocations, counted at BOTH live bindings (the
    defining resolver module and the identify_feature_group import), with the wrapper
    delegating to this module's original binding so nothing is double-counted.
    """
    gc.collect()
    calls: list[str] = []

    def counting_snapshot(mapping_arg: FeatureGroupEnvironmentMapping) -> ResolutionEnvironmentSnapshot:
        calls.append("call")
        return snapshot_from_mapping(mapping_arg)

    monkeypatch.setattr(resolver_module, "snapshot_from_mapping", counting_snapshot)
    monkeypatch.setattr(identify_module, "snapshot_from_mapping", counting_snapshot)

    mapping: FeatureGroupEnvironmentMapping = {EngineProbe722H: {CfwEngine722H}}
    collector = _off_collector(EngineProbe722H)
    features = Features([Feature(ENGINE_FEATURE_ONE), Feature(ENGINE_FEATURE_TWO)])

    with (
        patch(
            "mloda.core.prepare.accessible_plugins.PreFilterPlugins.resolve_feature_group_compute_framework_limitations"
        ) as mocked_limitations,
        patch("mloda.core.core.engine.Engine.create_setup_execution_plan"),
    ):
        mocked_limitations.return_value = mapping
        engine = Engine(features, {CfwEngine722H}, None, plugin_collector=collector)
        engine.setup_features_recursion(features)

    # Premise guards: both requested features resolved through the same environment.
    assert len(engine.resolution_outcomes) == 2
    assert EngineProbe722H in engine.feature_group_collection
    assert len(engine.feature_group_collection[EngineProbe722H]) == 2

    assert len(calls) <= 1, f"snapshot_from_mapping ran {len(calls)} times; one Engine must build it at most once"
