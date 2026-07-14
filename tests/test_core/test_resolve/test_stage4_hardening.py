"""Failing tests for issue #722 Stage 4 hardening (final post-review fix round).

Pins review-mandated fixes across the diagnostics stack:

1.  mlodaAPI.diagnose never raises on a feature pinned outside the run set. Investigated:
    the pre-check builds SetupComputeFramework over Features([]) BEFORE the try, and the
    try catches only FeatureResolutionError, so the ValueError raised by prepare()'s
    per-feature run-set validation escapes today.
2.  mlodaAPI.diagnose never raises on an unknown framework name. Investigated: the
    unknown-name ValueError comes from the pre-check line itself, before the try.
3.  A VALUE_REJECTION detail is redacted from ResolutionOutcome.to_payload() while the
    in-memory Rejection.detail keeps the raw provider text (it feeds rendered errors).
4.  resolve_feature with an EXPLICIT compute_frameworks set never raises on a raising
    is_available probe. Investigated: build_resolution_environment guards availability
    sampling only on the compute_frameworks=None branch; the explicit-set branch lets
    run_compute_framework_pipeline sample unguarded. TARGET CONTRACT pin (passes today):
    PreFilterPlugins with an explicit set still propagates the raise, engine semantics
    unchanged.
5.  diagnose() returns the full pre-check environment: a collector-disabled probe keeps
    its DISABLED_BY_COLLECTOR record, instead of the engine's survivors-only snapshot.
6.  The engine's mapping-derived snapshot carries the collector's effective strict mode.
    Investigated: Engine.__init__ calls snapshot_from_mapping without strict_mode, so
    session.resolution_report().environment.snapshot.strict_mode is "off" today even for
    a collector explicitly set to "warn".
7.  render_provider_view carries failure metadata: the failure stage string and category
    for a raising criteria hook, and the stage of the multi-pin request-validation
    failure. Today the provider view prints only statuses and rejection reasons.
8.  render_user_view for the multi-pin request-validation failure must not claim a
    provider hook raised; its next-step line names the pin or the request instead.
9.  Mixed-rejection presentation: with one candidate rejected only by LINK_INDEX and one
    rejected only by capability, the "unsupported on all installed compute frameworks"
    error must not list the link-rejected candidate. TARGET CONTRACT pin (passes today):
    the sole-candidate texts stay untouched.

RED phase: tests 1, 2, 3, 4a, 5, 6, 7, 8, and 9a fail against the current
implementation; tests 4b and 9b pass and pin target contracts.

PINNED CHOICES beyond the issue brief (report these to the Green agent):
- Tests 1 and 2 pin ONLY that no exception escapes, the report is a ResolutionReport,
  and complete is False; whether the environment or a feature record carries the error
  stays free.
- Test 3 pins that the payload keeps the "value_rejection" reason while dropping the
  detail text, and that the in-memory detail field equals the raw provider text.
- Test 4a pins that the returned error identifies the failing availability probe (raw
  internal text or the framework class name), mirroring the Stage 3 None-branch pin.
- Test 6 pins snapshot.strict_mode == "warn" for a collector explicitly set to "warn".
- Test 8 pins that "provider hook" appears nowhere in the user view and that a
  next-step line contains "pin" or "request" (case-insensitive loose substrings).
- Test 9a pins that the link-rejected candidate's class name is absent from the ENTIRE
  error string, not merely from the candidate list clause.

Probe names use the unique probe722k_ prefix; probe classes carry the 722K suffix so
process-global scans in other tests never trip these fixtures. The is_available hook
cannot be name-gated: it raises only while ARMED by its test and is disarmed by fixture
teardown, exactly like the Stage 3 hardening fixtures (xdist-safe: per-process state).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterator

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import PropertyValueRejection
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.api.plugin_info import ResolvedFeature
from mloda.core.api.request import mlodaAPI
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping, PreFilterPlugins
from mloda.core.resolve.environment import EnvironmentProvenance
from mloda.core.resolve.identity import PluginIdentity
from mloda.core.resolve.outcome import RejectionReason, ResolutionOutcome, ResolutionStatus
from mloda.core.resolve.render import render_provider_view, render_user_view
from mloda.core.resolve.report import ResolutionReport
from mloda.core.resolve.request import ResolutionRequestSnapshot
from mloda.core.resolve.resolver import FeatureGroupResolver, snapshot_from_mapping
from mloda.provider import BaseInputData, DataCreator, FeatureSet
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


PINNED_FEATURE = "probe722k_pinned"
VALUE_REJECT_FEATURE = "probe722k_value_reject"
CLEAN_FEATURE = "probe722k_clean"
ENV_FEATURE = "probe722k_env"
CRITERIA_BOOM_FEATURE = "probe722k_criteria_boom"
MULTI_PIN_FEATURE = "probe722k_multi_pin"
MIXED_FEATURE = "probe722k_mixed"

# Fake secret-shaped rejection detail used to prove payload redaction; never a real secret.
VALUE_SECRET_TEXT = "rejected value secret-722k"  # nosec B105
CRITERIA_BOOM_TEXT = "criteria boom 722k"
AVAIL_BOOM_TEXT = "avail boom 722k"
RAISING_AVAIL_CLASS_NAME = "CfwRaisingAvail722K"

CAPABILITY_CLAUSE_TEXT = "unsupported on all installed compute frameworks"
LINK_CLAUSE_TEXT = "an index carried by the request's links"


# ---------------------------------------------------------------------------
# Mock compute frameworks (unique names, suffix 722K)
# ---------------------------------------------------------------------------


class CfwA722K(ComputeFramework):
    """Framework the pinned diagnose probe declares; excluded from the run set."""


class CfwB722K(ComputeFramework):
    """Framework forming the run set that excludes CfwA722K."""


class CfwGood722K(ComputeFramework):
    """Well-behaved framework for the explicit-set availability probes."""


class CfwMixed722K(ComputeFramework):
    """Framework shared by the mixed-rejection probes."""


# Hook keys armed per test via the arm_hook_722k fixture; disarmed on teardown. While
# disarmed, the hook degrades to an inert default, so the module-level class stays
# harmless in every other test of the process (xdist-safe: per-process state).
_ARMED_HOOKS_722K: set[str] = set()


class CfwRaisingAvail722K(ComputeFramework):
    """Framework whose is_available() raises while armed; unavailable otherwise.

    EVERY availability sampling in the process hits this hook, so the raise is armed
    only for the tests pinning it and the class is otherwise a plain unavailable
    framework.
    """

    @staticmethod
    def is_available() -> bool:
        if "is_available" in _ARMED_HOOKS_722K:
            raise RuntimeError(AVAIL_BOOM_TEXT)
        return False


@pytest.fixture
def arm_hook_722k() -> Iterator[Callable[[str], None]]:
    """Arm raising hooks for one test; teardown disarms them even when the raise propagated."""
    armed_keys: list[str] = []

    def _arm(key: str) -> None:
        _ARMED_HOOKS_722K.add(key)
        armed_keys.append(key)

    yield _arm
    for key in armed_keys:
        _ARMED_HOOKS_722K.discard(key)


# ---------------------------------------------------------------------------
# Probe feature groups (match ONLY their own probe722k_* names)
# ---------------------------------------------------------------------------


class _Probe722KBase(FeatureGroup):
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


class SourcePinnedA722K(FeatureGroup):
    """DataCreator source on CfwA722K; matches ONLY probe722k_pinned via its DataCreator."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwA722K}

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={PINNED_FEATURE})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {PINNED_FEATURE: [1, 2, 3]}


class SourceEnvEnabled722K(FeatureGroup):
    """DataCreator source on PythonDictFramework; the collector-enabled environment probe."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={ENV_FEATURE})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {ENV_FEATURE: [1, 2, 3]}


class ProbeEnvDisabled722K(_Probe722KBase):
    """Concrete probe the environment-provenance collector deliberately disables."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}


class ValueRejectingProbe722K(_Probe722KBase):
    """Raises PropertyValueRejection with a secret-shaped detail, gated on its name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwA722K}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if str(feature_name) == VALUE_REJECT_FEATURE:
            raise PropertyValueRejection(VALUE_SECRET_TEXT)
        return False


class CriteriaBoomProbe722K(_Probe722KBase):
    """Raises RuntimeError from its criteria hook, gated on its name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwA722K}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if str(feature_name) == CRITERIA_BOOM_FEATURE:
            raise RuntimeError(CRITERIA_BOOM_TEXT)
        return False


class CleanProbe722K(_Probe722KBase):
    """Single clean candidate on the well-behaved framework."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwGood722K}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == CLEAN_FEATURE


class ProbeMixedLinkRejected722K(_Probe722KBase):
    """Declares an index no request link carries: rejected ONLY at the LINK_INDEX stage."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwMixed722K}

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("probe722k_mixed_key",))]

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == MIXED_FEATURE


class ProbeMixedCapRejected722K(_Probe722KBase):
    """Rejects its framework via the capability hook: rejected ONLY at the CAPABILITY stage."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwMixed722K}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == MIXED_FEATURE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        # Gated on the probe name so process-global scans in other tests never trip the rejection.
        return str(feature_name) != MIXED_FEATURE


class ProbeMixedAnchorLeft722K(_Probe722KBase):
    """Anchors the left side of the foreign link; never matches a name."""


class ProbeMixedAnchorRight722K(_Probe722KBase):
    """Anchors the right side of the foreign link; never matches a name."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _off_collector(*probes: type[FeatureGroup]) -> PluginCollector:
    return PluginCollector.enabled_feature_groups(set(probes)).set_strict_mode("off")


def _mixed_link() -> Link:
    """A link between two other groups whose indexes the indexed probe does not support."""
    return Link.inner(
        JoinSpec(ProbeMixedAnchorLeft722K, "probe722k_left_id"),
        JoinSpec(ProbeMixedAnchorRight722K, "probe722k_right_id"),
    )


def _resolve(
    mapping: FeatureGroupEnvironmentMapping,
    feature: Feature,
    links: set[Link] | None = None,
) -> ResolutionOutcome:
    request = ResolutionRequestSnapshot.from_feature(feature, links=links)
    return FeatureGroupResolver().resolve(request, snapshot_from_mapping(mapping))


def _criteria_failed_outcome() -> ResolutionOutcome:
    """FAILED outcome from a raising match_feature_group_criteria hook."""
    outcome = _resolve({CriteriaBoomProbe722K: {CfwA722K}}, Feature(CRITERIA_BOOM_FEATURE))
    assert outcome.status is ResolutionStatus.FAILED  # premise guard
    assert any(failure.stage == "match_feature_group_criteria" for failure in outcome.failures)  # premise guard
    return outcome


def _multi_pin_outcome() -> ResolutionOutcome:
    """FAILED outcome from the multi-pin request validation, before any candidate runs."""
    options = Options()
    request = ResolutionRequestSnapshot(
        feature_name=MULTI_PIN_FEATURE,
        domain=None,
        feature_group_scope=None,
        framework_pin=None,
        pinned_frameworks=(CfwA722K, CfwB722K),
        group_option_keys=frozenset(),
        context_option_keys=frozenset(),
        inherited_group_keys=options.inherited_group_keys,
        dependency_path=(),
        links=(),
        data_access_collection=None,
        options=options,
    )
    outcome = FeatureGroupResolver().resolve(request, snapshot_from_mapping({CleanProbe722K: {CfwGood722K}}))
    assert outcome.status is ResolutionStatus.FAILED  # premise guard
    assert len(outcome.failures) == 1  # premise guard
    assert outcome.failures[0].stage == "validate_request"  # premise guard
    return outcome


# ---------------------------------------------------------------------------
# 1. diagnose never raises on a feature pinned outside the run set
# ---------------------------------------------------------------------------


def test_probe722k_diagnose_never_raises_on_feature_pinned_outside_run_set() -> None:
    """A framework pin outside the run set yields a complete=False report, never a raise.

    Investigated: the pin survives the diagnose pre-check because that pre-check runs
    SetupComputeFramework over Features([]); the ValueError then raised by prepare()'s
    per-feature run-set validation is not a FeatureResolutionError, so it escapes today.
    """
    report = mlodaAPI.diagnose(
        [Feature(PINNED_FEATURE, compute_framework=CfwA722K.get_class_name())],
        compute_frameworks={CfwB722K},
        plugin_collector=_off_collector(SourcePinnedA722K),
    )

    assert report is not None
    assert isinstance(report, ResolutionReport)
    assert report.complete is False


# ---------------------------------------------------------------------------
# 2. diagnose never raises on an unknown framework name
# ---------------------------------------------------------------------------


def test_probe722k_diagnose_never_raises_on_unknown_framework_name() -> None:
    """An unknown compute framework name yields a complete=False report, never a raise.

    Investigated: the unknown-name ValueError comes from the SetupComputeFramework
    pre-check line itself, BEFORE the try that narrows to FeatureResolutionError.
    """
    report = mlodaAPI.diagnose(
        [Feature(PINNED_FEATURE)],
        compute_frameworks=["NoSuchFramework722K"],
        plugin_collector=_off_collector(SourcePinnedA722K),
    )

    assert report is not None
    assert isinstance(report, ResolutionReport)
    assert report.complete is False


# ---------------------------------------------------------------------------
# 3. VALUE_REJECTION payload redaction
# ---------------------------------------------------------------------------


def test_probe722k_value_rejection_payload_redacts_detail() -> None:
    """The raw rejection text stays on the in-memory detail; to_payload() never carries it."""
    outcome = _resolve({ValueRejectingProbe722K: {CfwA722K}}, Feature(VALUE_REJECT_FEATURE))

    assert outcome.status is ResolutionStatus.NOT_FOUND  # premise guard
    identity = PluginIdentity.from_class(ValueRejectingProbe722K)
    candidates = [candidate for candidate in outcome.candidates if candidate.identity == identity]
    assert len(candidates) == 1
    rejections = [
        rejection for rejection in candidates[0].rejections if rejection.reason is RejectionReason.VALUE_REJECTION
    ]
    assert len(rejections) == 1
    # The in-memory detail keeps the raw provider text: the rendered errors depend on it.
    assert rejections[0].detail == VALUE_SECRET_TEXT

    dumped = json.dumps(outcome.to_payload())
    assert RejectionReason.VALUE_REJECTION.value in dumped  # the structured reason stays visible
    assert "secret-722k" not in dumped


# ---------------------------------------------------------------------------
# 4. Explicit-set availability guard
# ---------------------------------------------------------------------------


def test_probe722k_resolve_feature_explicit_set_never_raises_on_raising_availability(
    arm_hook_722k: Callable[[str], None],
) -> None:
    """With an EXPLICIT compute_frameworks set, a raising is_available probe becomes
    ResolvedFeature.error, never a raise.

    Investigated: build_resolution_environment guards availability sampling only when
    compute_frameworks is None; the explicit-set branch reaches
    run_compute_framework_pipeline with available=None, which samples unguarded.
    """
    arm_hook_722k("is_available")
    assert CfwRaisingAvail722K.__name__ == RAISING_AVAIL_CLASS_NAME  # premise guard for the error assert below
    collector = _off_collector(CleanProbe722K)

    result = resolve_feature(CLEAN_FEATURE, compute_frameworks={CfwGood722K}, plugin_collector=collector)

    assert isinstance(result, ResolvedFeature)
    assert result.feature_group is None
    assert result.error is not None
    # The rendered error must identify the failing availability probe (raw internal text
    # or the framework identity), mirroring the Stage 3 None-branch pin.
    assert AVAIL_BOOM_TEXT in result.error or RAISING_AVAIL_CLASS_NAME in result.error


def test_probe722k_prefilter_explicit_set_availability_raise_still_propagates(
    arm_hook_722k: Callable[[str], None],
) -> None:
    """TARGET CONTRACT (passes today): the ENGINE path keeps propagating the raise.

    PreFilterPlugins with an explicit framework set samples availability through the
    shared pipeline; its raise-through semantics are pinned unchanged so the diagnostics
    guard cannot silently swallow engine-path failures.
    """
    arm_hook_722k("is_available")
    collector = _off_collector(CleanProbe722K)

    with pytest.raises(RuntimeError, match=AVAIL_BOOM_TEXT):
        PreFilterPlugins({CfwGood722K}, collector)


# ---------------------------------------------------------------------------
# 5. diagnose environment provenance: full pre-check outcome, not survivors-only
# ---------------------------------------------------------------------------


def test_probe722k_diagnose_environment_records_disabled_probes() -> None:
    """The diagnose report's environment keeps the DISABLED_BY_COLLECTOR record of a
    probe the collector disabled, proving the full pre-check outcome is returned and
    not the engine's synthesized survivors-only snapshot."""
    report = mlodaAPI.diagnose(
        [Feature(ENV_FEATURE)],
        compute_frameworks={PythonDictFramework},
        plugin_collector=_off_collector(SourceEnvEnabled722K),
    )

    assert report.complete is True  # premise guard: the request itself resolves
    assert report.environment is not None
    snapshot = report.environment.snapshot
    assert snapshot is not None

    disabled_identity = PluginIdentity.from_class(ProbeEnvDisabled722K)
    disabled_records = [record for record in snapshot.records if record.identity == disabled_identity]
    assert len(disabled_records) == 1, "the disabled probe must keep an environment record"
    assert disabled_records[0].provenance is EnvironmentProvenance.DISABLED_BY_COLLECTOR

    enabled_identity = PluginIdentity.from_class(SourceEnvEnabled722K)
    enabled_records = [record for record in snapshot.records if record.identity == enabled_identity]
    assert len(enabled_records) == 1
    assert enabled_records[0].provenance is EnvironmentProvenance.ACCESSIBLE


# ---------------------------------------------------------------------------
# 6. Engine snapshot strict-mode metadata
# ---------------------------------------------------------------------------


def test_probe722k_session_report_snapshot_carries_collector_strict_mode() -> None:
    """The session report's snapshot carries the collector's effective strict mode.

    INVESTIGATION RESULT: Engine.__init__ builds its mapping-derived snapshot via
    snapshot_from_mapping(self.accessible_plugins) without passing strict_mode, so the
    snapshot reads "off" today even though the collector explicitly runs in "warn".
    """
    collector = PluginCollector.enabled_feature_groups({SourceEnvEnabled722K}).set_strict_mode("warn")
    session = mlodaAPI.prepare(
        [Feature(ENV_FEATURE)],
        compute_frameworks={PythonDictFramework},
        plugin_collector=collector,
    )

    report = session.resolution_report()
    assert report.environment is not None
    snapshot = report.environment.snapshot
    assert snapshot is not None
    assert snapshot.strict_mode == "warn"


# ---------------------------------------------------------------------------
# 7. Provider renderer failure metadata
# ---------------------------------------------------------------------------


def test_probe722k_provider_view_carries_failure_stage_and_category() -> None:
    """The provider view names the failing stage and category of a FAILED outcome."""
    criteria_text = render_provider_view(_criteria_failed_outcome())
    assert "match_feature_group_criteria" in criteria_text
    assert "RuntimeError" in criteria_text

    multi_pin_text = render_provider_view(_multi_pin_outcome())
    assert "validate_request" in multi_pin_text


# ---------------------------------------------------------------------------
# 8. User renderer next-step accuracy
# ---------------------------------------------------------------------------


def test_probe722k_user_view_multi_pin_does_not_blame_provider_hook() -> None:
    """The user view of the multi-pin failure never claims a provider hook raised;
    its next-step line points at the pin or the request instead."""
    text = render_user_view(_multi_pin_outcome())

    assert "provider hook" not in text.lower()

    next_step_lines = [line for line in text.splitlines() if line.lower().startswith("next step")]
    assert next_step_lines, "the user view must keep a next-step line"
    assert any("pin" in line.lower() or "request" in line.lower() for line in next_step_lines)


# ---------------------------------------------------------------------------
# 9. Mixed-rejection presentation
# ---------------------------------------------------------------------------


def test_probe722k_mixed_rejection_capability_clause_excludes_link_rejected_candidate() -> None:
    """The capability error of a mixed rejection never lists the link-rejected candidate.

    One candidate is rejected ONLY by LINK_INDEX (indexed, foreign links provided), the
    other ONLY by capability; the "unsupported on all installed compute frameworks"
    text must not name the link-rejected class anywhere in the error.
    """
    collector = _off_collector(ProbeMixedLinkRejected722K, ProbeMixedCapRejected722K)

    result = resolve_feature(MIXED_FEATURE, links={_mixed_link()}, plugin_collector=collector)

    assert result.feature_group is None
    assert result.error is not None
    assert CAPABILITY_CLAUSE_TEXT in result.error  # premise guard: the capability clause is chosen
    assert ProbeMixedCapRejected722K.get_class_name() in result.error
    assert ProbeMixedLinkRejected722K.get_class_name() not in result.error


def test_probe722k_sole_candidate_rejection_texts_unchanged() -> None:
    """TARGET CONTRACT (passes today): the sole-candidate error texts stay untouched."""
    cap_only = resolve_feature(
        MIXED_FEATURE, links={_mixed_link()}, plugin_collector=_off_collector(ProbeMixedCapRejected722K)
    )
    assert cap_only.feature_group is None
    assert cap_only.error is not None
    assert CAPABILITY_CLAUSE_TEXT in cap_only.error
    assert ProbeMixedCapRejected722K.get_class_name() in cap_only.error

    link_only = resolve_feature(
        MIXED_FEATURE, links={_mixed_link()}, plugin_collector=_off_collector(ProbeMixedLinkRejected722K)
    )
    assert link_only.feature_group is None
    assert link_only.error is not None
    assert LINK_CLAUSE_TEXT in link_only.error
    assert ProbeMixedLinkRejected722K.get_class_name() in link_only.error
