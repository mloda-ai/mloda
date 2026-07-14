"""Failing tests for issue #722 final review fix round.

Pins three accepted findings from the final review:

A.  The engine's failure-path enrichment re-invokes provider hooks after the decision
    was already made (mloda/core/prepare/identify_feature_group.py):
    1. _capability_rejection_message calls split_frameworks_by_capability, re-invoking
       compute_framework_definition / is_available / supports_compute_framework on the
       failure path. A capability hook that returns False on its first call and raises
       on later calls leaks that raise through IdentifyFeatureGroupClass instead of the
       established capability rejection error (test 1), and even a well-behaved hook
       runs twice per candidate/framework (test 2, today the counter reads 2).
    2. _input_feature_forwarding_hint re-invokes match_feature_group_criteria with bare
       AND actual options while building the no-match error. Investigated for the Red
       phase: the resolver consumes the FIRST junk-key call (CRITERIA rejection), the
       hint's accepts_bare call carries no junk key, and its rejects_actual call is the
       SECOND junk-key call, so the raising path IS hit for this shape and the raise
       leaks today (test 3).
B.  build_resolution_environment samples framework availability BEFORE
    run_feature_group_pipeline, so an armed raising is_available probe masks the
    strict-mode feature-group fatal; the strict-mode message must take precedence
    (test 4).
C.  mlodaAPI.diagnose's failure branch returns err.report, whose environment is the
    engine's synthesized survivors-only outcome; it must carry the full pre-check
    environment, including the DISABLED_BY_COLLECTOR record of a collector-disabled
    probe, alongside the failing feature record (test 5).

RED phase: all five tests fail against the current implementation.
- Test 1 fails with the leaked RuntimeError("second call 722m") escaping
  IdentifyFeatureGroupClass instead of FeatureResolutionError.
- Test 2 fails on the invocation counter: 2 today, pinned to exactly 1.
- Test 3 fails with the leaked RuntimeError("forward rematch boom 722m") escaping the
  no-match error building.
- Test 4 fails on the message: availability_failure text today, strict-mode text pinned.
- Test 5 fails on provenance: the survivors-only snapshot has no record for the
  disabled probe.

PINNED CHOICES (report these to the Green agent):
- Tests 1 and 2 pin the established capability rejection text ("Unsupported compute
  framework(s) for feature ...") and that the rejected framework is named; HOW the
  message is built (structured outcome data vs a guarded re-invocation) stays free,
  but test 2 forbids a second hook invocation.
- Test 3 pins only that the error remains the no-match FeatureResolutionError and that
  the provider raise does not leak; whether the forwarding hint still appears stays free.
- Test 4 pins the exact leading clause "Strict mode filtered out all FeatureGroups" and
  that the availability boom text is absent; error ordering inside the outcome stays free.
- Test 5 pins complete=False, the failing feature record (dependency path ends with the
  missing feature, outcome NOT_FOUND), and the DISABLED_BY_COLLECTOR record.

Probe names use the unique probe722m_ prefix; probe classes carry the 722M suffix so
process-global scans in other tests never trip these fixtures. Hooks that must raise do
so only while ARMED by their test and are disarmed by fixture teardown, following the
Stage 3/4 hardening fixture pattern (xdist-safe: per-process state, name-gated hooks).
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.api.plugin_info import ResolvedFeature
from mloda.core.api.request import mlodaAPI
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.core.resolve.environment import EnvironmentProvenance
from mloda.core.resolve.identity import PluginIdentity
from mloda.core.resolve.outcome import FeatureResolutionError, ResolutionStatus
from mloda.core.resolve.report import ResolutionReport
from mloda.provider import BaseInputData, DataCreator, FeatureSet
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


CAP_BOOM_FEATURE = "probe722m_cap_boom"
CAP_ONCE_FEATURE = "probe722m_cap_once"
FORWARD_FEATURE = "probe722m_forward"
PRECEDENCE_FEATURE = "probe722m_precedence"
DIAG_ENV_FEATURE = "probe722m_diag_env"
DIAG_MISSING_FEATURE = "probe722m_diag_missing"

JUNK_KEY = "junk722m"

SECOND_CALL_TEXT = "second call 722m"
FORWARD_BOOM_TEXT = "forward rematch boom 722m"
AVAIL_PREC_BOOM_TEXT = "avail precedence boom 722m"

CAPABILITY_REJECTION_TEXT = "Unsupported compute framework(s) for feature"
NO_MATCH_TEXT = "No feature groups found for feature name"
# Exact leading clause of _STRICT_MODE_FEATURE_GROUPS_MESSAGE in mloda/core/resolve/environment.py.
STRICT_MODE_TEXT = "Strict mode filtered out all FeatureGroups"


# ---------------------------------------------------------------------------
# Armed-hook state (Stage 3/4 fixture pattern; per-process, xdist-safe)
# ---------------------------------------------------------------------------

# Hook keys armed per test via the arm_hook_722m fixture; disarmed on teardown. While
# disarmed, every hook below degrades to an inert default, so the module-level classes
# stay harmless in every other test of the process.
_ARMED_HOOKS_722M: set[str] = set()

# supports_compute_framework invocations of the second-call probe, keyed by framework
# class name; cleared by the owning test before each engine/debug pass.
_CAP_SECOND_CALL_COUNTS_722M: dict[str, int] = {}

# supports_compute_framework invocations of the counting probe; cleared per test.
_CAP_ONCE_CALLS_722M: list[str] = []

# match_feature_group_criteria invocations of the forwarding probe that carried the
# junk group key; cleared per test.
_FORWARD_JUNK_CALLS_722M: list[str] = []


@pytest.fixture
def arm_hook_722m() -> Iterator[Callable[[str], None]]:
    """Arm raising hooks for one test; teardown disarms them even when the raise propagated."""
    armed_keys: list[str] = []

    def _arm(key: str) -> None:
        _ARMED_HOOKS_722M.add(key)
        armed_keys.append(key)

    yield _arm
    for key in armed_keys:
        _ARMED_HOOKS_722M.discard(key)


# ---------------------------------------------------------------------------
# Mock compute frameworks (unique names, suffix 722M)
# ---------------------------------------------------------------------------


class CfwCapBoom722M(ComputeFramework):
    """Framework declared by the second-call capability probe."""


class CfwCapOnce722M(ComputeFramework):
    """Framework declared by the counting capability probe."""


class CfwForward722M(ComputeFramework):
    """Framework declared by the forwarding-hint probe."""


class CfwRaisingAvailPrec722M(ComputeFramework):
    """Framework whose is_available() raises while armed; unavailable otherwise.

    EVERY availability sampling in the process hits this hook, so the raise is armed
    only for the precedence test and the class is otherwise a plain unavailable
    framework.
    """

    @staticmethod
    def is_available() -> bool:
        if "avail_precedence" in _ARMED_HOOKS_722M:
            raise RuntimeError(AVAIL_PREC_BOOM_TEXT)
        return False


# ---------------------------------------------------------------------------
# Probe feature groups (match ONLY their own probe722m_* names)
# ---------------------------------------------------------------------------


class _Probe722MBase(FeatureGroup):
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


class ProbeCapSecondCallBoom722M(_Probe722MBase):
    """Capability hook: False on the FIRST call per framework, raises on later calls while armed."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwCapBoom722M}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == CAP_BOOM_FEATURE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        # Gated on the probe name AND the armed flag so nothing else ever trips the raise.
        if str(feature_name) != CAP_BOOM_FEATURE or "cap_second_call" not in _ARMED_HOOKS_722M:
            return False
        key = compute_framework.get_class_name()
        count = _CAP_SECOND_CALL_COUNTS_722M.get(key, 0) + 1
        _CAP_SECOND_CALL_COUNTS_722M[key] = count
        if count > 1:
            raise RuntimeError(SECOND_CALL_TEXT)
        return False


class ProbeCapCountingOnce722M(_Probe722MBase):
    """Capability hook: always False, counting its own invocations; never raises."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwCapOnce722M}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == CAP_ONCE_FEATURE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        if str(feature_name) == CAP_ONCE_FEATURE:
            _CAP_ONCE_CALLS_722M.append(compute_framework.get_class_name())
        return False


class ProbeForwardHintBoom722M(_Probe722MBase):
    """Accepts bare options for its name; junk-key options: False on the first call,
    raises on later calls while armed (inert False while disarmed)."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwForward722M}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        if str(feature_name) != FORWARD_FEATURE:
            return False
        if JUNK_KEY not in options.group:
            return True
        if "forward_junk" not in _ARMED_HOOKS_722M:
            return False
        _FORWARD_JUNK_CALLS_722M.append("call")
        if len(_FORWARD_JUNK_CALLS_722M) > 1:
            raise RuntimeError(FORWARD_BOOM_TEXT)
        return False


class ProbeStrictPrecedence722M(_Probe722MBase):
    """Concrete probe the strict-mode precedence collector enables; strict mode drops it."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == PRECEDENCE_FEATURE


class SourceDiagEnabled722M(FeatureGroup):
    """DataCreator source on PythonDictFramework; the collector-enabled diagnose probe."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={DIAG_ENV_FEATURE})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {DIAG_ENV_FEATURE: [1, 2, 3]}


class ProbeDiagDisabled722M(_Probe722MBase):
    """Concrete probe the diagnose-provenance collector deliberately disables."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _off_collector(*probes: type[FeatureGroup]) -> PluginCollector:
    return PluginCollector.enabled_feature_groups(set(probes)).set_strict_mode("off")


# ---------------------------------------------------------------------------
# 1. Second-call capability hook cannot leak from the failure-path enrichment
# ---------------------------------------------------------------------------


def test_probe722m_capability_second_call_raise_does_not_leak_from_error_building(
    arm_hook_722m: Callable[[str], None],
) -> None:
    """The engine's capability rejection error is built without leaking a raise from a
    second hook invocation; the debug path stays non-raising for the same probe.

    Today _capability_rejection_message re-invokes supports_compute_framework via
    split_frameworks_by_capability, so the second call raises and the RuntimeError
    escapes IdentifyFeatureGroupClass instead of the FeatureResolutionError.
    """
    arm_hook_722m("cap_second_call")
    _CAP_SECOND_CALL_COUNTS_722M.clear()
    mapping: FeatureGroupEnvironmentMapping = {ProbeCapSecondCallBoom722M: {CfwCapBoom722M}}

    with pytest.raises(FeatureResolutionError) as excinfo:
        IdentifyFeatureGroupClass(
            feature=Feature(CAP_BOOM_FEATURE),
            accessible_plugins=mapping,
            links=None,
        )

    message = str(excinfo.value)
    assert CAPABILITY_REJECTION_TEXT in message
    assert CfwCapBoom722M.get_class_name() in message
    assert SECOND_CALL_TEXT not in message

    # The debug path over the same probe must also stay non-raising (fresh counter,
    # so the hook's first call capability-rejects the framework again).
    _CAP_SECOND_CALL_COUNTS_722M.clear()
    result = resolve_feature(CAP_BOOM_FEATURE, plugin_collector=_off_collector(ProbeCapSecondCallBoom722M))
    assert isinstance(result, ResolvedFeature)
    assert result.feature_group is None
    assert result.error is not None


# ---------------------------------------------------------------------------
# 2. Capability hooks run exactly once per candidate/framework on the failure path
# ---------------------------------------------------------------------------


def test_probe722m_capability_hook_invoked_exactly_once_on_failure_path() -> None:
    """One engine identification invokes supports_compute_framework exactly once per
    candidate/framework, even when the capability rejection error is being built.

    Today it is 2: once in the resolver and once more in split_frameworks_by_capability
    while _build_no_feature_group_error renders the message.
    """
    _CAP_ONCE_CALLS_722M.clear()
    mapping: FeatureGroupEnvironmentMapping = {ProbeCapCountingOnce722M: {CfwCapOnce722M}}

    with pytest.raises(FeatureResolutionError, match="Unsupported compute framework"):
        IdentifyFeatureGroupClass(
            feature=Feature(CAP_ONCE_FEATURE),
            accessible_plugins=mapping,
            links=None,
        )

    assert len(_CAP_ONCE_CALLS_722M) == 1, (
        f"supports_compute_framework ran {len(_CAP_ONCE_CALLS_722M)} times; "
        f"the failure path must not re-invoke the capability hook"
    )


# ---------------------------------------------------------------------------
# 3. Forwarding-hint re-match cannot leak from the no-match error building
# ---------------------------------------------------------------------------


def test_probe722m_forwarding_hint_rematch_raise_does_not_leak(
    arm_hook_722m: Callable[[str], None],
) -> None:
    """The no-match error is built without leaking a raise from the forwarding-hint
    re-match; the error remains the no-match FeatureResolutionError.

    Investigated for the Red phase: the resolver's criteria evaluation consumes the
    FIRST junk-key call (CRITERIA rejection), _input_feature_forwarding_hint's
    accepts_bare call carries no junk key (returns True), and its rejects_actual call
    is the SECOND junk-key call, which raises. Today that RuntimeError escapes
    IdentifyFeatureGroupClass instead of the no-match error.
    """
    arm_hook_722m("forward_junk")
    _FORWARD_JUNK_CALLS_722M.clear()
    mapping: FeatureGroupEnvironmentMapping = {ProbeForwardHintBoom722M: {CfwForward722M}}

    with pytest.raises(FeatureResolutionError) as excinfo:
        IdentifyFeatureGroupClass(
            feature=Feature(FORWARD_FEATURE, options={JUNK_KEY: "junk-value"}),
            accessible_plugins=mapping,
            links=None,
        )

    message = str(excinfo.value)
    assert NO_MATCH_TEXT in message
    assert FORWARD_FEATURE in message
    assert FORWARD_BOOM_TEXT not in message


# ---------------------------------------------------------------------------
# 4. Environment error precedence: strict-mode fatal over availability failure
# ---------------------------------------------------------------------------


def test_probe722m_strict_mode_error_takes_precedence_over_availability_failure(
    arm_hook_722m: Callable[[str], None],
) -> None:
    """A strict-mode feature-group fatal outranks a raising availability probe.

    Today build_resolution_environment samples availability BEFORE running the
    feature-group pipeline, so the armed is_available raise becomes the
    availability_failure error and masks the strict-mode message.
    """
    arm_hook_722m("avail_precedence")
    collector = (
        PluginCollector.enabled_feature_groups({ProbeStrictPrecedence722M})
        .set_strict_mode("strict")
        .set_registry(PluginRegistry())
    )

    result = resolve_feature(PRECEDENCE_FEATURE, plugin_collector=collector)

    assert isinstance(result, ResolvedFeature)
    assert result.feature_group is None
    assert result.error is not None
    assert STRICT_MODE_TEXT in result.error
    assert AVAIL_PREC_BOOM_TEXT not in result.error


# ---------------------------------------------------------------------------
# 5. diagnose failure branch carries the full pre-check environment
# ---------------------------------------------------------------------------


def test_probe722m_diagnose_failure_branch_carries_full_precheck_environment() -> None:
    """The failure-branch diagnose report keeps the DISABLED_BY_COLLECTOR record of a
    collector-disabled probe alongside the failing feature record.

    Today the FeatureResolutionError branch returns err.report, whose environment is
    the engine's synthesized survivors-only outcome, so the disabled probe has no
    record at all.
    """
    report = mlodaAPI.diagnose(
        [Feature(DIAG_MISSING_FEATURE)],
        compute_frameworks={PythonDictFramework},
        plugin_collector=_off_collector(SourceDiagEnabled722M),
    )

    assert isinstance(report, ResolutionReport)
    assert report.complete is False

    failing = [
        record
        for record in report.features
        if record.dependency_path and record.dependency_path[-1] == DIAG_MISSING_FEATURE
    ]
    assert len(failing) == 1, "the failing feature record must be present in the report"
    assert failing[0].outcome.status is ResolutionStatus.NOT_FOUND

    assert report.environment is not None
    snapshot = report.environment.snapshot
    assert snapshot is not None

    enabled_identity = PluginIdentity.from_class(SourceDiagEnabled722M)
    enabled_records = [record for record in snapshot.records if record.identity == enabled_identity]
    assert len(enabled_records) == 1  # premise guard: the enabled probe is in the environment
    assert enabled_records[0].provenance is EnvironmentProvenance.ACCESSIBLE

    disabled_identity = PluginIdentity.from_class(ProbeDiagDisabled722M)
    disabled_records = [record for record in snapshot.records if record.identity == disabled_identity]
    assert len(disabled_records) == 1, "the disabled probe must keep an environment record"
    assert disabled_records[0].provenance is EnvironmentProvenance.DISABLED_BY_COLLECTOR
