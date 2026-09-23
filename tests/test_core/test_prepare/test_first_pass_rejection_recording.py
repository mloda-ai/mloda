"""First-pass rejection recording for feature resolution failures (board issue os-005, mloda#798).

Matching RECORDS each rejection as it happens: ``record_match_rejection`` writes into the
``MATCH_REJECTION_REASONS`` recorder the engine activates around its filter loop, the first reason
per owner wins, and the failure facts render from that recording. The engine never replays diagnosis
through ``_strict_validation_rejection_reason``; that method stays a standalone diagnostic facade.

All names carry an ``os005r`` suffix: test feature groups become global subclasses and the suite runs
in parallel, so a shared name would leak into another module's candidate universe. Every group here is
inert for unrelated features (it matches only its own unique name or option keys), so no disarm
fixture is needed.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from typing import Any, cast

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.match_rejection import (
    MATCH_REJECTION_REASONS,
    MatchRejection,
    record_match_rejection,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.components.property_spec import property_spec
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import FeatureResolutionError
from mloda.core.prepare.resolution_types import Elimination, EvaluationResult
from mloda.user import mlodaAPI
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise


STRICT_FEATURE_OS005R = "strict_recording_feature_os005r"
MISSING_OPTION_FEATURE_OS005R = "missing_option_source_os005r__sum_os005rmiss"
GUARD_FEATURE_OS005R = "guard_recording_feature_os005r"
FACADE_FEATURE_OS005R = "facade_probe_feature_os005r"

STRICT_REJECTION_REASON_OS005R = "Property value '14' failed validation for 'window_size_os005r'"
MISSING_OPTION_REASON_OS005R = "required option(s) some_key_os005r are absent after declared defaults and name bindings"
GUARD_REJECTION_REASON_OS005R = "Property value 'ok_os005r' rejected by match_guard for 'guarded_key_os005r'"
FACADE_SENTINEL_REASON_OS005R = "facade sentinel reason os005r"

# The name-collision groups carry an os005c suffix of their own so their names and option keys stay
# unique to this test; the shared class NAME is the point of the collision scenario.
COLLIDING_FEATURE_OS005C = "colliding_recording_feature_os005c"
COLLIDING_NAME_OS005C = "CollidingRejectFGOs005c"
COLLIDING_MODULE_A_OS005C = "tests.colliding_reject_module_a_os005c"
COLLIDING_MODULE_B_OS005C = "tests.colliding_reject_module_b_os005c"
COLLIDE_A_REASON_OS005C = "Property value 'bogus_os005c' not found in mapping for 'collide_a_os005c'"
COLLIDE_B_REASON_OS005C = "Property value 'bogus_os005c' not found in mapping for 'collide_b_os005c'"

# Every value the strict element_validator judged, across the WHOLE failed resolution. Reset per test.
VALIDATOR_CALLS_OS005R: list[Any] = []

# Every feature name the facade override was asked about. Reset per test.
FACADE_CALLS_OS005R: list[str] = []


def _counting_window_validator_os005r(value: Any) -> bool:
    """Count one judgment, then accept ints in (0, 13]."""
    VALIDATOR_CALLS_OS005R.append(value)
    return isinstance(value, int) and 0 < value <= 13


class RecorderFwOneOs005r(ComputeFramework):
    """Dummy compute framework for the recording tests."""


class StrictRecordingFGOs005r(FeatureChainParserMixin, FeatureGroup):
    """Config-path group whose strict validator counts every call it receives."""

    PROPERTY_MAPPING = {
        "window_size_os005r": property_spec(
            "Size of window",
            strict=True,
            context=False,
            element_validator=_counting_window_validator_os005r,
        ),
        DefaultOptionKeys.in_features: property_spec("source", context=True),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class MissingOptionRecordingFGOs005r(FeatureChainParserMixin, FeatureGroup):
    """Name-path group whose required options-only key is absent: a MISSING-option rejection."""

    PREFIX_PATTERN = r".*__(?P<op_os005r>\w+)_os005rmiss$"
    PROPERTY_MAPPING = {
        "op_os005r": property_spec("operation carried by the name", context=True),
        "some_key_os005r": property_spec("required, options-only", context=True),
        DefaultOptionKeys.in_features: property_spec("source", context=True),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class GuardRecordingFGOs005r(FeatureChainParserMixin, FeatureGroup):
    """Group whose strict spec carries a match_guard that rejects every value."""

    PROPERTY_MAPPING = {
        "guarded_key_os005r": property_spec(
            "strict key whose guard rejects every value",
            strict=True,
            allowed_values=("ok_os005r",),
            match_guard=lambda _value: False,
        ),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class FacadeProbeFGOs005r(FeatureGroup):
    """Never-matching candidate whose rejection facade records every call and returns a sentinel.

    The sentinel is gated on this group's own feature name, so a class leaked into another test's
    universe injects nothing there.
    """

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return False

    @classmethod
    def _strict_validation_rejection_reason(cls, feature_name: str | FeatureName, options: Options) -> str | None:
        FACADE_CALLS_OS005R.append(str(feature_name))
        if str(feature_name) == FACADE_FEATURE_OS005R:
            return FACADE_SENTINEL_REASON_OS005R
        return None

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


def _is_int_geq1_mge(value: Any) -> bool:
    """Accepts only a real ``int`` (not ``bool``) of 1 or more."""
    if isinstance(value, bool):
        return False
    return isinstance(value, int) and value >= 1


EXPECTED_STR_FEATURE_MGE = "expected_guard_str_mge"
EXPECTED_LIST_FEATURE_MGE = "expected_guard_list_mge"
EXPECTED_NONE_FEATURE_MGE = "expected_guard_none_mge"
EXPECTED_STRICT_FEATURE_MGE = "expected_guard_strict_mge"
NO_EXPECTED_FEATURE_MGE = "no_expected_guard_mge"
EXPECTED_E2E_FEATURE_MGE = "expected_guard_e2e_mge"

EXPECTED_STR_REASON_MGE = "option 'concurrency_mge' must be a whole number of 1 or more, got str '4'"
EXPECTED_LIST_REASON_MGE = "option 'concurrency_mge' must be a whole number of 1 or more, got list"
EXPECTED_NONE_REASON_MGE = "option 'concurrency_none_mge' must be a whole number of 1 or more, got None"
EXPECTED_STRICT_REASON_MGE = "option 'concurrency_strict_mge' must be a whole number of 1 or more, got str '4'"


class ExpectedGuardFGMge(FeatureChainParserMixin, FeatureGroup):
    """Non-strict spec with ``expected``: a guard rejection is still reportable."""

    PROPERTY_MAPPING = {
        "concurrency_mge": property_spec(
            "concurrency count",
            match_guard=_is_int_geq1_mge,
            expected="a whole number of 1 or more",
        ),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class ExpectedGuardNoneFGMge(FeatureChainParserMixin, FeatureGroup):
    """Non-strict spec with ``expected`` and ``allow_explicit_none``: an explicit None reaches the guard."""

    PROPERTY_MAPPING = {
        "concurrency_none_mge": property_spec(
            "concurrency count",
            match_guard=_is_int_geq1_mge,
            expected="a whole number of 1 or more",
            allow_explicit_none=True,
        ),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class ExpectedGuardStrictFGMge(FeatureChainParserMixin, FeatureGroup):
    """Strict spec with ``expected``: the strict path uses the new expected-based text too."""

    PROPERTY_MAPPING = {
        "concurrency_strict_mge": property_spec(
            "concurrency count",
            strict=True,
            allowed_values=("4",),
            match_guard=_is_int_geq1_mge,
            expected="a whole number of 1 or more",
        ),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class NoExpectedGuardFGMge(FeatureChainParserMixin, FeatureGroup):
    """Non-strict spec with no ``expected``: a guard rejection stays silent, as today."""

    PROPERTY_MAPPING = {
        "concurrency_bare_mge": property_spec(
            "concurrency count",
            match_guard=_is_int_geq1_mge,
        ),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class ExpectedGuardEndToEndFGMge(FeatureChainParserMixin, FeatureGroup):
    """End-to-end counterpart of ``ExpectedGuardFGMge``, run through ``mlodaAPI.run_all``."""

    PROPERTY_MAPPING = {
        "concurrency_e2e_mge": property_spec(
            "concurrency count",
            match_guard=_is_int_geq1_mge,
            expected="a whole number of 1 or more",
        ),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


def _build_colliding_rejection_groups_os005c() -> tuple[type[FeatureGroup], type[FeatureGroup]]:
    """Build two same-named candidates across modules, each strictly rejecting its OWN option key.

    Sharing a ``__name__`` is a supported scenario, so a recording keyed by class name would collapse
    the two into one slot. Each group's strict key is unique (``collide_a_os005c`` / ``collide_b_os005c``),
    so the same failed feature yields a DIFFERENT rejection reason per class. Leaked into another test's
    universe the groups are inert: each matches only when its own required collide key is present.
    """

    def make(module: str, key: str) -> type[FeatureGroup]:
        def input_features(self: Any, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return None

        namespace: dict[str, Any] = {
            "__module__": module,
            "__doc__": "Same-named candidate whose strict key rejects the provided value.",
            "PROPERTY_MAPPING": {
                key: property_spec(
                    "strict key accepting only its ok sentinel",
                    strict=True,
                    allowed_values=("ok_os005c",),
                    context=False,
                ),
                DefaultOptionKeys.in_features: property_spec("source", context=True),
            },
            "input_features": input_features,
        }
        created: Any = type(COLLIDING_NAME_OS005C, (FeatureChainParserMixin, FeatureGroup), namespace)
        return cast(type[FeatureGroup], created)

    return (
        make(COLLIDING_MODULE_A_OS005C, "collide_a_os005c"),
        make(COLLIDING_MODULE_B_OS005C, "collide_b_os005c"),
    )


@pytest.fixture(autouse=True)
def _reset_recording_state() -> Iterator[None]:
    """Recorder and counter state must not leak between tests, in either direction."""
    token = MATCH_REJECTION_REASONS.set(None)
    VALIDATOR_CALLS_OS005R.clear()
    FACADE_CALLS_OS005R.clear()
    yield
    MATCH_REJECTION_REASONS.reset(token)
    VALIDATOR_CALLS_OS005R.clear()
    FACADE_CALLS_OS005R.clear()


def _failed_result(feature: Feature, accessible_plugins: FeatureGroupEnvironmentMapping) -> EvaluationResult:
    """Run one engine attempt that must fail and return the structured result its single pass produced."""
    with pytest.raises(FeatureResolutionError) as exc_info:
        evaluate_or_raise(feature=feature, accessible_plugins=accessible_plugins, links=None)
    return exc_info.value.result


class TestFirstPassRejectionRecording:
    """The failure eliminations carry the rejections the real match pass produced, without a replay."""

    def test_strict_value_rejection_is_reported_and_the_validator_runs_once(self) -> None:
        """One failed resolution judges the bad value exactly once: match only, no diagnosis replay."""
        feature = Feature(
            STRICT_FEATURE_OS005R,
            Options(context={DefaultOptionKeys.in_features: "src", "window_size_os005r": 14}),
        )
        accessible_plugins: FeatureGroupEnvironmentMapping = {StrictRecordingFGOs005r: {RecorderFwOneOs005r}}

        result = _failed_result(feature, accessible_plugins)

        assert result.eliminations == {
            StrictRecordingFGOs005r: Elimination(stage="value_rejection", reason=STRICT_REJECTION_REASON_OS005R)
        }
        assert VALIDATOR_CALLS_OS005R == [14]

    def test_missing_required_option_reason_is_reported_from_the_first_pass(self) -> None:
        """The name-path presence non-match records its reason as it happens."""
        feature = Feature(MISSING_OPTION_FEATURE_OS005R)
        accessible_plugins: FeatureGroupEnvironmentMapping = {MissingOptionRecordingFGOs005r: {RecorderFwOneOs005r}}

        result = _failed_result(feature, accessible_plugins)

        assert result.eliminations == {
            MissingOptionRecordingFGOs005r: Elimination(stage="value_rejection", reason=MISSING_OPTION_REASON_OS005R)
        }

    def test_strict_match_guard_rejection_is_reported_from_the_first_pass(self) -> None:
        """A guard rejection on a strict spec records the same message the facade produces."""
        feature = Feature(GUARD_FEATURE_OS005R, Options(context={"guarded_key_os005r": "ok_os005r"}))
        accessible_plugins: FeatureGroupEnvironmentMapping = {GuardRecordingFGOs005r: {RecorderFwOneOs005r}}

        result = _failed_result(feature, accessible_plugins)

        assert result.eliminations == {
            GuardRecordingFGOs005r: Elimination(stage="value_rejection", reason=GUARD_REJECTION_REASON_OS005R)
        }

    def test_same_named_candidates_each_keep_their_own_recorded_reason(self) -> None:
        """Two candidates sharing a __name__ each report the reason their OWN match produced.

        A recording keyed by class name collapses both classes into one first-wins slot: the second
        class's distinct reason is dropped and the shared reason is attributed to both candidates.
        Keyed by the class OBJECT, the eliminations carry two entries with the same name and two reasons.
        """
        group_a, group_b = _build_colliding_rejection_groups_os005c()
        feature = Feature(
            COLLIDING_FEATURE_OS005C,
            Options(
                context={
                    DefaultOptionKeys.in_features: "src",
                    "collide_a_os005c": "bogus_os005c",
                    "collide_b_os005c": "bogus_os005c",
                }
            ),
        )
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            group_a: {RecorderFwOneOs005r},
            group_b: {RecorderFwOneOs005r},
        }

        result = _failed_result(feature, accessible_plugins)

        assert result.eliminations == {
            group_a: Elimination(stage="value_rejection", reason=COLLIDE_A_REASON_OS005C),
            group_b: Elimination(stage="value_rejection", reason=COLLIDE_B_REASON_OS005C),
        }


class TestExpectedGuardRejectionRecording:
    """``expected`` names what the guard accepts; the recorded reason uses that text."""

    def test_non_strict_expected_str_scalar_is_reported(self) -> None:
        """A non-strict spec with ``expected`` still records: str '4' names its own type and repr."""
        feature = Feature(EXPECTED_STR_FEATURE_MGE, Options(context={"concurrency_mge": "4"}))
        accessible_plugins: FeatureGroupEnvironmentMapping = {ExpectedGuardFGMge: {RecorderFwOneOs005r}}

        result = _failed_result(feature, accessible_plugins)

        assert result.eliminations == {
            ExpectedGuardFGMge: Elimination(stage="value_rejection", reason=EXPECTED_STR_REASON_MGE)
        }

    def test_non_strict_expected_list_value_carries_no_value_text(self) -> None:
        """A list value names only its type: no value text is echoed."""
        feature = Feature(EXPECTED_LIST_FEATURE_MGE, Options(context={"concurrency_mge": [1, 2]}))
        accessible_plugins: FeatureGroupEnvironmentMapping = {ExpectedGuardFGMge: {RecorderFwOneOs005r}}

        result = _failed_result(feature, accessible_plugins)

        assert result.eliminations == {
            ExpectedGuardFGMge: Elimination(stage="value_rejection", reason=EXPECTED_LIST_REASON_MGE)
        }

    def test_explicit_none_with_allow_explicit_none_is_reported(self) -> None:
        """An opted-in explicit ``None`` reaches the guard and is reported as 'got None'."""
        feature = Feature(EXPECTED_NONE_FEATURE_MGE, Options(context={"concurrency_none_mge": None}))
        accessible_plugins: FeatureGroupEnvironmentMapping = {ExpectedGuardNoneFGMge: {RecorderFwOneOs005r}}

        result = _failed_result(feature, accessible_plugins)

        assert result.eliminations == {
            ExpectedGuardNoneFGMge: Elimination(stage="value_rejection", reason=EXPECTED_NONE_REASON_MGE)
        }

    def test_strict_expected_uses_the_new_expected_text(self) -> None:
        """A strict spec with ``expected`` uses the new text too, not the old strict-only wording."""
        feature = Feature(EXPECTED_STRICT_FEATURE_MGE, Options(context={"concurrency_strict_mge": "4"}))
        accessible_plugins: FeatureGroupEnvironmentMapping = {ExpectedGuardStrictFGMge: {RecorderFwOneOs005r}}

        result = _failed_result(feature, accessible_plugins)

        assert result.eliminations == {
            ExpectedGuardStrictFGMge: Elimination(stage="value_rejection", reason=EXPECTED_STRICT_REASON_MGE)
        }

    def test_non_strict_without_expected_stays_silent(self) -> None:
        """A non-strict guard rejection with no ``expected`` records nothing, exactly as today."""
        feature = Feature(NO_EXPECTED_FEATURE_MGE, Options(context={"concurrency_bare_mge": "4"}))
        accessible_plugins: FeatureGroupEnvironmentMapping = {NoExpectedGuardFGMge: {RecorderFwOneOs005r}}

        result = _failed_result(feature, accessible_plugins)

        assert result.eliminations == {}

    def test_facade_parity_for_str_scalar(self) -> None:
        """The facade returns the identical reason for the str-scalar case."""
        options = Options(context={"concurrency_mge": "4"})

        assert ExpectedGuardFGMge._strict_validation_rejection_reason(EXPECTED_STR_FEATURE_MGE, options) == (
            EXPECTED_STR_REASON_MGE
        )

    def test_facade_parity_for_list_value(self) -> None:
        """The facade returns the identical reason for the list-value case."""
        options = Options(context={"concurrency_mge": [1, 2]})

        assert ExpectedGuardFGMge._strict_validation_rejection_reason(EXPECTED_LIST_FEATURE_MGE, options) == (
            EXPECTED_LIST_REASON_MGE
        )

    def test_facade_parity_for_explicit_none(self) -> None:
        """The facade returns the identical reason for the explicit-None case."""
        options = Options(context={"concurrency_none_mge": None})

        assert ExpectedGuardNoneFGMge._strict_validation_rejection_reason(EXPECTED_NONE_FEATURE_MGE, options) == (
            EXPECTED_NONE_REASON_MGE
        )

    def test_facade_parity_for_strict_expected(self) -> None:
        """The facade returns the identical reason for the strict-expected case."""
        options = Options(context={"concurrency_strict_mge": "4"})

        assert ExpectedGuardStrictFGMge._strict_validation_rejection_reason(EXPECTED_STRICT_FEATURE_MGE, options) == (
            EXPECTED_STRICT_REASON_MGE
        )

    def test_facade_parity_for_non_strict_without_expected_is_none(self) -> None:
        """The facade returns ``None`` for the non-strict, no-``expected`` case, matching the match pass."""
        options = Options(context={"concurrency_bare_mge": "4"})

        assert NoExpectedGuardFGMge._strict_validation_rejection_reason(NO_EXPECTED_FEATURE_MGE, options) is None

    def test_end_to_end_expected_guard_rejection_near_miss_line(self) -> None:
        """``mlodaAPI.run_all`` surfaces the exact near-miss line for an ``expected`` guard rejection."""
        with pytest.raises(FeatureResolutionError) as exc_info:
            mlodaAPI.run_all(
                [Feature(EXPECTED_E2E_FEATURE_MGE, Options(context={"concurrency_e2e_mge": "4"}))],
                compute_frameworks={RecorderFwOneOs005r},
                plugin_collector=PluginCollector.enabled_feature_groups({ExpectedGuardEndToEndFGMge}),
            )

        message = str(exc_info.value)
        assert (
            "  - ExpectedGuardEndToEndFGMge (option value): option 'concurrency_e2e_mge' must be "
            "a whole number of 1 or more, got str '4'"
        ) in message


class TestEngineNeverCallsTheFacade:
    """``_strict_validation_rejection_reason`` stays a standalone diagnostic; the engine never calls it."""

    def test_the_engine_never_calls_the_rejection_facade(self) -> None:
        """A failed resolution neither calls the facade nor lets its sentinel reach the facts."""
        feature = Feature(FACADE_FEATURE_OS005R)
        accessible_plugins: FeatureGroupEnvironmentMapping = {FacadeProbeFGOs005r: {RecorderFwOneOs005r}}

        result = _failed_result(feature, accessible_plugins)

        assert FACADE_CALLS_OS005R == []
        assert result.eliminations == {}

    def test_the_facade_still_answers_standalone_calls(self) -> None:
        """Called directly, the facade keeps working: many tests use it as a diagnostic."""
        reason = FacadeProbeFGOs005r._strict_validation_rejection_reason(FACADE_FEATURE_OS005R, Options())

        assert reason == FACADE_SENTINEL_REASON_OS005R
        assert FACADE_CALLS_OS005R == [FACADE_FEATURE_OS005R]


class TestRecorderActivation:
    """The recorder is inactive by default; only the engine's activation makes recording effective."""

    def test_recorder_defaults_to_inactive(self) -> None:
        """In a fresh context the recorder holds None: recording is off."""
        assert contextvars.Context().run(MATCH_REJECTION_REASONS.get) is None

    def test_record_match_rejection_is_a_no_op_while_inactive(self) -> None:
        """Recording outside an active evaluation changes nothing."""
        record_match_rejection("InertOwnerOs005r", "inert reason os005r")

        assert MATCH_REJECTION_REASONS.get() is None

    def test_direct_matcher_call_outside_the_engine_records_nothing(self) -> None:
        """A rejecting match outside the engine stays side-effect free."""
        options = Options(context={DefaultOptionKeys.in_features: "src", "window_size_os005r": 14})

        assert StrictRecordingFGOs005r.match_feature_group_criteria(STRICT_FEATURE_OS005R, options) is False
        assert MATCH_REJECTION_REASONS.get() is None

    def test_the_first_recorded_reason_per_owner_wins(self) -> None:
        """With the recorder active, a second reason for the same owner never overwrites the first."""
        token = MATCH_REJECTION_REASONS.set({})
        record_match_rejection("FirstWinsOwnerOs005r", "first reason os005r")
        record_match_rejection("FirstWinsOwnerOs005r", "second reason os005r")
        recorded = MATCH_REJECTION_REASONS.get()
        MATCH_REJECTION_REASONS.reset(token)

        assert recorded == {"FirstWinsOwnerOs005r": MatchRejection(reason="first reason os005r")}
