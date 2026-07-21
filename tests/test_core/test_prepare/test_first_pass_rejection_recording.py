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
from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    MATCH_REJECTION_REASONS,
    record_match_rejection,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import property_spec
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import FeatureResolutionError, RenderFacts
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise


STRICT_FEATURE_OS005R = "strict_recording_feature_os005r"
MISSING_OPTION_FEATURE_OS005R = "missing_option_source_os005r__sum_os005rmiss"
GUARD_FEATURE_OS005R = "guard_recording_feature_os005r"
FACADE_FEATURE_OS005R = "facade_probe_feature_os005r"

STRICT_REJECTION_REASON_OS005R = "Property value '14' failed validation for 'window_size_os005r'"
MISSING_OPTION_REASON_OS005R = "required option(s) some_key_os005r are absent after declared defaults and name bindings"
GUARD_REJECTION_REASON_OS005R = "Property value 'ok_os005r' rejected by match_guard for 'guarded_key_os005r'"
FACADE_SENTINEL_REASON_OS005R = "facade sentinel reason os005r"

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

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class MissingOptionRecordingFGOs005r(FeatureChainParserMixin, FeatureGroup):
    """Name-path group whose required options-only key is absent: a MISSING-option rejection."""

    PREFIX_PATTERN = r".*__(?P<op_os005r>\w+)_os005rmiss$"
    PROPERTY_MAPPING = {
        "op_os005r": property_spec("operation carried by the name", context=True),
        "some_key_os005r": property_spec("required, options-only", context=True),
        DefaultOptionKeys.in_features: property_spec("source", context=True),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
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

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
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
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return False

    @classmethod
    def _strict_validation_rejection_reason(cls, feature_name: str | FeatureName, options: Options) -> str | None:
        FACADE_CALLS_OS005R.append(str(feature_name))
        if str(feature_name) == FACADE_FEATURE_OS005R:
            return FACADE_SENTINEL_REASON_OS005R
        return None

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


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


def _failed_facts(feature: Feature, accessible_plugins: FeatureGroupEnvironmentMapping) -> RenderFacts:
    """Run one engine attempt that must fail and return the facts its single pass captured."""
    with pytest.raises(FeatureResolutionError) as exc_info:
        evaluate_or_raise(feature=feature, accessible_plugins=accessible_plugins, links=None)
    return exc_info.value.result.facts


class TestFirstPassRejectionRecording:
    """The failure facts carry the rejections the real match pass produced, without a replay."""

    def test_strict_value_rejection_is_reported_and_the_validator_runs_once(self) -> None:
        """One failed resolution judges the bad value exactly once: match only, no diagnosis replay."""
        feature = Feature(
            STRICT_FEATURE_OS005R,
            Options(context={DefaultOptionKeys.in_features: "src", "window_size_os005r": 14}),
        )
        accessible_plugins: FeatureGroupEnvironmentMapping = {StrictRecordingFGOs005r: {RecorderFwOneOs005r}}

        facts = _failed_facts(feature, accessible_plugins)

        assert facts.value_rejections == (("StrictRecordingFGOs005r", STRICT_REJECTION_REASON_OS005R),)
        assert VALIDATOR_CALLS_OS005R == [14]

    def test_missing_required_option_reason_is_reported_from_the_first_pass(self) -> None:
        """The name-path presence non-match records its reason as it happens."""
        feature = Feature(MISSING_OPTION_FEATURE_OS005R)
        accessible_plugins: FeatureGroupEnvironmentMapping = {MissingOptionRecordingFGOs005r: {RecorderFwOneOs005r}}

        facts = _failed_facts(feature, accessible_plugins)

        assert facts.value_rejections == (("MissingOptionRecordingFGOs005r", MISSING_OPTION_REASON_OS005R),)

    def test_strict_match_guard_rejection_is_reported_from_the_first_pass(self) -> None:
        """A guard rejection on a strict spec records the same message the facade produces."""
        feature = Feature(GUARD_FEATURE_OS005R, Options(context={"guarded_key_os005r": "ok_os005r"}))
        accessible_plugins: FeatureGroupEnvironmentMapping = {GuardRecordingFGOs005r: {RecorderFwOneOs005r}}

        facts = _failed_facts(feature, accessible_plugins)

        assert facts.value_rejections == (("GuardRecordingFGOs005r", GUARD_REJECTION_REASON_OS005R),)


class TestEngineNeverCallsTheFacade:
    """``_strict_validation_rejection_reason`` stays a standalone diagnostic; the engine never calls it."""

    def test_the_engine_never_calls_the_rejection_facade(self) -> None:
        """A failed resolution neither calls the facade nor lets its sentinel reach the facts."""
        feature = Feature(FACADE_FEATURE_OS005R)
        accessible_plugins: FeatureGroupEnvironmentMapping = {FacadeProbeFGOs005r: {RecorderFwOneOs005r}}

        facts = _failed_facts(feature, accessible_plugins)

        assert FACADE_CALLS_OS005R == []
        assert facts.value_rejections == ()

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

        assert recorded == {"FirstWinsOwnerOs005r": "first reason os005r"}
