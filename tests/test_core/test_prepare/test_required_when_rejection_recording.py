"""A ``required_when`` non-match must be DIAGNOSABLE: it records its reason like the sibling presence rules.

``record_match_rejection`` is the seam the engine activates per candidate around its filter loop
(``IdentifyFeatureGroupClass._filter_feature_group_by_criteria``): a reason recorded while a candidate
matches becomes that candidate's ``value_rejection`` ``Elimination`` and reaches the resolution-failure
report. Two presence rules already record there, the name-path required-presence rule
(``FeatureChainParser._check_name_path_required_presence``) and the strict ``match_guard``
(``FeatureChainParserMixin``). ``feature_chain_author_guards.check_required_when`` only logs at debug and records
nothing, so a feature group that a declared ``required_when`` turned into a non-match is invisible in
the failure report. See ``test_first_pass_rejection_recording.py`` for the seam's full contract.

Names carry an ``rwrec`` suffix: a test feature group becomes a global subclass and the suite runs in
parallel, so a shared name would leak into another module's candidate universe. The group here is inert
for unrelated features: it matches only its own unique class name, and even then its ``required_when``
guard turns the match into a non-match unless its own unique option keys are present.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Optional

import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import MATCH_REJECTION_REASONS
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import FeatureResolutionError
from mloda.core.prepare.resolution_types import EvaluationResult
from mloda.provider import PropertySpec
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise


RWREC_REQUIRED_KEY = "rwrec_required_key"
RWREC_COMPANION_KEY = "rwrec_companion_key"


class RwrecFwOne(ComputeFramework):
    """Dummy compute framework for the required_when recording tests."""


class RwrecRequiredWhenFG(FeatureGroup):
    """Pattern-less group whose declared key is required exactly when its companion key is absent."""

    PROPERTY_MAPPING = {
        RWREC_REQUIRED_KEY: PropertySpec(
            "Required whenever the companion key is absent.",
            context=False,
            default=None,
            required_when=(lambda options: options.get(RWREC_COMPANION_KEY) is None),
        ),
        RWREC_COMPANION_KEY: PropertySpec("The alternative to the required key.", context=False, default=None),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


RWREC_FEATURE = RwrecRequiredWhenFG.get_class_name()


@pytest.fixture(autouse=True)
def _reset_recording_state() -> Iterator[None]:
    """Recorder state must not leak between tests, in either direction."""
    token = MATCH_REJECTION_REASONS.set(None)
    yield
    MATCH_REJECTION_REASONS.reset(token)


def _failed_result(feature: Feature, accessible_plugins: FeatureGroupEnvironmentMapping) -> EvaluationResult:
    """Run one engine attempt that must fail and return the structured result its single pass produced."""
    with pytest.raises(FeatureResolutionError) as exc_info:
        evaluate_or_raise(feature=feature, accessible_plugins=accessible_plugins, links=None)
    return exc_info.value.result


class TestRequiredWhenRejectionRecording:
    """A declared requirement that fires is reported as a near-miss, naming the key and its owner."""

    def test_required_when_non_match_is_reported_as_a_near_miss(self) -> None:
        """The failed resolution carries a value_rejection naming the missing key and the owning class."""
        feature = Feature(RWREC_FEATURE)
        accessible_plugins: FeatureGroupEnvironmentMapping = {RwrecRequiredWhenFG: {RwrecFwOne}}

        result = _failed_result(feature, accessible_plugins)

        elimination = result.eliminations.get(RwrecRequiredWhenFG)
        assert elimination is not None, (
            "the required_when non-match recorded no reason, so the failure report cannot explain it; "
            f"eliminations={result.eliminations}"
        )
        assert elimination.stage == "value_rejection"
        assert RWREC_REQUIRED_KEY in elimination.reason, elimination.reason
        assert RwrecRequiredWhenFG.__name__ in elimination.reason, elimination.reason

    def test_the_reason_is_recorded_under_the_owning_class_name(self) -> None:
        """With the recorder active, the non-match records exactly one reason keyed by the owning class."""
        token = MATCH_REJECTION_REASONS.set({})
        matched = RwrecRequiredWhenFG.match_feature_group_criteria(RWREC_FEATURE, Options())
        recorded = MATCH_REJECTION_REASONS.get()
        MATCH_REJECTION_REASONS.reset(token)

        assert matched is False
        assert recorded is not None
        assert list(recorded) == [RwrecRequiredWhenFG.__name__], recorded

    def test_a_satisfied_requirement_records_nothing(self) -> None:
        """Control: with the companion key present the predicate is unsatisfied, so nothing is recorded."""
        token = MATCH_REJECTION_REASONS.set({})
        matched = RwrecRequiredWhenFG.match_feature_group_criteria(
            RWREC_FEATURE, Options({RWREC_COMPANION_KEY: "rwrec_value"})
        )
        recorded = MATCH_REJECTION_REASONS.get()
        MATCH_REJECTION_REASONS.reset(token)

        assert matched is True
        assert recorded == {}

    def test_a_direct_match_outside_an_evaluation_records_nothing(self) -> None:
        """Activation contract: recording is inert outside the engine's per-candidate window."""
        assert RwrecRequiredWhenFG.match_feature_group_criteria(RWREC_FEATURE, Options()) is False
        assert MATCH_REJECTION_REASONS.get() is None
