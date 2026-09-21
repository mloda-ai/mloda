"""The MIN/MAX_IN_FEATURES gate counts the sources the feature name carries (#944).

When the name identifies the group, input_features splits the name and never reads the
in_features option, so the gate must count the same sources the name path would use.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.match_rejection import MATCH_REJECTION_REASONS, MatchRejection
from mloda.provider import DefaultOptionKeys, PropertySpec
from mloda.user import Feature, Options

# A dict is uncountable for get_in_features: it raises TypeError instead of yielding features.
JUNK_IN_FEATURES: dict[str, int] = {"a": 1}


class _NameSourceGate944(FeatureChainParserMixin):
    """Two to three in_features, with the sources readable from the name."""

    PREFIX_PATTERN = r".*__([\w]+)_gate944$"
    MIN_IN_FEATURES = 2
    MAX_IN_FEATURES = 3
    PROPERTY_MAPPING = {
        "operation": PropertySpec(
            "Operation to apply",
            allowed_values={"op1": "Operation 1"},
            context=True,
            strict_validation=True,
        )
    }


def _options(in_features: Any = None) -> Options:
    context: dict[str, Any] = {"operation": "op1"}
    if in_features is not None:
        context[DefaultOptionKeys.in_features] = in_features
    return Options(context=context)


class TestNameSourcesDriveTheGate:
    """The name path counts the name's own sources, not the option value."""

    def test_junk_in_features_option_does_not_reject_a_name_carried_source_count(self) -> None:
        """The name carries two sources, inside MIN=2 / MAX=3; the uncountable option is not consulted."""
        result = _NameSourceGate944.match_feature_group_criteria("f1&f2__op1_gate944", _options(JUNK_IN_FEATURES))

        assert result is True

    def test_name_source_count_below_min_is_a_non_match(self) -> None:
        """One name source is below MIN=2, even though the option value would have passed the gate."""
        result = _NameSourceGate944.match_feature_group_criteria("f1__op1_gate944", _options(["a", "b"]))

        assert result is False

    def test_name_source_count_above_max_is_a_non_match(self) -> None:
        """Four name sources exceed MAX=3, even though the option value would have passed the gate."""
        result = _NameSourceGate944.match_feature_group_criteria("f1&f2&f3&f4__op1_gate944", _options(["a", "b"]))

        assert result is False

    def test_name_source_count_below_min_without_any_option_is_a_non_match(self) -> None:
        """No in_features option at all: the name's single source still fails MIN=2."""
        result = _NameSourceGate944.match_feature_group_criteria("f1__op1_gate944", _options())

        assert result is False

    def test_name_source_count_inside_range_still_matches(self) -> None:
        """Regression pin: a name-carried count inside MIN/MAX matches with no option present."""
        result = _NameSourceGate944.match_feature_group_criteria("f1&f2&f3__op1_gate944", _options())

        assert result is True


class TestOptionPathGateUnchanged:
    """Without a name that identifies the group, the option value is still what gets counted."""

    def test_uncountable_option_value_is_still_a_non_match(self) -> None:
        result = _NameSourceGate944.match_feature_group_criteria("any_name", _options(JUNK_IN_FEATURES))

        assert result is False

    def test_option_count_below_min_is_still_a_non_match(self) -> None:
        result = _NameSourceGate944.match_feature_group_criteria("any_name", _options("single_feature"))

        assert result is False

    def test_option_count_above_max_is_still_a_non_match(self) -> None:
        in_features = frozenset({Feature("a"), Feature("b"), Feature("c"), Feature("d")})
        result = _NameSourceGate944.match_feature_group_criteria("any_name", _options(in_features))

        assert result is False

    def test_option_count_inside_range_still_matches(self) -> None:
        in_features = frozenset({Feature("a"), Feature("b")})
        result = _NameSourceGate944.match_feature_group_criteria("any_name", _options(in_features))

        assert result is True


OWNER_944 = "_NameSourceGate944"
BELOW_MIN_NAME_944 = "f1__op1_gate944"
ABOVE_MAX_NAME_944 = "f1&f2&f3&f4__op1_gate944"
BELOW_MIN_REASON_944 = f"Feature '{BELOW_MIN_NAME_944}' requires at least 2 in_feature(s), but found 1"
ABOVE_MAX_REASON_944 = f"Feature '{ABOVE_MAX_NAME_944}' allows at most 3 in_feature(s), but found 4"


@pytest.fixture
def rejection_window() -> Iterator[dict[str, MatchRejection]]:
    """Open a per-test recording window and always close it again."""
    reasons: dict[str, MatchRejection] = {}
    token = MATCH_REJECTION_REASONS.set(reasons)
    yield reasons
    MATCH_REJECTION_REASONS.reset(token)


class TestNameSourceCountRejectionIsRecorded:
    """A name-carried count outside MIN/MAX is a reportable near-miss, not a silent non-match."""

    def test_below_min_records_the_actionable_reason(self, rejection_window: dict[str, MatchRejection]) -> None:
        """The reason keeps the pre-gate wording: the declared MIN and the count the name carries."""
        result = _NameSourceGate944.match_feature_group_criteria(BELOW_MIN_NAME_944, _options())

        assert result is False
        assert rejection_window == {OWNER_944: MatchRejection(reason=BELOW_MIN_REASON_944, stage="value_rejection")}

    def test_above_max_records_the_actionable_reason(self, rejection_window: dict[str, MatchRejection]) -> None:
        """The reason keeps the pre-gate wording: the declared MAX and the count the name carries."""
        result = _NameSourceGate944.match_feature_group_criteria(ABOVE_MAX_NAME_944, _options())

        assert result is False
        assert rejection_window == {OWNER_944: MatchRejection(reason=ABOVE_MAX_REASON_944, stage="value_rejection")}

    def test_below_min_records_the_name_count_even_with_a_passing_option(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """The option value would pass the gate, so the reason must report the name's count, not the option's."""
        _NameSourceGate944.match_feature_group_criteria(BELOW_MIN_NAME_944, _options(["a", "b"]))

        assert rejection_window == {OWNER_944: MatchRejection(reason=BELOW_MIN_REASON_944, stage="value_rejection")}

    def test_a_count_inside_the_range_records_nothing(self, rejection_window: dict[str, MatchRejection]) -> None:
        result = _NameSourceGate944.match_feature_group_criteria("f1&f2&f3__op1_gate944", _options())

        assert result is True
        assert rejection_window == {}

    def test_the_option_path_still_records_nothing(self, rejection_window: dict[str, MatchRejection]) -> None:
        """Pinned contrast: only the name path reports; the option path stays a silent non-match."""
        result = _NameSourceGate944.match_feature_group_criteria("any_name", _options("single_feature"))

        assert result is False
        assert rejection_window == {}


class _OptionsOnlyGroupM951(FeatureChainParserMixin):
    """Options-only group: one declared non-in_features key, no in_features key, default MIN_IN_FEATURES."""

    PROPERTY_MAPPING = {"mode_m951": PropertySpec("Mode", allowed_values={"fast_m951": "Fast"}, context=True)}


class _AllOptionalGroupM951(FeatureChainParserMixin):
    """All-optional mapping, so it matches options that never mention its keys."""

    PROPERTY_MAPPING = {"tuning_m951": PropertySpec("Tuning", default=None, context=True)}


class TestOptionPathZeroSourceRecordsNoRejection:
    """The option path records no rejection reason for a zero-source non-match, whatever the options address."""

    def test_addressed_group_without_in_features_does_not_match_and_records_nothing(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """The option path keeps its "not mine" meaning; the definition-time warning is the mitigation."""
        options = Options(context={"mode_m951": "fast_m951"})

        result = _OptionsOnlyGroupM951.match_feature_group_criteria("any_name_m951", options)

        assert result is False
        assert rejection_window == {}

    def test_options_addressing_none_of_the_group_record_nothing(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        result = _AllOptionalGroupM951.match_feature_group_criteria("any_name_m951", Options())

        assert result is False
        assert rejection_window == {}

    def test_unrelated_options_record_nothing(self, rejection_window: dict[str, MatchRejection]) -> None:
        options = Options(context={"unrelated_key_m951": "x"})

        result = _AllOptionalGroupM951.match_feature_group_criteria("any_name_m951", options)

        assert result is False
        assert rejection_window == {}

    def test_supplied_in_features_still_matches_and_records_nothing(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        options = Options(context={"mode_m951": "fast_m951", DefaultOptionKeys.in_features: "src_m951"})

        result = _OptionsOnlyGroupM951.match_feature_group_criteria("any_name_m951", options)

        assert result is True
        assert rejection_window == {}
