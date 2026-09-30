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
from mloda.core.abstract_plugins.components.match_rejection import MATCH_REJECTION_REASONS, NAME_STAGE, MatchRejection
from mloda.provider import DefaultOptionKeys, PropertySpec
from mloda.user import Feature, FeatureName, Options
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup

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


class _NameSourceGuardGate944(FeatureChainParserMixin):
    """Second key's guard rejects every value; an out-of-range count must win over the guard's own reason."""

    PREFIX_PATTERN = r".*__([\w]+)_guardgate944$"
    MIN_IN_FEATURES = 2
    MAX_IN_FEATURES = 3
    PROPERTY_MAPPING = {
        "operation": PropertySpec(
            "Operation to apply",
            allowed_values={"op1": "Operation 1"},
            context=True,
            strict_validation=True,
        ),
        "guarded_key_gate944": PropertySpec(
            "guarded key whose guard rejects every value",
            allowed_values=("ok_gate944",),
            strict_validation=True,
            match_guard=lambda _value: False,
        ),
    }


def _options(in_features: Any = None) -> Options:
    context: dict[str, Any] = {"operation": "op1"}
    if in_features is not None:
        context[DefaultOptionKeys.in_features] = in_features
    return Options(context=context)


class TestNameSourcesDriveTheGate:
    """The name path counts the name's own sources, not the option value."""

    def test_junk_in_features_option_does_not_reject_a_name_carried_source_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The name carries two sources, inside MIN=2 / MAX=3; the gate reads the name, not the option.

        The env var downgrades the contradiction error so only the count gate is under test.
        """
        monkeypatch.setenv("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "1")

        result = _NameSourceGate944.match_feature_group_criteria("f1&f2__op1_gate944", _options(JUNK_IN_FEATURES))

        assert result is True

    def test_junk_in_features_option_aborts_a_name_match(self) -> None:
        """An in_features value the matcher cannot resolve contradicts the name."""
        with pytest.raises(ValueError) as exc_info:
            _NameSourceGate944.match_feature_group_criteria("f1&f2__op1_gate944", _options(JUNK_IN_FEATURES))

        assert "in_features" in str(exc_info.value)

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


class TestDeclaredInFeaturesAgreeWithName:
    """A declared in_features must list exactly the name's direct sources."""

    NAME = "f1&f2__op1_gate944"

    def test_equal_list_matches(self) -> None:
        assert _NameSourceGate944.match_feature_group_criteria(self.NAME, _options(["f1", "f2"])) is True

    def test_reordered_list_aborts(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _NameSourceGate944.match_feature_group_criteria(self.NAME, _options(["f2", "f1"]))

        message = str(exc_info.value)
        assert "in_features" in message
        assert "f1" in message
        assert "f2" in message
        assert "direct source" in message

    def test_message_does_not_carry_a_long_raw_value(self) -> None:
        long_value = "n" * 200

        with pytest.raises(ValueError) as exc_info:
            _NameSourceGate944.match_feature_group_criteria(self.NAME, _options([long_value, "b"]))

        assert long_value not in str(exc_info.value)

    def test_message_omits_the_root_source_hint_for_a_flat_name(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _NameSourceGate944.match_feature_group_criteria(self.NAME, _options(["p1", "p2"]))

        assert "root source" not in str(exc_info.value)

    def test_message_says_the_root_source_when_the_name_sources_are_chained(self) -> None:
        options = Options(context={DefaultOptionKeys.in_features: ["s"]})

        with pytest.raises(ValueError) as exc_info:
            AggregatedFeatureGroup.match_feature_group_criteria("s__mean_imputed__sum_aggr", options)

        assert "not the root source" in str(exc_info.value)

    def test_message_says_order_matters_for_the_same_names_reordered(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _NameSourceGate944.match_feature_group_criteria(self.NAME, _options(["f2", "f1"]))

        assert "order" in str(exc_info.value).lower()

    @pytest.mark.parametrize("in_features", [frozenset({"f1", "f2"}), {"f2", "f1"}])
    def test_set_in_any_order_matches(self, in_features: Any) -> None:
        assert _NameSourceGate944.match_feature_group_criteria(self.NAME, _options(in_features)) is True

    def test_feature_objects_compare_by_name(self) -> None:
        assert _NameSourceGate944.match_feature_group_criteria(self.NAME, _options([Feature("f1"), Feature("f2")]))

    def test_different_names_abort(self) -> None:
        with pytest.raises(ValueError):
            _NameSourceGate944.match_feature_group_criteria(self.NAME, _options(["a", "b"]))

    def test_empty_list_is_ignored(self) -> None:
        assert _NameSourceGate944.match_feature_group_criteria(self.NAME, _options([])) is True

    def test_env_var_downgrades_to_a_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "1")

        assert _NameSourceGate944.match_feature_group_criteria(self.NAME, _options(["a", "b"])) is True

    @staticmethod
    def _declared() -> list[Feature]:
        return [
            Feature("f1", options=Options(context={"k1": 1}), feature_group="ScopeOne944"),
            Feature("f2", options=Options(context={"k2": 2}), feature_group="ScopeTwo944"),
        ]

    def test_agreeing_list_returns_the_declared_features(self) -> None:
        declared = self._declared()
        result = _NameSourceGate944().input_features(_options(declared), FeatureName(self.NAME))

        assert result is not None
        by_name = {str(f.name): f for f in result}
        assert set(by_name) == {"f1", "f2"}
        assert by_name["f1"].options.get("k1") == 1
        assert by_name["f2"].options.get("k2") == 2
        assert by_name["f1"].feature_group_scope == "ScopeOne944"
        assert by_name["f2"].feature_group_scope == "ScopeTwo944"

    @pytest.mark.parametrize("container", [set, frozenset])
    def test_agreeing_set_returns_the_declared_features(self, container: Any) -> None:
        result = _NameSourceGate944().input_features(_options(container(self._declared())), FeatureName(self.NAME))

        assert result is not None
        by_name = {str(f.name): f for f in result}
        assert set(by_name) == {"f1", "f2"}
        assert by_name["f1"].options.get("k1") == 1
        assert by_name["f2"].feature_group_scope == "ScopeTwo944"

    def test_agreeing_features_are_copies_not_the_declared_objects(self) -> None:
        declared = self._declared()
        result = _NameSourceGate944().input_features(_options(declared), FeatureName(self.NAME))

        assert result is not None
        by_name = {str(f.name): f for f in result}
        for original in declared:
            returned = by_name[str(original.name)]
            assert returned == original
            assert returned is not original

        by_name["f1"].options.add_to_group("mutated", 1)
        assert declared[0].options.get("mutated") is None

    def test_string_entries_yield_bare_named_features(self) -> None:
        result = _NameSourceGate944().input_features(_options(["f1", "f2"]), FeatureName(self.NAME))

        assert result is not None
        assert {str(f.name) for f in result} == {"f1", "f2"}
        assert all(f.feature_group_scope is None for f in result)

    def test_no_in_features_returns_bare_features(self) -> None:
        result = _NameSourceGate944().input_features(_options(), FeatureName(self.NAME))

        assert result is not None
        assert {str(f.name) for f in result} == {"f1", "f2"}
        assert all(f.feature_group_scope is None for f in result)

    def test_downgraded_reordered_list_returns_the_bare_name_features(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "1")
        declared = list(reversed(self._declared()))

        result = _NameSourceGate944().input_features(_options(declared), FeatureName(self.NAME))

        assert result is not None
        assert {str(f.name) for f in result} == {"f1", "f2"}
        assert all(f.feature_group_scope is None for f in result)
        assert all(f.options.get("k1") is None and f.options.get("k2") is None for f in result)

    def test_inherited_in_features_returns_the_bare_name_features(self) -> None:
        options = _options(self._declared())
        options.inherited_context_keys = frozenset({DefaultOptionKeys.in_features.value})

        result = _NameSourceGate944().input_features(options, FeatureName(self.NAME))

        assert result is not None
        assert all(f.feature_group_scope is None for f in result)
        assert all(f.options.get("k1") is None and f.options.get("k2") is None for f in result)


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
        assert rejection_window == {OWNER_944: MatchRejection(reason=BELOW_MIN_REASON_944, stage=NAME_STAGE)}
        assert (
            _NameSourceGate944._strict_validation_rejection_reason(BELOW_MIN_NAME_944, _options())
            == BELOW_MIN_REASON_944
        )

    def test_facade_call_inside_an_open_window_records_nothing(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """The facade is a standalone diagnostic: calling it must never write into an active recording window."""
        reason = _NameSourceGate944._strict_validation_rejection_reason(BELOW_MIN_NAME_944, _options())

        assert reason == BELOW_MIN_REASON_944
        assert rejection_window == {}

    def test_above_max_records_the_actionable_reason(self, rejection_window: dict[str, MatchRejection]) -> None:
        """The reason keeps the pre-gate wording: the declared MAX and the count the name carries."""
        result = _NameSourceGate944.match_feature_group_criteria(ABOVE_MAX_NAME_944, _options())

        assert result is False
        assert rejection_window == {OWNER_944: MatchRejection(reason=ABOVE_MAX_REASON_944, stage=NAME_STAGE)}

    def test_below_min_records_the_name_count_even_with_a_passing_option(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """The option value would pass the gate, so the reason must report the name's count, not the option's."""
        _NameSourceGate944.match_feature_group_criteria(BELOW_MIN_NAME_944, _options(["a", "b"]))

        assert rejection_window == {OWNER_944: MatchRejection(reason=BELOW_MIN_REASON_944, stage=NAME_STAGE)}

    def test_a_count_inside_the_range_records_nothing(self, rejection_window: dict[str, MatchRejection]) -> None:
        result = _NameSourceGate944.match_feature_group_criteria("f1&f2&f3__op1_gate944", _options())

        assert result is True
        assert rejection_window == {}

    def test_the_option_path_still_records_nothing(self, rejection_window: dict[str, MatchRejection]) -> None:
        """Pinned contrast: only the name path reports; the option path stays a silent non-match."""
        result = _NameSourceGate944.match_feature_group_criteria("any_name", _options("single_feature"))

        assert result is False
        assert rejection_window == {}

    def test_the_count_reason_wins_over_a_rejected_guard(self, rejection_window: dict[str, MatchRejection]) -> None:
        """A too-few name count AND a guard-rejected key: the recorded reason is the count reason, never the guard's."""
        context = {"operation": "op1", "guarded_key_gate944": "ok_gate944"}
        options = Options(context=context)
        count_reason = "Feature 'f1__op1_guardgate944' requires at least 2 in_feature(s), but found 1"

        result = _NameSourceGuardGate944.match_feature_group_criteria("f1__op1_guardgate944", options)

        assert result is False
        assert rejection_window == {"_NameSourceGuardGate944": MatchRejection(reason=count_reason, stage=NAME_STAGE)}
        assert (
            _NameSourceGuardGate944._strict_validation_rejection_reason("f1__op1_guardgate944", options) == count_reason
        )


EMPTY_OPERAND_NAMES_1716 = ["&f2__op1_gate944", "f1&__op1_gate944", "f1&&f2__op1_gate944"]


class TestEmptyOperandIsRejected:
    """An empty operand in the name is a recorded non-match; an empty config operand is a silent one."""

    @pytest.mark.parametrize("name", EMPTY_OPERAND_NAMES_1716)
    def test_name_with_an_empty_operand_is_a_recorded_non_match(
        self, name: str, rejection_window: dict[str, MatchRejection]
    ) -> None:
        result = _NameSourceGate944.match_feature_group_criteria(name, _options())

        assert result is False
        recorded = rejection_window[OWNER_944]
        assert recorded.stage == NAME_STAGE
        assert "empty in_feature" in recorded.reason

    @pytest.mark.parametrize("name", EMPTY_OPERAND_NAMES_1716)
    def test_strict_validation_reason_reports_the_same_reason(
        self, name: str, rejection_window: dict[str, MatchRejection]
    ) -> None:
        _NameSourceGate944.match_feature_group_criteria(name, _options())

        reason = _NameSourceGate944._strict_validation_rejection_reason(name, _options())

        assert reason == rejection_window[OWNER_944].reason

    @pytest.mark.parametrize("name", EMPTY_OPERAND_NAMES_1716)
    def test_input_features_raises_the_reason(self, name: str, rejection_window: dict[str, MatchRejection]) -> None:
        _NameSourceGate944.match_feature_group_criteria(name, _options())
        expected = rejection_window[OWNER_944].reason

        with pytest.raises(ValueError) as exc_info:
            _NameSourceGate944().input_features(_options(), FeatureName(name))

        assert str(exc_info.value) == expected

    def test_empty_operand_in_features_option_is_a_silent_non_match(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
        result = _NameSourceGate944.match_feature_group_criteria("any_name", _options(["", "f2"]))

        assert result is False
        assert rejection_window == {}

    def test_empty_operand_in_features_option_makes_input_features_raise(self) -> None:
        with pytest.raises(ValueError):
            _NameSourceGate944().input_features(_options(["", "f2"]), FeatureName("any_name"))


class TestSourceFeaturesReason:
    """source_features_reason checks empty operands first, then the count."""

    def test_empty_operand_reason_names_the_feature_and_says_empty(self) -> None:
        reason = _NameSourceGate944.source_features_reason("&f2__op1_gate944", ("", "f2"))

        assert reason is not None
        assert "empty" in reason
        assert "&f2__op1_gate944" in reason

    def test_empty_operand_wins_over_a_count_violation(self) -> None:
        reason = _NameSourceGate944.source_features_reason("&__op1_gate944", ("",))

        assert reason is not None
        assert "empty" in reason

    def test_count_violation_returns_the_count_reason(self) -> None:
        assert _NameSourceGate944.source_features_reason(BELOW_MIN_NAME_944, ("f1",)) == BELOW_MIN_REASON_944

    def test_valid_sources_have_no_reason(self) -> None:
        assert _NameSourceGate944.source_features_reason("f1&f2__op1_gate944", ("f1", "f2")) is None


class _OptionsOnlyGroupM951(FeatureChainParserMixin):
    """Options-only group: no in_features key, default MIN_IN_FEATURES."""

    PROPERTY_MAPPING = {"mode_m951": PropertySpec("Mode", allowed_values={"fast_m951": "Fast"}, context=True)}


class _AllOptionalGroupM951(FeatureChainParserMixin):
    """All-optional mapping."""

    PROPERTY_MAPPING = {"tuning_m951": PropertySpec("Tuning", default=None, context=True)}


class TestOptionPathZeroSourceRecordsNoRejection:
    """The option path records no rejection reason for a zero-source non-match."""

    def test_addressed_group_without_in_features_does_not_match_and_records_nothing(
        self, rejection_window: dict[str, MatchRejection]
    ) -> None:
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
