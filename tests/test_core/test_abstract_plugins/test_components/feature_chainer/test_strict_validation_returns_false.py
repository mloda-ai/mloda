"""Tests that strict_validation ValueError is caught and returns False."""

from typing import Any

import pytest

from mloda.user import FeatureName, Options
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
from mloda.core.abstract_plugins.components.utils import escalate_match_abort
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import PropertySpec, property_spec


class BaseMode(FeatureChainParserMixin):
    """Base class with strict validation on 'mode' key."""

    MIN_IN_FEATURES = 0

    PROPERTY_MAPPING = {
        "mode": PropertySpec(
            "Mode of operation",
            allowed_values={"mode_a": "Mode A", "mode_b": "Mode B"},
            context=False,
            strict_validation=True,
        )
    }


class SubGroupA(BaseMode):
    """Subclass that only accepts mode_a."""

    PROPERTY_MAPPING = {
        "mode": PropertySpec(
            "Mode of operation",
            allowed_values={"mode_a": "Mode A"},
            context=False,
            strict_validation=True,
        )
    }


class SubGroupB(BaseMode):
    """Subclass that only accepts mode_b."""

    PROPERTY_MAPPING = {
        "mode": PropertySpec(
            "Mode of operation",
            allowed_values={"mode_b": "Mode B"},
            context=False,
            strict_validation=True,
        )
    }


class SubGroupWithElementValidator(BaseMode):
    """Subclass with a custom element validator."""

    PROPERTY_MAPPING = {
        "mode": PropertySpec(
            "Mode of operation",
            context=False,
            strict_validation=True,
            element_validator=lambda v: v.startswith("valid_"),
        )
    }


class MalformedPrefixFeatureGroup(FeatureChainParserMixin):
    """PREFIX_PATTERN matches names with no chain separator, triggering a parse ValueError unrelated to option validation."""

    PREFIX_PATTERN = r"^malformed_prefix_(\w+)$"


class TestStrictValidationReturnsFalse:
    """Tests that strict_validation failures return False instead of raising ValueError."""

    def test_returns_false_for_nonmatching_subclass(self) -> None:
        """SubGroupA with mode=mode_b should return False, not raise ValueError."""
        feature_name = FeatureName("any_feature")
        options = Options(group={"mode": "mode_b"})

        result = SubGroupA.match_feature_group_criteria(feature_name, options)

        assert result is False

    def test_returns_true_for_matching_subclass(self) -> None:
        """SubGroupA with mode=mode_a should return True."""
        feature_name = FeatureName("any_feature")
        options = Options(group={"mode": "mode_a"})

        result = SubGroupA.match_feature_group_criteria(feature_name, options)

        assert result is True

    def test_element_validator_failure_returns_false(self) -> None:
        """Custom element_validator rejection should return False, not raise ValueError."""
        feature_name = FeatureName("any_feature")
        options = Options(group={"mode": "invalid_prefix"})

        result = SubGroupWithElementValidator.match_feature_group_criteria(feature_name, options)

        assert result is False


class TestStrictValidationRejectionReason:
    """Tests that _strict_validation_rejection_reason surfaces the discarded ValueError message."""

    def test_membership_check_rejection_returns_message(self) -> None:
        """SubGroupA with mode=mode_b should return the membership-check message, not None."""
        feature_name = FeatureName("any_feature")
        options = Options(group={"mode": "mode_b"})

        assert SubGroupA.match_feature_group_criteria(feature_name, options) is False

        reason = SubGroupA._strict_validation_rejection_reason(feature_name, options)

        assert reason is not None
        assert "mode_b" in reason
        assert "mode" in reason

    def test_matching_value_returns_none(self) -> None:
        """SubGroupA with mode=mode_a matches, so there is no rejection reason."""
        feature_name = FeatureName("any_feature")
        options = Options(group={"mode": "mode_a"})

        reason = SubGroupA._strict_validation_rejection_reason(feature_name, options)

        assert reason is None

    def test_element_validator_rejection_returns_message(self) -> None:
        """SubGroupWithElementValidator with an invalid mode returns the element_validator message."""
        feature_name = FeatureName("any_feature")
        options = Options(group={"mode": "invalid_prefix"})

        reason = SubGroupWithElementValidator._strict_validation_rejection_reason(feature_name, options)

        assert reason is not None
        assert "invalid_prefix" in reason
        assert "mode" in reason

    def test_unrelated_candidate_returns_none(self) -> None:
        """When the relevant property is entirely absent, this is a non-match, not a rejection."""
        feature_name = FeatureName("any_feature")
        options = Options(group={})

        reason = SubGroupA._strict_validation_rejection_reason(feature_name, options)

        assert reason is None

    def test_malformed_prefix_match_returns_none(self) -> None:
        """PREFIX_PATTERN matches but the name has no chain separator: this is a
        malformed-name parse ValueError, not an option-value rejection, so it must
        return None rather than the parser's "has no source feature" message."""
        result = MalformedPrefixFeatureGroup._strict_validation_rejection_reason("malformed_prefix_test", Options())

        assert result is None


def _is_positive_int_pgv(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _raising_guard_pgv(value: Any) -> bool:
    raise TypeError("cannot judge")


class PlainStrictModePgv(FeatureGroup):
    """Plain group (no mixin) with a strict, optional value space."""

    PROPERTY_MAPPING = {
        "mode": PropertySpec(
            "Mode of operation",
            allowed_values={"mode_a": "Mode A"},
            context=False,
            strict_validation=True,
            default=None,
        )
    }


class PlainElementValidatorPgv(FeatureGroup):
    """Plain group whose strict spec judges values with an element_validator."""

    PROPERTY_MAPPING = {
        "mode": PropertySpec(
            "Mode of operation",
            context=False,
            strict_validation=True,
            element_validator=lambda v: v.startswith("valid_"),
            default=None,
        )
    }


class PlainGuardedPgv(FeatureGroup):
    """Plain group with a match_guard and no required key."""

    PROPERTY_MAPPING = {"limit_pgv": property_spec("positive count", match_guard=_is_positive_int_pgv, default=None)}


class PlainRaisingGuardPgv(FeatureGroup):
    """Plain group whose match_guard cannot judge the value."""

    PROPERTY_MAPPING = {"limit_pgv": property_spec("positive count", match_guard=_raising_guard_pgv, default=None)}


class TestPlainGroupStrictValidation:
    """A plain group judges present option values like a mixin group: a rejection is a non-match."""

    def test_rejected_value_is_a_non_match(self) -> None:
        name = PlainStrictModePgv.get_class_name()

        assert PlainStrictModePgv.match_feature_group_criteria(name, Options(group={"mode": "mode_b"})) is False

    def test_accepted_value_matches(self) -> None:
        name = PlainStrictModePgv.get_class_name()

        assert PlainStrictModePgv.match_feature_group_criteria(name, Options(group={"mode": "mode_a"})) is True

    def test_absent_optional_value_matches(self) -> None:
        name = PlainStrictModePgv.get_class_name()

        assert PlainStrictModePgv.match_feature_group_criteria(name, Options()) is True

    def test_element_validator_rejection_is_a_non_match(self) -> None:
        name = PlainElementValidatorPgv.get_class_name()

        assert (
            PlainElementValidatorPgv.match_feature_group_criteria(name, Options(group={"mode": "invalid_prefix"}))
            is False
        )
        assert PlainElementValidatorPgv.match_feature_group_criteria(name, Options(group={"mode": "valid_x"})) is True

    def test_rejection_reason_is_recorded(self, rejection_window: dict[str, MatchRejection]) -> None:
        name = PlainStrictModePgv.get_class_name()

        assert PlainStrictModePgv.match_feature_group_criteria(name, Options(group={"mode": "mode_b"})) is False

        rejection = rejection_window[name]
        assert rejection.stage == "value_rejection"
        assert "mode_b" in rejection.reason
        assert "'mode'" in rejection.reason

    def test_unmarked_value_error_is_contained_as_a_non_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def reject(cls: type[FeatureChainParser], options: Options, property_mapping: dict[str, Any]) -> None:
            raise ValueError("unmarked")

        monkeypatch.setattr(FeatureChainParser, "_validate_present_option_values", classmethod(reject))
        name = PlainStrictModePgv.get_class_name()

        assert PlainStrictModePgv.match_feature_group_criteria(name, Options(group={"mode": "mode_a"})) is False

    def test_marked_abort_crosses_the_containment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def abort(cls: type[FeatureChainParser], options: Options, property_mapping: dict[str, Any]) -> None:
            raise escalate_match_abort(ValueError("marked"))

        monkeypatch.setattr(FeatureChainParser, "_validate_present_option_values", classmethod(abort))
        name = PlainStrictModePgv.get_class_name()

        with pytest.raises(ValueError, match="marked"):
            PlainStrictModePgv.match_feature_group_criteria(name, Options(group={"mode": "mode_a"}))


class TestPlainGroupMatchGuard:
    """A plain group enforces its match_guard in the default matcher, and a delegating override keeps it."""

    @pytest.mark.parametrize(("value", "expected"), [(4, True), (0, False), ("4", False), (True, False)])
    def test_guard_decides_the_match(self, value: Any, expected: bool) -> None:
        name = PlainGuardedPgv.get_class_name()

        assert PlainGuardedPgv.match_feature_group_criteria(name, Options(context={"limit_pgv": value})) is expected

    def test_absent_value_skips_the_guard(self) -> None:
        assert PlainGuardedPgv.match_feature_group_criteria(PlainGuardedPgv.get_class_name(), Options()) is True

    def test_raising_guard_counts_as_a_rejection(self) -> None:
        name = PlainRaisingGuardPgv.get_class_name()

        assert PlainRaisingGuardPgv.match_feature_group_criteria(name, Options(context={"limit_pgv": 4})) is False

    def test_guard_sees_the_value_once_through_a_delegating_override(self) -> None:
        calls: list[Any] = []

        def counting_guard(value: Any) -> bool:
            calls.append(value)
            return bool(value != 0)

        class CountedGuardParent(FeatureGroup):
            PROPERTY_MAPPING = {"limit_pgv": property_spec("count", match_guard=counting_guard, default=None)}

        class DelegatingGuardChild(CountedGuardParent):
            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: FeatureName | str,
                options: Options,
                data_access_collection: DataAccessCollection | None = None,
            ) -> bool:
                return super().match_feature_group_criteria(feature_name, options, data_access_collection)

        name = DelegatingGuardChild.get_class_name()

        assert DelegatingGuardChild.match_feature_group_criteria(name, Options(context={"limit_pgv": 0})) is False
        assert calls == [0]
