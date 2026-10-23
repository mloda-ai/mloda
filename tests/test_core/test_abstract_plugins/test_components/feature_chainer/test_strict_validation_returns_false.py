"""Tests that strict_validation ValueError is caught and returns False."""

from typing import Any, NoReturn

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


class _RaisingVerdict:
    """A verdict whose truthiness raises the given exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def __bool__(self) -> bool:
        raise escalate_match_abort(self._exc)


def _raise_value_as_marked_abort(value: Any) -> bool:
    raise escalate_match_abort(value)


def _verdict_raising_on_truthiness(value: Any) -> Any:
    return _RaisingVerdict(value)


class MarkedElementValidatorMode(BaseMode):
    """Mode whose element_validator raises the marked abort held in the value."""

    PROPERTY_MAPPING = {
        "mode": PropertySpec(
            "Mode of operation",
            context=False,
            strict_validation=True,
            element_validator=_raise_value_as_marked_abort,
        )
    }


class MarkedVerdictMode(BaseMode):
    """Mode whose element_validator returns a verdict that raises on truthiness."""

    PROPERTY_MAPPING = {
        "mode": PropertySpec(
            "Mode of operation",
            context=False,
            strict_validation=True,
            element_validator=_verdict_raising_on_truthiness,
        )
    }


_RESOLVE_MARKER = escalate_match_abort(ValueError("marked resolve_feature_name abort"))


class MarkedResolveMode(BaseMode):
    """Mode whose resolve_feature_name raises a marked match abort."""

    PREFIX_PATTERN = r"^marked_resolve__(?P<mode>\w+)$"

    @classmethod
    def resolve_feature_name(cls, feature_name: Any) -> NoReturn:
        raise _RESOLVE_MARKER


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

    @pytest.mark.parametrize(
        ("group", "make_value"),
        [
            (MarkedElementValidatorMode, lambda marker: marker),
            (SubGroupA, lambda marker: _MarkedAbortEqValue(marker)),
            (MarkedVerdictMode, lambda marker: marker),
        ],
        ids=["element_validator", "membership_eq", "verdict_bool"],
    )
    def test_marked_abort_from_value_check_propagates(self, group: Any, make_value: Any) -> None:
        marker = escalate_match_abort(ValueError("marked diagnostic abort"))

        with pytest.raises(ValueError) as exc_info:
            group._strict_validation_rejection_reason(
                FeatureName("any_feature"), Options(group={"mode": make_value(marker)})
            )

        assert exc_info.value is marker

    def test_marked_abort_from_resolve_feature_name_propagates(self) -> None:
        feature_name = FeatureName("any_feature")
        options = Options(group={"mode": "mode_a"})

        with pytest.raises(ValueError) as diagnostic_info:
            MarkedResolveMode._strict_validation_rejection_reason(feature_name, options)
        with pytest.raises(ValueError) as match_info:
            MarkedResolveMode.match_feature_group_criteria(feature_name, options)

        assert diagnostic_info.value is _RESOLVE_MARKER
        assert match_info.value is _RESOLVE_MARKER

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


def _marked_abort_validator(value: Any) -> bool:
    raise escalate_match_abort(ValueError("marked"))


class PlainMarkedAbortValidatorPgv(FeatureGroup):
    """Plain group whose element_validator raises a marked abort."""

    PROPERTY_MAPPING = {
        "mode": PropertySpec(
            "Mode of operation",
            context=False,
            strict_validation=True,
            element_validator=_marked_abort_validator,
            default=None,
        )
    }


class _MarkedAbortEqValue:
    """Hash-collides with a member so membership runs __eq__, which raises a marked abort."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def __hash__(self) -> int:
        return hash("mode_a")

    def __eq__(self, other: object) -> bool:
        raise escalate_match_abort(self._exc)


class PlainGuardedPgv(FeatureGroup):
    """Plain group with a match_guard and no required key."""

    PROPERTY_MAPPING = {"limit_pgv": property_spec("positive count", match_guard=_is_positive_int_pgv, default=None)}


class PlainRaisingGuardPgv(FeatureGroup):
    """Plain group whose match_guard cannot judge the value."""

    PROPERTY_MAPPING = {"limit_pgv": property_spec("positive count", match_guard=_raising_guard_pgv, default=None)}


class WideningGuardedPgv(PlainGuardedPgv):
    """Widens name matching to `_score` and keeps the option declarations."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name).endswith("_score") and cls.passes_option_declarations(options)


class WideningStrictModePgv(PlainStrictModePgv):
    """Widens name matching to `_score` and keeps the option declarations."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name).endswith("_score") and cls.passes_option_declarations(options)


class MixinWideningGuardedPgv(FeatureChainParserMixin, FeatureGroup):
    """Mixin group with a match_guard and a widening override."""

    MIN_IN_FEATURES = 0
    PROPERTY_MAPPING = {
        "limit_pgv": property_spec("positive count", match_guard=_is_positive_int_pgv, default=None),
        "mode": PropertySpec(
            "Mode of operation",
            allowed_values={"mode_a": "Mode A"},
            context=False,
            strict_validation=True,
            default=None,
        ),
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name).endswith("_score") and cls.passes_option_declarations(options)


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

    @pytest.mark.parametrize(
        ("group", "name"),
        [(PlainStrictModePgv, None), (WideningStrictModePgv, "x_score")],
    )
    def test_rejection_reason_is_recorded(
        self, rejection_window: dict[str, MatchRejection], group: type[FeatureGroup], name: str | None
    ) -> None:
        key = group.get_class_name()

        assert group.match_feature_group_criteria(name or key, Options(group={"mode": "mode_b"})) is False

        rejection = rejection_window[key]
        assert rejection.stage == "value_rejection"
        assert "mode_b" in rejection.reason
        assert "'mode'" in rejection.reason

    def test_unmarked_value_error_is_contained_as_a_non_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def reject(cls: type[FeatureChainParser], options: Options, property_mapping: dict[str, Any]) -> None:
            raise ValueError("unmarked")

        monkeypatch.setattr(FeatureChainParser, "_validate_present_option_values", classmethod(reject))
        name = PlainStrictModePgv.get_class_name()

        assert PlainStrictModePgv.match_feature_group_criteria(name, Options(group={"mode": "mode_a"})) is False

    def test_unmarked_value_error_records_its_message_as_the_reason(
        self, monkeypatch: pytest.MonkeyPatch, rejection_window: dict[str, MatchRejection]
    ) -> None:
        def reject(cls: type[FeatureChainParser], options: Options, property_mapping: dict[str, Any]) -> None:
            raise ValueError("unmarked reason")

        monkeypatch.setattr(FeatureChainParser, "_validate_present_option_values", classmethod(reject))
        name = PlainStrictModePgv.get_class_name()

        assert PlainStrictModePgv.match_feature_group_criteria(name, Options(group={"mode": "mode_a"})) is False

        rejection = rejection_window[name]
        assert rejection.stage == "value_rejection"
        assert rejection.reason == "unmarked reason"

    def test_marked_abort_crosses_the_containment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def abort(cls: type[FeatureChainParser], options: Options, property_mapping: dict[str, Any]) -> None:
            raise escalate_match_abort(ValueError("marked"))

        monkeypatch.setattr(FeatureChainParser, "_validate_present_option_values", classmethod(abort))
        name = PlainStrictModePgv.get_class_name()

        with pytest.raises(ValueError, match="marked"):
            PlainStrictModePgv.match_feature_group_criteria(name, Options(group={"mode": "mode_a"}))

    @pytest.mark.parametrize(
        "group, value, exc_type",
        [
            (PlainMarkedAbortValidatorPgv, "mode_a", ValueError),
            (PlainStrictModePgv, _MarkedAbortEqValue(ValueError("marked")), ValueError),
            (PlainStrictModePgv, _MarkedAbortEqValue(TypeError("marked")), TypeError),
        ],
        ids=["element_validator", "eq_value_error", "eq_type_error"],
    )
    def test_marked_abort_from_value_check_propagates(self, group: Any, value: Any, exc_type: type) -> None:
        with pytest.raises(exc_type, match="marked"):
            group.match_feature_group_criteria(group.get_class_name(), Options(group={"mode": value}))


class TestPlainGroupMatchGuard:
    """A plain group enforces its match_guard in the default matcher, and a delegating override keeps it."""

    @pytest.mark.parametrize(("value", "expected"), [(4, True), (0, False), ("4", False), (True, False)])
    def test_guard_decides_the_match(self, value: Any, expected: bool) -> None:
        name = PlainGuardedPgv.get_class_name()

        assert PlainGuardedPgv.match_feature_group_criteria(name, Options(context={"limit_pgv": value})) is expected

    @pytest.mark.parametrize(
        ("name", "value", "expected"),
        [("x_score", 4, True), ("x_score", 0, False), ("x_score", True, False), ("x_other", 4, False)],
    )
    def test_widened_name_is_decided_by_the_guard(self, name: str, value: Any, expected: bool) -> None:
        assert WideningGuardedPgv.match_feature_group_criteria(name, Options(context={"limit_pgv": value})) is expected

    def test_widened_strict_value_is_enforced(self) -> None:
        assert WideningStrictModePgv.match_feature_group_criteria("x_score", Options(group={"mode": "mode_a"})) is True
        assert WideningStrictModePgv.match_feature_group_criteria("x_score", Options(group={"mode": "mode_b"})) is False

    @pytest.mark.parametrize(
        ("options", "expected"),
        [
            (Options(context={"limit_pgv": 4}), True),
            (Options(context={"limit_pgv": 0}), False),
            (Options(group={"mode": "mode_b"}), False),
        ],
    )
    def test_mixin_group_widened_name_keeps_declarations(self, options: Options, expected: bool) -> None:
        assert MixinWideningGuardedPgv.match_feature_group_criteria("x_score", options) is expected

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
