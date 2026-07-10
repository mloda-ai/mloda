"""Tests that strict_validation ValueError is caught and returns False."""

from mloda.user import FeatureName, Options
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.provider import DefaultOptionKeys


class BaseMode(FeatureChainParserMixin):
    """Base class with strict validation on 'mode' key."""

    PROPERTY_MAPPING = {
        "mode": {
            "mode_a": "Mode A",
            "mode_b": "Mode B",
            DefaultOptionKeys.strict_validation: True,
        }
    }


class SubGroupA(BaseMode):
    """Subclass that only accepts mode_a."""

    PROPERTY_MAPPING = {
        "mode": {
            "mode_a": "Mode A",
            DefaultOptionKeys.strict_validation: True,
        }
    }


class SubGroupB(BaseMode):
    """Subclass that only accepts mode_b."""

    PROPERTY_MAPPING = {
        "mode": {
            "mode_b": "Mode B",
            DefaultOptionKeys.strict_validation: True,
        }
    }


class SubGroupWithValidationFunction(BaseMode):
    """Subclass with a custom validation function."""

    PROPERTY_MAPPING = {
        "mode": {
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.validation_function: lambda v: v.startswith("valid_"),
        }
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

    def test_validation_function_failure_returns_false(self) -> None:
        """Custom validation_function rejection should return False, not raise ValueError."""
        feature_name = FeatureName("any_feature")
        options = Options(group={"mode": "invalid_prefix"})

        result = SubGroupWithValidationFunction.match_feature_group_criteria(feature_name, options)

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

    def test_validation_function_rejection_returns_message(self) -> None:
        """SubGroupWithValidationFunction with an invalid mode returns the validation_function message."""
        feature_name = FeatureName("any_feature")
        options = Options(group={"mode": "invalid_prefix"})

        reason = SubGroupWithValidationFunction._strict_validation_rejection_reason(feature_name, options)

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
