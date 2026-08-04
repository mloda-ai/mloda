"""The MIN/MAX_IN_FEATURES gate counts the sources the feature name carries (#944).

When the name identifies the group, input_features splits the name and never reads the
in_features option, so the gate must count the same sources the name path would use.
"""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
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
