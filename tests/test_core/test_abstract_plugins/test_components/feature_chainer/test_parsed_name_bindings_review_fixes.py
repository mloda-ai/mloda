"""Regression tests for the #770 review fixes.

FIX A unifies the "name string-identifies this group" predicate across the three match sites, so a
named capture after an ABSENT optional first positional group is fully bound (guard, diagnostic,
forwarded-mismatch), while a positional pattern that matches only via an absent optional first group
keeps the pre-#770 required-presence rejection. FIX B makes the definition-time ambiguity guard
per-pattern and flattens list-valued pattern attributes.

Each corrected-behavior test fails against the committed code and passes once the fixes land. Fixture
names carry an "rf770" marker so they cannot collide with the pnb770 fixtures or the registry.
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.options import Options


SCALER_KEY = "scaler_rf770"
OP_KEY = "op_rf770"
KEY_A = "keya_rf770"
KEY_B = "keyb_rf770"

NAMED_GUARD_NAME = "standard__x_guard_rf770"
NAMED_FWD_NAME = "standard__x_fwd_rf770"
POSITIONAL_NAME = "x__op_rf770"


def _rejects_standard(value: Any) -> bool:
    """match_guard: every scaler but 'standard' passes."""
    return bool(value != "standard")


def _inherited_child_options(consumer_group: dict[str, Any]) -> Options:
    """Build child options exactly like the engine does: inherit_from the consumer."""
    child_options = Options()
    child_options.inherit_from(Options(group=consumer_group))
    return child_options


class NamedOptionalFirstGuardedGroup(FeatureChainParserMixin):
    """Named capture after an OPTIONAL first positional group, whose legacy positional value is None."""

    PREFIX_PATTERN = rf"^(opt_)?(?P<{SCALER_KEY}>minmax|standard)__.+_guard_rf770$"

    PROPERTY_MAPPING: dict[str, PropertySpec] = {
        SCALER_KEY: PropertySpec(
            "Scaler carried by a named capture; a guard rejects the 'standard' value",
            allowed_values={"minmax", "standard"},
            context=False,
            strict_validation=True,
            match_guard=_rejects_standard,
        )
    }


class NamedOptionalFirstForwardedGroup(FeatureChainParserMixin):
    """Same optional-first named shape, group-categorized so a forwarded value is protected."""

    PREFIX_PATTERN = rf"^(opt_)?(?P<{SCALER_KEY}>minmax|standard)__.+_fwd_rf770$"

    PROPERTY_MAPPING: dict[str, PropertySpec] = {
        SCALER_KEY: PropertySpec(
            "Scaler carried by a named capture, group-categorized for forwarding",
            allowed_values={"minmax", "standard"},
            context=False,
            strict_validation=True,
        )
    }


class PositionalOptionalFirstGroup(FeatureChainParserMixin):
    """Positional pattern whose only capture is an absent optional first group; one required strict key."""

    PREFIX_PATTERN = r".*__(foo_)?op_rf770$"

    PROPERTY_MAPPING: dict[str, PropertySpec] = {
        OP_KEY: PropertySpec(
            "Required strict key that the name never carries",
            allowed_values={"alpha", "beta"},
            context=True,
            strict_validation=True,
        )
    }


class TestFixANamedOptionalFirst:
    """A named capture identifies the group even when the legacy positional value is absent."""

    def test_minmax_value_matches_without_options(self) -> None:
        """Precondition: the fixture claims the chained name when the guard has nothing to reject."""
        assert NamedOptionalFirstGuardedGroup.match_feature_group_criteria("minmax__x_guard_rf770", Options()) is True

    def test_match_guard_sees_the_named_binding(self) -> None:
        """The guard rejecting the name-carried scaler makes the match fail, though the legacy op is None."""
        result = NamedOptionalFirstGuardedGroup.match_feature_group_criteria(NAMED_GUARD_NAME, Options())

        assert result is False

    def test_rejection_reason_agrees_with_the_match_decision(self) -> None:
        """The diagnostic replay rejects exactly when the match does: both reject the guarded scaler."""
        result = NamedOptionalFirstGuardedGroup.match_feature_group_criteria(NAMED_GUARD_NAME, Options())
        reason = NamedOptionalFirstGuardedGroup._strict_validation_rejection_reason(NAMED_GUARD_NAME, Options())

        assert result is False
        assert reason is not None
        assert SCALER_KEY in reason

    def test_forwarded_scaler_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A forwarded scaler that contradicts the name-parsed one is a hard error, not a silent override."""
        monkeypatch.delenv("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", raising=False)
        child_options = _inherited_child_options({SCALER_KEY: "minmax"})
        assert child_options.inherited_group_keys == frozenset({SCALER_KEY})  # precondition

        with pytest.raises(ValueError) as exc_info:
            NamedOptionalFirstForwardedGroup.match_feature_group_criteria(NAMED_FWD_NAME, child_options)

        message = str(exc_info.value)
        assert SCALER_KEY in message
        assert "minmax" in message
        assert "standard" in message
        assert "forward_group_exclude" in message

    def test_forwarded_fixture_matches_without_options(self) -> None:
        """Precondition: the forwarded fixture claims the chained name on its own."""
        assert NamedOptionalFirstForwardedGroup.match_feature_group_criteria(NAMED_FWD_NAME, Options()) is True

    def test_agreeing_forwarded_scaler_matches(self) -> None:
        """Guard against over-rejecting: an equal forwarded value is no mismatch."""
        child_options = _inherited_child_options({SCALER_KEY: "standard"})

        assert NamedOptionalFirstForwardedGroup.match_feature_group_criteria(NAMED_FWD_NAME, child_options) is True


class TestFixAPositionalOptionalFirst:
    """A positional pattern matching only via an absent optional first group keeps the pre-#770 rejection."""

    def test_parser_empty_options_is_a_non_match(self) -> None:
        """The string path no longer matches on the pattern alone: required presence still guards it."""
        result = FeatureChainParser.match_configuration_feature_chain_parser(
            POSITIONAL_NAME,
            Options(),
            PositionalOptionalFirstGroup.PROPERTY_MAPPING,
            [PositionalOptionalFirstGroup.PREFIX_PATTERN],
        )

        assert result is False

    def test_mixin_empty_options_is_a_non_match(self) -> None:
        """The mixer agrees: no match-then-fail for the empty-options case."""
        result = PositionalOptionalFirstGroup.match_feature_group_criteria(POSITIONAL_NAME, Options())

        assert result is False

    def test_required_key_present_matches(self) -> None:
        """Guard against over-rejecting: the required key present and valid still matches."""
        result = PositionalOptionalFirstGroup.match_feature_group_criteria(
            POSITIONAL_NAME, Options(context={OP_KEY: "alpha"})
        )

        assert result is True


class TestFixBPerPatternAmbiguity:
    """The definition-time ambiguity guard is per-pattern and flattens list-valued pattern attributes."""

    def test_positional_prefix_plus_named_suffix_raises(self) -> None:
        """A named SUFFIX no longer excuses an ambiguous positional PREFIX: the positional pattern is checked."""
        with pytest.raises(ValueError) as exc_info:

            class _PositionalPlusNamedRf770(FeatureChainParserMixin):
                PREFIX_PATTERN = r".*__(\w+)_posrf770$"
                SUFFIX_PATTERN = r".*__(?P<something_rf770>\w+)_sufrf770$"
                PROPERTY_MAPPING = {
                    KEY_A: PropertySpec("a", allowed_values=("pca", "sharedval_rf770")),
                    KEY_B: PropertySpec("b", allowed_values=("sharedval_rf770", "auto")),
                }

        message = str(exc_info.value)
        assert KEY_A in message
        assert KEY_B in message
        assert "sharedval_rf770" in message

    def test_list_form_positional_pattern_raises(self) -> None:
        """The list-form PREFIX_PATTERN is flattened to its element and checked, not skipped as zero-capture."""
        with pytest.raises(ValueError) as exc_info:

            class _ListFormPositionalRf770(FeatureChainParserMixin):
                PREFIX_PATTERN = [r".*__(shared_rf770)_listx_rf770$"]
                PROPERTY_MAPPING = {
                    KEY_A: PropertySpec("a", allowed_values=("shared_rf770", "pca")),
                    KEY_B: PropertySpec("b", allowed_values=("shared_rf770", "auto")),
                }

        assert "shared_rf770" in str(exc_info.value)

    def test_shipped_style_single_positional_no_overlap_does_not_raise(self) -> None:
        """Guard against over-rejecting: a single positional pattern with disjoint keys still defines cleanly."""

        class _ShippedStylePositionalRf770(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(\w+)_shipped_rf770$"
            PROPERTY_MAPPING = {
                KEY_A: PropertySpec("a", allowed_values=("pca", "tsne")),
                KEY_B: PropertySpec("b", allowed_values=("auto", "arpack")),
            }

        assert _ShippedStylePositionalRf770.PREFIX_PATTERN.endswith("_shipped_rf770$")
