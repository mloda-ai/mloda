"""An explicit None becomes a distinct, honored option value only when a spec opts in (#768).

Today the option layer reads an explicit ``None`` exactly like an absent key ("None means
absent"). ``allow_explicit_none=True`` is the per-spec opt-in that flips two behaviors for that one
key: an explicit ``None`` now FLOWS THROUGH the parser's value validation (a strict spec can accept
or reject it), and an explicit ``None`` counts as PRESENT for the required-presence check. Every
flagless spec is unchanged: an explicit ``None`` still reads as absent.

These tests drive the parser entry points directly with hand-built mappings: the config path
(``match_configuration_feature_chain_parser``), the ``required_when`` guard (``check_required_when``),
and the ``match_guard`` sweep (``_first_rejecting_guard``). They need no FeatureGroup subclass and
leave the registry untouched (the direct-call style of ``test_property_spec_type.py``).
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    FeatureChainParser,
    PropertyValueRejection,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.options import Options


def _accepts_none(value: Any) -> bool:
    """A value space whose accepted set is ``None`` plus the positive ints (#768)."""
    return value is None or (isinstance(value, int) and value > 0)


def _always_required(options: Any) -> bool:
    """A ``required_when`` predicate that always demands the option (#768)."""
    return True


def _rejects_none(value: Any) -> bool:
    """A ``match_guard`` that rejects None and accepts every other value (#768)."""
    return value is not None


class TestExplicitNoneFlowsThroughStrictValidation:
    """Under strict + opt-in, an explicit None is validated like any other value (#768)."""

    def test_opted_in_none_outside_value_space_is_rejected(self) -> None:
        """C1: an opted-in explicit None outside the strict value space is validated and REJECTED."""
        property_mapping = {
            "k": PropertySpec(
                "k", allowed_values=("a", "b"), strict_validation=True, default="a", allow_explicit_none=True
            )
        }

        with pytest.raises(PropertyValueRejection):
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={"k": None}), property_mapping
            )

    def test_flagless_none_is_absent_so_the_default_makes_it_skippable(self) -> None:
        """C4 contrast (DoD 3): without the flag the same explicit None reads as absent and is not validated."""
        property_mapping = {"k": PropertySpec("k", allowed_values=("a", "b"), strict_validation=True, default="a")}

        assert (
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={"k": None}), property_mapping
            )
            is True
        )

    def test_opted_in_none_inside_value_space_is_accepted(self) -> None:
        """C2: an opted-in explicit None a strict validator accepts is validated and matches."""
        property_mapping = {
            "k": PropertySpec("k", strict_validation=True, element_validator=_accepts_none, allow_explicit_none=True)
        }

        assert (
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={"k": None}), property_mapping
            )
            is True
        )


class TestExplicitNoneSatisfiesRequiredPresence:
    """Under opt-in, an explicit None satisfies a required key's presence; absent is still absent (#768)."""

    def test_opted_in_absent_key_is_still_required(self) -> None:
        """C3: a truly absent opted-in required key is still a non-match (absent stays absent)."""
        property_mapping = {"k": PropertySpec("k", allow_explicit_none=True)}

        assert (
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={}), property_mapping
            )
            is False
        )

    def test_opted_in_explicit_none_satisfies_presence(self) -> None:
        """C3: an explicit None satisfies the required key once the spec opts in."""
        property_mapping = {"k": PropertySpec("k", allow_explicit_none=True)}

        assert (
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={"k": None}), property_mapping
            )
            is True
        )

    def test_flagless_explicit_none_reads_as_absent(self) -> None:
        """C3 contrast (DoD 3): without the flag an explicit None reads as absent, so the required key is unmet."""
        property_mapping = {"k": PropertySpec("k")}

        assert (
            FeatureChainParser.match_configuration_feature_chain_parser(
                "any_feature", Options(context={"k": None}), property_mapping
            )
            is False
        )


class TestRequiredWhenHonorsExplicitNone:
    """check_required_when reads an explicit None as present only when the spec opts in (#768)."""

    def test_opted_in_explicit_none_satisfies_required_when(self) -> None:
        """D1: an opted-in satisfied requirement is met by an explicit None, which now counts as present."""
        property_mapping = {"k": PropertySpec("k", required_when=_always_required, allow_explicit_none=True)}

        assert (
            FeatureChainParser.check_required_when(
                "Owner", "any_feature", [], property_mapping, Options(context={"k": None})
            )
            is True
        )

    def test_flagless_explicit_none_leaves_requirement_unmet(self) -> None:
        """D2 contrast (DoD 3): without the flag the explicit None reads as absent, so the requirement is unmet."""
        property_mapping = {"k": PropertySpec("k", required_when=_always_required)}

        assert (
            FeatureChainParser.check_required_when(
                "Owner", "any_feature", [], property_mapping, Options(context={"k": None})
            )
            is False
        )

    def test_opted_in_absent_key_leaves_requirement_unmet(self) -> None:
        """D3: absent stays absent; an opted-in required key that is truly absent is still unmet."""
        property_mapping = {"k": PropertySpec("k", required_when=_always_required, allow_explicit_none=True)}

        assert (
            FeatureChainParser.check_required_when("Owner", "any_feature", [], property_mapping, Options(context={}))
            is False
        )


class TestMatchGuardHonorsExplicitNone:
    """_first_rejecting_guard runs a match_guard on an explicit None only when the spec opts in (#768)."""

    def test_opted_in_explicit_none_runs_the_guard_and_records_rejection(self) -> None:
        """E1: an opted-in guard that rejects None runs on the explicit None and records the rejection."""
        property_mapping = {"k": PropertySpec("k", match_guard=_rejects_none, allow_explicit_none=True)}

        rejection = FeatureChainParserMixin._first_rejecting_guard(Options(context={"k": None}), property_mapping)
        assert rejection == ("k", None)

    def test_flagless_explicit_none_skips_the_guard(self) -> None:
        """E2 contrast (DoD 3): without the flag the guard is skipped for the None, so nothing is rejected."""
        property_mapping = {"k": PropertySpec("k", match_guard=_rejects_none)}

        rejection = FeatureChainParserMixin._first_rejecting_guard(Options(context={"k": None}), property_mapping)
        assert rejection is None
