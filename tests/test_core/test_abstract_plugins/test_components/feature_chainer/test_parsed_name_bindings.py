"""Structured parsed-name bindings for PROPERTY_MAPPING (issue #770).

``parse_name`` returns the parse as FACTS (``ParsedFeatureName``), and ``bind_name_captures`` turns those
facts into PROPERTY_MAPPING bindings by NAME instead of by a reverse allowed_values lookup. That makes a
secondary capture and an element_validator-only spec reachable from the feature name.

All fixture names carry a "pnb770" marker so they cannot collide with other tests in the global registry.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_chainer.parsed_feature_name import ParsedFeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import PropertySpec
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.scaling.base import ScalingFeatureGroup
from mloda_plugins.feature_group.experimental.text_cleaning.base import TextCleaningFeatureGroup


ALGORITHM_KEY = "algorithm_pnb770"
SIZE_KEY = "size_pnb770"
SOLVER_KEY = "solver_pnb770"
NOTES_KEY = "notes_pnb770"

ALGORITHMS = {"pca": "PCA", "tsne": "t-SNE"}


def _is_digit_string(value: Any) -> bool:
    """Accept a non-empty digit string, the shape a regex capture yields."""
    return isinstance(value, str) and value.isdigit()


def _rejects_tsne(value: Any) -> bool:
    """match_guard: everything but tsne passes."""
    return bool(value != "tsne")


def _needs_notes(options: Options) -> bool:
    """required_when: notes is required once the name carries size 3."""
    return bool(options.get(SIZE_KEY) == "3")


def _algorithm_spec(**overrides: Any) -> PropertySpec:
    """A strict allowed_values spec for the algorithm key."""
    kwargs: dict[str, Any] = {
        "allowed_values": ALGORITHMS,
        "context": True,
        "strict_validation": True,
    }
    kwargs.update(overrides)
    return PropertySpec("Algorithm of the pnb770 fixture", **kwargs)


def _size_spec() -> PropertySpec:
    """An element_validator-only spec: it declares NO allowed_values, so no reverse lookup can reach it."""
    return PropertySpec(
        "Size of the pnb770 fixture",
        context=True,
        strict_validation=True,
        element_validator=_is_digit_string,
    )


class NamedCaptureGroup(FeatureChainParserMixin):
    """Two named captures: algorithm has a value space, size has only an element_validator."""

    PREFIX_PATTERN = rf".*__(?P<{ALGORITHM_KEY}>pca|tsne)_(?P<{SIZE_KEY}>\d+)d_pnb770$"

    PROPERTY_MAPPING: dict[str, PropertySpec] = {
        ALGORITHM_KEY: _algorithm_spec(),
        SIZE_KEY: _size_spec(),
    }


class OptionalCaptureGroup(FeatureChainParserMixin):
    """The size group is optional, so it may not participate in a match."""

    PREFIX_PATTERN = rf".*__(?P<{ALGORITHM_KEY}>pca|tsne)(?:_(?P<{SIZE_KEY}>\d+)d)?_optional_pnb770$"

    PROPERTY_MAPPING: dict[str, PropertySpec] = {
        ALGORITHM_KEY: _algorithm_spec(),
        SIZE_KEY: _size_spec(),
    }


class UnmappedNamedCaptureGroup(FeatureChainParserMixin):
    """A named capture that is no mapping key, whose VALUE would satisfy the legacy reverse lookup."""

    PREFIX_PATTERN = r".*__(?P<unmapped_pnb770>pca|tsne)_unmapped_pnb770$"

    PROPERTY_MAPPING: dict[str, PropertySpec] = {ALGORITHM_KEY: _algorithm_spec()}


class LegacyPositionalGroup(FeatureChainParserMixin):
    """No named capture at all: the legacy allowed_values fallback still binds group 1."""

    PREFIX_PATTERN = r".*__(pca|tsne)_legacy_pnb770$"

    PROPERTY_MAPPING: dict[str, PropertySpec] = {
        ALGORITHM_KEY: _algorithm_spec(),
        NOTES_KEY: PropertySpec("Free text, no value space", context=True, strict_validation=False, default=""),
    }


class RequiredWhenNamedGroup(FeatureChainParserMixin):
    """The SECONDARY capture drives a required_when predicate."""

    PREFIX_PATTERN = rf".*__(?P<{ALGORITHM_KEY}>pca|tsne)_(?P<{SIZE_KEY}>\d+)d_reqwhen_pnb770$"

    PROPERTY_MAPPING: dict[str, PropertySpec] = {
        ALGORITHM_KEY: _algorithm_spec(),
        SIZE_KEY: _size_spec(),
        NOTES_KEY: PropertySpec("Notes", context=True, strict_validation=False, required_when=_needs_notes),
    }


class GuardedNamedGroup(FeatureChainParserMixin):
    """A match_guard on a key the feature name carries."""

    PREFIX_PATTERN = rf".*__(?P<{ALGORITHM_KEY}>pca|tsne)_guard_pnb770$"

    PROPERTY_MAPPING: dict[str, PropertySpec] = {
        ALGORITHM_KEY: _algorithm_spec(strict_validation=False, match_guard=_rejects_tsne)
    }


class StrictNamedGroup(FeatureChainParserMixin):
    """The capture is wider than the value space, so the name can carry a rejected value."""

    PREFIX_PATTERN = rf".*__(?P<{ALGORITHM_KEY}>\w+)_strict_pnb770$"

    PROPERTY_MAPPING: dict[str, PropertySpec] = {ALGORITHM_KEY: _algorithm_spec()}


class ForwardedSecondaryGroup(FeatureChainParserMixin):
    """Both keys are group-categorized (context=False): only group options flow through forwarding."""

    PREFIX_PATTERN = rf".*__(?P<{ALGORITHM_KEY}>pca|tsne)_(?P<{SOLVER_KEY}>auto|arpack)_fwd_pnb770$"

    PROPERTY_MAPPING: dict[str, PropertySpec] = {
        ALGORITHM_KEY: _algorithm_spec(context=False),
        SOLVER_KEY: PropertySpec(
            "Solver of the pnb770 fixture",
            allowed_values={"auto": "Auto", "arpack": "ARPACK"},
            context=False,
            strict_validation=True,
        ),
    }


FORWARDED_FEATURE_NAME = "sales__pca_auto_fwd_pnb770"


def _inherited_child_options(consumer_group: dict[str, Any]) -> Options:
    """Build child options exactly like the engine does: inherit_from the consumer."""
    child_options = Options()
    child_options.inherit_from(Options(group=consumer_group))
    return child_options


class TestParsedFeatureNameShape:
    """The parse result is a frozen record of facts, mirroring what ``re`` reports."""

    def test_fields_carry_the_parse(self) -> None:
        """matched, source_feature, operation_part, named_captures and positional_captures are the shape."""
        parsed = FeatureChainParser.parse_name("f0__pca_2d_pnb770", [NamedCaptureGroup.PREFIX_PATTERN])

        assert parsed.matched is True
        assert parsed.source_feature == "f0"
        assert parsed.operation_part == "pca_2d_pnb770"
        assert parsed.named_captures == {ALGORITHM_KEY: "pca", SIZE_KEY: "2"}
        assert parsed.positional_captures == ("pca", "2")

    def test_result_is_frozen(self) -> None:
        """The parse is a fact, so it cannot be rewritten after the fact."""
        parsed = ParsedFeatureName(matched=True, source_feature="f0", operation_part="pca_2d_pnb770")

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(parsed, "matched", False)

    def test_defaults_describe_a_captureless_miss(self) -> None:
        """Only ``matched`` has no default; everything else defaults to empty."""
        parsed = ParsedFeatureName(matched=False)

        assert parsed.source_feature is None
        assert parsed.operation_part is None
        assert parsed.named_captures == {}
        assert parsed.positional_captures == ()

    def test_named_capture_appears_in_both_capture_views(self) -> None:
        """A named group is reported by name AND by position, exactly as ``re`` does."""
        pattern = NamedCaptureGroup.PREFIX_PATTERN
        parsed = FeatureChainParser.parse_name("f0__tsne_7d_pnb770", [pattern])
        match = re.match(pattern, "f0__tsne_7d_pnb770")

        assert match is not None
        assert parsed.named_captures == match.groupdict()
        assert parsed.positional_captures == match.groups()

    def test_non_participating_optional_group_is_none_in_both_views(self) -> None:
        """An optional group that did not participate is None by name and by position."""
        parsed = FeatureChainParser.parse_name("f0__pca_optional_pnb770", [OptionalCaptureGroup.PREFIX_PATTERN])

        assert parsed.matched is True
        assert parsed.named_captures == {ALGORITHM_KEY: "pca", SIZE_KEY: None}
        assert parsed.positional_captures == ("pca", None)


class TestParseName:
    """parse_name keeps today's matching semantics and fabricates nothing."""

    def test_no_pattern_match_is_a_miss(self) -> None:
        """A name no pattern matches yields matched=False and no facts."""
        parsed = FeatureChainParser.parse_name("unrelated_pnb770", [NamedCaptureGroup.PREFIX_PATTERN])

        assert parsed.matched is False
        assert parsed.source_feature is None
        assert parsed.named_captures == {}
        assert parsed.positional_captures == ()

    def test_pattern_match_without_source_feature_raises(self) -> None:
        """A matched name with nothing before the separator still raises today's ValueError."""
        with pytest.raises(ValueError, match="but has no source feature: orphan_pnb770_thing"):
            FeatureChainParser.parse_name("orphan_pnb770_thing", [r"^orphan_pnb770_(\w+)$"])

    def test_captureless_match_has_no_captures(self) -> None:
        """A captureless pattern matches with zero captures of either kind."""
        parsed = FeatureChainParser.parse_name("x__cleaned_text", [r".*__cleaned_text$"])

        assert parsed.matched is True
        assert parsed.source_feature == "x"
        assert parsed.named_captures == {}
        assert parsed.positional_captures == ()

    def test_captureless_match_carries_the_raw_suffix_not_a_fabricated_token(self) -> None:
        """operation_part is the raw text after the separator; the legacy 'cleaned' token is nowhere."""
        parsed = FeatureChainParser.parse_name("x__cleaned_text", [r".*__cleaned_text$"])

        assert parsed.operation_part == "cleaned_text"
        assert "cleaned" not in dataclasses.asdict(parsed).values()


class TestLegacyParseFeatureNameAdapter:
    """parse_feature_name keeps returning exactly today's tuples: it is public API."""

    def test_positional_capture_tuple_unchanged(self) -> None:
        """Group 1 plus the source feature, as always."""
        assert FeatureChainParser.parse_feature_name(
            "f0__pca_legacy_pnb770", [LegacyPositionalGroup.PREFIX_PATTERN]
        ) == (
            "pca",
            "f0",
        )

    def test_named_capture_tuple_uses_the_first_group(self) -> None:
        """A named group is group 1 too, so the legacy tuple is unchanged."""
        assert FeatureChainParser.parse_feature_name("f0__pca_2d_pnb770", [NamedCaptureGroup.PREFIX_PATTERN]) == (
            "pca",
            "f0",
        )

    def test_captureless_no_longer_fabricates(self) -> None:
        """#772: a captureless match no longer fabricates a token; operation_config is None, the source stays."""
        assert FeatureChainParser.parse_feature_name("x__cleaned_text", [r".*__cleaned_text$"]) == (None, "x")

    def test_miss_tuple_unchanged(self) -> None:
        """No pattern match is still (None, None)."""
        assert FeatureChainParser.parse_feature_name("unrelated_pnb770", [NamedCaptureGroup.PREFIX_PATTERN]) == (
            None,
            None,
        )

    def test_no_source_feature_valueerror_unchanged(self) -> None:
        """match_parser_criteria depends on this raise, so the adapter must not swallow it."""
        with pytest.raises(ValueError, match="but has no source feature: orphan_pnb770_thing"):
            FeatureChainParser.parse_feature_name("orphan_pnb770_thing", [r"^orphan_pnb770_(\w+)$"])


class TestBindNameCaptures:
    """Binding is by name, deterministic, and documented: no first-matching-allowed_values search."""

    def test_named_capture_binds_to_the_same_named_key(self) -> None:
        """A named capture binds to the PROPERTY_MAPPING key of the same name."""
        parsed = FeatureChainParser.parse_name("f0__pca_2d_pnb770", [NamedCaptureGroup.PREFIX_PATTERN])

        bindings = FeatureChainParser.bind_name_captures(parsed, NamedCaptureGroup.PROPERTY_MAPPING)

        assert bindings[ALGORITHM_KEY] == "pca"

    def test_named_capture_binds_to_element_validator_only_spec(self) -> None:
        """The core fix: a spec with NO allowed_values receives a name-derived value."""
        parsed = FeatureChainParser.parse_name("f0__pca_2d_pnb770", [NamedCaptureGroup.PREFIX_PATTERN])

        bindings = FeatureChainParser.bind_name_captures(parsed, NamedCaptureGroup.PROPERTY_MAPPING)

        assert bindings[SIZE_KEY] == "2"

    def test_multiple_named_captures_bind_to_separate_keys(self) -> None:
        """Every capture reaches its own key, not just the first one."""
        parsed = FeatureChainParser.parse_name("f0__tsne_7d_pnb770", [NamedCaptureGroup.PREFIX_PATTERN])

        bindings = FeatureChainParser.bind_name_captures(parsed, NamedCaptureGroup.PROPERTY_MAPPING)

        assert bindings == {ALGORITHM_KEY: "tsne", SIZE_KEY: "7"}

    def test_named_capture_that_is_no_mapping_key_is_ignored(self) -> None:
        """Patterns may use named groups for other purposes, so an unmapped name binds nothing."""
        parsed = FeatureChainParser.parse_name("f0__pca_unmapped_pnb770", [UnmappedNamedCaptureGroup.PREFIX_PATTERN])

        bindings = FeatureChainParser.bind_name_captures(parsed, UnmappedNamedCaptureGroup.PROPERTY_MAPPING)

        assert bindings == {}

    def test_named_pattern_never_falls_back_to_the_reverse_lookup(self) -> None:
        """'pca' is in algorithm's allowed_values, but a named pattern binds by name ONLY."""
        parsed = FeatureChainParser.parse_name("f0__pca_unmapped_pnb770", [UnmappedNamedCaptureGroup.PREFIX_PATTERN])

        bindings = FeatureChainParser.bind_name_captures(parsed, UnmappedNamedCaptureGroup.PROPERTY_MAPPING)

        assert ALGORITHM_KEY not in bindings

    def test_non_participating_named_capture_binds_nothing(self) -> None:
        """A None capture has no value to bind."""
        parsed = FeatureChainParser.parse_name("f0__pca_optional_pnb770", [OptionalCaptureGroup.PREFIX_PATTERN])

        bindings = FeatureChainParser.bind_name_captures(parsed, OptionalCaptureGroup.PROPERTY_MAPPING)

        assert bindings == {ALGORITHM_KEY: "pca"}

    def test_legacy_positional_fallback_binds_group_one(self) -> None:
        """With no named capture anywhere, group 1 still binds via allowed_values membership."""
        parsed = FeatureChainParser.parse_name("f0__pca_legacy_pnb770", [LegacyPositionalGroup.PREFIX_PATTERN])

        bindings = FeatureChainParser.bind_name_captures(parsed, LegacyPositionalGroup.PROPERTY_MAPPING)

        assert bindings == {ALGORITHM_KEY: "pca"}

    def test_legacy_fallback_binds_nothing_when_no_value_space_contains_the_value(self) -> None:
        """The fallback only ever binds a value that is already a member of an allowed_values."""
        parsed = FeatureChainParser.parse_name("f0__bogus_strict_pnb770", [r".*__(\w+)_strict_pnb770$"])

        bindings = FeatureChainParser.bind_name_captures(parsed, LegacyPositionalGroup.PROPERTY_MAPPING)

        assert bindings == {}

    def test_miss_binds_nothing(self) -> None:
        """An unmatched name has no captures to bind."""
        parsed = FeatureChainParser.parse_name("unrelated_pnb770", [NamedCaptureGroup.PREFIX_PATTERN])

        assert FeatureChainParser.bind_name_captures(parsed, NamedCaptureGroup.PROPERTY_MAPPING) == {}


class TestDefinitionTimeAmbiguity:
    """Legacy positional binding over intersecting value spaces is order-dependent: reject it at definition."""

    def test_positional_capture_with_overlapping_value_spaces_raises(self) -> None:
        """Two keys reachable by the same captured string make the binding a guess."""
        with pytest.raises(ValueError) as exc_info:

            class _OverlappingPositionalPnb770(FeatureChainParserMixin):
                PREFIX_PATTERN = r".*__(\w+)_overlap_pnb770$"
                PROPERTY_MAPPING = {
                    ALGORITHM_KEY: PropertySpec("a", allowed_values=("pca", "shared_pnb770")),
                    SOLVER_KEY: PropertySpec("b", allowed_values=("shared_pnb770", "auto")),
                }

        message = str(exc_info.value)
        assert "_OverlappingPositionalPnb770" in message
        assert ALGORITHM_KEY in message
        assert SOLVER_KEY in message
        assert "shared_pnb770" in message
        assert "(?P<" in message

    def test_feature_group_definition_is_validated_too(self) -> None:
        """The check runs from FeatureGroup.__init_subclass__, not only from the mixin."""
        with pytest.raises(ValueError) as exc_info:

            class _OverlappingFeatureGroupPnb770(FeatureGroup):
                PREFIX_PATTERN = r".*__(\w+)_overlapfg_pnb770$"
                PROPERTY_MAPPING = {
                    ALGORITHM_KEY: PropertySpec("a", allowed_values=("pca", "shared_pnb770")),
                    SOLVER_KEY: PropertySpec("b", allowed_values=("shared_pnb770", "auto")),
                }

        assert "shared_pnb770" in str(exc_info.value)

    def test_named_captures_make_the_same_overlap_unambiguous(self) -> None:
        """Named binding is explicit, so intersecting value spaces are no problem at all."""

        class _OverlappingNamedPnb770(FeatureChainParserMixin):
            PREFIX_PATTERN = rf".*__(?P<{ALGORITHM_KEY}>\w+)_(?P<{SOLVER_KEY}>\w+)_named_pnb770$"
            PROPERTY_MAPPING = {
                ALGORITHM_KEY: PropertySpec("a", allowed_values=("pca", "shared_pnb770")),
                SOLVER_KEY: PropertySpec("b", allowed_values=("shared_pnb770", "auto")),
            }

        parsed = FeatureChainParser.parse_name(
            "f0__shared_pnb770_auto_named_pnb770", [_OverlappingNamedPnb770.PREFIX_PATTERN]
        )
        bindings = FeatureChainParser.bind_name_captures(parsed, _OverlappingNamedPnb770.PROPERTY_MAPPING)

        assert bindings == {ALGORITHM_KEY: "shared_pnb770", SOLVER_KEY: "auto"}

    def test_non_str_overlap_does_not_raise(self) -> None:
        """A capture is always a str, so a non-str member is unreachable and cannot be ambiguous."""

        class _NonStrOverlapPnb770(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__(\w+)_nonstr_pnb770$"
            PROPERTY_MAPPING = {
                ALGORITHM_KEY: PropertySpec("a", allowed_values=(1, 2)),
                SOLVER_KEY: PropertySpec("b", allowed_values=(2, 3)),
            }

        assert _NonStrOverlapPnb770.PREFIX_PATTERN.endswith("_nonstr_pnb770$")

    def test_captureless_pattern_with_overlap_does_not_raise(self) -> None:
        """With no capture group there is no value to misbind."""

        class _CapturelessOverlapPnb770(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__captureless_pnb770$"
            PROPERTY_MAPPING = {
                ALGORITHM_KEY: PropertySpec("a", allowed_values=("shared_pnb770",)),
                SOLVER_KEY: PropertySpec("b", allowed_values=("shared_pnb770",)),
            }

        assert _CapturelessOverlapPnb770.PREFIX_PATTERN.endswith("__captureless_pnb770$")

    def test_uncompilable_pattern_does_not_break_class_definition(self) -> None:
        """A pattern this check cannot compile is treated as declaring no named group, never a definition crash."""

        class _BrokenPatternPnb770(FeatureChainParserMixin):
            PREFIX_PATTERN = r".*__([unclosed_pnb770$"
            PROPERTY_MAPPING = {ALGORITHM_KEY: _algorithm_spec()}

        assert _BrokenPatternPnb770.PROPERTY_MAPPING[ALGORITHM_KEY].strict_validation is True


class TestBuildEffectiveOptions:
    """Every binding is merged at once, and nothing to merge means the very same object back."""

    def test_all_bindings_are_merged(self) -> None:
        """Merging no longer stops after the first key."""
        effective = FeatureChainParser.build_effective_options(
            "f0__pca_2d_pnb770",
            [NamedCaptureGroup.PREFIX_PATTERN],
            NamedCaptureGroup.PROPERTY_MAPPING,
            Options(),
        )

        assert effective.get(ALGORITHM_KEY) == "pca"
        assert effective.get(SIZE_KEY) == "2"

    def test_present_option_wins_over_a_binding(self) -> None:
        """An explicit option is never overwritten by a name-derived value."""
        options = Options(context={ALGORITHM_KEY: "tsne"})

        effective = FeatureChainParser.build_effective_options(
            "f0__pca_2d_pnb770", [NamedCaptureGroup.PREFIX_PATTERN], NamedCaptureGroup.PROPERTY_MAPPING, options
        )

        assert effective.get(ALGORITHM_KEY) == "tsne"
        assert effective.get(SIZE_KEY) == "2"

    def test_nothing_to_merge_returns_the_same_object(self) -> None:
        """Identity, not a copy: an unrelated name has nothing to contribute."""
        options = Options(context={NOTES_KEY: "x"})

        effective = FeatureChainParser.build_effective_options(
            "unrelated_pnb770", [NamedCaptureGroup.PREFIX_PATTERN], NamedCaptureGroup.PROPERTY_MAPPING, options
        )

        assert effective is options

    def test_unbindable_named_pattern_returns_the_same_object(self) -> None:
        """A pattern whose named captures bind to nothing leaves the options untouched."""
        options = Options()

        effective = FeatureChainParser.build_effective_options(
            "f0__pca_unmapped_pnb770",
            [UnmappedNamedCaptureGroup.PREFIX_PATTERN],
            UnmappedNamedCaptureGroup.PROPERTY_MAPPING,
            options,
        )

        assert effective is options

    def test_propagate_context_keys_survive_the_merge(self) -> None:
        """Regression: the merged Options keeps the propagation contract of the original."""
        options = Options(context={NOTES_KEY: "x"}, propagate_context_keys=frozenset({NOTES_KEY}))

        effective = FeatureChainParser.build_effective_options(
            "f0__pca_2d_pnb770", [NamedCaptureGroup.PREFIX_PATTERN], NamedCaptureGroup.PROPERTY_MAPPING, options
        )

        assert effective.propagate_context_keys == frozenset({NOTES_KEY})
        assert effective.get(SIZE_KEY) == "2"


class TestBoundValuesAreVisible:
    """A name-bound value is a real value: required_when, match_guard and strict validation all see it."""

    def test_required_when_sees_a_bound_secondary_capture(self) -> None:
        """The predicate reads the size the name carries, so notes becomes required."""
        result = RequiredWhenNamedGroup.match_feature_group_criteria("f0__pca_3d_reqwhen_pnb770", Options())

        assert result is False

    def test_required_when_satisfied_by_the_present_option(self) -> None:
        """The same feature matches once the conditionally required option is there."""
        result = RequiredWhenNamedGroup.match_feature_group_criteria(
            "f0__pca_3d_reqwhen_pnb770", Options(context={NOTES_KEY: "n"})
        )

        assert result is True

    def test_required_when_stays_off_when_the_bound_value_does_not_trigger_it(self) -> None:
        """Guard against over-rejecting: a non-triggering bound value requires nothing."""
        result = RequiredWhenNamedGroup.match_feature_group_criteria("f0__pca_2d_reqwhen_pnb770", Options())

        assert result is True

    def test_match_guard_sees_a_bound_value(self) -> None:
        """A guard rejecting the name-carried value is a non-match."""
        result = GuardedNamedGroup.match_feature_group_criteria("f0__tsne_guard_pnb770", Options())

        assert result is False

    def test_match_guard_accepts_a_bound_value_it_allows(self) -> None:
        """The guard only rejects what it rejects."""
        result = GuardedNamedGroup.match_feature_group_criteria("f0__pca_guard_pnb770", Options())

        assert result is True

    def test_strict_validation_sees_a_bound_value(self) -> None:
        """A bound value outside the strict value space is a non-match."""
        result = StrictNamedGroup.match_feature_group_criteria("f0__bogus_strict_pnb770", Options())

        assert result is False

    def test_strict_validation_accepts_a_bound_member(self) -> None:
        """A bound member of the value space still matches."""
        result = StrictNamedGroup.match_feature_group_criteria("f0__pca_strict_pnb770", Options())

        assert result is True

    def test_rejection_reason_reports_the_bound_value(self) -> None:
        """The diagnostic replay must not disagree with the match decision."""
        reason = StrictNamedGroup._strict_validation_rejection_reason("f0__bogus_strict_pnb770", Options())

        assert reason is not None
        assert ALGORITHM_KEY in reason
        assert "bogus" in reason


class TestForwardedMismatchOverBindings:
    """Forwarded-mismatch protection covers every bound capture, not only group 1."""

    def test_fixture_matches_without_options(self) -> None:
        """Precondition: the fixture claims the chained name via string parsing."""
        assert ForwardedSecondaryGroup.match_feature_group_criteria(FORWARDED_FEATURE_NAME, Options()) is True

    def test_forwarded_secondary_capture_mismatch_raises(self) -> None:
        """The solver key is bound by a SECONDARY capture, which today's single reverse lookup misses."""
        child_options = _inherited_child_options({SOLVER_KEY: "arpack"})
        assert child_options.inherited_group_keys == frozenset({SOLVER_KEY})  # precondition

        with pytest.raises(ValueError) as exc_info:
            ForwardedSecondaryGroup.match_feature_group_criteria(FORWARDED_FEATURE_NAME, child_options)

        message = str(exc_info.value)
        assert FORWARDED_FEATURE_NAME in message
        assert SOLVER_KEY in message
        assert "arpack" in message
        assert "auto" in message
        assert "forward_group_exclude" in message
        assert "MLODA_ALLOW_FORWARDED_NAME_MISMATCH" in message

    def test_forwarded_secondary_capture_equal_value_matches(self) -> None:
        """An agreeing forwarded value is no mismatch."""
        child_options = _inherited_child_options({SOLVER_KEY: "auto"})

        assert ForwardedSecondaryGroup.match_feature_group_criteria(FORWARDED_FEATURE_NAME, child_options) is True

    def test_forwarded_primary_capture_mismatch_still_raises(self) -> None:
        """Group 1 keeps the protection it has today."""
        child_options = _inherited_child_options({ALGORITHM_KEY: "tsne"})

        with pytest.raises(ValueError, match="forward_group_exclude"):
            ForwardedSecondaryGroup.match_feature_group_criteria(FORWARDED_FEATURE_NAME, child_options)

    def test_forwarded_singleton_unwrap_still_applies_to_a_secondary_capture(self) -> None:
        """#764 semantics: a forwarded singleton equals its sole element."""
        child_options = _inherited_child_options({SOLVER_KEY: ["auto"]})

        assert ForwardedSecondaryGroup.match_feature_group_criteria(FORWARDED_FEATURE_NAME, child_options) is True

    def test_author_set_secondary_value_does_not_raise(self) -> None:
        """Only FORWARDED values are protected; an author-set one keeps today's name precedence."""
        child_options = Options(group={SOLVER_KEY: "arpack"})
        assert child_options.inherited_group_keys == frozenset()  # precondition

        assert ForwardedSecondaryGroup.match_feature_group_criteria(FORWARDED_FEATURE_NAME, child_options) is True

    def test_env_var_downgrades_the_secondary_mismatch_to_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The escape hatch covers the secondary capture too."""
        monkeypatch.setenv("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "1")
        child_options = _inherited_child_options({SOLVER_KEY: "arpack"})

        with caplog.at_level(logging.WARNING):
            result = ForwardedSecondaryGroup.match_feature_group_criteria(FORWARDED_FEATURE_NAME, child_options)

        assert result is True
        records = [record for record in caplog.records if SOLVER_KEY in record.getMessage()]
        assert len(records) == 1


class TestShippedPluginsUnchanged:
    """Shipped PREFIX_PATTERNs stay positional and must parse byte-for-byte as before."""

    def test_aggregated_feature_group_parses_unchanged(self) -> None:
        """Group 1 is the aggregation type, everything before the separator is the source."""
        parsed = FeatureChainParser.parse_feature_name("sales__sum_aggr", [AggregatedFeatureGroup.PREFIX_PATTERN])

        assert parsed == ("sum", "sales")

    def test_aggregated_feature_group_still_matches(self) -> None:
        """The shipped matcher keeps claiming its chained name with no options at all."""
        assert AggregatedFeatureGroup.match_feature_group_criteria("sales__sum_aggr", Options()) is True

    def test_scaling_feature_group_parses_unchanged(self) -> None:
        """The alternation capture is group 1, as before."""
        parsed = FeatureChainParser.parse_feature_name("income__standard_scaled", [ScalingFeatureGroup.PREFIX_PATTERN])

        assert parsed == ("standard", "income")

    def test_scaling_feature_group_rejects_an_unknown_scaler(self) -> None:
        """A value outside the alternation is still no match for the pattern."""
        parsed = FeatureChainParser.parse_feature_name("income__bogus_scaled", [ScalingFeatureGroup.PREFIX_PATTERN])

        assert parsed == (None, None)

    def test_text_cleaning_captureless_no_longer_fabricates(self) -> None:
        """The only captureless shipped pattern: #772 retires the fabrication, so operation_config is None."""
        parsed = FeatureChainParser.parse_feature_name("text__cleaned_text", [TextCleaningFeatureGroup.PREFIX_PATTERN])

        assert parsed == (None, "text")
