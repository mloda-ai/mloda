"""Captureless PREFIX_PATTERNs bind nothing and no longer fabricate an operation token (issue #772).

#770 fabricated a token for a captureless match via ``operation_part.split("_")[0]`` (e.g. "cleaned"
from ``x__cleaned_text``) and fed it into reverse-lookup binding, the ``_name_identifies_group`` gate,
the public ``parse_feature_name`` adapter, and the forwarded-mismatch check. #772 retires that
fabrication: a captureless pattern is a pure RECOGNITION predicate. It still identifies its feature
group, but it binds nothing, injects nothing into effective options, and cannot be steered by an
unrelated allowed value. A class that declares a captureless pattern with a non-empty PROPERTY_MAPPING
must opt in to the recognition-only form with ``RECOGNITION_ONLY_PATTERN = True`` or it is warned at
class-definition time.

All fixture markers carry a "c772" suffix so they cannot collide in the global plugin registry.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import PropertySpec
from mloda_plugins.feature_group.experimental.text_cleaning.base import TextCleaningFeatureGroup

CLEANED_TEXT_NAME = "x__cleaned_text"
CLEANED_TEXT_PATTERN = r".*__cleaned_text$"
FEATURE_CHAIN_PARSER_LOGGER = "mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser"

# "cleaned" is exactly the token the retired code fabricated from ``__cleaned_text``.
FABRICATED_TOKEN = "cleaned"  # nosec B105
OPERATION_KEY_C772 = "operation_c772"


def _inherited_child_options(consumer_group: dict[str, Any]) -> Options:
    """Build child options exactly like the engine does: inherit_from the consumer (records provenance)."""
    child_options = Options()
    child_options.inherit_from(Options(group=consumer_group))
    return child_options


class TestLegacyOperationConfigNoFabrication:
    """``_legacy_operation_config`` is the first positional capture, or None; it fabricates nothing."""

    def test_captureless_returns_none(self) -> None:
        """A captureless match has zero positional captures, so there is no operation value to return."""
        parsed = FeatureChainParser.parse_name(CLEANED_TEXT_NAME, [CLEANED_TEXT_PATTERN])

        assert FeatureChainParser._legacy_operation_config(parsed) is None

    def test_positional_still_returns_group_one(self) -> None:
        """A positional pattern is unchanged: group 1 is the operation value."""
        parsed = FeatureChainParser.parse_name("f0__pca_legacy_c772", [r".*__(pca|tsne)_legacy_c772$"])

        assert FeatureChainParser._legacy_operation_config(parsed) == "pca"


class TestCapturelessStillIdentifiesGroup:
    """A captureless match is a recognition predicate: it identifies the group with no fabricated token."""

    def test_captureless_match_identifies_group_without_mapping(self) -> None:
        """Zero positional captures plus a match still identifies the group, even without a mapping."""
        parsed = FeatureChainParser.parse_name(CLEANED_TEXT_NAME, [CLEANED_TEXT_PATTERN])

        assert FeatureChainParser._name_identifies_group(parsed, None) is True

    def test_captureless_match_identifies_group_with_mapping(self) -> None:
        """The shipped captureless mapping binds nothing, so recognition alone must identify the group."""
        parsed = FeatureChainParser.parse_name(CLEANED_TEXT_NAME, [CLEANED_TEXT_PATTERN])

        assert FeatureChainParser._name_identifies_group(parsed, TextCleaningFeatureGroup.PROPERTY_MAPPING) is True

    def test_text_cleaning_match_criteria_still_true(self) -> None:
        """The recognition predicate keeps claiming its chained name with no options at all."""
        assert TextCleaningFeatureGroup.match_feature_group_criteria("review__cleaned_text", Options()) is True

    def test_optional_first_positional_group_does_not_identify(self) -> None:
        """A POSITIONAL optional-first group that did not participate still does NOT identify (unchanged)."""
        parsed = FeatureChainParser.parse_name("x__optfirst_c772", [r".*__(?:(pca|tsne)_)?optfirst_c772$"])

        assert parsed.positional_captures == (None,)  # precondition: one positional group, absent
        assert FeatureChainParser._name_identifies_group(parsed, None) is False


class TestCapturelessBindsNothing:
    """An unrelated allowed value cannot change a captureless pattern's behavior (DoD item 6)."""

    def test_bind_name_captures_ignores_unrelated_allowed_value(self) -> None:
        """Even when a key's allowed_values contains the old fabricated token, a captureless match binds nothing."""
        parsed = FeatureChainParser.parse_name(CLEANED_TEXT_NAME, [CLEANED_TEXT_PATTERN])
        property_mapping = {
            OPERATION_KEY_C772: PropertySpec(
                "op", allowed_values=(FABRICATED_TOKEN, "other"), context=True, strict_validation=True
            ),
        }

        assert FeatureChainParser.bind_name_captures(parsed, property_mapping) == {}

    def test_build_effective_options_injects_nothing(self) -> None:
        """Nothing to bind means nothing to merge: the original options come back by identity, untouched."""
        options = Options()
        property_mapping = {
            OPERATION_KEY_C772: PropertySpec(
                "op", allowed_values=(FABRICATED_TOKEN,), context=True, strict_validation=True
            ),
        }

        effective = FeatureChainParser.build_effective_options(
            CLEANED_TEXT_NAME, [CLEANED_TEXT_PATTERN], property_mapping, options
        )

        assert effective is options
        assert effective.get(OPERATION_KEY_C772) is None


class TestCapturelessDefinitionDiagnostic:
    """A captureless pattern with a non-empty mapping must opt in to the recognition-only form."""

    def test_captureless_mapping_without_marker_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Defining such a class without the marker emits a definition-time WARNING mentioning the marker."""
        with caplog.at_level(logging.WARNING):

            class _CapturelessNoMarkerC772(FeatureChainParserMixin):
                PREFIX_PATTERN = r".*__recognize_c772$"
                PROPERTY_MAPPING = {
                    OPERATION_KEY_C772: PropertySpec(
                        "op", allowed_values=("normalize",), context=True, strict_validation=True
                    ),
                }

            assert _CapturelessNoMarkerC772.PREFIX_PATTERN.endswith("__recognize_c772$")

        warnings = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING
            and record.name == FEATURE_CHAIN_PARSER_LOGGER
            and "RECOGNITION_ONLY_PATTERN" in record.getMessage()
        ]
        assert warnings

    def test_recognition_only_marker_suppresses_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """The explicit no-binding form (RECOGNITION_ONLY_PATTERN=True) emits no such definition-time warning."""
        with caplog.at_level(logging.WARNING):

            class _CapturelessWithMarkerC772(FeatureChainParserMixin):
                RECOGNITION_ONLY_PATTERN = True
                PREFIX_PATTERN = r".*__recognize2_c772$"
                PROPERTY_MAPPING = {
                    OPERATION_KEY_C772: PropertySpec(
                        "op", allowed_values=("normalize",), context=True, strict_validation=True
                    ),
                }

            assert _CapturelessWithMarkerC772.RECOGNITION_ONLY_PATTERN is True

        warnings = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING
            and record.name == FEATURE_CHAIN_PARSER_LOGGER
            and "RECOGNITION_ONLY_PATTERN" in record.getMessage()
        ]
        assert not warnings


class TestCapturelessForwardedMismatchAbsent:
    """A captureless pattern binds nothing, so forwarded-mismatch protection never fires for it."""

    def test_forwarded_option_does_not_raise_for_captureless(self) -> None:
        """A consumer-forwarded PROPERTY_MAPPING key present on the child is no mismatch: the match stays True."""
        child_options = _inherited_child_options({TextCleaningFeatureGroup.CLEANING_OPERATIONS: "normalize"})
        # precondition: the key really arrived via forwarding (provenance recorded)
        assert TextCleaningFeatureGroup.CLEANING_OPERATIONS in child_options.inherited_group_keys

        result = TextCleaningFeatureGroup.match_feature_group_criteria("review__cleaned_text", child_options)

        assert result is True


class TestTextCleaningMigration:
    """The only shipped captureless plugin declares itself recognition-only."""

    def test_recognition_only_pattern_is_true(self) -> None:
        """TextCleaningFeatureGroup opts in to the recognition-only form."""
        assert TextCleaningFeatureGroup.RECOGNITION_ONLY_PATTERN is True
