"""Definition-time diagnostic for a "universal configuration matcher" PROPERTY_MAPPING (issue #771).

A FeatureGroup whose PROPERTY_MAPPING has ZERO unconditionally-required keys, declares no capturing
pattern, and inherits the mixin's configuration matcher will match ANY feature name with EMPTY
options (Finding 10 of #750). That silent over-matching is what this diagnostic warns about at
class-definition time. The warning names the escape hatch ``ALLOW_UNIVERSAL_MATCHER`` and the class.

Authors silence it legitimately by: declaring an unconditionally-required key, supplying a genuinely
discriminating ``match_feature_group_criteria``, gating with ``required_when`` predicates that fire
for empty options, or opting in with ``ALLOW_UNIVERSAL_MATCHER = True``.

Requiredness (final PropertySpec semantics, already in the repo):
- ``default=NO_DEFAULT`` AND ``required_when is None``  -> unconditionally required.
- any declared ``default`` (including ``default=None``)  -> optional.
- ``required_when=<predicate>``                          -> conditionally required (not unconditional).
``FeatureChainParser._can_skip_required_check(spec)`` is True for the optional-or-conditional cases.

Every fixture carries a "u771" marker in its class name, keys, and values so it cannot collide in the
global plugin registry; the captureless test (test_captureless_no_binding.py) uses "c772" the same way.
The universal fixtures declare NO PREFIX_PATTERN/SUFFIX_PATTERN (a pure-config matcher), so the ONLY
diagnostic in scope is the universal-matcher warning, never the captureless one.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import PropertySpec
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup

FEATURE_CHAIN_PARSER_LOGGER = "mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser"

# A name that no fixture pattern would ever recognize, used as the "unrelated probe".
UNRELATED_NAME_U771 = "some_unrelated_feature_u771"

# Case 5's two mutually-exclusive conditional keys (modeled on the sklearn PIPELINE_NAME/PIPELINE_STEPS
# pattern): each is required only when the other is absent, so EMPTY options leaves both required.
COND_KEY_A_U771 = "pipe_a_u771e"
COND_KEY_B_U771 = "pipe_b_u771e"


def _universal_matcher_warnings(
    caplog: pytest.LogCaptureFixture, class_name: str | None = None
) -> list[logging.LogRecord]:
    """The ALLOW_UNIVERSAL_MATCHER definition-time warnings, optionally scoped to one class name.

    Filters exactly like the captureless test filters RECOGNITION_ONLY_PATTERN records: by logger
    name, WARNING level, and the marker substring in the rendered message.
    """
    records = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and record.name == FEATURE_CHAIN_PARSER_LOGGER
        and "ALLOW_UNIVERSAL_MATCHER" in record.getMessage()
    ]
    if class_name is not None:
        records = [record for record in records if class_name in record.getMessage()]
    return records


class TestUniversalMatcherWarns:
    """The guard warns when an inherited config matcher matches any feature name with empty options."""

    def test_inherited_all_declared_default_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 1: every key declares ``default=None``, no pattern -> universal matcher -> WARNS."""
        with caplog.at_level(logging.WARNING):

            class _InheritedAllDefaultU771a(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "opt_a_u771a": PropertySpec("optional a", default=None),
                    "opt_b_u771a": PropertySpec("optional b", default=None),
                }

            assert "opt_a_u771a" in _InheritedAllDefaultU771a.PROPERTY_MAPPING

        warnings = _universal_matcher_warnings(caplog, "_InheritedAllDefaultU771a")
        assert warnings, "expected an ALLOW_UNIVERSAL_MATCHER warning naming the class"

    def test_declared_default_value_is_optional_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 6: a concrete declared default (not just None) still counts as optional -> WARNS."""
        with caplog.at_level(logging.WARNING):

            class _DeclaredDefaultU771f(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "defaulted_u771f": PropertySpec(
                        "declared concrete default",
                        default="x_u771",
                        strict_validation=True,
                        allowed_values=("x_u771",),
                    ),
                }

            # Precondition: a concrete declared default makes the key optional, not required.
            spec = _DeclaredDefaultU771f.PROPERTY_MAPPING["defaulted_u771f"]
            assert FeatureChainParser._can_skip_required_check(spec) is True

        warnings = _universal_matcher_warnings(caplog, "_DeclaredDefaultU771f")
        assert warnings, "a declared default is optional, so the class is still a universal matcher"

    def test_passthrough_override_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 4: an override that only delegates via super() stays universal -> WARNS.

        This is the DoD "distinguish genuine custom from pass-through" case: a pass-through matcher is
        as universal as the inherited one, so the guard must still fire.
        """
        with caplog.at_level(logging.WARNING):

            class _PassThroughU771d(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"opt_u771d": PropertySpec("optional", default=None)}

                @classmethod
                def match_feature_group_criteria(
                    cls,
                    feature_name: str | FeatureName,
                    options: Options,
                    data_access_collection: Any = None,
                ) -> bool:
                    return super().match_feature_group_criteria(feature_name, options, data_access_collection)

            # Precondition: the pass-through override still matches an unrelated name with empty options.
            assert _PassThroughU771d.match_feature_group_criteria("anything_u771d", Options()) is True

        warnings = _universal_matcher_warnings(caplog, "_PassThroughU771d")
        assert warnings, "a pass-through override is still a universal matcher and must warn"


class TestUniversalMatcherDoesNotWarn:
    """The guard stays quiet whenever the class is NOT a universal matcher, or opts out explicitly."""

    def test_unconditionally_required_key_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 2: one unconditionally-required key (NO_DEFAULT, no required_when) -> NO WARN."""
        with caplog.at_level(logging.WARNING):

            class _RequiredKeyU771b(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "required_u771b": PropertySpec(
                        "required op", allowed_values=("agg_u771b",), strict_validation=True
                    ),
                    "opt_u771b": PropertySpec("optional", default=None),
                }

            # Precondition: the first key is unconditionally required (cannot be skipped).
            required_spec = _RequiredKeyU771b.PROPERTY_MAPPING["required_u771b"]
            assert FeatureChainParser._can_skip_required_check(required_spec) is False

        assert not _universal_matcher_warnings(caplog), "a required key means the matcher is not universal"

    def test_genuine_custom_matcher_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 3: a real, discriminating override rejects the unrelated probe -> NO WARN."""
        with caplog.at_level(logging.WARNING):

            class _GenuineCustomU771c(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"opt_u771c": PropertySpec("optional", default=None)}

                @classmethod
                def match_feature_group_criteria(
                    cls,
                    feature_name: str | FeatureName,
                    options: Options,
                    data_access_collection: Any = None,
                ) -> bool:
                    return str(feature_name) == "specific_u771c"

            # Precondition: the custom matcher really discriminates (True only for its own name).
            assert _GenuineCustomU771c.match_feature_group_criteria("specific_u771c", Options()) is True
            assert _GenuineCustomU771c.match_feature_group_criteria(UNRELATED_NAME_U771, Options()) is False

        assert not _universal_matcher_warnings(caplog), "a genuinely custom matcher is not universal"

    def test_conditional_requirement_fires_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 5: mutually-exclusive required_when keys make empty options a non-match -> NO WARN."""
        with caplog.at_level(logging.WARNING):

            class _ConditionalU771e(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    COND_KEY_A_U771: PropertySpec(
                        "a, required when b is absent",
                        default=None,
                        required_when=(lambda options: options.get(COND_KEY_B_U771) is None),
                    ),
                    COND_KEY_B_U771: PropertySpec(
                        "b, required when a is absent",
                        default=None,
                        required_when=(lambda options: options.get(COND_KEY_A_U771) is None),
                    ),
                }

            # Precondition: with EMPTY options at least one conditional key is required, so the
            # config match fails and the class is therefore NOT a universal matcher.
            assert _ConditionalU771e.match_feature_group_criteria(UNRELATED_NAME_U771, Options()) is False

        assert not _universal_matcher_warnings(caplog), "a conditional requirement that fires is not universal"

    def test_escape_hatch_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """Case 7: ALLOW_UNIVERSAL_MATCHER = True opts out of the diagnostic -> NO WARN."""
        with caplog.at_level(logging.WARNING):

            class _EscapeHatchU771g(FeatureChainParserMixin, FeatureGroup):
                ALLOW_UNIVERSAL_MATCHER = True
                PROPERTY_MAPPING = {"opt_u771g": PropertySpec("optional", default=None)}

            assert _EscapeHatchU771g.ALLOW_UNIVERSAL_MATCHER is True

        assert not _universal_matcher_warnings(caplog), "the escape hatch must silence the diagnostic"


class TestUniversalMatcherMotivation:
    """Behavioral pins that document WHY the guard exists and keep shipped plugins out of scope."""

    def test_inherited_all_optional_matches_unrelated_name(self) -> None:
        """Case 8: without the escape hatch, an all-optional inherited matcher claims an unrelated name.

        This is the motivation for the diagnostic and holds both before and after implementation: the
        guard only warns, it does not change matching behavior.
        """

        class _UniversalMotivationU771h(FeatureChainParserMixin, FeatureGroup):
            PROPERTY_MAPPING = {"opt_u771h": PropertySpec("optional", default=None)}

        assert _UniversalMotivationU771h.match_feature_group_criteria(UNRELATED_NAME_U771, Options()) is True

    def test_shipped_aggregated_feature_group_is_not_universal(self) -> None:
        """Case 9: a representative shipped plugin has an unconditionally-required key -> out of scope.

        Pins that shipped plugins do not trip the diagnostic; passes before and after implementation.
        """
        assert any(
            not FeatureChainParser._can_skip_required_check(spec)
            for spec in AggregatedFeatureGroup.PROPERTY_MAPPING.values()
        ), "AggregatedFeatureGroup must keep an unconditionally-required key so it is not a universal matcher"
