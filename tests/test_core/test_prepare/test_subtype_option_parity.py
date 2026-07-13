"""Failing pinning tests for SubtypeDeclaration matcher parity and hardening (issue #639 follow-up).

Defect 1: resolve_subtype reads options.get(key) raw, while
match_configuration_feature_chain_parser normalizes singleton containers
(a frozenset element is unwrapped and validated) and treats a declared
PROPERTY_MAPPING default as an accepted match. A config-accepted feature must
therefore resolve to the same normalized subtype the matcher validated, and
supports_compute_framework must gate on it.

Defect 2: resolve_subtype promises "never raises", but a malformed
PREFIX_PATTERN regex propagates re.error out of parse_feature_name.
"""

from typing import Optional

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.provider import FeatureGroup, SubtypeDeclaration, property_spec


SBPAR_KEY = "sbpar_aggregation"
SBPAR_DEFAULT_KEY = "sbpar_defaulted_aggregation"
SBPAR_BETA_SUPPORTED = frozenset({"max"})
SBPAR_UNCHAINED = "sbpar_unchained_feature"


class SbparFwAlpha(ComputeFramework):
    """First dummy compute framework for option-parity tests."""


class SbparFwBeta(ComputeFramework):
    """Second dummy compute framework for option-parity tests; 'sum' is narrowed off it."""


class SbparWindowFG(FeatureGroup):
    """Shape A keyed family; 'sum' is declared unsupported on SbparFwBeta."""

    SUBTYPES = SubtypeDeclaration(
        key=SBPAR_KEY,
        supported={SbparFwBeta.get_class_name(): SBPAR_BETA_SUPPORTED},
    )
    PROPERTY_MAPPING = {
        SBPAR_KEY: property_spec(
            "Aggregation subtype.",
            strict=True,
            allowed_values={"sum": "Sum", "max": "Max"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SbparFwAlpha, SbparFwBeta}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SbparDefaultedFG(FeatureGroup):
    """Shape A keyed family whose key declares default='sum'; 'sum' is narrowed off SbparFwBeta."""

    SUBTYPES = SubtypeDeclaration(
        key=SBPAR_DEFAULT_KEY,
        supported={SbparFwBeta.get_class_name(): SBPAR_BETA_SUPPORTED},
    )
    PROPERTY_MAPPING = {
        SBPAR_DEFAULT_KEY: property_spec(
            "Aggregation subtype with a declared default.",
            strict=True,
            allowed_values={"sum": "Sum", "max": "Max"},
            default="sum",
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SbparFwAlpha, SbparFwBeta}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SbparBadPrefixFG(FeatureGroup):
    """Keyed family whose PREFIX_PATTERN is a malformed regex; must never leak re.error."""

    SUBTYPES = SubtypeDeclaration(
        key=SBPAR_KEY,
        supported={SbparFwBeta.get_class_name(): SBPAR_BETA_SUPPORTED},
    )
    PREFIX_PATTERN = r"*__(bad"
    PROPERTY_MAPPING = {
        SBPAR_KEY: property_spec(
            "Aggregation subtype.",
            strict=True,
            allowed_values={"sum": "Sum", "max": "Max"},
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SbparFwAlpha, SbparFwBeta}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class TestSbparSingletonContainerParity:
    """A singleton frozenset option value is matcher-accepted as 'sum'; resolution must mirror it."""

    def test_matcher_accepts_singleton_frozenset(self) -> None:
        # Empirical anchor: the matcher unwraps frozenset elements and validates 'sum'.
        options = Options(group={SBPAR_KEY: frozenset({"sum"})})
        accepted = FeatureChainParser.match_configuration_feature_chain_parser(
            SBPAR_UNCHAINED, options, property_mapping=SbparWindowFG.PROPERTY_MAPPING
        )
        assert accepted is True

    def test_resolve_subtype_normalizes_singleton_frozenset(self) -> None:
        options = Options(group={SBPAR_KEY: frozenset({"sum"})})
        assert SbparWindowFG.resolve_subtype(SBPAR_UNCHAINED, options) == "sum"

    def test_singleton_frozenset_gates_supports_compute_framework(self) -> None:
        options = Options(group={SBPAR_KEY: frozenset({"sum"})})
        gate = SbparWindowFG.supports_compute_framework
        assert gate(SBPAR_UNCHAINED, options, SbparFwBeta) is False
        assert gate(SBPAR_UNCHAINED, options, SbparFwAlpha) is True

    def test_plain_string_option_baseline_still_gates(self) -> None:
        # Baseline sanity: the raw-string path already agrees with the matcher.
        options = Options(group={SBPAR_KEY: "sum"})
        assert SbparWindowFG.resolve_subtype(SBPAR_UNCHAINED, options) == "sum"
        assert SbparWindowFG.supports_compute_framework(SBPAR_UNCHAINED, options, SbparFwBeta) is False


class TestSbparPropertyMappingDefaultParity:
    """An absent option with a declared default is matcher-accepted; the default must resolve."""

    def test_matcher_accepts_absent_option_via_default(self) -> None:
        # Empirical anchor: a key with default='sum' passes validation with empty options.
        accepted = FeatureChainParser.match_configuration_feature_chain_parser(
            SBPAR_UNCHAINED, Options(), property_mapping=SbparDefaultedFG.PROPERTY_MAPPING
        )
        assert accepted is True

    def test_resolve_subtype_applies_declared_default(self) -> None:
        assert SbparDefaultedFG.resolve_subtype(SBPAR_UNCHAINED, Options()) == "sum"

    def test_declared_default_gates_supports_compute_framework(self) -> None:
        gate = SbparDefaultedFG.supports_compute_framework
        assert gate(SBPAR_UNCHAINED, Options(), SbparFwBeta) is False
        assert gate(SBPAR_UNCHAINED, Options(), SbparFwAlpha) is True

    def test_explicit_option_overrides_declared_default(self) -> None:
        options = Options(group={SBPAR_DEFAULT_KEY: "max"})
        assert SbparDefaultedFG.resolve_subtype(SBPAR_UNCHAINED, options) == "max"
        assert SbparDefaultedFG.supports_compute_framework(SBPAR_UNCHAINED, options, SbparFwBeta) is True


class TestSbparMalformedPrefixPatternNeverRaises:
    """resolve_subtype degrades to None on a malformed PREFIX_PATTERN instead of raising re.error."""

    def test_resolve_subtype_returns_none_for_chained_name(self) -> None:
        assert SbparBadPrefixFG.resolve_subtype("value__sum_x", Options()) is None

    def test_supports_compute_framework_stays_open(self) -> None:
        gate = SbparBadPrefixFG.supports_compute_framework
        assert gate("value__sum_x", Options(), SbparFwBeta) is True
        assert gate("value__sum_x", Options(), SbparFwAlpha) is True
