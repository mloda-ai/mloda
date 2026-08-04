"""Pinning tests for the pure failure renderer on the evaluation seam (issue #791).

``IdentifyFeatureGroupClass.evaluate(...)`` captures every fact a resolution-failure message needs
during its single first pass: per-candidate compute-framework capability
(``EvaluationResult.candidate_frameworks``) plus the remaining facts (``EvaluationResult.facts``).
``render_resolution_failure(result, feature)`` is then a PURE projection of that result: it reads
only the result and the ``Feature``, never re-runs matching and never calls a provider-overridable
hook. The feature groups below count every such hook, so a renderer that calls one is caught.

All names are suffixed ``_791`` because test feature groups become global subclasses.
"""

import inspect
import logging
from abc import abstractmethod
from ast import literal_eval
from collections.abc import Callable, Iterator
from difflib import get_close_matches
from typing import Any, ClassVar, Optional, cast, get_args

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import PropertyValueRejection
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import property_spec
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.prepare import resolution_types
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.core.prepare.resolution_failure_renderer import _STAGE_LABELS, render_resolution_failure
from mloda.core.prepare.resolution_types import (
    CandidateFrameworks,
    Elimination,
    EliminationStage,
    EvaluationResult,
    RenderFacts,
)


SUCCESS_FEATURE_791 = "renderer_success_791"
MULTIPLE_FEATURE_791 = "renderer_multiple_791"
CAPABILITY_FEATURE_791 = "renderer_capability_791"
ABSTRACT_FEATURE_791 = "renderer_abstract_791"
KNOWN_FEATURE_791 = "renderer_known_feature_791"
TYPO_FEATURE_791 = "renderer_knwon_feature_791"
SCOPED_NO_MATCH_FEATURE_791 = "renderer_scoped_no_match_791"
FORWARDING_FEATURE_791 = "renderer_forwarding_791"
RAISING_DOMAIN_FEATURE_791 = "renderer_raising_domain_791"
RAISING_ABSTRACT_FEATURE_791 = "renderer_raising_abstract_791"
NARROW_ENABLED_FEATURE_791 = "renderer_narrow_enabled_791"
NONE_ENABLED_FEATURE_791 = "renderer_none_enabled_791"
TIE_FEATURE_791 = "renderer_tie_791"
TIE_CAPABILITY_FEATURE_791 = "renderer_tie_capability_791"
MISSING_OPTION_FEATURE_791 = "renderer_missing_option_791__sum_renderer791m"
STRANDED_FEATURE_791 = "renderer_stranded_791"
CUTOFF_FEATURE_791 = "renderer_cutoff_791"
CUTOFF_CATALOG_NAME_791 = "renderer_cutoff_threshold_group_791"
DUPLICATE_TYPO_FEATURE_791 = "renderer_duplcate_791"
DUPLICATE_CATALOG_NAME_791 = "renderer_duplicate_791"
DUPLICATE_SPARE_NAME_791 = "renderer_duplicate_spare_791"
EMPTY_CATALOG_FEATURE_791 = "renderer_empty_catalog_791"
DECLARED_UNMATCHED_FEATURE_791 = "renderer_declared_unmatched_791"
WIDE_FEATURE_791 = "renderer_wide_791"
DEAD_SIBLING_FEATURE_791 = "renderer_dead_sibling_revenue_791"
DEAD_SIBLING_SPARE_791 = "renderer_dead_sibling_profit_791"
SHARED_TYPO_FEATURE_791 = "renderer_shared_naem_791"
SHARED_LIVE_NAME_791 = "renderer_shared_name_791"
SHARED_DEAD_NAME_791 = "renderer_shared_dead_791"
VALUE_STAGE_FEATURE_791 = "renderer_value_stage_791"
VALUE_STAGE_SPARE_791 = "renderer_value_stage_spare_791"
CAPABILITY_STAGE_FEATURE_791 = "renderer_capability_stage_791"
CAPABILITY_STAGE_SPARE_791 = "renderer_capability_stage_spare_791"
RAISING_DEAD_NAMES_FEATURE_791 = "renderer_raising_dead_names_791"
RAISING_DEAD_NAMES_SPARE_791 = "renderer_raising_dead_names_spare_791"
DISABLED_PAIR_FEATURE_791 = "renderer_disabled_pair_revenue_791"
DISABLED_PAIR_SPARE_791 = "renderer_disabled_pair_profit_791"

# The name-blind gates. Each pair is a requested name nothing matches plus the sibling name the group that
# loses at the gate declares, so only the gate can decide whether that sibling is worth suggesting.
DOMAIN_GATE_FEATURE_791 = "renderer_domain_gate_revenue_791"
DOMAIN_GATE_SIBLING_791 = "renderer_domain_gate_profit_791"
SCOPE_GATE_FEATURE_791 = "renderer_scope_gate_revenue_791"
SCOPE_GATE_SIBLING_791 = "renderer_scope_gate_profit_791"
VALUE_DOMAIN_FEATURE_791 = "renderer_value_domain_revenue_791"
VALUE_DOMAIN_SPARE_791 = "renderer_value_domain_profit_791"
DEGRADED_DOMAIN_FEATURE_791 = "renderer_degraded_domain_revenue_791"
DEGRADED_DOMAIN_SPARE_791 = "renderer_degraded_domain_profit_791"
MALFORMED_DOMAIN_FEATURE_791 = "renderer_malformed_domain_revenue_791"
MALFORMED_DOMAIN_SPARE_791 = "renderer_malformed_domain_profit_791"

# A get_domain() return that is not a Domain at all: the decision gate compares it and drops the candidate.
MALFORMED_DOMAIN_VALUE_791 = "renderer_not_a_domain_791"

# The name an abstract base and a framework-less concrete group both declare, plus a typo of it that
# nothing matches: only whether the abstract base counts as a live declarer decides the suggestion.
ABSTRACT_DECLARER_NAME_791 = "renderer_uninstantiable_revenue_791"
ABSTRACT_DECLARER_TYPO_791 = "renderer_uninstantiable_revneue_791"

# The requested domain is declared by no candidate at all, so every domain-carrying request below fails it.
REQUESTED_DOMAIN_791 = "renderer_requested_domain_791"
OTHER_DOMAIN_791 = "renderer_other_domain_791"

# Scope of the name-blind scope gate: a healthy group of this module, so the scope names a real accessible class.
GATE_SCOPE_791 = "RendererKnownNamesFG791"

VALUE_DOMAIN_REJECTION_REASON_791 = "ValueRejectingCrossDomainFG791 declines every value of this option"

# Built around the class names below, because the default matcher also owns a name by class-name PREFIX.
# The request drops the trailing underscore, so nothing matches it while both declared names stay close to it.
LIVE_PREFIX_FEATURE_791 = "RendererLivePrefixFG791sum_791"
LIVE_PREFIX_COVERED_791 = "RendererLivePrefixFG791_sum_791"
DEAD_PREFIX_UNCOVERED_791 = "RendererDeadPrefixFG791_sum_791"

# The two names the default matcher owns by class identity: the class name itself and its prefix. Neither is
# declared by anything, so only the candidate that carries them can keep them suggestible. The request is a
# typo of the class name, close to both and matching neither.
DEAD_CLASS_NAME_791 = "RendererCrossDomainNameFG791"
DEAD_CLASS_PREFIX_791 = "RendererCrossDomainNameFG791_"
DEAD_CLASS_NAME_TYPO_791 = "RendererCrossDoaminNameFG791"

VALUE_STAGE_REJECTION_REASON_791 = "renderer_value_stage_791 declines every value of this option"

# The stages whose gate CAN see the feature name, so a sibling name of a candidate eliminated there may still
# resolve. Pinned here as the complement of NAME_INDEPENDENT_STAGES: a tenth stage fails the partition test.
NAME_DEPENDENT_STAGES_791: frozenset[EliminationStage] = frozenset(
    {"value_rejection", "input_data", "matcher_error", "capability", "framework_pin"}
)

# Stands in for a ninth stage shipped without a near-miss label: no entry of the label table covers this token.
UNLABELED_STAGE_791 = "renderer_unlabeled_stage_791"
UNLABELED_STAGE_FEATURE_791 = "renderer_unlabeled_stage_feature_791"
UNLABELED_STAGE_REASON_791 = "eliminated at a stage this build has no label for"

# The renderer's own module logger: the degrade must be reported by the module that degrades.
RENDERER_LOGGER_791 = "mloda.core.prepare.resolution_failure_renderer"

# The matcher's own module logger: a degraded domain read happens during evaluate(), never during rendering.
IDENTIFY_LOGGER_791 = "mloda.core.prepare.identify_feature_group"

# Eight names, all close to WIDE_FEATURE_791, so only the cut can bound the rendered line.
WIDE_CATALOG_NAMES_791 = frozenset(
    {
        "renderer_wide_alpha_791",
        "renderer_wide_bravo_791",
        "renderer_wide_delta_791",
        "renderer_wide_echo_791",
        "renderer_wide_gamma_791",
        "renderer_wide_kilo_791",
        "renderer_wide_lima_791",
        "renderer_wide_zulu_791",
    }
)

MAX_RENDERED_SUGGESTIONS_791 = 5

HEALTHY_DOMAIN_791 = "renderer_healthy_domain_791"
BOOM_SUPPORTED_NAME_791 = "renderer_boom_supported_name_791"

# Same-named tie candidates get an explicit __module__ so only the module can break the sort tie.
TIE_MODULE_A_791 = "tests.renderer_tie_module_a_791"
TIE_MODULE_B_791 = "tests.renderer_tie_module_b_791"

TROUBLESHOOTING_LINE = (
    "For troubleshooting guide, see: "
    "https://mloda-ai.github.io/mloda/in_depth/troubleshooting/feature-group-resolution-errors/"
)

SUGGESTION_PREFIX = "Did you mean one of: "

# Call counters for every provider-overridable hook, keyed "<ClassName>.<hook>". Reset per test.
HOOK_CALLS: dict[str, int] = {}

# supports_compute_framework calls keyed by the (candidate, framework) PAIR it was asked about.
# HOOK_CALLS only knows the hook name, so it cannot see a pair being asked twice. Reset per test.
PAIR_CALLS: dict[tuple[str, str], int] = {}


def _record(class_name: str, hook: str) -> None:
    """Count one call of a provider-overridable hook."""
    key = f"{class_name}.{hook}"
    HOOK_CALLS[key] = HOOK_CALLS.get(key, 0) + 1


def _record_pair(class_name: str, framework_name: str) -> None:
    """Count one capability-hook call for a single (candidate, framework) pair."""
    key = (class_name, framework_name)
    PAIR_CALLS[key] = PAIR_CALLS.get(key, 0) + 1


def _malformed(value: Any) -> Any:
    """Return a deliberately ill-typed value.

    A provider's annotation is a promise, not a guarantee: hiding the value from the type checker is how a
    real plugin's runtime bug reaches mloda. The gate that compares it still decides, so the core must too.
    """
    return value


class RendererFwOne791(ComputeFramework):
    """First dummy compute framework for the failure-renderer tests."""


class RendererFwTwo791(ComputeFramework):
    """Second dummy compute framework for the failure-renderer tests."""


class RendererFwThree791(ComputeFramework):
    """Third dummy compute framework, used only to pin a feature away from every candidate."""


class CountingFeatureGroup791(FeatureGroup):
    """Feature group base that counts every provider-overridable hook the renderer must not call."""

    MATCHES: ClassVar[frozenset[str]] = frozenset()
    DOMAIN_NAME: ClassVar[Optional[str]] = None
    FRAMEWORK_RULE: ClassVar[Optional[set[type[ComputeFramework]]]] = None
    SUPPORTED_FRAMEWORKS: ClassVar[Optional[frozenset[str]]] = None
    SUPPORTED_NAMES: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        _record(cls.get_class_name(), "match_feature_group_criteria")
        return str(feature_name) in cls.MATCHES

    @classmethod
    def get_domain(cls) -> Domain:
        _record(cls.get_class_name(), "get_domain")
        if cls.DOMAIN_NAME is None:
            return Domain.get_default_domain()
        return Domain(cls.DOMAIN_NAME)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        _record(cls.get_class_name(), "compute_framework_rule")
        return cls.FRAMEWORK_RULE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        _record(cls.get_class_name(), "supports_compute_framework")
        _record_pair(cls.get_class_name(), compute_framework.get_class_name())
        if cls.SUPPORTED_FRAMEWORKS is None:
            return True
        return compute_framework.get_class_name() in cls.SUPPORTED_FRAMEWORKS

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        _record(cls.get_class_name(), "index_columns")
        return None

    @classmethod
    def supports_index(cls, index: Index) -> Optional[bool]:
        _record(cls.get_class_name(), "supports_index")
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        _record(cls.get_class_name(), "feature_names_supported")
        return set(cls.SUPPORTED_NAMES)

    @classmethod
    def prefix(cls) -> str:
        _record(cls.get_class_name(), "prefix")
        return f"{cls.get_class_name()}_"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class RendererSuccessFG791(CountingFeatureGroup791):
    """Sole winner of the success scenario."""

    MATCHES = frozenset({SUCCESS_FEATURE_791})
    FRAMEWORK_RULE = {RendererFwOne791}


class RendererMultipleAFG791(CountingFeatureGroup791):
    """First of two unrelated siblings matching the same name, in domain 'renderer_domain_a_791'."""

    MATCHES = frozenset({MULTIPLE_FEATURE_791})
    DOMAIN_NAME = "renderer_domain_a_791"
    FRAMEWORK_RULE = {RendererFwOne791}


class RendererMultipleBFG791(CountingFeatureGroup791):
    """Second sibling matching the same name, in domain 'renderer_domain_b_791'."""

    MATCHES = frozenset({MULTIPLE_FEATURE_791})
    DOMAIN_NAME = "renderer_domain_b_791"
    FRAMEWORK_RULE = {RendererFwOne791}


class RendererCapabilityAFG791(CountingFeatureGroup791):
    """Mirrored capability candidate: supports RendererFwOne791, rejects RendererFwTwo791."""

    MATCHES = frozenset({CAPABILITY_FEATURE_791})
    FRAMEWORK_RULE = {RendererFwOne791, RendererFwTwo791}
    SUPPORTED_FRAMEWORKS = frozenset({"RendererFwOne791"})


class RendererCapabilityBFG791(CountingFeatureGroup791):
    """Mirrored capability candidate: supports RendererFwTwo791, rejects RendererFwOne791."""

    MATCHES = frozenset({CAPABILITY_FEATURE_791})
    FRAMEWORK_RULE = {RendererFwOne791, RendererFwTwo791}
    SUPPORTED_FRAMEWORKS = frozenset({"RendererFwTwo791"})


class RendererAbstractBaseFG791(CountingFeatureGroup791):
    """Abstract base that matches the name but can never be instantiated."""

    MATCHES = frozenset({ABSTRACT_FEATURE_791})
    FRAMEWORK_RULE = {RendererFwOne791}

    @classmethod
    @abstractmethod
    def _renderer_abstract_hook_791(cls) -> str:
        """Abstract hook that keeps this base uninstantiable."""


class RendererConcreteSubFG791(RendererAbstractBaseFG791):
    """Concrete implementation of the abstract base, declaring two compute frameworks."""

    MATCHES = frozenset({ABSTRACT_FEATURE_791})
    FRAMEWORK_RULE = {RendererFwOne791, RendererFwTwo791}

    @classmethod
    def _renderer_abstract_hook_791(cls) -> str:
        return "concrete"


class RendererKnownNamesFG791(CountingFeatureGroup791):
    """Name catalog for the 'Did you mean' suggestion."""

    MATCHES = frozenset({KNOWN_FEATURE_791})
    SUPPORTED_NAMES = frozenset({KNOWN_FEATURE_791})
    FRAMEWORK_RULE = {RendererFwOne791}


class RendererBareOnlyFG791(CountingFeatureGroup791):
    """Matches its bare name only: any group option makes it reject the feature."""

    MATCHES = frozenset({FORWARDING_FEATURE_791})
    FRAMEWORK_RULE = {RendererFwOne791}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        matched = super().match_feature_group_criteria(feature_name, options, data_access_collection)
        return matched and not options.group


class RendererStrictFG791(FeatureChainParserMixin, FeatureGroup):
    """Config-based group whose strict 'window_size' validator rejects out-of-range values."""

    PROPERTY_MAPPING = {
        "window_size": property_spec(
            "Size of window",
            strict=True,
            context=False,
            element_validator=lambda v: isinstance(v, int) and 0 < v <= 13,
        ),
        DefaultOptionKeys.in_features: property_spec("source", context=True),
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: str | FeatureName,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        _record(cls.get_class_name(), "match_feature_group_criteria")
        return super().match_feature_group_criteria(feature_name, options, data_access_collection)

    @classmethod
    def get_domain(cls) -> Domain:
        _record(cls.get_class_name(), "get_domain")
        return super().get_domain()

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        _record(cls.get_class_name(), "compute_framework_rule")
        return {RendererFwOne791}

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        _record(cls.get_class_name(), "supports_compute_framework")
        return super().supports_compute_framework(feature_name, options, compute_framework)

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        _record(cls.get_class_name(), "index_columns")
        return None

    @classmethod
    def supports_index(cls, index: Index) -> Optional[bool]:
        _record(cls.get_class_name(), "supports_index")
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        _record(cls.get_class_name(), "feature_names_supported")
        return set()

    @classmethod
    def prefix(cls) -> str:
        _record(cls.get_class_name(), "prefix")
        return f"{cls.get_class_name()}_"

    @classmethod
    def _strict_validation_rejection_reason(cls, feature_name: str | FeatureName, options: Options) -> str | None:
        _record(cls.get_class_name(), "_strict_validation_rejection_reason")
        return super()._strict_validation_rejection_reason(feature_name, options)

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


WINDOW_REJECTION_REASON = "Property value '14' failed validation for 'window_size'"


class RendererMissingOptionFG791(FeatureChainParserMixin, FeatureGroup):
    """Name-path group whose required options-only key is absent: a MISSING-option rejection, not a wrong value."""

    PREFIX_PATTERN = r".*__(?P<op_791m>\w+)_renderer791m$"
    PROPERTY_MAPPING = {
        "op_791m": property_spec("operation carried by the name", context=True),
        "some_key_791m": property_spec("required, options-only", context=True),
        DefaultOptionKeys.in_features: property_spec("source", context=True),
    }

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


MISSING_OPTION_REASON_791 = "required option(s) some_key_791m are absent after declared defaults and name bindings"


class RendererNarrowEnabledFG791(CountingFeatureGroup791):
    """Declares two available frameworks; the run enables only the one this candidate rejects."""

    MATCHES = frozenset({NARROW_ENABLED_FEATURE_791})
    FRAMEWORK_RULE = {RendererFwOne791, RendererFwTwo791}
    SUPPORTED_FRAMEWORKS = frozenset({"RendererFwTwo791"})


class RendererNoneEnabledFG791(CountingFeatureGroup791):
    """Declares two available frameworks; the run enables neither of them."""

    MATCHES = frozenset({NONE_ENABLED_FEATURE_791})
    FRAMEWORK_RULE = {RendererFwOne791, RendererFwTwo791}
    SUPPORTED_FRAMEWORKS = frozenset({"RendererFwTwo791"})


class RendererStrandedFG791(CountingFeatureGroup791):
    """Declares the requested name itself, then loses every framework the run could have enabled."""

    MATCHES = frozenset({STRANDED_FEATURE_791})
    SUPPORTED_NAMES = frozenset({STRANDED_FEATURE_791})
    FRAMEWORK_RULE = {RendererFwOne791, RendererFwTwo791}


class RendererCutoffAFG791(CountingFeatureGroup791):
    """Eliminated candidate declaring the requested name, so the name catalog carries the exact echo."""

    MATCHES = frozenset({CUTOFF_FEATURE_791})
    SUPPORTED_NAMES = frozenset({CUTOFF_FEATURE_791})
    FRAMEWORK_RULE = {RendererFwOne791, RendererFwTwo791}


class RendererCutoffBFG791(CountingFeatureGroup791):
    """Second eliminated candidate, contributing two more droppable hints: its class name and its prefix."""

    MATCHES = frozenset({CUTOFF_FEATURE_791})
    FRAMEWORK_RULE = {RendererFwOne791, RendererFwTwo791}


class SpareNameCatalogFG791(CountingFeatureGroup791):
    """Healthy catalog group, named far enough from the request that only its supported name can be suggested."""

    MATCHES = frozenset({CUTOFF_CATALOG_NAME_791})
    SUPPORTED_NAMES = frozenset({CUTOFF_CATALOG_NAME_791})
    FRAMEWORK_RULE = {RendererFwOne791}


class RendererDuplicateNameFG791(CountingFeatureGroup791):
    """Catalog candidate declaring the shared name; the four siblings below declare exactly the same one."""

    SUPPORTED_NAMES = frozenset({DUPLICATE_CATALOG_NAME_791})
    FRAMEWORK_RULE = {RendererFwOne791}


class RendererDuplicateSiblingAFG791(RendererDuplicateNameFG791):
    """Second contributor of the shared name."""


class RendererDuplicateSiblingBFG791(RendererDuplicateNameFG791):
    """Third contributor of the shared name."""


class RendererDuplicateSiblingCFG791(RendererDuplicateNameFG791):
    """Fourth contributor of the shared name."""


class RendererDuplicateSiblingDFG791(RendererDuplicateNameFG791):
    """Fifth contributor of the shared name, enough copies to fill every suggestion slot."""


class RendererDuplicateSpareFG791(CountingFeatureGroup791):
    """Catalog candidate holding the one genuinely different close name the copies push out."""

    SUPPORTED_NAMES = frozenset({DUPLICATE_SPARE_NAME_791})
    FRAMEWORK_RULE = {RendererFwOne791}


class RendererDeclaredUnmatchedFG791(CountingFeatureGroup791):
    """Declares the requested name but matches nothing, so nothing records it as an elimination."""

    SUPPORTED_NAMES = frozenset({DECLARED_UNMATCHED_FEATURE_791})
    FRAMEWORK_RULE = {RendererFwOne791}


class RendererWideCatalogFG791(CountingFeatureGroup791):
    """Catalog candidate declaring more close names than the message may ever list."""

    SUPPORTED_NAMES = WIDE_CATALOG_NAMES_791
    FRAMEWORK_RULE = {RendererFwOne791}


class RendererDeadSiblingFG791(CountingFeatureGroup791):
    """The reported repro: declares the requested name AND a sibling, then loses every framework."""

    MATCHES = frozenset({DEAD_SIBLING_FEATURE_791})
    SUPPORTED_NAMES = frozenset({DEAD_SIBLING_FEATURE_791, DEAD_SIBLING_SPARE_791})
    FRAMEWORK_RULE = {RendererFwOne791, RendererFwTwo791}


class RendererSharedDeadFG791(CountingFeatureGroup791):
    """Dead contributor of two names: one it shares with the live group below, one only it declares."""

    MATCHES = frozenset({SHARED_TYPO_FEATURE_791})
    SUPPORTED_NAMES = frozenset({SHARED_LIVE_NAME_791, SHARED_DEAD_NAME_791})
    FRAMEWORK_RULE = {RendererFwOne791, RendererFwTwo791}


class RendererSharedLiveFG791(CountingFeatureGroup791):
    """Live contributor of the shared name: it matches nothing here, so nothing ever eliminates it."""

    SUPPORTED_NAMES = frozenset({SHARED_LIVE_NAME_791})
    FRAMEWORK_RULE = {RendererFwOne791}


class RendererValueStageFG791(CountingFeatureGroup791):
    """Eliminated at value_rejection, a name-DEPENDENT stage: it declined THIS name's value."""

    MATCHES = frozenset({VALUE_STAGE_FEATURE_791})
    SUPPORTED_NAMES = frozenset({VALUE_STAGE_SPARE_791})
    FRAMEWORK_RULE = {RendererFwOne791}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        # Name-guarded, so this globally visible class stays inert for every other name it is asked about.
        if not super().match_feature_group_criteria(feature_name, options, data_access_collection):
            return False
        raise PropertyValueRejection(VALUE_STAGE_REJECTION_REASON_791)


class RendererCapabilityStageFG791(CountingFeatureGroup791):
    """Eliminated at capability: the per-feature hook rejected the one framework the run enabled."""

    MATCHES = frozenset({CAPABILITY_STAGE_FEATURE_791})
    SUPPORTED_NAMES = frozenset({CAPABILITY_STAGE_SPARE_791})
    FRAMEWORK_RULE = {RendererFwOne791}
    SUPPORTED_FRAMEWORKS = frozenset({"RendererFwThree791"})


class RendererDisabledPairFG791(CountingFeatureGroup791):
    """Declares and matches BOTH names, then loses every framework: whichever one is requested, it is eliminated."""

    MATCHES = frozenset({DISABLED_PAIR_FEATURE_791, DISABLED_PAIR_SPARE_791})
    SUPPORTED_NAMES = frozenset({DISABLED_PAIR_FEATURE_791, DISABLED_PAIR_SPARE_791})
    FRAMEWORK_RULE = {RendererFwOne791, RendererFwTwo791}


class SpareNoFrameworkFG791(CountingFeatureGroup791):
    """Declares the spare name and matches nothing, so nothing records the empty framework set that kills it.

    Named far from both feature names on purpose: only its declared name may ever reach the suggestion line.
    """

    SUPPORTED_NAMES = frozenset({DISABLED_PAIR_SPARE_791})
    FRAMEWORK_RULE = {RendererFwOne791}


class RendererLivePrefixFG791(CountingFeatureGroup791):
    """Live group matching by class-name prefix, the default rule, so it owns names it never declares."""

    FRAMEWORK_RULE = {RendererFwOne791}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        _record(cls.get_class_name(), "match_feature_group_criteria")
        return cls.feature_name_contains_class_name_as_prefix(str(feature_name))


class RendererDeadPrefixFG791(CountingFeatureGroup791):
    """Dead declarer of two names: one the live prefix above covers, one only its own dead prefix covers."""

    MATCHES = frozenset({LIVE_PREFIX_FEATURE_791})
    SUPPORTED_NAMES = frozenset({LIVE_PREFIX_COVERED_791, DEAD_PREFIX_UNCOVERED_791})
    FRAMEWORK_RULE = {RendererFwOne791, RendererFwTwo791}


class CrossDomainDeclarerFG791(CountingFeatureGroup791):
    """Declares and matches the sibling name only, from a domain no request here asks for.

    Named far from both feature names on purpose: only its declared name may ever reach the suggestion line.
    """

    MATCHES = frozenset({DOMAIN_GATE_SIBLING_791})
    SUPPORTED_NAMES = frozenset({DOMAIN_GATE_SIBLING_791})
    DOMAIN_NAME = OTHER_DOMAIN_791
    FRAMEWORK_RULE = {RendererFwOne791}


class RendererCrossDomainNameFG791(CountingFeatureGroup791):
    """Declares no name at all and owns two by class identity, from a domain no request here asks for.

    Named CLOSE to its request on purpose: the class name and the prefix are the only names it can contribute.
    """

    DOMAIN_NAME = OTHER_DOMAIN_791
    FRAMEWORK_RULE = {RendererFwOne791}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        _record(cls.get_class_name(), "match_feature_group_criteria")
        # The two class-identity rules of the default matcher, and the two names the catalog captures for it.
        name = str(feature_name)
        return cls.feature_name_equal_to_class_name(name) or cls.feature_name_contains_class_name_as_prefix(name)


class RendererLiveNameDeclarerFG791(CountingFeatureGroup791):
    """Live declarer of the wrong-domain group's two class-identity names, in the requested domain."""

    SUPPORTED_NAMES = frozenset({DEAD_CLASS_NAME_791, DEAD_CLASS_PREFIX_791})
    DOMAIN_NAME = REQUESTED_DOMAIN_791
    FRAMEWORK_RULE = {RendererFwOne791}


class OutsideScopeDeclarerFG791(CountingFeatureGroup791):
    """Declares and matches the sibling name only, from outside the scope every request here asks for.

    Named far from both feature names on purpose: only its declared name may ever reach the suggestion line.
    """

    MATCHES = frozenset({SCOPE_GATE_SIBLING_791})
    SUPPORTED_NAMES = frozenset({SCOPE_GATE_SIBLING_791})
    FRAMEWORK_RULE = {RendererFwOne791}


class SpareAbstractBaseFG791(CountingFeatureGroup791):
    """Abstract base declaring and matching one name, in the requested domain: it can never be identified.

    Named far from both feature names on purpose: only its declared name may ever reach the suggestion line.
    """

    MATCHES = frozenset({ABSTRACT_DECLARER_NAME_791})
    SUPPORTED_NAMES = frozenset({ABSTRACT_DECLARER_NAME_791})
    DOMAIN_NAME = REQUESTED_DOMAIN_791
    FRAMEWORK_RULE = {RendererFwOne791}

    @classmethod
    @abstractmethod
    def _renderer_spare_abstract_hook_791(cls) -> str:
        """Abstract hook that keeps this base uninstantiable."""


class SpareDeadTwinFG791(CountingFeatureGroup791):
    """Concrete declarer of the same name, left without a single enabled framework by every run below.

    Named far from both feature names on purpose: only its declared name may ever reach the suggestion line.
    """

    MATCHES = frozenset({ABSTRACT_DECLARER_NAME_791})
    SUPPORTED_NAMES = frozenset({ABSTRACT_DECLARER_NAME_791})
    FRAMEWORK_RULE = {RendererFwOne791}


class ValueRejectingCrossDomainFG791(CountingFeatureGroup791):
    """Declines THIS name's value at the matcher AND sits in another domain: name-dependent record, name-blind gate."""

    MATCHES = frozenset({VALUE_DOMAIN_FEATURE_791})
    SUPPORTED_NAMES = frozenset({VALUE_DOMAIN_SPARE_791})
    DOMAIN_NAME = OTHER_DOMAIN_791
    FRAMEWORK_RULE = {RendererFwOne791}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        # Name-guarded, so this globally visible class stays inert for every other name it is asked about.
        if not super().match_feature_group_criteria(feature_name, options, data_access_collection):
            return False
        raise PropertyValueRejection(VALUE_DOMAIN_REJECTION_REASON_791)


class RendererDomainBoom791(RuntimeError):
    """Raised by a provider's get_domain() hook."""


class RendererPrefixBoom791(RuntimeError):
    """Raised by a provider's prefix() hook."""


class RendererNamesBoom791(RuntimeError):
    """Raised by a provider's feature_names_supported() hook."""


class RendererRejectionBoom791(RuntimeError):
    """Raised by a provider's _strict_validation_rejection_reason() hook."""


class RendererFrameworkRuleBoom791(RuntimeError):
    """Raised by a provider's compute_framework_rule() hook."""


class RaisingHookGroup791(CountingFeatureGroup791):
    """Base for the groups whose fact-capture hook raises or returns a malformed value.

    Its subclasses are ALWAYS built inside a function.

    ``ARMED`` is what makes that safe. A group built per test still outlives it: pytest keeps a failing
    test's traceback, and that traceback keeps the builder's frame (and so the class) alive and globally
    visible in ``FeatureGroup.__subclasses__()``. A group whose declaration hook raised forever would then
    take down every later test in the worker that enumerates the plugin universe -- ``PreFilterPlugins``
    fails closed on a raising ``compute_framework_definition()``. The autouse fixture disarms every group
    it built, so a leaked class is inert.
    """

    ARMED: ClassVar[bool] = True


RAISING_GROUPS_BUILT: list[type[RaisingHookGroup791]] = []


def _armed(group: type[RaisingHookGroup791]) -> type[RaisingHookGroup791]:
    """Track a freshly built raising group so the autouse fixture can disarm it after the test."""
    RAISING_GROUPS_BUILT.append(group)
    return group


def _build_raising_domain_groups() -> tuple[type[CountingFeatureGroup791], type[CountingFeatureGroup791]]:
    """Build a (raising get_domain, healthy get_domain) pair that both match the same feature name."""

    class RendererRaisingDomainFG791(RaisingHookGroup791):
        """Candidate whose get_domain() hook raises."""

        MATCHES = frozenset({RAISING_DOMAIN_FEATURE_791})
        FRAMEWORK_RULE = {RendererFwOne791}

        @classmethod
        def get_domain(cls) -> Domain:
            _record(cls.get_class_name(), "get_domain")
            if cls.ARMED:
                raise RendererDomainBoom791("get_domain exploded 791")
            return Domain.get_default_domain()

    class RendererHealthyDomainFG791(CountingFeatureGroup791):
        """Candidate whose get_domain() hook works, standing next to the raising one."""

        MATCHES = frozenset({RAISING_DOMAIN_FEATURE_791})
        DOMAIN_NAME = HEALTHY_DOMAIN_791
        FRAMEWORK_RULE = {RendererFwOne791}

    return _armed(RendererRaisingDomainFG791), RendererHealthyDomainFG791


def _build_unreadable_domain_group() -> type[CountingFeatureGroup791]:
    """Build a non-matching declarer of a sibling name whose get_domain() raises, so no gate can judge it.

    Named far from both feature names on purpose: only its declared name may ever reach the suggestion line.
    """

    class UnreadableDomainFG791(RaisingHookGroup791):
        """Declarer whose domain can never be read, matching nothing."""

        SUPPORTED_NAMES = frozenset({DEGRADED_DOMAIN_SPARE_791})
        FRAMEWORK_RULE = {RendererFwOne791}

        @classmethod
        def get_domain(cls) -> Domain:
            _record(cls.get_class_name(), "get_domain")
            if cls.ARMED:
                raise RendererDomainBoom791("get_domain exploded 791")
            return Domain.get_default_domain()

    return _armed(UnreadableDomainFG791)


def _build_malformed_domain_group() -> type[CountingFeatureGroup791]:
    """Build a declarer of a sibling name whose get_domain() returns a bare str instead of a Domain.

    Named far from both feature names on purpose: only its declared name may ever reach the suggestion line.
    """

    class BadDomainReturnFG791(RaisingHookGroup791):
        """Declarer whose domain read is well-formed as a call and malformed as a value."""

        MATCHES = frozenset({MALFORMED_DOMAIN_SPARE_791})
        SUPPORTED_NAMES = frozenset({MALFORMED_DOMAIN_SPARE_791})
        FRAMEWORK_RULE = {RendererFwOne791}

        @classmethod
        def get_domain(cls) -> Domain:
            _record(cls.get_class_name(), "get_domain")
            if cls.ARMED:
                bad_domain: Domain = _malformed(MALFORMED_DOMAIN_VALUE_791)
                return bad_domain
            return Domain.get_default_domain()

    return _armed(BadDomainReturnFG791)


def _build_raising_prefix_group() -> type[CountingFeatureGroup791]:
    """Build a catalog group whose prefix() hook raises."""

    class RendererRaisingPrefixFG791(RaisingHookGroup791):
        """Catalog candidate whose prefix() hook raises."""

        FRAMEWORK_RULE = {RendererFwOne791}

        @classmethod
        def prefix(cls) -> str:
            _record(cls.get_class_name(), "prefix")
            if cls.ARMED:
                raise RendererPrefixBoom791("prefix exploded 791")
            return f"{cls.get_class_name()}_"

    return _armed(RendererRaisingPrefixFG791)


def _build_raising_names_group() -> type[CountingFeatureGroup791]:
    """Build a catalog group whose feature_names_supported() hook raises."""

    class RendererRaisingNamesFG791(RaisingHookGroup791):
        """Catalog candidate whose feature_names_supported() hook raises."""

        FRAMEWORK_RULE = {RendererFwOne791}
        SUPPORTED_NAMES = frozenset({BOOM_SUPPORTED_NAME_791})

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            _record(cls.get_class_name(), "feature_names_supported")
            if cls.ARMED:
                raise RendererNamesBoom791("feature_names_supported exploded 791")
            return set()

    return _armed(RendererRaisingNamesFG791)


def _build_raising_dead_names_group() -> type[CountingFeatureGroup791]:
    """Build a group that is eliminated with no framework left AND whose feature_names_supported() raises."""

    class RendererRaisingDeadNamesFG791(RaisingHookGroup791):
        """Dead candidate whose declared names can never be read."""

        MATCHES = frozenset({RAISING_DEAD_NAMES_FEATURE_791})
        SUPPORTED_NAMES = frozenset({RAISING_DEAD_NAMES_SPARE_791})
        FRAMEWORK_RULE = {RendererFwOne791, RendererFwTwo791}

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            _record(cls.get_class_name(), "feature_names_supported")
            if cls.ARMED:
                raise RendererNamesBoom791("feature_names_supported exploded 791")
            return set(cls.SUPPORTED_NAMES)

    return _armed(RendererRaisingDeadNamesFG791)


def _build_raising_rejection_group() -> type[CountingFeatureGroup791]:
    """Build a group whose _strict_validation_rejection_reason() hook raises."""

    class RendererRaisingRejectionFG791(RaisingHookGroup791):
        """Candidate whose value-rejection diagnostic hook raises."""

        FRAMEWORK_RULE = {RendererFwOne791}

        @classmethod
        def _strict_validation_rejection_reason(cls, feature_name: str | FeatureName, options: Options) -> str | None:
            _record(cls.get_class_name(), "_strict_validation_rejection_reason")
            if cls.ARMED:
                raise RendererRejectionBoom791("_strict_validation_rejection_reason exploded 791")
            return None

    return _armed(RendererRaisingRejectionFG791)


def _build_raising_framework_rule_groups() -> tuple[type[CountingFeatureGroup791], type[CountingFeatureGroup791]]:
    """Build an abstract base plus a concrete subclass whose compute_framework_rule() hook raises."""

    class RendererRaisingAbstractBaseFG791(RaisingHookGroup791):
        """Abstract base that matches the name but can never be instantiated."""

        MATCHES = frozenset({RAISING_ABSTRACT_FEATURE_791})
        FRAMEWORK_RULE = {RendererFwOne791}

        @classmethod
        @abstractmethod
        def _renderer_raising_abstract_hook_791(cls) -> str:
            """Abstract hook that keeps this base uninstantiable."""

    class RendererRaisingConcreteSubFG791(RendererRaisingAbstractBaseFG791):
        """Concrete implementation whose framework declaration raises."""

        MATCHES = frozenset({RAISING_ABSTRACT_FEATURE_791})

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            _record(cls.get_class_name(), "compute_framework_rule")
            if cls.ARMED:
                raise RendererFrameworkRuleBoom791("compute_framework_rule exploded 791")
            return cls.FRAMEWORK_RULE

        @classmethod
        def _renderer_raising_abstract_hook_791(cls) -> str:
            return "concrete"

    return RendererRaisingAbstractBaseFG791, _armed(RendererRaisingConcreteSubFG791)


def _make_tie_group(module: str, namespace: dict[str, Any]) -> type[CountingFeatureGroup791]:
    """Build a candidate named RendererTieFG791 in the given module, so only __module__ breaks the sort tie."""
    created: Any = type("RendererTieFG791", (CountingFeatureGroup791,), {"__module__": module, **namespace})
    return cast(type[CountingFeatureGroup791], created)


def _build_tie_domain_groups() -> tuple[type[CountingFeatureGroup791], type[CountingFeatureGroup791]]:
    """Build two same-named 'multiple' candidates that differ only in module and domain."""
    group_a = _make_tie_group(
        TIE_MODULE_A_791,
        {
            "MATCHES": frozenset({TIE_FEATURE_791}),
            "DOMAIN_NAME": "renderer_tie_domain_a_791",
            "FRAMEWORK_RULE": {RendererFwOne791},
        },
    )
    group_b = _make_tie_group(
        TIE_MODULE_B_791,
        {
            "MATCHES": frozenset({TIE_FEATURE_791}),
            "DOMAIN_NAME": "renderer_tie_domain_b_791",
            "FRAMEWORK_RULE": {RendererFwOne791},
        },
    )
    return group_a, group_b


def _build_tie_capability_groups() -> tuple[type[CountingFeatureGroup791], type[CountingFeatureGroup791]]:
    """Build two same-named capability candidates that differ only in module and supported framework."""
    group_a = _make_tie_group(
        TIE_MODULE_A_791,
        {
            "MATCHES": frozenset({TIE_CAPABILITY_FEATURE_791}),
            "FRAMEWORK_RULE": {RendererFwOne791, RendererFwTwo791},
            "SUPPORTED_FRAMEWORKS": frozenset({"RendererFwOne791"}),
        },
    )
    group_b = _make_tie_group(
        TIE_MODULE_B_791,
        {
            "MATCHES": frozenset({TIE_CAPABILITY_FEATURE_791}),
            "FRAMEWORK_RULE": {RendererFwOne791, RendererFwTwo791},
            "SUPPORTED_FRAMEWORKS": frozenset({"RendererFwTwo791"}),
        },
    )
    return group_a, group_b


Scenario = tuple[Feature, FeatureGroupEnvironmentMapping]


def success_scenario() -> Scenario:
    """Exactly one winner."""
    return Feature(SUCCESS_FEATURE_791), {RendererSuccessFG791: {RendererFwOne791}}


def multiple_scenario() -> Scenario:
    """Two identified candidates. Inserted B-before-A so a sorted rendering is observable."""
    return (
        Feature(MULTIPLE_FEATURE_791),
        {RendererMultipleBFG791: {RendererFwOne791}, RendererMultipleAFG791: {RendererFwOne791}},
    )


def capability_scenario() -> Scenario:
    """Mirrored capability rejection: the pin to a third framework keeps both candidates out."""
    return (
        Feature(CAPABILITY_FEATURE_791, compute_framework="RendererFwThree791"),
        {
            RendererCapabilityBFG791: {RendererFwOne791, RendererFwTwo791},
            RendererCapabilityAFG791: {RendererFwOne791, RendererFwTwo791},
        },
    )


def abstract_with_frameworks_scenario() -> Scenario:
    """Abstract base matched; its concrete subclass is accessible but has no enabled framework."""
    return (
        Feature(ABSTRACT_FEATURE_791),
        {RendererAbstractBaseFG791: {RendererFwOne791}, RendererConcreteSubFG791: set()},
    )


def abstract_bare_scenario() -> Scenario:
    """Abstract base matched with no concrete implementation accessible at all."""
    return Feature(ABSTRACT_FEATURE_791), {RendererAbstractBaseFG791: {RendererFwOne791}}


def ordinary_none_scenario() -> Scenario:
    """No match: one candidate rejects an option value, another supplies the name catalog."""
    feature = Feature(
        TYPO_FEATURE_791,
        Options(context={DefaultOptionKeys.in_features: "src", "window_size": 14}),
    )
    return feature, {RendererStrictFG791: {RendererFwOne791}, RendererKnownNamesFG791: {RendererFwOne791}}


def missing_option_scenario() -> Scenario:
    """No match: the sole candidate rejects for a MISSING required option."""
    return Feature(MISSING_OPTION_FEATURE_791), {RendererMissingOptionFG791: {RendererFwOne791}}


def scoped_none_scenario() -> Scenario:
    """No match while scoped to a feature group."""
    feature = Feature(SCOPED_NO_MATCH_FEATURE_791, feature_group="RendererKnownNamesFG791")
    return feature, {RendererKnownNamesFG791: {RendererFwOne791}}


def capability_narrow_enabled_scenario() -> Scenario:
    """Shape A: two declared frameworks, only the rejected one is enabled for this run."""
    return Feature(NARROW_ENABLED_FEATURE_791), {RendererNarrowEnabledFG791: {RendererFwOne791}}


def capability_none_enabled_scenario() -> Scenario:
    """Shape B: two declared frameworks, none enabled for this run."""
    return Feature(NONE_ENABLED_FEATURE_791), {RendererNoneEnabledFG791: set()}


def stranded_supported_name_scenario() -> Scenario:
    """The requested name is declared by the one candidate the run then eliminates: nothing is left to suggest."""
    return Feature(STRANDED_FEATURE_791), {RendererStrandedFG791: set()}


def suggestion_cut_scenario() -> Scenario:
    """Five droppable close matches (the echo plus both eliminated candidates' hints) outrank the useful name."""
    return (
        Feature(CUTOFF_FEATURE_791),
        {
            RendererCutoffAFG791: set(),
            RendererCutoffBFG791: set(),
            SpareNameCatalogFG791: {RendererFwOne791},
        },
    )


def duplicate_catalog_scenario() -> Scenario:
    """Five candidates declare the SAME name, so its copies alone can fill every suggestion slot."""
    return (
        Feature(DUPLICATE_TYPO_FEATURE_791),
        {
            RendererDuplicateNameFG791: {RendererFwOne791},
            RendererDuplicateSiblingAFG791: {RendererFwOne791},
            RendererDuplicateSiblingBFG791: {RendererFwOne791},
            RendererDuplicateSiblingCFG791: {RendererFwOne791},
            RendererDuplicateSiblingDFG791: {RendererFwOne791},
            RendererDuplicateSpareFG791: {RendererFwOne791},
        },
    )


def empty_catalog_scenario() -> Scenario:
    """No accessible plugin at all, so no name was ever captured and the catalog is empty."""
    return Feature(EMPTY_CATALOG_FEATURE_791), {}


def declared_unmatched_scenario() -> Scenario:
    """The requested name is declared by a candidate that simply does not match it, so nothing is eliminated."""
    return Feature(DECLARED_UNMATCHED_FEATURE_791), {RendererDeclaredUnmatchedFG791: {RendererFwOne791}}


def wide_catalog_scenario() -> Scenario:
    """More close names than slots, none of them droppable, so only the cut can bound the line."""
    return Feature(WIDE_FEATURE_791), {RendererWideCatalogFG791: {RendererFwOne791}}


def dead_sibling_scenario() -> Scenario:
    """The reported repro: the sole candidate declaring both names loses every framework the run could enable."""
    return Feature(DEAD_SIBLING_FEATURE_791), {RendererDeadSiblingFG791: set()}


def shared_dead_and_live_name_scenario() -> Scenario:
    """One name is declared by both a dead and a live candidate, the other only by the dead one."""
    return (
        Feature(SHARED_TYPO_FEATURE_791),
        {RendererSharedDeadFG791: set(), RendererSharedLiveFG791: {RendererFwOne791}},
    )


def value_stage_scenario() -> Scenario:
    """A value_rejection near-miss that keeps an enabled framework, so a sibling name could still resolve to it."""
    return Feature(VALUE_STAGE_FEATURE_791), {RendererValueStageFG791: {RendererFwOne791}}


def value_stage_without_frameworks_scenario() -> Scenario:
    """The same value_rejection candidate minus every enabled framework: no name can resolve to it."""
    return Feature(VALUE_STAGE_FEATURE_791), {RendererValueStageFG791: set()}


def capability_stage_scenario() -> Scenario:
    """A capability near-miss: the per-feature hook rejected the enabled framework for THIS name."""
    return Feature(CAPABILITY_STAGE_FEATURE_791), {RendererCapabilityStageFG791: {RendererFwOne791}}


def disabled_pair_scenario() -> Scenario:
    """Two groups with no enabled framework; only the one that matched the request is recorded as eliminated."""
    return (
        Feature(DISABLED_PAIR_FEATURE_791),
        {RendererDisabledPairFG791: set(), SpareNoFrameworkFG791: set()},
    )


def disabled_pair_reverse_scenario() -> Scenario:
    """The same two groups, now requesting the spare name the other direction suggests."""
    return (
        Feature(DISABLED_PAIR_SPARE_791),
        {RendererDisabledPairFG791: set(), SpareNoFrameworkFG791: set()},
    )


def live_prefix_scenario() -> Scenario:
    """A dead group's two names next to a live group whose prefix owns one of them."""
    return (
        Feature(LIVE_PREFIX_FEATURE_791),
        {RendererDeadPrefixFG791: set(), RendererLivePrefixFG791: {RendererFwOne791}},
    )


def live_prefix_success_scenario() -> Scenario:
    """The same two groups, requesting the covered name: the live group's prefix resolves it."""
    return (
        Feature(LIVE_PREFIX_COVERED_791),
        {RendererDeadPrefixFG791: set(), RendererLivePrefixFG791: {RendererFwOne791}},
    )


def domain_gate_scenario() -> Scenario:
    """A domain-carrying request nothing matches, next to a declarer of the close sibling name in another domain."""
    return (
        Feature(DOMAIN_GATE_FEATURE_791, domain=REQUESTED_DOMAIN_791),
        {CrossDomainDeclarerFG791: {RendererFwOne791}},
    )


def domain_gate_sibling_scenario() -> Scenario:
    """The same request domain, now asking for the sibling name a suggestion would hand back."""
    return (
        Feature(DOMAIN_GATE_SIBLING_791, domain=REQUESTED_DOMAIN_791),
        {CrossDomainDeclarerFG791: {RendererFwOne791}},
    )


def domainless_gate_scenario() -> Scenario:
    """The same declarer on a request carrying NO domain, where the domain gate cannot fire at all."""
    return Feature(DOMAIN_GATE_FEATURE_791), {CrossDomainDeclarerFG791: {RendererFwOne791}}


def domain_hook_cost_scenario() -> Scenario:
    """One candidate the decision pass compares by domain, one it never matches, on a domain-carrying request."""
    return (
        Feature(DOMAIN_GATE_SIBLING_791, domain=REQUESTED_DOMAIN_791),
        {CrossDomainDeclarerFG791: {RendererFwOne791}, ValueRejectingCrossDomainFG791: {RendererFwOne791}},
    )


def scope_gate_scenario() -> Scenario:
    """A scoped request nothing matches, next to a declarer of the close sibling name outside that scope."""
    return (
        Feature(SCOPE_GATE_FEATURE_791, feature_group=GATE_SCOPE_791),
        {OutsideScopeDeclarerFG791: {RendererFwOne791}, RendererKnownNamesFG791: {RendererFwOne791}},
    )


def scope_gate_sibling_scenario() -> Scenario:
    """The same scope, now asking for the sibling name a suggestion would hand back."""
    return (
        Feature(SCOPE_GATE_SIBLING_791, feature_group=GATE_SCOPE_791),
        {OutsideScopeDeclarerFG791: {RendererFwOne791}, RendererKnownNamesFG791: {RendererFwOne791}},
    )


def value_rejection_cross_domain_scenario() -> Scenario:
    """A candidate recorded at a name-DEPENDENT stage that the name-blind domain gate kills anyway."""
    return (
        Feature(VALUE_DOMAIN_FEATURE_791, domain=REQUESTED_DOMAIN_791),
        {ValueRejectingCrossDomainFG791: {RendererFwOne791}},
    )


def degraded_domain_scenario() -> Scenario:
    """A domain-carrying request whose only declarer of the close sibling name cannot report its domain."""
    return (
        Feature(DEGRADED_DOMAIN_FEATURE_791, domain=REQUESTED_DOMAIN_791),
        {_build_unreadable_domain_group(): {RendererFwOne791}},
    )


def malformed_domain_scenario() -> Scenario:
    """A domain-carrying request whose only declarer of the close sibling name returns a non-Domain domain."""
    return (
        Feature(MALFORMED_DOMAIN_FEATURE_791, domain=REQUESTED_DOMAIN_791),
        {_build_malformed_domain_group(): {RendererFwOne791}},
    )


def malformed_domain_sibling_scenario() -> Scenario:
    """The same request domain, now asking for the sibling name a suggestion would hand back."""
    return (
        Feature(MALFORMED_DOMAIN_SPARE_791, domain=REQUESTED_DOMAIN_791),
        {_build_malformed_domain_group(): {RendererFwOne791}},
    )


def dead_class_name_scenario() -> Scenario:
    """A domain-carrying request nothing matches, next to a wrong-domain group its typo names."""
    return (
        Feature(DEAD_CLASS_NAME_TYPO_791, domain=REQUESTED_DOMAIN_791),
        {RendererCrossDomainNameFG791: {RendererFwOne791}},
    )


def dead_class_name_with_live_declarer_scenario() -> Scenario:
    """The same request, now with a live group declaring both names the wrong-domain group owns."""
    return (
        Feature(DEAD_CLASS_NAME_TYPO_791, domain=REQUESTED_DOMAIN_791),
        {RendererCrossDomainNameFG791: {RendererFwOne791}, RendererLiveNameDeclarerFG791: {RendererFwOne791}},
    )


def dead_class_name_echo_scenario(name: str) -> Scenario:
    """The same request domain, now asking for one of the two names a suggestion would hand back."""
    return Feature(name, domain=REQUESTED_DOMAIN_791), {RendererCrossDomainNameFG791: {RendererFwOne791}}


def abstract_declarer_typo_scenario() -> Scenario:
    """A typo nothing matches, next to an abstract declarer of the close name and a framework-less twin."""
    return (
        Feature(ABSTRACT_DECLARER_TYPO_791),
        {SpareAbstractBaseFG791: {RendererFwOne791}, SpareDeadTwinFG791: set()},
    )


def abstract_declarer_name_scenario() -> Scenario:
    """The same two groups, now asking for the name a suggestion would hand back."""
    return (
        Feature(ABSTRACT_DECLARER_NAME_791),
        {SpareAbstractBaseFG791: {RendererFwOne791}, SpareDeadTwinFG791: set()},
    )


def abstract_domain_gate_scenario() -> Scenario:
    """An abstract-only failure on a request that carries the domain the abstract base declares."""
    return (
        Feature(ABSTRACT_DECLARER_NAME_791, domain=REQUESTED_DOMAIN_791),
        {SpareAbstractBaseFG791: {RendererFwOne791}},
    )


def raising_dead_names_scenario() -> Scenario:
    """A dead candidate whose feature_names_supported() raises, so it can contribute no name at all."""
    return Feature(RAISING_DEAD_NAMES_FEATURE_791), {_build_raising_dead_names_group(): set()}


def raising_domain_multiple_scenario() -> Scenario:
    """A 'multiple' failure where one identified candidate's get_domain() raises. The request has no domain."""
    raising, healthy = _build_raising_domain_groups()
    return Feature(RAISING_DOMAIN_FEATURE_791), {raising: {RendererFwOne791}, healthy: {RendererFwOne791}}


def raising_prefix_none_scenario() -> Scenario:
    """An ordinary-none failure where one catalog group's prefix() raises."""
    return (
        Feature(TYPO_FEATURE_791),
        {_build_raising_prefix_group(): {RendererFwOne791}, RendererKnownNamesFG791: {RendererFwOne791}},
    )


def raising_names_none_scenario() -> Scenario:
    """An ordinary-none failure where one catalog group's feature_names_supported() raises."""
    return (
        Feature(TYPO_FEATURE_791),
        {_build_raising_names_group(): {RendererFwOne791}, RendererKnownNamesFG791: {RendererFwOne791}},
    )


def raising_rejection_none_scenario() -> Scenario:
    """An ordinary-none failure next to a candidate whose value-rejection diagnostic would raise if consulted."""
    feature = Feature(
        TYPO_FEATURE_791,
        Options(context={DefaultOptionKeys.in_features: "src", "window_size": 14}),
    )
    return feature, {_build_raising_rejection_group(): {RendererFwOne791}, RendererStrictFG791: {RendererFwOne791}}


def raising_framework_rule_abstract_scenario() -> Scenario:
    """An abstract-only failure where the concrete subclass's compute_framework_rule() raises."""
    base, concrete = _build_raising_framework_rule_groups()
    return Feature(RAISING_ABSTRACT_FEATURE_791), {base: {RendererFwOne791}, concrete: set()}


FAILING_SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "multiple": multiple_scenario,
    "capability": capability_scenario,
    "abstract_with_frameworks": abstract_with_frameworks_scenario,
    "abstract_bare": abstract_bare_scenario,
    "ordinary_none": ordinary_none_scenario,
    "scoped_none": scoped_none_scenario,
    "capability_narrow_enabled": capability_narrow_enabled_scenario,
    "capability_none_enabled": capability_none_enabled_scenario,
    "dead_sibling": dead_sibling_scenario,
    "shared_dead_and_live_name": shared_dead_and_live_name_scenario,
    "value_stage": value_stage_scenario,
    "value_stage_without_frameworks": value_stage_without_frameworks_scenario,
    "capability_stage": capability_stage_scenario,
    "disabled_pair": disabled_pair_scenario,
    "disabled_pair_reverse": disabled_pair_reverse_scenario,
    "live_prefix": live_prefix_scenario,
    "domain_gate": domain_gate_scenario,
    "scope_gate": scope_gate_scenario,
    "value_rejection_cross_domain": value_rejection_cross_domain_scenario,
    "degraded_domain": degraded_domain_scenario,
    "malformed_domain": malformed_domain_scenario,
    "dead_class_name": dead_class_name_scenario,
    "abstract_declarer_typo": abstract_declarer_typo_scenario,
    "abstract_domain_gate": abstract_domain_gate_scenario,
    "raising_dead_names": raising_dead_names_scenario,
    "raising_domain_multiple": raising_domain_multiple_scenario,
    "raising_prefix_none": raising_prefix_none_scenario,
    "raising_names_none": raising_names_none_scenario,
    "raising_rejection_none": raising_rejection_none_scenario,
    "raising_framework_rule_abstract": raising_framework_rule_abstract_scenario,
}

# Every (candidate, framework) pair the capability hook may be asked about during one evaluate(), and how
# often: exactly once. Run-only: the hook is asked solely over the frameworks the run enabled for the
# candidate, so a declared-but-disabled framework is never asked about at all.
CAPABILITY_PAIR_EXPECTATIONS: dict[str, dict[tuple[str, str], int]] = {
    "capability": {
        ("RendererCapabilityAFG791", "RendererFwOne791"): 1,
        ("RendererCapabilityAFG791", "RendererFwTwo791"): 1,
        ("RendererCapabilityBFG791", "RendererFwOne791"): 1,
        ("RendererCapabilityBFG791", "RendererFwTwo791"): 1,
    },
    "capability_narrow_enabled": {
        ("RendererNarrowEnabledFG791", "RendererFwOne791"): 1,
    },
    "capability_none_enabled": {},
}


def _evaluate(scenario: Scenario) -> EvaluationResult:
    feature, accessible_plugins = scenario
    return IdentifyFeatureGroupClass.evaluate(feature=feature, accessible_plugins=accessible_plugins, links=None)


def _render(scenario: Scenario) -> str:
    feature, _ = scenario
    message = render_resolution_failure(_evaluate(scenario), feature)
    assert message is not None
    return message


def _suggestions(message: str) -> list[str] | None:
    """Names listed on the 'Did you mean' line, or None when the renderer omitted the line entirely."""
    line = next((line for line in message.split("\n") if line.startswith(SUGGESTION_PREFIX)), None)
    if line is None:
        return None
    return cast(list[str], literal_eval(line[len(SUGGESTION_PREFIX) : -1]))


@pytest.fixture(autouse=True)
def reset_hook_calls() -> Iterator[None]:
    """Counter state must not leak between tests, and no raising hook may outlive the test that built it."""
    HOOK_CALLS.clear()
    PAIR_CALLS.clear()
    yield
    for group in RAISING_GROUPS_BUILT:
        group.ARMED = False
    RAISING_GROUPS_BUILT.clear()


class TestRenderResolutionFailureMessages:
    """The renderer projects each failure kind into the message shape the engine raises today."""

    def test_success_renders_none(self) -> None:
        """A successful evaluation has nothing to render."""
        scenario = success_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        assert result.failure_kind is None
        assert render_resolution_failure(result, feature) is None

    def test_multiple_lists_every_identified_candidate_with_its_domain(self) -> None:
        """Sorted one line per identified candidate, then the troubleshooting URL."""
        module = RendererMultipleAFG791.__module__

        message = _render(multiple_scenario())

        assert message == (
            f"Multiple feature groups found for feature '{MULTIPLE_FEATURE_791}':\n"
            f"  - RendererMultipleAFG791 ({module}) [domain: renderer_domain_a_791]\n"
            f"  - RendererMultipleBFG791 ({module}) [domain: renderer_domain_b_791]\n"
            f"{TROUBLESHOOTING_LINE}"
        )

    def test_capability_rejection_renders_one_line_per_candidate(self) -> None:
        """Each pinned-out candidate names its OWN supported frameworks in one sorted near-miss line."""
        message = _render(capability_scenario())

        assert message.startswith(f"No feature groups found for feature name: '{CAPABILITY_FEATURE_791}'.")
        assert (
            f"Feature group(s) eliminated while matching '{CAPABILITY_FEATURE_791}':\n"
            "  - RendererCapabilityAFG791 (compute framework pin): pinned compute framework 'RendererFwThree791'"
            " is not among its supported ['RendererFwOne791']\n"
            "  - RendererCapabilityBFG791 (compute framework pin): pinned compute framework 'RendererFwThree791'"
            " is not among its supported ['RendererFwTwo791']"
        ) in message

    def test_abstract_only_names_concrete_implementation_frameworks(self) -> None:
        """Abstract-only names the concrete frameworks, then appends the near-miss line for the concrete subclass."""
        message = _render(abstract_with_frameworks_scenario())

        assert message == (
            f"No feature groups found for feature name: '{ABSTRACT_FEATURE_791}'. "
            "Its concrete implementations require compute framework(s) "
            "['RendererFwOne791', 'RendererFwTwo791'], "
            "none of which are available or enabled for this run.\n"
            f"Feature group(s) eliminated while matching '{ABSTRACT_FEATURE_791}':\n"
            "  - RendererConcreteSubFG791 (compute framework): none of its compute frameworks are enabled for this run"
        )

    def test_abstract_only_without_concrete_implementation(self) -> None:
        """Abstract-only with no concrete implementation keeps the bare variant."""
        message = _render(abstract_bare_scenario())

        assert message == (
            f"No feature groups found for feature name: '{ABSTRACT_FEATURE_791}'. "
            "Only abstract feature group base(s) matched, which cannot be instantiated; "
            "no concrete implementation is available or enabled."
        )

    def test_ordinary_none_renders_rejections_suggestion_and_pointers(self) -> None:
        """The near-miss block, then 'Did you mean', then the resolve_feature and troubleshooting lines."""
        message = _render(ordinary_none_scenario())

        lines = message.split("\n")
        assert lines[0] == f"No feature groups found for feature name: '{TYPO_FEATURE_791}'."
        assert lines[1] == f"Feature group(s) eliminated while matching '{TYPO_FEATURE_791}':"
        assert lines[2] == f"  - RendererStrictFG791 (option value): {WINDOW_REJECTION_REASON}"
        assert lines[3].startswith("Did you mean one of: [")
        assert f"'{KNOWN_FEATURE_791}'" in lines[3]
        assert lines[4] == "Use resolve_feature(name, options=...) to debug feature resolution."
        assert lines[5] == TROUBLESHOOTING_LINE
        assert len(lines) == 6

    def test_missing_option_rejection_renders_under_the_options_heading(self) -> None:
        """A MISSING required option is a value_rejection too, so it lands in the near-miss block as an option value."""
        scenario = missing_option_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        assert result.failure_kind == "none"
        assert result.eliminations == {
            RendererMissingOptionFG791: Elimination(stage="value_rejection", reason=MISSING_OPTION_REASON_791)
        }

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert f"Feature group(s) eliminated while matching '{MISSING_OPTION_FEATURE_791}':" in message
        assert f"  - RendererMissingOptionFG791 (option value): {MISSING_OPTION_REASON_791}" in message

    def test_scoped_none_renders_callout_and_scoped_pointer(self) -> None:
        """A scoped no-match carries the scope callout and the scoped resolve_feature pointer."""
        message = _render(scoped_none_scenario())

        assert message.startswith(
            f"No feature groups found for feature name: '{SCOPED_NO_MATCH_FEATURE_791}'. "
            "Scoped to feature group: 'RendererKnownNamesFG791'."
        )
        assert "Use resolve_feature(name, options=..., feature_group=...) to debug feature resolution." in message
        assert message.endswith(TROUBLESHOOTING_LINE)

    def test_forwarding_hint_is_dropped(self) -> None:
        """The forwarding hint needs a speculative second match, so the pure renderer drops it."""
        feature = Feature(FORWARDING_FEATURE_791, Options(group={"query_text": "hi", "top_k": 5}))
        accessible_plugins: FeatureGroupEnvironmentMapping = {RendererBareOnlyFG791: {RendererFwOne791}}

        message = render_resolution_failure(
            IdentifyFeatureGroupClass.evaluate(feature=feature, accessible_plugins=accessible_plugins, links=None),
            feature,
        )

        assert message is not None
        assert "forward_group" not in message


class TestPerCandidateCorrelation:
    """A candidate's frameworks stay correlated to that candidate, never merged into a union."""

    def test_near_miss_lines_name_only_each_candidates_own_frameworks(self) -> None:
        """Mirrored candidates: each near-miss line names only its OWN supported framework, never the union."""
        message = _render(capability_scenario())

        a_line = next(line for line in message.split("\n") if "RendererCapabilityAFG791" in line)
        b_line = next(line for line in message.split("\n") if "RendererCapabilityBFG791" in line)

        assert a_line == (
            "  - RendererCapabilityAFG791 (compute framework pin): pinned compute framework 'RendererFwThree791'"
            " is not among its supported ['RendererFwOne791']"
        )
        assert b_line == (
            "  - RendererCapabilityBFG791 (compute framework pin): pinned compute framework 'RendererFwThree791'"
            " is not among its supported ['RendererFwTwo791']"
        )
        # The cross-candidate union today's message builds must never appear.
        assert "['RendererFwOne791', 'RendererFwTwo791']" not in message

    def test_candidate_frameworks_are_captured_per_candidate(self) -> None:
        """evaluate() correlates supported/rejected frameworks with their own candidate."""
        result = _evaluate(capability_scenario())

        assert result.candidate_frameworks == {
            RendererCapabilityAFG791: CandidateFrameworks(
                supported=frozenset({RendererFwOne791}),
                rejected=frozenset({RendererFwTwo791}),
            ),
            RendererCapabilityBFG791: CandidateFrameworks(
                supported=frozenset({RendererFwTwo791}),
                rejected=frozenset({RendererFwOne791}),
            ),
        }


class TestRenderDeterminism:
    """Rendering is a pure function of the result: same input, same string."""

    @pytest.mark.parametrize("scenario_name", sorted(FAILING_SCENARIOS))
    def test_repeated_rendering_returns_identical_strings(self, scenario_name: str) -> None:
        """Rendering the same result twice returns the identical string."""
        scenario = FAILING_SCENARIOS[scenario_name]()
        feature, _ = scenario
        result = _evaluate(scenario)

        first = render_resolution_failure(result, feature)
        second = render_resolution_failure(result, feature)

        assert first is not None
        assert first == second

    def test_candidate_lines_are_sorted(self) -> None:
        """Candidate lines are sorted, independent of the accessible_plugins insertion order."""
        multiple_lines = [line for line in _render(multiple_scenario()).split("\n") if line.startswith("  - ")]
        capability_lines = [line for line in _render(capability_scenario()).split("\n") if line.startswith("  - ")]

        assert multiple_lines == sorted(multiple_lines)
        assert len(multiple_lines) == 2
        assert capability_lines == sorted(capability_lines)
        assert len(capability_lines) == 2


class TestRendererCallsNoProviderHook:
    """The core DoD: rendering touches no provider-overridable hook, for every failure kind."""

    @pytest.mark.parametrize("scenario_name", sorted(FAILING_SCENARIOS))
    def test_rendering_leaves_every_hook_counter_unchanged(self, scenario_name: str) -> None:
        """evaluate() may call the hooks; repeated rendering afterwards must call none."""
        scenario = FAILING_SCENARIOS[scenario_name]()
        feature, _ = scenario
        result = _evaluate(scenario)
        assert result.failure_kind is not None

        snapshot = dict(HOOK_CALLS)
        assert snapshot, "the fixture feature groups must count at least one hook call during evaluate()"

        for _ in range(3):
            assert render_resolution_failure(result, feature) is not None

        assert HOOK_CALLS == snapshot


class TestFactsCapturedDuringEvaluate:
    """evaluate() captures the render facts in its first pass, and only when it has no winner."""

    def test_success_leaves_facts_at_the_empty_default(self) -> None:
        """No capture on the success path; candidate frameworks still correlate to the winner."""
        result = _evaluate(success_scenario())

        assert result.facts == RenderFacts()
        assert result.candidate_frameworks == {
            RendererSuccessFG791: CandidateFrameworks(supported=frozenset({RendererFwOne791}), rejected=frozenset())
        }

    def test_domains_captured_for_multiple(self) -> None:
        """The 'multiple' kind captures the domain NAME of every identified candidate."""
        result = _evaluate(multiple_scenario())

        assert result.failure_kind == "multiple"
        assert result.facts.domains == {
            RendererMultipleAFG791: "renderer_domain_a_791",
            RendererMultipleBFG791: "renderer_domain_b_791",
        }

    def test_concrete_frameworks_captured_for_abstract_only(self) -> None:
        """The abstract-only kind captures the declared frameworks of accessible concrete subclasses."""
        result = _evaluate(abstract_with_frameworks_scenario())

        assert result.failure_kind == "abstract_only"
        assert result.facts.concrete_frameworks == ("RendererFwOne791", "RendererFwTwo791")

    def test_no_concrete_frameworks_without_accessible_implementation(self) -> None:
        """No accessible concrete subclass leaves concrete_frameworks empty."""
        result = _evaluate(abstract_bare_scenario())

        assert result.failure_kind == "abstract_only"
        assert result.facts.concrete_frameworks == ()

    def test_ordinary_none_records_value_rejection_elimination_and_captures_known_names(self) -> None:
        """The ordinary-none kind records the value rejection as an elimination and captures the name catalog."""
        result = _evaluate(ordinary_none_scenario())

        assert result.failure_kind == "none"
        assert result.eliminations == {
            RendererStrictFG791: Elimination(stage="value_rejection", reason=WINDOW_REJECTION_REASON)
        }
        assert KNOWN_FEATURE_791 in result.facts.known_names
        assert "RendererKnownNamesFG791" in result.facts.known_names
        assert "RendererStrictFG791_" in result.facts.known_names


class TestFactCaptureNeverTakesEvaluateDown:
    """Fact capture is best-effort rendering data, never a decision input.

    evaluate() now calls hooks its decision pass never called (get_domain on a domain-less request,
    prefix, feature_names_supported, compute_framework_definition).
    A provider whose hook raises must degrade that one fact only: evaluate() still returns its result and
    the renderer still returns a message, exactly as the guarded message builders behaved before #791.
    """

    def test_raising_get_domain_still_lists_both_candidates(self) -> None:
        """A degraded domain drops only the '[domain: ...]' suffix of its own candidate line."""
        scenario = raising_domain_multiple_scenario()
        feature, accessible_plugins = scenario
        raising, healthy = list(accessible_plugins)
        module = raising.__module__

        result = _evaluate(scenario)

        assert result.failure_kind == "multiple"
        assert render_resolution_failure(result, feature) == (
            f"Multiple feature groups found for feature '{RAISING_DOMAIN_FEATURE_791}':\n"
            f"  - {healthy.__name__} ({module}) [domain: {HEALTHY_DOMAIN_791}]\n"
            f"  - {raising.__name__} ({module})\n"
            f"{TROUBLESHOOTING_LINE}"
        )

    def test_raising_prefix_only_costs_that_group_its_prefix(self) -> None:
        """A raising prefix() contributes no name; the healthy group next to it still contributes its own."""
        scenario = raising_prefix_none_scenario()
        feature, _ = scenario

        result = _evaluate(scenario)

        assert result.failure_kind == "none"
        assert "RendererRaisingPrefixFG791_" not in result.facts.known_names
        assert KNOWN_FEATURE_791 in result.facts.known_names
        assert "RendererKnownNamesFG791_" in result.facts.known_names

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert f"'{KNOWN_FEATURE_791}'" in message

    def test_raising_feature_names_supported_only_costs_that_group_its_names(self) -> None:
        """A raising feature_names_supported() contributes no name, and the catalog still renders."""
        scenario = raising_names_none_scenario()
        feature, _ = scenario

        result = _evaluate(scenario)

        assert result.failure_kind == "none"
        assert BOOM_SUPPORTED_NAME_791 not in result.facts.known_names
        assert KNOWN_FEATURE_791 in result.facts.known_names

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert f"'{KNOWN_FEATURE_791}'" in message

    def test_the_raising_rejection_hook_is_never_consulted(self) -> None:
        """The engine renders rejections recorded in the first pass, so a raising hook simply never fires.

        The healthy strict candidate's rejection was recorded while matching and still renders; the
        hostile diagnostic is never called, so there is nothing to degrade.
        """
        scenario = raising_rejection_none_scenario()
        feature, _ = scenario

        result = _evaluate(scenario)

        assert result.failure_kind == "none"
        assert "RendererRaisingRejectionFG791._strict_validation_rejection_reason" not in HOOK_CALLS
        assert result.eliminations == {
            RendererStrictFG791: Elimination(stage="value_rejection", reason=WINDOW_REJECTION_REASON)
        }

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert f"  - RendererStrictFG791 (option value): {WINDOW_REJECTION_REASON}" in message
        assert "RendererRaisingRejectionFG791 (option value)" not in message

    def test_raising_compute_framework_rule_falls_back_to_the_bare_abstract_message(self) -> None:
        """No framework name could be captured, so the abstract-only message takes its bare variant."""
        scenario = raising_framework_rule_abstract_scenario()
        feature, _ = scenario

        result = _evaluate(scenario)

        assert result.failure_kind == "abstract_only"
        assert result.facts.concrete_frameworks == ()
        assert render_resolution_failure(result, feature) == (
            f"No feature groups found for feature name: '{RAISING_ABSTRACT_FEATURE_791}'. "
            "Only abstract feature group base(s) matched, which cannot be instantiated; "
            "no concrete implementation is available or enabled.\n"
            f"Feature group(s) eliminated while matching '{RAISING_ABSTRACT_FEATURE_791}':\n"
            "  - RendererRaisingConcreteSubFG791 (compute framework): "
            "none of its compute frameworks are enabled for this run"
        )

    def test_resolve_feature_still_reports_candidates_when_get_domain_raises(self) -> None:
        """End-to-end parity: a raising capture hook must not empty resolve_feature's candidates."""
        raising, healthy = _build_raising_domain_groups()
        enabled: set[type[FeatureGroup]] = {raising, healthy}

        result = resolve_feature(
            Feature(RAISING_DOMAIN_FEATURE_791),
            plugin_collector=PluginCollector.enabled_feature_groups(enabled),
        )

        assert result.feature_group is None
        assert set(result.candidates) == enabled
        assert result.error is not None


class TestCapabilityRenderingUniverse:
    """Run-only: the capability near-miss names ONLY the run-enabled frameworks the candidate rejected.

    ``candidate_frameworks`` is the decision fact and stays the run's own (narrower) split of the
    frameworks that were enabled. Rendering inherits that narrowing: a declared-but-not-enabled framework
    is never named, and an all-disabled candidate renders under the ordinary-none message.
    """

    def test_not_enabled_framework_still_renders_as_supported(self) -> None:
        """Shape A: only the enabled framework is named; the supported-but-not-enabled one is dropped."""
        scenario = capability_narrow_enabled_scenario()
        feature, _ = scenario

        result = _evaluate(scenario)

        # The decision fact stays the run's own split: only RendererFwOne791 was enabled, and it was rejected.
        assert result.candidate_frameworks == {
            RendererNarrowEnabledFG791: CandidateFrameworks(
                supported=frozenset(), rejected=frozenset({RendererFwOne791})
            )
        }
        message = render_resolution_failure(result, feature)
        assert message is not None
        assert message.startswith(f"No feature groups found for feature name: '{NARROW_ENABLED_FEATURE_791}'.")
        assert (
            "  - RendererNarrowEnabledFG791 (compute framework): "
            "supports_compute_framework rejected ['RendererFwOne791']"
        ) in message
        # The declared-but-not-enabled framework is no longer part of the run-only universe.
        assert "RendererFwTwo791" not in message

    def test_no_enabled_framework_still_renders_the_capability_message(self) -> None:
        """Shape B: nothing enabled renders the frameworks_not_enabled near-miss under the ordinary-none message."""
        scenario = capability_none_enabled_scenario()
        feature, _ = scenario

        result = _evaluate(scenario)

        # Nothing was enabled, so the decision loop split nothing: the decision fact is empty on both sides.
        assert result.candidate_frameworks == {
            RendererNoneEnabledFG791: CandidateFrameworks(supported=frozenset(), rejected=frozenset())
        }
        message = render_resolution_failure(result, feature)

        assert message is not None
        assert message.startswith(f"No feature groups found for feature name: '{NONE_ENABLED_FEATURE_791}'.")
        assert (
            "  - RendererNoneEnabledFG791 (compute framework): none of its compute frameworks are enabled for this run"
        ) in message
        assert "RendererFwTwo791" not in message

    @pytest.mark.parametrize("scenario_name", sorted(CAPABILITY_PAIR_EXPECTATIONS))
    def test_capability_hook_is_asked_once_per_candidate_framework_pair(self, scenario_name: str) -> None:
        """The decision loop already split the enabled frameworks; capture may only ask about the rest."""
        _evaluate(FAILING_SCENARIOS[scenario_name]())

        repeated = sorted(pair for pair, count in PAIR_CALLS.items() if count != 1)
        assert not repeated, f"the capability hook was asked more than once for {repeated}"
        assert PAIR_CALLS == CAPABILITY_PAIR_EXPECTATIONS[scenario_name]


class TestSortTiesAreStable:
    """Two candidates sharing a __name__ across modules must not fall back to insertion order."""

    def test_multiple_tie_sorts_by_module_and_ignores_insertion_order(self) -> None:
        """Same-named 'multiple' candidates render module-sorted, whichever way they were inserted."""
        group_a, group_b = _build_tie_domain_groups()
        feature = Feature(TIE_FEATURE_791)
        a_first: FeatureGroupEnvironmentMapping = {group_a: {RendererFwOne791}, group_b: {RendererFwOne791}}
        b_first: FeatureGroupEnvironmentMapping = {group_b: {RendererFwOne791}, group_a: {RendererFwOne791}}

        expected = (
            f"Multiple feature groups found for feature '{TIE_FEATURE_791}':\n"
            f"  - RendererTieFG791 ({TIE_MODULE_A_791}) [domain: renderer_tie_domain_a_791]\n"
            f"  - RendererTieFG791 ({TIE_MODULE_B_791}) [domain: renderer_tie_domain_b_791]\n"
            f"{TROUBLESHOOTING_LINE}"
        )

        assert _render((feature, a_first)) == expected
        assert _render((feature, b_first)) == expected

    def test_capability_tie_sorts_by_module_and_ignores_insertion_order(self) -> None:
        """Same-named capability candidates render module-sorted, whichever way they were inserted.

        Every close match here is a candidate the near-miss block already named (its class name or prefix), so
        the 'Did you mean' line is suppressed entirely rather than echoing the eliminated candidates back.
        """
        group_a, group_b = _build_tie_capability_groups()
        feature = Feature(TIE_CAPABILITY_FEATURE_791, compute_framework="RendererFwThree791")
        both = {RendererFwOne791, RendererFwTwo791}
        a_first: FeatureGroupEnvironmentMapping = {group_a: set(both), group_b: set(both)}
        b_first: FeatureGroupEnvironmentMapping = {group_b: set(both), group_a: set(both)}

        expected = (
            f"No feature groups found for feature name: '{TIE_CAPABILITY_FEATURE_791}'.\n"
            f"Feature group(s) eliminated while matching '{TIE_CAPABILITY_FEATURE_791}':\n"
            "  - RendererTieFG791 (compute framework pin): pinned compute framework 'RendererFwThree791'"
            " is not among its supported ['RendererFwOne791']\n"
            "  - RendererTieFG791 (compute framework pin): pinned compute framework 'RendererFwThree791'"
            " is not among its supported ['RendererFwTwo791']\n"
            "Use resolve_feature(name, options=...) to debug feature resolution.\n"
            f"{TROUBLESHOOTING_LINE}"
        )

        assert _render((feature, a_first)) == expected
        assert _render((feature, b_first)) == expected
        # The suppressed suggestion must not echo the already-named eliminated candidate or its prefix.
        assert "Did you mean" not in _render((feature, a_first))


class TestSuggestionsNeverEchoTheRequestedName:
    """A suggestion equal to the requested name is worthless, whichever candidate contributed it."""

    def test_eliminated_candidates_supported_name_is_never_suggested_back(self) -> None:
        """A candidate declaring the requested name is eliminated: the message must not hand that name back."""
        scenario = stranded_supported_name_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        # The name catalog carries the requested name, and the eliminated hints (class name and prefix only)
        # cannot reach it, so suppression has to key on the requested name itself.
        assert STRANDED_FEATURE_791 in result.facts.known_names
        assert STRANDED_FEATURE_791 not in result.facts.eliminated_hints

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert "Did you mean" not in message
        assert message == (
            f"No feature groups found for feature name: '{STRANDED_FEATURE_791}'.\n"
            f"Feature group(s) eliminated while matching '{STRANDED_FEATURE_791}':\n"
            "  - RendererStrandedFG791 (compute framework): none of its compute frameworks are enabled for this run\n"
            "Use resolve_feature(name, options=...) to debug feature resolution.\n"
            f"{TROUBLESHOOTING_LINE}"
        )

    def test_droppable_matches_do_not_consume_the_suggestion_slots(self) -> None:
        """Suggestions are filtered before the cut, so a useful name ranked behind five droppable ones survives."""
        scenario = suggestion_cut_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        droppable = {CUTOFF_FEATURE_791, *result.facts.eliminated_hints}
        # Premise of the scenario: cutting to five first spends every slot on a droppable name.
        cut_first = get_close_matches(CUTOFF_FEATURE_791, list(result.facts.known_names), n=5, cutoff=0.5)
        assert len(cut_first) == 5
        assert set(cut_first) <= droppable
        assert CUTOFF_CATALOG_NAME_791 not in cut_first

        message = render_resolution_failure(result, feature)
        assert message is not None
        suggestions = _suggestions(message)

        assert suggestions is not None
        assert CUTOFF_CATALOG_NAME_791 in suggestions
        assert droppable.isdisjoint(suggestions)
        # Both eliminated candidates are still named where they belong, in the near-miss block.
        assert "  - RendererCutoffAFG791 (compute framework):" in message
        assert "  - RendererCutoffBFG791 (compute framework):" in message

    def test_a_never_eliminated_candidates_same_name_is_suppressed_too(self) -> None:
        """A candidate can declare the requested name without ever being eliminated; the echo is dropped anyway.

        Nothing was eliminated here, so widening eliminated_hints to the eliminated candidates' supported
        names would suppress nothing: only keying on the requested name itself does.
        """
        scenario = declared_unmatched_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        assert result.facts.eliminated_hints == frozenset()
        assert DECLARED_UNMATCHED_FEATURE_791 in result.facts.known_names

        message = render_resolution_failure(result, feature)
        assert message is not None
        suggestions = _suggestions(message)

        assert suggestions is not None
        assert DECLARED_UNMATCHED_FEATURE_791 not in suggestions
        # The line still renders; it just carries the candidate's class name and prefix instead of the echo.
        assert set(suggestions) == {"RendererDeclaredUnmatchedFG791", "RendererDeclaredUnmatchedFG791_"}


class TestSuggestionSlotsListDistinctNames:
    """The suggestion line lists distinct useful names, and the cut to five is what bounds it."""

    def test_duplicate_catalog_entries_do_not_spend_the_suggestion_slots(self) -> None:
        """One name declared by five candidates enters the catalog five times, but may fill only one slot."""
        scenario = duplicate_catalog_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        # Premise of the scenario: ranking the catalog as captured spends every slot on copies of one name.
        known_names = list(result.facts.known_names)
        copies = get_close_matches(DUPLICATE_TYPO_FEATURE_791, known_names, n=MAX_RENDERED_SUGGESTIONS_791, cutoff=0.5)
        assert copies == [DUPLICATE_CATALOG_NAME_791] * MAX_RENDERED_SUGGESTIONS_791

        message = render_resolution_failure(result, feature)
        assert message is not None
        suggestions = _suggestions(message)

        assert suggestions is not None
        assert len(suggestions) == len(set(suggestions))
        assert DUPLICATE_CATALOG_NAME_791 in suggestions
        assert DUPLICATE_SPARE_NAME_791 in suggestions

    def test_an_empty_name_catalog_renders_no_suggestion_line(self) -> None:
        """No accessible plugin leaves the catalog empty: ranking it must not raise, and no line renders."""
        scenario = empty_catalog_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        assert result.failure_kind == "none"
        assert result.facts.known_names == ()

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert _suggestions(message) is None
        assert message == (
            f"No feature groups found for feature name: '{EMPTY_CATALOG_FEATURE_791}'.\n"
            "Use resolve_feature(name, options=...) to debug feature resolution.\n"
            f"{TROUBLESHOOTING_LINE}"
        )

    def test_more_surviving_close_names_than_slots_are_cut_to_five(self) -> None:
        """Nothing is droppable here, so the cut is the only bound: exactly five of the pool are listed."""
        scenario = wide_catalog_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        # Premise of the scenario: more than five distinct names clear the cutoff and none of them is droppable.
        known_names = sorted(set(result.facts.known_names))
        ranked = get_close_matches(WIDE_FEATURE_791, known_names, n=len(known_names), cutoff=0.5)
        assert len(ranked) > MAX_RENDERED_SUGGESTIONS_791
        assert result.facts.eliminated_hints == frozenset()
        assert WIDE_FEATURE_791 not in known_names

        message = render_resolution_failure(result, feature)
        assert message is not None
        suggestions = _suggestions(message)

        assert suggestions is not None
        assert len(suggestions) == MAX_RENDERED_SUGGESTIONS_791
        assert set(suggestions) <= WIDE_CATALOG_NAMES_791


class TestSuggestionsNeverPointAtADeadGroupsSiblingName:
    """A name only dead groups declare cannot resolve either, so suggesting it hands back the same failure.

    A candidate is dead when it was eliminated AND either the gate that dropped it structurally could not see
    the feature name (NAME_INDEPENDENT_STAGES), or it has no accessible compute framework left at all. The
    rule is per NAME, not per candidate: one live declarer is enough to keep a name suggestible.
    """

    def test_a_dead_groups_sibling_name_is_never_suggested(self) -> None:
        """The reported repro: following the suggested sibling would fail with a byte-identical message."""
        scenario = dead_sibling_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        assert result.eliminations[RendererDeadSiblingFG791].stage == "frameworks_not_enabled"
        # Premise: the requested name and the eliminated hints cannot reach the sibling, so only this rule can.
        droppable = {DEAD_SIBLING_FEATURE_791, *result.facts.eliminated_hints}
        survivors = [name for name in dict.fromkeys(result.facts.known_names) if name not in droppable]
        ranked = get_close_matches(DEAD_SIBLING_FEATURE_791, survivors, n=MAX_RENDERED_SUGGESTIONS_791, cutoff=0.5)
        assert ranked == [DEAD_SIBLING_SPARE_791]
        assert DEAD_SIBLING_SPARE_791 in result.facts.dead_only_names

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert message == (
            f"No feature groups found for feature name: '{DEAD_SIBLING_FEATURE_791}'.\n"
            f"Feature group(s) eliminated while matching '{DEAD_SIBLING_FEATURE_791}':\n"
            "  - RendererDeadSiblingFG791 (compute framework): "
            "none of its compute frameworks are enabled for this run\n"
            "Use resolve_feature(name, options=...) to debug feature resolution.\n"
            f"{TROUBLESHOOTING_LINE}"
        )

    def test_a_name_a_live_group_also_declares_is_still_suggested(self) -> None:
        """Suppression keys on the name: the live declarer would resolve it, so only the dead-only name goes."""
        scenario = shared_dead_and_live_name_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        assert result.eliminations[RendererSharedDeadFG791].stage == "frameworks_not_enabled"
        assert RendererSharedLiveFG791 not in result.eliminations
        assert SHARED_LIVE_NAME_791 not in result.facts.dead_only_names
        assert SHARED_DEAD_NAME_791 in result.facts.dead_only_names

        message = render_resolution_failure(result, feature)
        assert message is not None
        suggestions = _suggestions(message)

        assert suggestions is not None
        assert set(suggestions) == {SHARED_LIVE_NAME_791, "RendererSharedLiveFG791", "RendererSharedLiveFG791_"}

    def test_a_value_rejection_candidates_sibling_name_is_still_suggested(self) -> None:
        """value_rejection is name-DEPENDENT: the candidate declined this name's value, not the sibling's."""
        scenario = value_stage_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        assert result.eliminations == {
            RendererValueStageFG791: Elimination(stage="value_rejection", reason=VALUE_STAGE_REJECTION_REASON_791)
        }
        assert VALUE_STAGE_SPARE_791 not in result.facts.dead_only_names

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert _suggestions(message) == [VALUE_STAGE_SPARE_791]

    def test_a_capability_candidates_sibling_name_is_still_suggested(self) -> None:
        """capability comes from supports_compute_framework(feature.name, ...), so a sibling name may pass it."""
        scenario = capability_stage_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        assert result.eliminations[RendererCapabilityStageFG791].stage == "capability"
        assert CAPABILITY_STAGE_SPARE_791 not in result.facts.dead_only_names

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert _suggestions(message) == [CAPABILITY_STAGE_SPARE_791]

    def test_a_name_dependent_stage_without_any_framework_is_dead_anyway(self) -> None:
        """Same candidate and same stage as above, minus every framework: no name can reach it, so it is dead."""
        scenario = value_stage_without_frameworks_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        assert result.eliminations[RendererValueStageFG791].stage == "value_rejection"
        assert VALUE_STAGE_SPARE_791 in result.facts.known_names
        assert VALUE_STAGE_SPARE_791 in result.facts.dead_only_names

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert _suggestions(message) is None

    def test_a_raising_names_hook_on_a_dead_candidate_contributes_nothing(self) -> None:
        """A dead candidate whose name hook raises declares no dead name, and takes nothing else down."""
        scenario = raising_dead_names_scenario()
        feature, accessible_plugins = scenario
        (raising,) = accessible_plugins

        result = _evaluate(scenario)

        assert result.eliminations[raising].stage == "frameworks_not_enabled"
        assert RAISING_DEAD_NAMES_SPARE_791 not in result.facts.known_names
        # The declared names could not be read, so all this dead candidate contributes is its class identity:
        # the hook that raised names nothing at all.
        assert RAISING_DEAD_NAMES_SPARE_791 not in result.facts.dead_only_names
        assert result.facts.dead_only_names == frozenset({raising.get_class_name(), raising.prefix()})

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert message == (
            f"No feature groups found for feature name: '{RAISING_DEAD_NAMES_FEATURE_791}'.\n"
            f"Feature group(s) eliminated while matching '{RAISING_DEAD_NAMES_FEATURE_791}':\n"
            f"  - {raising.__name__} (compute framework): none of its compute frameworks are enabled for this run\n"
            "Use resolve_feature(name, options=...) to debug feature resolution.\n"
            f"{TROUBLESHOOTING_LINE}"
        )

    def test_feature_names_supported_is_read_once_per_candidate(self) -> None:
        """The name catalog and the dead-name capture read the same candidate, and share one hook call."""
        result = _evaluate(dead_sibling_scenario())

        # Both readers ran: the catalog holds the sibling and the dead-name capture claimed it.
        assert DEAD_SIBLING_SPARE_791 in result.facts.known_names
        assert DEAD_SIBLING_SPARE_791 in result.facts.dead_only_names
        assert HOOK_CALLS["RendererDeadSiblingFG791.feature_names_supported"] == 1


class TestAGroupWithoutAnyFrameworkIsDeadWithoutAnElimination:
    """A group with no accessible framework can never be identified, for ANY name, so it declares nothing live.

    It is only recorded as eliminated for the name it was asked about, so an elimination record is the wrong
    liveness test: a second such group that never matched would otherwise keep the whole dead pair suggestible.
    """

    def test_two_disabled_groups_never_suggest_each_others_names(self) -> None:
        """Neither direction may suggest the other name: both requests hit the same pair of disabled groups."""
        scenario = disabled_pair_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        assert result.eliminations[RendererDisabledPairFG791].stage == "frameworks_not_enabled"
        # The spare group never matched this name, so nothing records it: only its empty framework set kills it.
        assert SpareNoFrameworkFG791 not in result.eliminations

        # Premise: the spare name is the one suggestion this request ranks, so only this rule can drop it.
        droppable = {DISABLED_PAIR_FEATURE_791, *result.facts.eliminated_hints}
        survivors = [name for name in dict.fromkeys(result.facts.known_names) if name not in droppable]
        ranked = get_close_matches(DISABLED_PAIR_FEATURE_791, survivors, n=MAX_RENDERED_SUGGESTIONS_791, cutoff=0.5)
        assert ranked == [DISABLED_PAIR_SPARE_791]

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert message == (
            f"No feature groups found for feature name: '{DISABLED_PAIR_FEATURE_791}'.\n"
            f"Feature group(s) eliminated while matching '{DISABLED_PAIR_FEATURE_791}':\n"
            "  - RendererDisabledPairFG791 (compute framework): "
            "none of its compute frameworks are enabled for this run\n"
            "Use resolve_feature(name, options=...) to debug feature resolution.\n"
            f"{TROUBLESHOOTING_LINE}"
        )
        assert DISABLED_PAIR_SPARE_791 in result.facts.dead_only_names

        # The other direction of the ping-pong: following the spare name may not point back at the first one.
        reverse = disabled_pair_reverse_scenario()
        reverse_feature, _ = reverse
        reverse_result = _evaluate(reverse)

        reverse_message = render_resolution_failure(reverse_result, reverse_feature)
        assert reverse_message is not None
        assert _suggestions(reverse_message) is None
        assert DISABLED_PAIR_FEATURE_791 in reverse_result.facts.dead_only_names


class TestALivePrefixKeepsACoveredNameSuggestible:
    """The default matcher also owns names by class-name prefix, so a live group covers names it never declares.

    Subtracting only the exact names live groups declare misses that coverage and suppresses a name that resolves.
    """

    def test_only_the_name_no_live_prefix_covers_is_suppressed(self) -> None:
        """One dead group, two dead names: the live prefix keeps one suggestible, the other stays suppressed."""
        scenario = live_prefix_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        assert result.eliminations[RendererDeadPrefixFG791].stage == "frameworks_not_enabled"
        assert RendererLivePrefixFG791 not in result.eliminations
        # Premise: the live prefix covers one of the two dead names, and no live prefix covers the other.
        assert LIVE_PREFIX_COVERED_791.startswith(RendererLivePrefixFG791.prefix())
        assert not DEAD_PREFIX_UNCOVERED_791.startswith(RendererLivePrefixFG791.prefix())

        # Coverage is resolution, not just text: requesting the covered name identifies that one live group.
        success = _evaluate(live_prefix_success_scenario())
        assert success.failure_kind is None
        assert set(success.identified) == {RendererLivePrefixFG791}

        message = render_resolution_failure(result, feature)
        assert message is not None
        suggestions = _suggestions(message)

        assert suggestions is not None
        assert set(suggestions) == {LIVE_PREFIX_COVERED_791, "RendererLivePrefixFG791", "RendererLivePrefixFG791_"}
        assert LIVE_PREFIX_COVERED_791 not in result.facts.dead_only_names
        assert DEAD_PREFIX_UNCOVERED_791 in result.facts.dead_only_names


class TestANameBlindGateKillsEveryNameItsCandidateDeclares:
    """domain and scope are name-blind, so a candidate that loses at either resolves NO name it declares.

    An elimination is recorded only for a candidate that first matched the requested name, so a wrong-domain or
    out-of-scope group that never matched carries no record at all. Reading deadness off the record alone counts
    such a group as a live declarer, and the suggestion then hands back a name that fails at the very same gate.
    """

    def test_a_wrong_domain_declarers_sibling_name_is_never_suggested(self) -> None:
        """The declarer never matched the requested name, so only the gate itself can say it is dead."""
        scenario = domain_gate_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        # Premise: nothing matched, so this candidate carries no elimination record to judge it by.
        assert result.failure_kind == "none"
        assert result.eliminations == {}
        # Premise: the sibling is the one name this request ranks, so only this rule can drop it.
        survivors = [name for name in dict.fromkeys(result.facts.known_names) if name != DOMAIN_GATE_FEATURE_791]
        ranked = get_close_matches(DOMAIN_GATE_FEATURE_791, survivors, n=MAX_RENDERED_SUGGESTIONS_791, cutoff=0.5)
        assert ranked == [DOMAIN_GATE_SIBLING_791]

        # Suppression is resolution, not just text: requesting the sibling fails at the same gate.
        sibling = _evaluate(domain_gate_sibling_scenario())
        assert sibling.failure_kind == "none"
        assert sibling.eliminations[CrossDomainDeclarerFG791].stage == "domain"

        assert DOMAIN_GATE_SIBLING_791 in result.facts.dead_only_names

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert message == (
            f"No feature groups found for feature name: '{DOMAIN_GATE_FEATURE_791}'. "
            f"Requested domain: '{REQUESTED_DOMAIN_791}'.\n"
            "Use resolve_feature(name, options=...) to debug feature resolution.\n"
            f"{TROUBLESHOOTING_LINE}"
        )

    def test_an_out_of_scope_declarers_sibling_name_is_never_suggested(self) -> None:
        """Same shape at the scope gate: the out-of-scope name goes, the in-scope group's own name stays."""
        scenario = scope_gate_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        # Premise: nothing matched, so this candidate carries no elimination record to judge it by.
        assert result.failure_kind == "none"
        assert result.eliminations == {}
        # Premise: the out-of-scope sibling outranks the in-scope group's own name, so only this rule can drop it.
        survivors = [name for name in dict.fromkeys(result.facts.known_names) if name != SCOPE_GATE_FEATURE_791]
        ranked = get_close_matches(SCOPE_GATE_FEATURE_791, survivors, n=MAX_RENDERED_SUGGESTIONS_791, cutoff=0.5)
        assert ranked == [SCOPE_GATE_SIBLING_791, KNOWN_FEATURE_791]

        # Suppression is resolution, not just text: requesting the sibling fails at the same gate.
        sibling = _evaluate(scope_gate_sibling_scenario())
        assert sibling.failure_kind == "none"
        assert sibling.eliminations[OutsideScopeDeclarerFG791].stage == "scope"

        assert SCOPE_GATE_SIBLING_791 in result.facts.dead_only_names
        # The in-scope group is untouched by the gate, so its own name stays suggestible.
        assert KNOWN_FEATURE_791 not in result.facts.dead_only_names

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert message == (
            f"No feature groups found for feature name: '{SCOPE_GATE_FEATURE_791}'. "
            f"Scoped to feature group: '{GATE_SCOPE_791}'.\n"
            f"Did you mean one of: ['{KNOWN_FEATURE_791}']?\n"
            "Use resolve_feature(name, options=..., feature_group=...) to debug feature resolution.\n"
            f"{TROUBLESHOOTING_LINE}"
        )

    def test_a_name_dependent_record_does_not_revive_a_wrong_domain_candidate(self) -> None:
        """The record says value_rejection, but the domain gate kills every name the candidate declares."""
        scenario = value_rejection_cross_domain_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        # Premise: the recorded stage is name-DEPENDENT, so the record alone keeps this candidate live.
        assert result.eliminations == {
            ValueRejectingCrossDomainFG791: Elimination(
                stage="value_rejection", reason=VALUE_DOMAIN_REJECTION_REASON_791
            )
        }
        assert "value_rejection" in NAME_DEPENDENT_STAGES_791
        # Premise: the spare is the one name this request ranks, so only this rule can drop it.
        droppable = {VALUE_DOMAIN_FEATURE_791, *result.facts.eliminated_hints}
        survivors = [name for name in dict.fromkeys(result.facts.known_names) if name not in droppable]
        ranked = get_close_matches(VALUE_DOMAIN_FEATURE_791, survivors, n=MAX_RENDERED_SUGGESTIONS_791, cutoff=0.5)
        assert ranked == [VALUE_DOMAIN_SPARE_791]

        assert VALUE_DOMAIN_SPARE_791 in result.facts.dead_only_names

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert message == (
            f"No feature groups found for feature name: '{VALUE_DOMAIN_FEATURE_791}'. "
            f"Requested domain: '{REQUESTED_DOMAIN_791}'.\n"
            f"Feature group(s) eliminated while matching '{VALUE_DOMAIN_FEATURE_791}':\n"
            f"  - ValueRejectingCrossDomainFG791 (option value): {VALUE_DOMAIN_REJECTION_REASON_791}\n"
            "Use resolve_feature(name, options=...) to debug feature resolution.\n"
            f"{TROUBLESHOOTING_LINE}"
        )


class TestADegradedDomainReadNeverDecidesAgainstACandidate:
    """A domain that cannot be read is not a wrong domain: the candidate stays live and the degrade is reported."""

    def test_a_raising_get_domain_keeps_the_candidates_names_suggestible(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """evaluate() still returns, the message still renders, and the unreadable candidate keeps its names."""
        scenario = degraded_domain_scenario()
        feature, accessible_plugins = scenario
        (unreadable,) = accessible_plugins

        with caplog.at_level(logging.WARNING, logger=IDENTIFY_LOGGER_791):
            result = _evaluate(scenario)

        # Premise: the request carries a domain, and the candidate never matched, so no decision gate judged it.
        assert feature.domain is not None
        assert result.failure_kind == "none"
        assert result.eliminations == {}

        assert DEGRADED_DOMAIN_SPARE_791 not in result.facts.dead_only_names

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert _suggestions(message) == [DEGRADED_DOMAIN_SPARE_791]

        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING and record.name == IDENTIFY_LOGGER_791
        ]
        degraded_field = f"{unreadable.get_class_name()}.get_domain"
        assert [warning for warning in warnings if degraded_field in warning], (
            f"Expected a WARNING naming '{degraded_field}', got {warnings}"
        )


class TestTheNameBlindGateCaptureCostsNoExtraHookCall:
    """The capture reuses the decision pass's memoized domain outcome, and asks for nothing it cannot use."""

    def test_a_domainless_request_never_reads_a_non_matching_candidates_domain(self) -> None:
        """Without a requested domain the gate cannot fire, so the hook must not be called at all."""
        scenario = domainless_gate_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        # Premise: no domain to compare against, and the candidate never matched, so no gate can judge it.
        assert feature.domain is None
        assert result.failure_kind == "none"
        assert result.eliminations == {}
        # Premise: this candidate's counters are wired, so the assertion below is about the hook, not the key.
        assert HOOK_CALLS["CrossDomainDeclarerFG791.match_feature_group_criteria"] == 1

        assert "CrossDomainDeclarerFG791.get_domain" not in HOOK_CALLS
        # The gate cannot fire, so the candidate stays live and keeps every name it declares suggestible.
        assert result.facts.dead_only_names == frozenset()

    def test_each_candidates_domain_is_read_at_most_once_per_evaluation(self) -> None:
        """One decision-pass read plus one capture read of the same candidate is one hook call, not two."""
        scenario = domain_hook_cost_scenario()
        _, accessible_plugins = scenario
        result = _evaluate(scenario)

        # Premise: one candidate reached the decision-side domain gate, the other never matched at all.
        assert result.eliminations[CrossDomainDeclarerFG791].stage == "domain"
        assert ValueRejectingCrossDomainFG791 not in result.eliminations
        # Premise: the capture really does need the second candidate's domain, so the bound is not vacuous.
        assert VALUE_DOMAIN_SPARE_791 in result.facts.dead_only_names

        counts = {
            candidate.get_class_name(): HOOK_CALLS.get(f"{candidate.get_class_name()}.get_domain", 0)
            for candidate in accessible_plugins
        }
        # Both counts are pinned EXACTLY: an upper bound also holds when a hook is never called at all, so an
        # edit that stopped asking about the non-matching candidate entirely would slip through it.
        assert counts["CrossDomainDeclarerFG791"] == 1, counts
        assert counts["ValueRejectingCrossDomainFG791"] == 1, counts


class TestADomainCarryingFailureNamesTheRequestedDomain:
    """A request that carries a domain says so in its failure message, exactly as a scoped request does.

    The domain is a gate the user set and the reason nothing resolved, so a message that never mentions it
    describes a run the user did not ask for. The callout sits where ``scope_callout``'s does: on the sentence
    line, before any near-miss block. The pointer line is unchanged, because ``resolve_feature`` takes the
    domain on the Feature and has no domain keyword to name.
    """

    def test_a_none_failure_states_the_requested_domain(self) -> None:
        """The whole none message, with the domain the request carried on its sentence line."""
        scenario = domain_gate_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        # Premise: the request carries a domain, and that domain is why nothing resolved.
        assert feature.domain is not None
        assert feature.domain.name == REQUESTED_DOMAIN_791
        assert result.failure_kind == "none"

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert message.split("\n")[0].endswith(f"Requested domain: '{REQUESTED_DOMAIN_791}'.")
        assert message == (
            f"No feature groups found for feature name: '{DOMAIN_GATE_FEATURE_791}'. "
            f"Requested domain: '{REQUESTED_DOMAIN_791}'.\n"
            "Use resolve_feature(name, options=...) to debug feature resolution.\n"
            f"{TROUBLESHOOTING_LINE}"
        )

    def test_an_abstract_only_failure_states_the_requested_domain(self) -> None:
        """Same callout in the same place on the other message that carries one."""
        scenario = abstract_domain_gate_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        # Premise: the abstract base declares the requested domain, so the request reaches the abstract message.
        assert feature.domain is not None
        assert result.failure_kind == "abstract_only"

        message = render_resolution_failure(result, feature)
        assert message == (
            f"No feature groups found for feature name: '{ABSTRACT_DECLARER_NAME_791}'. "
            "Only abstract feature group base(s) matched, which cannot be instantiated; "
            "no concrete implementation is available or enabled. "
            f"Requested domain: '{REQUESTED_DOMAIN_791}'."
        )


class TestAnAbstractBaseIsNeverALiveDeclarer:
    """An accessible abstract base can be matched but never identified, so it keeps no name suggestible.

    ``_filter_loop`` parks an abstract candidate in ``abstract_matched`` and never in the identified mapping,
    whatever name it is asked about: exactly the predicate the dead-name capture computes. Counting it as a
    live declarer cancels a dead sibling's suppression, and the suggestion then hands back a name whose own
    message only says the base cannot be instantiated.
    """

    def test_an_abstract_declarer_does_not_keep_a_dead_siblings_name_suggestible(self) -> None:
        """The base is accessible and enabled, so only its abstractness can say it declares nothing live."""
        scenario = abstract_declarer_typo_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        # Premise: the base is abstract, and nothing else about this run kills it.
        assert inspect.isabstract(SpareAbstractBaseFG791)
        assert feature.domain is None
        assert feature.feature_group_scope is None
        # Premise: neither candidate matched the typo, so no elimination record judges either of them.
        assert result.failure_kind == "none"
        assert result.eliminations == {}
        # Premise: the shared name is the one name this request ranks, so only this rule can drop it.
        survivors = [name for name in dict.fromkeys(result.facts.known_names) if name != ABSTRACT_DECLARER_TYPO_791]
        ranked = get_close_matches(ABSTRACT_DECLARER_TYPO_791, survivors, n=MAX_RENDERED_SUGGESTIONS_791, cutoff=0.5)
        assert ranked == [ABSTRACT_DECLARER_NAME_791]

        # Suppression is resolution, not just text: following the suggestion renders the abstract-only message.
        sibling_feature, sibling_plugins = abstract_declarer_name_scenario()
        sibling = _evaluate((sibling_feature, sibling_plugins))
        assert sibling.failure_kind == "abstract_only"
        assert sibling.abstract_matched == {SpareAbstractBaseFG791}
        assert render_resolution_failure(sibling, sibling_feature) == (
            f"No feature groups found for feature name: '{ABSTRACT_DECLARER_NAME_791}'. "
            "Only abstract feature group base(s) matched, which cannot be instantiated; "
            "no concrete implementation is available or enabled.\n"
            f"Feature group(s) eliminated while matching '{ABSTRACT_DECLARER_NAME_791}':\n"
            "  - SpareDeadTwinFG791 (compute framework): none of its compute frameworks are enabled for this run"
        )

        assert ABSTRACT_DECLARER_NAME_791 in result.facts.dead_only_names

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert _suggestions(message) is None


class TestADeadCandidatesClassNameAndPrefixAreSuppressedToo:
    """A dead candidate owns two more names than it declares: its class name and its class-name prefix.

    The live branch of the difference contributes the whole catalog (class name, declared names, prefix) while
    the dead branch contributes only the declared names, and a candidate that never matched carries no
    eliminated_hints entry either. So nothing suppresses the two names the default matcher owns by class
    identity, and both are handed back although both are eliminated at the very same gate.
    """

    def test_a_dead_candidates_class_name_and_prefix_are_never_suggested(self) -> None:
        """The wrong-domain group declares no name at all, so only its class identity can be suggested."""
        scenario = dead_class_name_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        # Premise: the two constants really are this candidate's class-identity names.
        assert RendererCrossDomainNameFG791.get_class_name() == DEAD_CLASS_NAME_791
        assert RendererCrossDomainNameFG791.prefix() == DEAD_CLASS_PREFIX_791
        # Premise: nothing matched the typo, so no record and no eliminated hint can reach either name.
        assert result.failure_kind == "none"
        assert result.eliminations == {}
        assert result.facts.eliminated_hints == frozenset()
        # Premise: both names are what this request ranks, so only this rule can drop them.
        survivors = [name for name in dict.fromkeys(result.facts.known_names) if name != DEAD_CLASS_NAME_TYPO_791]
        ranked = get_close_matches(DEAD_CLASS_NAME_TYPO_791, survivors, n=MAX_RENDERED_SUGGESTIONS_791, cutoff=0.5)
        assert ranked == [DEAD_CLASS_NAME_791, DEAD_CLASS_PREFIX_791]

        # Suppression is resolution, not just text: both names really are eliminated at the same gate.
        for echo_name in (DEAD_CLASS_NAME_791, DEAD_CLASS_PREFIX_791):
            echo = _evaluate(dead_class_name_echo_scenario(echo_name))
            assert echo.failure_kind == "none"
            assert echo.eliminations[RendererCrossDomainNameFG791].stage == "domain"

        assert DEAD_CLASS_NAME_791 in result.facts.dead_only_names
        assert DEAD_CLASS_PREFIX_791 in result.facts.dead_only_names

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert _suggestions(message) is None

    def test_a_live_declarer_of_the_same_two_names_keeps_them_suggestible(self) -> None:
        """Still a difference, not a per-candidate drop: one live declarer keeps both names worth suggesting."""
        scenario = dead_class_name_with_live_declarer_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        # Premise: the live group matched nothing here, so nothing ever eliminates it.
        assert result.failure_kind == "none"
        assert RendererLiveNameDeclarerFG791 not in result.eliminations
        assert RendererLiveNameDeclarerFG791.get_domain() == Domain(REQUESTED_DOMAIN_791)

        assert result.facts.dead_only_names == frozenset()

        message = render_resolution_failure(result, feature)
        assert message is not None
        suggestions = _suggestions(message)

        assert suggestions is not None
        assert {DEAD_CLASS_NAME_791, DEAD_CLASS_PREFIX_791} <= set(suggestions)


class TestAMalformedDomainReturnIsDecidedLikeTheGateDecidesIt:
    """The decision gate is ``domain == feature.domain``, so a non-Domain return FAILS it for every name.

    A raise is the other case: there the gate raises rather than deciding, so nothing is known and the
    candidate stays live (TestADegradedDomainReadNeverDecidesAgainstACandidate). Reading both through one
    best-effort name that maps them onto the same None makes the capture disagree with the gate it retests:
    the same malformed read eliminates a name-matching candidate at 'domain' and keeps a non-matching one alive.
    """

    def test_a_malformed_get_domain_return_kills_the_names_it_declares(self) -> None:
        """The candidate never matched, so only the retest decides, and the gate would have decided against it."""
        scenario = malformed_domain_scenario()
        feature, accessible_plugins = scenario
        (malformed,) = accessible_plugins
        result = _evaluate(scenario)

        # Premise: the request carries a domain, and the candidate never matched, so no gate recorded it.
        assert feature.domain is not None
        assert result.failure_kind == "none"
        assert result.eliminations == {}
        # Premise: the return really is not a Domain, so the gate compares it and decides against it.
        assert not isinstance(malformed.get_domain(), Domain)
        # Premise: the spare is the one name this request ranks, so only this rule can drop it.
        survivors = [name for name in dict.fromkeys(result.facts.known_names) if name != MALFORMED_DOMAIN_FEATURE_791]
        ranked = get_close_matches(MALFORMED_DOMAIN_FEATURE_791, survivors, n=MAX_RENDERED_SUGGESTIONS_791, cutoff=0.5)
        assert ranked == [MALFORMED_DOMAIN_SPARE_791]

        # Suppression is resolution, not just text: requesting the spare is eliminated at the domain gate.
        sibling_feature, sibling_plugins = malformed_domain_sibling_scenario()
        (sibling_group,) = sibling_plugins
        sibling = _evaluate((sibling_feature, sibling_plugins))
        assert sibling.failure_kind == "none"
        assert sibling.eliminations[sibling_group].stage == "domain"

        assert MALFORMED_DOMAIN_SPARE_791 in result.facts.dead_only_names

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert _suggestions(message) is None


class TestEveryEliminationStageIsClassified:
    """A tenth stage must be classified and labelled before it ships, or it silently misrenders or misdrops names."""

    def test_the_two_stage_sets_partition_the_stage_literal(self) -> None:
        """NAME_INDEPENDENT_STAGES and its name-dependent complement cover EliminationStage exactly once."""
        name_independent = resolution_types.NAME_INDEPENDENT_STAGES
        stages = frozenset(get_args(EliminationStage))

        assert name_independent.isdisjoint(NAME_DEPENDENT_STAGES_791)
        assert name_independent | NAME_DEPENDENT_STAGES_791 == stages

    def test_every_stage_carries_a_near_miss_label(self) -> None:
        """dict[EliminationStage, str] does not force an entry per member, so the table is pinned complete here."""
        assert set(_STAGE_LABELS) == set(get_args(EliminationStage))


class TestAnUnlabeledStageStillRenders:
    """Rendering is a best-effort projection: every other field degrades via safe_field, so a label must too."""

    def test_a_stage_without_a_label_falls_back_to_its_raw_token(self) -> None:
        """The near-miss line names the candidate, the raw stage token and the reason; the rest still renders."""
        feature = Feature(UNLABELED_STAGE_FEATURE_791)
        result = EvaluationResult(
            identified={},
            eliminations={
                RendererSuccessFG791: Elimination(
                    stage=cast(EliminationStage, UNLABELED_STAGE_791),
                    reason=UNLABELED_STAGE_REASON_791,
                )
            },
        )

        # Premise: this failure reaches the near-miss block, and no label covers the stage it carries.
        assert result.failure_kind == "none"
        assert UNLABELED_STAGE_791 not in _STAGE_LABELS

        message = render_resolution_failure(result, feature)

        assert message == (
            f"No feature groups found for feature name: '{UNLABELED_STAGE_FEATURE_791}'.\n"
            f"Feature group(s) eliminated while matching '{UNLABELED_STAGE_FEATURE_791}':\n"
            f"  - RendererSuccessFG791 ({UNLABELED_STAGE_791}): {UNLABELED_STAGE_REASON_791}\n"
            "Use resolve_feature(name, options=...) to debug feature resolution.\n"
            f"{TROUBLESHOOTING_LINE}"
        )

    def test_a_stage_without_a_label_warns_and_still_renders(self, caplog: pytest.LogCaptureFixture) -> None:
        """A missing label is a build defect, not an expected degrade, so the renderer warns while degrading."""
        feature = Feature(UNLABELED_STAGE_FEATURE_791)
        result = EvaluationResult(
            identified={},
            eliminations={
                RendererSuccessFG791: Elimination(
                    stage=cast(EliminationStage, UNLABELED_STAGE_791),
                    reason=UNLABELED_STAGE_REASON_791,
                )
            },
        )

        # Premise: this failure reaches the near-miss block, and no label covers the stage it carries.
        assert result.failure_kind == "none"
        assert UNLABELED_STAGE_791 not in _STAGE_LABELS

        with caplog.at_level(logging.WARNING, logger=RENDERER_LOGGER_791):
            message = render_resolution_failure(result, feature)

        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.levelno == logging.WARNING and record.name == RENDERER_LOGGER_791
        ]
        assert len(warnings) == 1, f"Expected exactly one WARNING from the renderer, got {warnings}"
        assert UNLABELED_STAGE_791 in warnings[0], "The warning must name the stage token that carries no label"

        # The warning reports the degrade, it never replaces it: the whole message still renders unchanged.
        assert message == (
            f"No feature groups found for feature name: '{UNLABELED_STAGE_FEATURE_791}'.\n"
            f"Feature group(s) eliminated while matching '{UNLABELED_STAGE_FEATURE_791}':\n"
            f"  - RendererSuccessFG791 ({UNLABELED_STAGE_791}): {UNLABELED_STAGE_REASON_791}\n"
            "Use resolve_feature(name, options=...) to debug feature resolution.\n"
            f"{TROUBLESHOOTING_LINE}"
        )
