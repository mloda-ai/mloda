"""Pinning tests for the pure failure renderer on the evaluation seam (issue #791).

``IdentifyFeatureGroupClass.evaluate(...)`` captures every fact a resolution-failure message needs
during its single first pass: per-candidate compute-framework capability
(``EvaluationResult.candidate_frameworks``) plus the remaining facts (``EvaluationResult.facts``).
``render_resolution_failure(result, feature)`` is then a PURE projection of that result: it reads
only the result and the ``Feature``, never re-runs matching and never calls a provider-overridable
hook. The feature groups below count every such hook, so a renderer that calls one is caught.

All names are suffixed ``_791`` because test feature groups become global subclasses.
"""

from abc import abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, ClassVar, Optional, cast

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import property_spec
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import (
    CandidateFrameworks,
    EvaluationResult,
    IdentifyFeatureGroupClass,
    RenderFacts,
    render_resolution_failure,
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

HEALTHY_DOMAIN_791 = "renderer_healthy_domain_791"
BOOM_SUPPORTED_NAME_791 = "renderer_boom_supported_name_791"

# Same-named tie candidates get an explicit __module__ so only the module can break the sort tie.
TIE_MODULE_A_791 = "tests.renderer_tie_module_a_791"
TIE_MODULE_B_791 = "tests.renderer_tie_module_b_791"

TROUBLESHOOTING_LINE = (
    "For troubleshooting guide, see: "
    "https://mloda-ai.github.io/mloda/in_depth/troubleshooting/feature-group-resolution-errors/"
)

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
    """Base for the groups whose fact-capture hook raises. Its subclasses are ALWAYS built inside a function.

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
    "raising_domain_multiple": raising_domain_multiple_scenario,
    "raising_prefix_none": raising_prefix_none_scenario,
    "raising_names_none": raising_names_none_scenario,
    "raising_rejection_none": raising_rejection_none_scenario,
    "raising_framework_rule_abstract": raising_framework_rule_abstract_scenario,
}

# Every (candidate, framework) pair the capability hook may be asked about during one evaluate(), and how
# often: exactly once. The decision loop splits the ENABLED frameworks, capture only adds the pairs it did
# not cover, so the union is the candidate's declared+available frameworks and nothing is asked twice.
CAPABILITY_PAIR_EXPECTATIONS: dict[str, dict[tuple[str, str], int]] = {
    "capability": {
        ("RendererCapabilityAFG791", "RendererFwOne791"): 1,
        ("RendererCapabilityAFG791", "RendererFwTwo791"): 1,
        ("RendererCapabilityBFG791", "RendererFwOne791"): 1,
        ("RendererCapabilityBFG791", "RendererFwTwo791"): 1,
    },
    "capability_narrow_enabled": {
        ("RendererNarrowEnabledFG791", "RendererFwOne791"): 1,
        ("RendererNarrowEnabledFG791", "RendererFwTwo791"): 1,
    },
    "capability_none_enabled": {
        ("RendererNoneEnabledFG791", "RendererFwOne791"): 1,
        ("RendererNoneEnabledFG791", "RendererFwTwo791"): 1,
    },
}


def _evaluate(scenario: Scenario) -> EvaluationResult:
    feature, accessible_plugins = scenario
    return IdentifyFeatureGroupClass.evaluate(feature=feature, accessible_plugins=accessible_plugins, links=None)


def _render(scenario: Scenario) -> str:
    feature, _ = scenario
    message = render_resolution_failure(_evaluate(scenario), feature)
    assert message is not None
    return message


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
        """Each rejecting candidate names its OWN rejected and supported frameworks."""
        message = _render(capability_scenario())

        assert message == (
            f"Unsupported compute framework(s) for feature '{CAPABILITY_FEATURE_791}':\n"
            "  - RendererCapabilityAFG791: ['RendererFwTwo791']. Supported on: ['RendererFwOne791'].\n"
            "  - RendererCapabilityBFG791: ['RendererFwOne791']. Supported on: ['RendererFwTwo791'].\n"
            "Pin the feature to a supported compute framework or override supports_compute_framework."
        )

    def test_abstract_only_names_concrete_implementation_frameworks(self) -> None:
        """Abstract-only with accessible concrete subclasses names the frameworks they require."""
        message = _render(abstract_with_frameworks_scenario())

        assert message == (
            f"No feature groups found for feature name: '{ABSTRACT_FEATURE_791}'. "
            "Its concrete implementations require compute framework(s) "
            "['RendererFwOne791', 'RendererFwTwo791'], "
            "none of which are available or enabled for this run."
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
        """Value rejections, then 'Did you mean', then the resolve_feature and troubleshooting lines."""
        message = _render(ordinary_none_scenario())

        lines = message.split("\n")
        assert lines[0] == f"No feature groups found for feature name: '{TYPO_FEATURE_791}'."
        assert lines[1] == f"Feature group(s) rejected the supplied options while matching '{TYPO_FEATURE_791}':"
        assert lines[2] == f"  - RendererStrictFG791: {WINDOW_REJECTION_REASON}"
        assert lines[3].startswith("Did you mean one of: [")
        assert f"'{KNOWN_FEATURE_791}'" in lines[3]
        assert lines[4] == "Use resolve_feature(name, options=...) to debug feature resolution."
        assert lines[5] == TROUBLESHOOTING_LINE
        assert len(lines) == 6

    def test_missing_option_rejection_renders_under_the_options_heading(self) -> None:
        """A MISSING required option is not a wrong value, so the heading must cover both rejection kinds."""
        scenario = missing_option_scenario()
        feature, _ = scenario
        result = _evaluate(scenario)

        assert result.failure_kind == "none"
        assert result.facts.value_rejections == (("RendererMissingOptionFG791", MISSING_OPTION_REASON_791),)

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert f"Feature group(s) rejected the supplied options while matching '{MISSING_OPTION_FEATURE_791}':" in (
            message
        )
        assert f"  - RendererMissingOptionFG791: {MISSING_OPTION_REASON_791}" in message

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

    def test_capability_frameworks_are_never_unioned_across_candidates(self) -> None:
        """Mirrored candidates: no line may claim a framework is both supported and unsupported."""
        message = _render(capability_scenario())

        a_line = next(line for line in message.split("\n") if "RendererCapabilityAFG791" in line)
        b_line = next(line for line in message.split("\n") if "RendererCapabilityBFG791" in line)

        assert a_line == "  - RendererCapabilityAFG791: ['RendererFwTwo791']. Supported on: ['RendererFwOne791']."
        assert b_line == "  - RendererCapabilityBFG791: ['RendererFwOne791']. Supported on: ['RendererFwTwo791']."
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

    def test_value_rejections_and_known_names_captured_for_ordinary_none(self) -> None:
        """The ordinary-none kind captures the value rejections and the name catalog."""
        result = _evaluate(ordinary_none_scenario())

        assert result.failure_kind == "none"
        assert result.facts.value_rejections == (("RendererStrictFG791", WINDOW_REJECTION_REASON),)
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
        assert result.facts.value_rejections == (("RendererStrictFG791", WINDOW_REJECTION_REASON),)

        message = render_resolution_failure(result, feature)
        assert message is not None
        assert f"  - RendererStrictFG791: {WINDOW_REJECTION_REASON}" in message
        assert "RendererRaisingRejectionFG791:" not in message

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
            "no concrete implementation is available or enabled."
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
    """The capability message keeps today's universe: the candidate's DECLARED frameworks that are available.

    ``candidate_frameworks`` is the decision fact and stays the run's own (narrower) split of the
    frameworks that were enabled. Rendering must not inherit that narrowing, or a candidate loses the
    'Supported on' clause it has today, and an all-disabled candidate flips to the ordinary-none message.
    """

    def test_not_enabled_framework_still_renders_as_supported(self) -> None:
        """Shape A: the supported framework is available but not enabled, and must still be named."""
        scenario = capability_narrow_enabled_scenario()
        feature, _ = scenario

        result = _evaluate(scenario)

        # The decision fact stays the run's own split: only RendererFwOne791 was enabled, and it was rejected.
        assert result.candidate_frameworks == {
            RendererNarrowEnabledFG791: CandidateFrameworks(
                supported=frozenset(), rejected=frozenset({RendererFwOne791})
            )
        }
        assert render_resolution_failure(result, feature) == (
            f"Unsupported compute framework(s) for feature '{NARROW_ENABLED_FEATURE_791}':\n"
            "  - RendererNarrowEnabledFG791: ['RendererFwOne791']. Supported on: ['RendererFwTwo791'].\n"
            "Pin the feature to a supported compute framework or override supports_compute_framework."
        )

    def test_no_enabled_framework_still_renders_the_capability_message(self) -> None:
        """Shape B: an empty accessible set must not flip the failure to the ordinary-none message."""
        scenario = capability_none_enabled_scenario()
        feature, _ = scenario

        result = _evaluate(scenario)

        # Nothing was enabled, so the decision loop split nothing: the decision fact is empty on both sides.
        assert result.candidate_frameworks == {
            RendererNoneEnabledFG791: CandidateFrameworks(supported=frozenset(), rejected=frozenset())
        }
        message = render_resolution_failure(result, feature)

        assert message == (
            f"Unsupported compute framework(s) for feature '{NONE_ENABLED_FEATURE_791}':\n"
            "  - RendererNoneEnabledFG791: ['RendererFwOne791']. Supported on: ['RendererFwTwo791'].\n"
            "Pin the feature to a supported compute framework or override supports_compute_framework."
        )
        assert "No feature groups found" not in message

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
        """Same-named capability candidates render module-sorted, whichever way they were inserted."""
        group_a, group_b = _build_tie_capability_groups()
        feature = Feature(TIE_CAPABILITY_FEATURE_791, compute_framework="RendererFwThree791")
        both = {RendererFwOne791, RendererFwTwo791}
        a_first: FeatureGroupEnvironmentMapping = {group_a: set(both), group_b: set(both)}
        b_first: FeatureGroupEnvironmentMapping = {group_b: set(both), group_a: set(both)}

        expected = (
            f"Unsupported compute framework(s) for feature '{TIE_CAPABILITY_FEATURE_791}':\n"
            "  - RendererTieFG791: ['RendererFwTwo791']. Supported on: ['RendererFwOne791'].\n"
            "  - RendererTieFG791: ['RendererFwOne791']. Supported on: ['RendererFwTwo791'].\n"
            "Pin the feature to a supported compute framework or override supports_compute_framework."
        )

        assert _render((feature, a_first)) == expected
        assert _render((feature, b_first)) == expected
