"""Single-pass, exactly-once resolution failure paths (issue #782).

Pinned contract:

* ``IdentifyFeatureGroupClass.__init__`` raises ``render_resolution_failure(result, feature)`` built from
  the SAME ``EvaluationResult`` its single ``evaluate()`` pass produced. The legacy raising path
  (``validate`` and the ``_build_*``/``_*_message``/``_*_hint`` builders) and the module function
  ``split_frameworks_by_capability`` are gone.
* ``resolve_feature`` renders its ``error`` from the ``EvaluationResult`` it already computed.
  ``_engine_failure_message``, which constructed ``IdentifyFeatureGroupClass`` a second time purely to
  scrape its ``ValueError`` text, is gone.
* Both consumers therefore agree, string for string, on every failure kind.
* During ONE resolution attempt each decision-relevant provider hook runs at most once at its natural
  granularity: per candidate, or per (candidate, framework) / (candidate, index) pair.
* Rendering is a pure projection: it calls no provider hook and mutates neither the ``Feature`` nor its
  ``Options``. Reader-style matching, which can enrich ``Options`` while selecting data access, is the
  hazard this closes.

All names carry a ``_782`` suffix: test feature groups become global subclasses and the suite runs in
parallel, so a shared name would leak into another module's candidate universe.
"""

from abc import abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
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
from mloda.core.abstract_plugins.components.link import JoinSpec, Link
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api import plugin_docs
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.prepare import identify_feature_group
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping, PreFilterPlugins
from mloda.core.prepare.identify_feature_group import (
    IdentifyFeatureGroupClass,
    render_resolution_failure,
)
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise


MULTIPLE_FEATURE_782 = "single_pass_multiple_782"
DOMAIN_MULTIPLE_FEATURE_782 = "single_pass_domain_multiple_782"
ABSTRACT_FEATURE_782 = "single_pass_abstract_782"
CAPABILITY_FEATURE_782 = "single_pass_capability_782"
BARE_ONLY_FEATURE_782 = "single_pass_bare_only_782"
STRICT_FEATURE_782 = "single_pass_strict_782"
READER_FEATURE_782 = "single_pass_reader_782"
LINK_FEATURE_782 = "single_pass_link_782"
STATEFUL_MATCH_FEATURE_782 = "single_pass_stateful_match_782"
STATEFUL_SUPPORT_FEATURE_782 = "single_pass_stateful_support_782"
BAD_DOMAIN_FEATURE_782 = "single_pass_bad_domain_782"
BAD_DECLARATION_FEATURE_782 = "single_pass_bad_declaration_782"
BAD_CATALOG_FEATURE_782 = "single_pass_bad_catalog_typo_782"
TIE_REJECT_FEATURE_782 = "single_pass_tie_reject_782"

PROBE_DOMAIN_782 = "single_pass_domain_782"
HEALTHY_DOMAIN_782 = "single_pass_healthy_domain_782"

# Values a provider returns in violation of its own annotation.
BAD_NAME_VALUE_782 = 123
BAD_PREFIX_VALUE_782 = 456
BAD_REASON_VALUE_782 = 999
HEALTHY_REJECTION_REASON_782 = "single pass healthy rejection reason 782"
STRICT_REJECTION_REASON_782 = "Property value '14' failed validation for 'window_size_782'"

# Same-named tie candidates get an explicit __module__ so only the module can break the sort tie.
TIE_MODULE_A_782 = "tests.single_pass_tie_module_a_782"
TIE_MODULE_B_782 = "tests.single_pass_tie_module_b_782"

TROUBLESHOOTING_LINE_782 = (
    "For troubleshooting guide, see: "
    "https://mloda-ai.github.io/mloda/in_depth/troubleshooting/feature-group-resolution-errors/"
)

# Options key prefix a reader-style matcher stamps onto whatever Options it is handed, once per call.
OPTION_PROBE_PREFIX_782 = "single_pass_probe_782_"

LEFT_INDEX_782 = Index(("single_pass_left_782",))
RIGHT_INDEX_782 = Index(("single_pass_right_782",))

# Call counters for every provider-overridable hook, keyed "<ClassName>.<hook>". Reset per test.
HOOK_CALLS: dict[str, int] = {}

# supports_compute_framework calls keyed by the (candidate, framework) PAIR it was asked about.
PAIR_CALLS: dict[tuple[str, str], int] = {}

# supports_index calls keyed by the (candidate, index) PAIR it was asked about.
INDEX_CALLS: dict[tuple[str, str], int] = {}

# Per-class answer counters backing the deliberately stateful hooks. Reset per test.
STATEFUL_CALLS: dict[str, int] = {}


def _record(class_name: str, hook: str) -> None:
    """Count one call of a provider-overridable hook."""
    key = f"{class_name}.{hook}"
    HOOK_CALLS[key] = HOOK_CALLS.get(key, 0) + 1


def _record_pair(class_name: str, framework_name: str) -> None:
    """Count one capability-hook call for a single (candidate, framework) pair."""
    key = (class_name, framework_name)
    PAIR_CALLS[key] = PAIR_CALLS.get(key, 0) + 1


def _record_index(class_name: str, index: Index) -> None:
    """Count one supports_index call for a single (candidate, index) pair."""
    key = (class_name, str(index))
    INDEX_CALLS[key] = INDEX_CALLS.get(key, 0) + 1


def _next_stateful_answer(class_name: str) -> int:
    """Return how many times this class's stateful hook has been asked, counting from zero."""
    seen = STATEFUL_CALLS.get(class_name, 0)
    STATEFUL_CALLS[class_name] = seen + 1
    return seen


def _counts_for(hook: str) -> dict[str, int]:
    """Actual call count of one hook, keyed by the class it was asked of.

    Asserting against an EXACT expectation rather than an upper bound: "no counter exceeds one" also
    holds when a hook is never called at all, so an edit that stopped asking entirely would slip through.
    """
    suffix = f".{hook}"
    return {key[: -len(suffix)]: count for key, count in HOOK_CALLS.items() if key.endswith(suffix)}


def _once_each(class_names: frozenset[str]) -> dict[str, int]:
    """The expected counter shape: every named candidate asked exactly once."""
    return {name: 1 for name in sorted(class_names)}


def _malformed(value: Any) -> Any:
    """Return a deliberately ill-typed value.

    A provider's annotation is a promise, not a guarantee: hiding the value from the type checker is how
    a real plugin's runtime bug reaches mloda. The core must degrade the fact, not take the call down.
    """
    return value


class SinglePassFwOne_782(ComputeFramework):
    """First dummy compute framework for the single-pass tests."""


class SinglePassFwTwo_782(ComputeFramework):
    """Second dummy compute framework for the single-pass tests."""


class CountingSinglePassFG_782(FeatureGroup):
    """Feature group base counting every provider-overridable hook one resolution attempt may reach."""

    MATCHES: frozenset[str] = frozenset()
    DOMAIN_NAME: Optional[str] = None
    FRAMEWORK_RULE: Optional[set[type[ComputeFramework]]] = None
    SUPPORTED_FRAMEWORKS: Optional[frozenset[str]] = None
    SUPPORTED_NAMES: frozenset[str] = frozenset()

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
        _record_index(cls.get_class_name(), index)
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


class SinglePassMultipleAFG_782(CountingSinglePassFG_782):
    """First of two siblings matching the same name; both win, so the failure is 'multiple'."""

    MATCHES = frozenset({MULTIPLE_FEATURE_782, DOMAIN_MULTIPLE_FEATURE_782})
    DOMAIN_NAME = PROBE_DOMAIN_782
    FRAMEWORK_RULE = {SinglePassFwOne_782}


class SinglePassMultipleBFG_782(CountingSinglePassFG_782):
    """Second sibling matching the same name, sharing the domain so a domain-scoped request keeps both."""

    MATCHES = frozenset({MULTIPLE_FEATURE_782, DOMAIN_MULTIPLE_FEATURE_782})
    DOMAIN_NAME = PROBE_DOMAIN_782
    FRAMEWORK_RULE = {SinglePassFwOne_782}


class SinglePassAbstractBaseFG_782(CountingSinglePassFG_782):
    """Abstract base that matches the name but can never be instantiated."""

    MATCHES = frozenset({ABSTRACT_FEATURE_782})
    FRAMEWORK_RULE = {SinglePassFwOne_782}

    @classmethod
    @abstractmethod
    def _single_pass_abstract_hook_782(cls) -> str:
        """Abstract hook that keeps this base uninstantiable."""


class SinglePassConcreteSubFG_782(SinglePassAbstractBaseFG_782):
    """Concrete implementation that is ALSO criteria-matched, declaring only a framework the run disables.

    Both fact captures want its framework declaration: the capability capture (it is criteria-matched)
    and the abstract-only capture (it is an accessible concrete subclass of the matched base).
    """

    MATCHES = frozenset({ABSTRACT_FEATURE_782})
    FRAMEWORK_RULE = {SinglePassFwTwo_782}

    @classmethod
    def _single_pass_abstract_hook_782(cls) -> str:
        return "concrete"


class SinglePassCapabilityRejectFG_782(CountingSinglePassFG_782):
    """Declares both frameworks, supports only the one the run disables, so the enabled one is rejected."""

    MATCHES = frozenset({CAPABILITY_FEATURE_782})
    FRAMEWORK_RULE = {SinglePassFwOne_782, SinglePassFwTwo_782}
    SUPPORTED_FRAMEWORKS = frozenset({"SinglePassFwTwo_782"})


class SinglePassBareOnlyFG_782(CountingSinglePassFG_782):
    """Matches its bare name only: any group option makes it reject the feature."""

    MATCHES = frozenset({BARE_ONLY_FEATURE_782})
    FRAMEWORK_RULE = {SinglePassFwOne_782}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        matched = super().match_feature_group_criteria(feature_name, options, data_access_collection)
        return matched and not options.group


class SinglePassReaderFG_782(CountingSinglePassFG_782):
    """Reader-style matcher: consults the DataAccessCollection AND enriches the Options it is handed.

    The enrichment is the #782 hazard in miniature. Each call stamps a fresh probe key, so the number of
    probe keys left on ``feature.options`` is exactly the number of times the matcher was asked about it.
    """

    MATCHES = frozenset({READER_FEATURE_782})
    FRAMEWORK_RULE = {SinglePassFwOne_782}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        _record(cls.get_class_name(), "match_feature_group_criteria")
        options.add_to_context(f"{OPTION_PROBE_PREFIX_782}{len(_probe_keys(options))}", "seen")
        if data_access_collection is None or not data_access_collection.files:
            return False
        return str(feature_name) in cls.MATCHES and not options.group


class SinglePassLinkFG_782(CountingSinglePassFG_782):
    """Declares index columns and supports no index, so the link filter rules it out."""

    MATCHES = frozenset({LINK_FEATURE_782})
    FRAMEWORK_RULE = {SinglePassFwOne_782}

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        _record(cls.get_class_name(), "index_columns")
        return [LEFT_INDEX_782]

    @classmethod
    def supports_index(cls, index: Index) -> Optional[bool]:
        _record(cls.get_class_name(), "supports_index")
        _record_index(cls.get_class_name(), index)
        return False


class SinglePassStatefulMatchFG_782(CountingSinglePassFG_782):
    """Matches on its FIRST call and never again: a second evaluation would see a different universe."""

    MATCHES = frozenset({STATEFUL_MATCH_FEATURE_782})
    FRAMEWORK_RULE = {SinglePassFwOne_782, SinglePassFwTwo_782}
    SUPPORTED_FRAMEWORKS = frozenset({"SinglePassFwTwo_782"})

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        _record(cls.get_class_name(), "match_feature_group_criteria")
        if _next_stateful_answer(cls.get_class_name()) > 0:
            return False
        return str(feature_name) in cls.MATCHES


class SinglePassStatefulSupportFG_782(CountingSinglePassFG_782):
    """Rejects every framework on its FIRST answer and accepts them on every later one."""

    MATCHES = frozenset({STATEFUL_SUPPORT_FEATURE_782})
    FRAMEWORK_RULE = {SinglePassFwOne_782}

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        _record(cls.get_class_name(), "supports_compute_framework")
        _record_pair(cls.get_class_name(), compute_framework.get_class_name())
        return _next_stateful_answer(cls.get_class_name()) > 0


class NotAFramework_782:
    """Not a ComputeFramework at all. A malformed compute_framework_rule() can still name it."""


class MalformedValueGroup_782(CountingSinglePassFG_782):
    """Base for groups whose hook RETURNS a malformed value. Subclasses are ALWAYS built inside a function.

    ``ARMED`` is what makes that safe, mirroring the #791 raising-hook harness. A group built per test still
    outlives it: pytest keeps a failing test's traceback, which keeps the builder's frame (and so the class)
    alive and globally visible in ``FeatureGroup.__subclasses__()``. A permanently malformed name catalog
    would then break every later test in the worker whose failure path builds a catalog over the whole
    universe. The autouse fixture disarms every group it built, so a leaked class is inert.
    """

    ARMED: ClassVar[bool] = True


MALFORMED_GROUPS_BUILT_782: list[type[MalformedValueGroup_782]] = []


def _armed_782(group: type[MalformedValueGroup_782]) -> type[MalformedValueGroup_782]:
    """Track a freshly built malformed group so the autouse fixture can disarm it after the test."""
    MALFORMED_GROUPS_BUILT_782.append(group)
    return group


def _build_bad_names_group_782() -> type[CountingSinglePassFG_782]:
    """Build a catalog group whose feature_names_supported() returns a non-str, breaking its own annotation."""

    class SinglePassBadNamesFG_782(MalformedValueGroup_782):
        """Catalog candidate contributing a non-str feature name."""

        FRAMEWORK_RULE = {SinglePassFwOne_782}

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            _record(cls.get_class_name(), "feature_names_supported")
            if cls.ARMED:
                names: set[str] = _malformed({BAD_NAME_VALUE_782})
                return names
            return set()

    return _armed_782(SinglePassBadNamesFG_782)


def _build_bad_prefix_group_782() -> type[CountingSinglePassFG_782]:
    """Build a catalog group whose prefix() returns a non-str."""

    class SinglePassBadPrefixFG_782(MalformedValueGroup_782):
        """Catalog candidate contributing a non-str prefix."""

        FRAMEWORK_RULE = {SinglePassFwOne_782}

        @classmethod
        def prefix(cls) -> str:
            _record(cls.get_class_name(), "prefix")
            if cls.ARMED:
                bad_prefix: str = _malformed(BAD_PREFIX_VALUE_782)
                return bad_prefix
            return f"{cls.get_class_name()}_"

    return _armed_782(SinglePassBadPrefixFG_782)


def _build_bad_domain_groups_782() -> tuple[type[CountingSinglePassFG_782], type[CountingSinglePassFG_782]]:
    """Build a (malformed get_domain, healthy get_domain) pair that both match the same feature name."""

    class SinglePassBadDomainFG_782(MalformedValueGroup_782):
        """Candidate whose get_domain() returns a bare str instead of a Domain."""

        MATCHES = frozenset({BAD_DOMAIN_FEATURE_782})
        FRAMEWORK_RULE = {SinglePassFwOne_782}

        @classmethod
        def get_domain(cls) -> Domain:
            _record(cls.get_class_name(), "get_domain")
            if cls.ARMED:
                bad_domain: Domain = _malformed("single_pass_not_a_domain_782")
                return bad_domain
            return Domain.get_default_domain()

    class SinglePassHealthyDomainFG_782(CountingSinglePassFG_782):
        """Candidate whose get_domain() works, standing next to the malformed one."""

        MATCHES = frozenset({BAD_DOMAIN_FEATURE_782})
        DOMAIN_NAME = HEALTHY_DOMAIN_782
        FRAMEWORK_RULE = {SinglePassFwOne_782}

    return _armed_782(SinglePassBadDomainFG_782), SinglePassHealthyDomainFG_782


def _build_bad_declaration_groups_782() -> tuple[
    type[CountingSinglePassFG_782], type[CountingSinglePassFG_782], type[CountingSinglePassFG_782]
]:
    """Build an abstract base plus a malformed-declaring and a healthy concrete subclass."""

    class SinglePassBadDeclAbstractFG_782(CountingSinglePassFG_782):
        """Abstract base that matches the name but can never be instantiated."""

        MATCHES = frozenset({BAD_DECLARATION_FEATURE_782})
        FRAMEWORK_RULE = {SinglePassFwOne_782}

        @classmethod
        @abstractmethod
        def _single_pass_bad_decl_hook_782(cls) -> str:
            """Abstract hook that keeps this base uninstantiable."""

    class SinglePassBadDeclConcreteFG_782(SinglePassBadDeclAbstractFG_782, MalformedValueGroup_782):
        """Concrete implementation declaring a well-formed framework AND a class that is not one."""

        MATCHES = frozenset()

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            _record(cls.get_class_name(), "compute_framework_rule")
            if cls.ARMED:
                bad_rule: set[type[ComputeFramework]] = _malformed({SinglePassFwOne_782, NotAFramework_782})
                return bad_rule
            return {SinglePassFwOne_782}

        @classmethod
        def _single_pass_bad_decl_hook_782(cls) -> str:
            return "concrete"

    class SinglePassHealthyDeclConcreteFG_782(SinglePassBadDeclAbstractFG_782):
        """Concrete sibling whose declaration is well-formed; its framework must still be named."""

        MATCHES = frozenset()
        FRAMEWORK_RULE = {SinglePassFwTwo_782}

        @classmethod
        def _single_pass_bad_decl_hook_782(cls) -> str:
            return "concrete"

    return (
        SinglePassBadDeclAbstractFG_782,
        _armed_782(SinglePassBadDeclConcreteFG_782),
        SinglePassHealthyDeclConcreteFG_782,
    )


def _build_tie_rejection_groups_782() -> tuple[type[FeatureGroup], type[FeatureGroup]]:
    """Build two same-named candidates across modules, both overriding the rejection hook, one with a non-str reason.

    Sharing a __name__ once forced ``sorted()`` past the first tuple element and onto the hook-provided
    reason. Reasons now come only from first-pass recording, so the pin is that these hooks are never
    consulted at all; the hook counts its calls to prove it.
    """

    def rejection_hook(cls: Any, feature_name: Any, options: Any) -> Any:
        # Disarmed, the group reports no rejection at all, which is what makes a leaked one inert: these two
        # SHARE a __name__, so an armed leak would poison any later sorted() over the whole universe.
        _record(cls.get_class_name(), "_strict_validation_rejection_reason")
        return _malformed(cls.REASON) if cls.ARMED else None

    def make(module: str, reason: Any) -> type[FeatureGroup]:
        namespace: dict[str, Any] = {
            "__module__": module,
            "__doc__": "Same-named candidate carrying its own value-rejection reason.",
            "MATCHES": frozenset(),
            "FRAMEWORK_RULE": {SinglePassFwOne_782},
            "REASON": reason,
            "_strict_validation_rejection_reason": classmethod(rejection_hook),
        }
        created: Any = type("SinglePassTieRejectFG_782", (MalformedValueGroup_782,), namespace)
        _armed_782(cast(type[MalformedValueGroup_782], created))
        return cast(type[FeatureGroup], created)

    return make(TIE_MODULE_A_782, HEALTHY_REJECTION_REASON_782), make(TIE_MODULE_B_782, BAD_REASON_VALUE_782)


class SinglePassStrictFG_782(FeatureChainParserMixin, FeatureGroup):
    """Config-based group whose strict 'window_size_782' validator rejects out-of-range values."""

    PROPERTY_MAPPING = {
        "window_size_782": property_spec(
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
        return {SinglePassFwOne_782}

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        _record(cls.get_class_name(), "supports_compute_framework")
        _record_pair(cls.get_class_name(), compute_framework.get_class_name())
        return super().supports_compute_framework(feature_name, options, compute_framework)

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        _record(cls.get_class_name(), "index_columns")
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


def _probe_keys(options: Options) -> list[str]:
    """Every reader probe key currently stamped on these Options."""
    return sorted(key for key in options.context if key.startswith(OPTION_PROBE_PREFIX_782))


@dataclass(frozen=True)
class Scenario782:
    """One failure scenario, expressed so the engine and resolve_feature see the SAME environment."""

    feature: Feature
    feature_groups: frozenset[type[FeatureGroup]]
    frameworks: frozenset[type[ComputeFramework]] = frozenset({SinglePassFwOne_782})
    links: Optional[set[Link]] = None
    data_access_collection: Optional[DataAccessCollection] = None
    # Every accessible candidate the matcher must be asked about, exactly once. Drives the EXACT counter
    # expectation, so a hook that stops being called at all fails just as loudly as one called twice.
    candidates: frozenset[str] = field(default_factory=frozenset)


def _collector(scenario: Scenario782) -> PluginCollector:
    return PluginCollector.enabled_feature_groups(set(scenario.feature_groups))


def _accessible(scenario: Scenario782) -> FeatureGroupEnvironmentMapping:
    """Build the environment exactly as resolve_feature builds it, so both consumers share it."""
    return PreFilterPlugins(set(scenario.frameworks), _collector(scenario)).get_accessible_plugins()


def _reset_counters() -> None:
    HOOK_CALLS.clear()
    PAIR_CALLS.clear()
    INDEX_CALLS.clear()


def _engine_error(scenario: Scenario782) -> str:
    """Run ONE engine attempt and return its text.

    The environment is built first and the counters cleared afterwards, so they cover the resolution
    attempt only: PreFilterPlugins legitimately asks every group to declare its frameworks once.
    """
    accessible_plugins = _accessible(scenario)
    _reset_counters()

    with pytest.raises(ValueError) as exc_info:
        evaluate_or_raise(
            feature=scenario.feature,
            accessible_plugins=accessible_plugins,
            links=scenario.links,
            data_access_collection=scenario.data_access_collection,
        )
    return str(exc_info.value)


def _resolve_error(scenario: Scenario782) -> str:
    """Run ONE resolve_feature attempt and return its error.

    resolve_feature builds the environment itself, so its counters include that one declaration pass.
    """
    _reset_counters()

    resolved = resolve_feature(
        scenario.feature,
        plugin_collector=_collector(scenario),
        links=scenario.links,
        data_access_collection=scenario.data_access_collection,
        compute_frameworks=set(scenario.frameworks),
    )
    assert resolved.feature_group is None
    assert resolved.error is not None
    return resolved.error


def multiple_scenario(scope: Optional[str] = None) -> Scenario782:
    """Two identified candidates."""
    return Scenario782(
        feature=Feature(MULTIPLE_FEATURE_782, feature_group=scope),
        feature_groups=frozenset({SinglePassMultipleAFG_782, SinglePassMultipleBFG_782}),
        candidates=frozenset({"SinglePassMultipleAFG_782", "SinglePassMultipleBFG_782"}),
    )


def domain_multiple_scenario(scope: Optional[str] = None) -> Scenario782:
    """Two identified candidates for a request that CARRIES a domain, so the filter loop asks get_domain."""
    return Scenario782(
        feature=Feature(DOMAIN_MULTIPLE_FEATURE_782, domain=PROBE_DOMAIN_782, feature_group=scope),
        feature_groups=frozenset({SinglePassMultipleAFG_782, SinglePassMultipleBFG_782}),
        candidates=frozenset({"SinglePassMultipleAFG_782", "SinglePassMultipleBFG_782"}),
    )


def abstract_only_scenario(scope: Optional[str] = None) -> Scenario782:
    """Abstract base matched; its criteria-matched concrete subclass declares no enabled framework."""
    return Scenario782(
        feature=Feature(ABSTRACT_FEATURE_782, feature_group=scope),
        feature_groups=frozenset({SinglePassAbstractBaseFG_782, SinglePassConcreteSubFG_782}),
        # The abstract base is matched too: it is asked, then parked as abstract rather than criteria-matched.
        candidates=frozenset({"SinglePassAbstractBaseFG_782", "SinglePassConcreteSubFG_782"}),
    )


def capability_scenario(scope: Optional[str] = None) -> Scenario782:
    """The one enabled framework is the one the candidate rejects."""
    return Scenario782(
        feature=Feature(CAPABILITY_FEATURE_782, feature_group=scope),
        feature_groups=frozenset({SinglePassCapabilityRejectFG_782}),
        candidates=frozenset({"SinglePassCapabilityRejectFG_782"}),
    )


def ordinary_none_scenario(scope: Optional[str] = None) -> Scenario782:
    """No match: the candidate matches the bare name but rejects the feature's group options."""
    return Scenario782(
        feature=Feature(BARE_ONLY_FEATURE_782, Options(group={"top_k_782": 5}), feature_group=scope),
        feature_groups=frozenset({SinglePassBareOnlyFG_782}),
        candidates=frozenset({"SinglePassBareOnlyFG_782"}),
    )


def strict_none_scenario() -> Scenario782:
    """No match: the candidate rejects an option VALUE, which the message reports as a rejection line."""
    return Scenario782(
        feature=Feature(
            STRICT_FEATURE_782,
            Options(context={DefaultOptionKeys.in_features: "src", "window_size_782": 14}),
        ),
        feature_groups=frozenset({SinglePassStrictFG_782}),
        candidates=frozenset({"SinglePassStrictFG_782"}),
    )


def reader_scenario() -> Scenario782:
    """No match: a reader-style matcher accepts the bare name off a DataAccessCollection but not the options."""
    return Scenario782(
        feature=Feature(READER_FEATURE_782, Options(group={"top_k_782": 5})),
        feature_groups=frozenset({SinglePassReaderFG_782}),
        data_access_collection=DataAccessCollection(files={"single_pass_782.csv"}),
        candidates=frozenset({"SinglePassReaderFG_782"}),
    )


def link_scenario() -> Scenario782:
    """No match: the candidate declares index columns but supports neither of the link's indexes."""
    link = Link.inner(
        JoinSpec(SinglePassLinkFG_782, LEFT_INDEX_782),
        JoinSpec(SinglePassLinkFG_782, RIGHT_INDEX_782),
    )
    return Scenario782(
        feature=Feature(LINK_FEATURE_782),
        feature_groups=frozenset({SinglePassLinkFG_782}),
        links={link},
        candidates=frozenset({"SinglePassLinkFG_782"}),
    )


def stateful_match_scenario() -> Scenario782:
    """The candidate matches once, then never again."""
    return Scenario782(
        feature=Feature(STATEFUL_MATCH_FEATURE_782),
        feature_groups=frozenset({SinglePassStatefulMatchFG_782}),
        candidates=frozenset({"SinglePassStatefulMatchFG_782"}),
    )


def stateful_support_scenario() -> Scenario782:
    """The candidate rejects its framework once, then accepts it forever."""
    return Scenario782(
        feature=Feature(STATEFUL_SUPPORT_FEATURE_782),
        feature_groups=frozenset({SinglePassStatefulSupportFG_782}),
        candidates=frozenset({"SinglePassStatefulSupportFG_782"}),
    )


# Every failure kind named in the issue's definition of done, unscoped and scoped.
PAIRED_SCENARIOS: dict[str, Callable[[], Scenario782]] = {
    "multiple": multiple_scenario,
    "abstract_only": abstract_only_scenario,
    "capability": capability_scenario,
    "ordinary_none": ordinary_none_scenario,
    "scoped_multiple": lambda: multiple_scenario(scope="CountingSinglePassFG_782"),
    "scoped_abstract_only": lambda: abstract_only_scenario(scope="SinglePassAbstractBaseFG_782"),
    "scoped_capability": lambda: capability_scenario(scope="SinglePassCapabilityRejectFG_782"),
    "scoped_ordinary_none": lambda: ordinary_none_scenario(scope="SinglePassBareOnlyFG_782"),
}

# Scenarios whose failure branch also wants the hook its decision loop already called.
COUNTED_SCENARIOS: dict[str, Callable[[], Scenario782]] = {
    **PAIRED_SCENARIOS,
    "domain_multiple": domain_multiple_scenario,
    "strict_none": strict_none_scenario,
    "reader": reader_scenario,
    "link": link_scenario,
}


def bad_catalog_scenario(group: type[FeatureGroup]) -> Scenario782:
    """No match, so the failure path builds a name catalog over a candidate with a malformed one."""
    return Scenario782(
        feature=Feature(BAD_CATALOG_FEATURE_782),
        feature_groups=frozenset({group}),
        candidates=frozenset({group.get_class_name()}),
    )


@pytest.fixture(autouse=True)
def reset_counters() -> Iterator[None]:
    """Counter and stateful-hook state must not leak between tests.

    No malformed group may outlive the test that built it: it stays globally visible in the FeatureGroup
    tree, and any later test whose failure path enumerates the universe would trip over it.
    """
    _reset_counters()
    STATEFUL_CALLS.clear()
    yield
    _reset_counters()
    STATEFUL_CALLS.clear()
    for group in MALFORMED_GROUPS_BUILT_782:
        group.ARMED = False
    MALFORMED_GROUPS_BUILT_782.clear()


class TestEngineAndResolveFeatureAgree:
    """Definition of done: both consumers render every failure from the same EvaluationResult."""

    @pytest.mark.parametrize("scenario_name", sorted(PAIRED_SCENARIOS))
    def test_engine_and_resolve_feature_report_the_same_message(self, scenario_name: str) -> None:
        """The engine's ValueError text and resolve_feature's error are the SAME string."""
        engine_message = _engine_error(PAIRED_SCENARIOS[scenario_name]())
        resolve_message = _resolve_error(PAIRED_SCENARIOS[scenario_name]())

        assert engine_message == resolve_message

    @pytest.mark.parametrize("scenario_name", sorted(PAIRED_SCENARIOS))
    def test_the_shared_message_describes_its_failure_kind(self, scenario_name: str) -> None:
        """Agreement is worthless if both sides degrade to the same generic text."""
        message = _engine_error(PAIRED_SCENARIOS[scenario_name]())

        if "multiple" in scenario_name:
            assert message.startswith("Multiple feature groups found for feature")
        elif "capability" in scenario_name:
            # Capability rejection is now a near-miss line under the ordinary none message, not its own message.
            assert message.startswith("No feature groups found for feature name:")
            assert "(compute framework): supports_compute_framework rejected" in message
        else:
            assert message.startswith("No feature groups found for feature name:")

    @pytest.mark.parametrize("scenario_name", sorted(name for name in PAIRED_SCENARIOS if name.startswith("scoped")))
    def test_the_scoped_message_carries_exactly_one_scope_callout(self, scenario_name: str) -> None:
        """A scoped request names its scope once; a second render would append it twice."""
        message = _engine_error(PAIRED_SCENARIOS[scenario_name]())

        assert message.count("Scoped to feature group:") == 1


class TestDecisionHooksRunAtMostOncePerCandidate:
    """The heart of #782: one resolution attempt asks each decision hook once per candidate."""

    @pytest.mark.parametrize("scenario_name", sorted(COUNTED_SCENARIOS))
    def test_engine_asks_match_feature_group_criteria_once_per_candidate(self, scenario_name: str) -> None:
        """The speculative bare/actual re-match of the old forwarding hint is gone."""
        scenario = COUNTED_SCENARIOS[scenario_name]()

        _engine_error(scenario)

        assert _counts_for("match_feature_group_criteria") == _once_each(scenario.candidates)

    @pytest.mark.parametrize("scenario_name", sorted(COUNTED_SCENARIOS))
    def test_resolve_feature_asks_match_feature_group_criteria_once_per_candidate(self, scenario_name: str) -> None:
        """resolve_feature evaluates once; it must not construct the engine again to scrape a message."""
        scenario = COUNTED_SCENARIOS[scenario_name]()

        _resolve_error(scenario)

        assert _counts_for("match_feature_group_criteria") == _once_each(scenario.candidates)

    def test_engine_asks_get_domain_once_when_the_request_carries_a_domain(self) -> None:
        """The filter loop needs the domain to decide and the message needs it to render: capture it once."""
        scenario = domain_multiple_scenario()

        _engine_error(scenario)

        assert _counts_for("get_domain") == _once_each(scenario.candidates)

    def test_resolve_feature_asks_get_domain_once_when_the_request_carries_a_domain(self) -> None:
        """Same budget across the diagnostic API."""
        scenario = domain_multiple_scenario()

        _resolve_error(scenario)

        assert _counts_for("get_domain") == _once_each(scenario.candidates)

    def test_engine_asks_compute_framework_rule_once_per_candidate(self) -> None:
        """The concrete subclass is criteria-matched AND an abstract-only implementation: still one ask.

        Only the concrete subclass is asked: the abstract base is parked before the framework split, and
        the abstract-only capture skips abstract candidates.
        """
        _engine_error(abstract_only_scenario())

        assert _counts_for("compute_framework_rule") == {"SinglePassConcreteSubFG_782": 1}

    def test_resolve_feature_asks_compute_framework_rule_once_per_candidate_after_the_environment_build(
        self,
    ) -> None:
        """PreFilterPlugins declares every group's frameworks once while building the environment.

        That build precedes the resolution attempt, so it is not a second ask. The attempt itself adds
        exactly one, and only for the concrete subclass.
        """
        _resolve_error(abstract_only_scenario())

        assert _counts_for("compute_framework_rule") == {
            "SinglePassAbstractBaseFG_782": 1,
            "SinglePassConcreteSubFG_782": 2,
        }

    def test_engine_asks_the_capability_hook_once_per_candidate_framework_pair(self) -> None:
        """Run-only: the hook is asked solely over the run-enabled frameworks, so the disabled FwTwo is dropped."""
        _engine_error(capability_scenario())

        assert PAIR_CALLS == {
            ("SinglePassCapabilityRejectFG_782", "SinglePassFwOne_782"): 1,
        }

    def test_resolve_feature_asks_the_capability_hook_once_per_candidate_framework_pair(self) -> None:
        """Same budget across the diagnostic API."""
        _resolve_error(capability_scenario())

        assert PAIR_CALLS == {
            ("SinglePassCapabilityRejectFG_782", "SinglePassFwOne_782"): 1,
        }

    def test_engine_asks_the_capability_hook_once_for_the_abstract_only_concrete_subclass(self) -> None:
        """Run-only: the subclass declares only a run-disabled framework, so the hook is not asked at all."""
        _engine_error(abstract_only_scenario())

        assert PAIR_CALLS == {}

    def test_engine_asks_the_link_hooks_once_per_candidate_and_index(self) -> None:
        """index_columns is per candidate; supports_index is per (candidate, index)."""
        scenario = link_scenario()

        _engine_error(scenario)

        assert _counts_for("index_columns") == _once_each(scenario.candidates)
        assert INDEX_CALLS == {
            ("SinglePassLinkFG_782", str(LEFT_INDEX_782)): 1,
            ("SinglePassLinkFG_782", str(RIGHT_INDEX_782)): 1,
        }

    def test_resolve_feature_asks_the_link_hooks_once_per_candidate_and_index(self) -> None:
        """The second evaluation doubled both; one pass must not."""
        scenario = link_scenario()

        _resolve_error(scenario)

        assert _counts_for("index_columns") == _once_each(scenario.candidates)
        assert INDEX_CALLS == {
            ("SinglePassLinkFG_782", str(LEFT_INDEX_782)): 1,
            ("SinglePassLinkFG_782", str(RIGHT_INDEX_782)): 1,
        }

    def test_engine_renders_the_value_rejection_from_the_first_pass_without_the_hook(self) -> None:
        """The rejection reason is rendered from the first pass; the hook is not consulted."""
        scenario = strict_none_scenario()

        message = _engine_error(scenario)

        assert _counts_for("_strict_validation_rejection_reason") == {}
        assert f"  - SinglePassStrictFG_782 (option value): {STRICT_REJECTION_REASON_782}" in message

    def test_resolve_feature_renders_the_value_rejection_from_the_first_pass_without_the_hook(self) -> None:
        """Same contract across the diagnostic API: rendered from the first pass, hook not consulted."""
        scenario = strict_none_scenario()

        message = _resolve_error(scenario)

        assert _counts_for("_strict_validation_rejection_reason") == {}
        assert f"  - SinglePassStrictFG_782 (option value): {STRICT_REJECTION_REASON_782}" in message

    def test_engine_asks_a_reader_style_matcher_once_per_candidate(self) -> None:
        """A matcher that consults a DataAccessCollection is asked once, like any other."""
        _engine_error(reader_scenario())

        assert HOOK_CALLS.get("SinglePassReaderFG_782.match_feature_group_criteria") == 1

    def test_resolve_feature_asks_a_reader_style_matcher_once_per_candidate(self) -> None:
        """Same budget across the diagnostic API."""
        _resolve_error(reader_scenario())

        assert HOOK_CALLS.get("SinglePassReaderFG_782.match_feature_group_criteria") == 1


class TestStatefulHookCannotChangeTheResultOrItsMessage:
    """A hook that would answer differently on a second call is never asked a second time."""

    def test_stateful_matcher_message_describes_the_first_answer(self) -> None:
        """First answer: matched, then rejected on its only enabled framework. That is what both must report."""
        engine_message = _engine_error(stateful_match_scenario())
        STATEFUL_CALLS.clear()
        resolve_message = _resolve_error(stateful_match_scenario())

        assert engine_message == resolve_message
        assert engine_message.startswith("No feature groups found for feature name:")
        assert "SinglePassFwOne_782" in engine_message

    def test_stateful_matcher_keeps_its_candidate_in_resolve_feature(self) -> None:
        """The reported candidates and the reported error must describe the same evaluation."""
        scenario = stateful_match_scenario()

        resolved = resolve_feature(
            scenario.feature,
            plugin_collector=_collector(scenario),
            compute_frameworks=set(scenario.frameworks),
        )

        assert resolved.error is not None
        assert [candidate.get_class_name() for candidate in resolved.candidates] == ["SinglePassStatefulMatchFG_782"]
        assert resolved.error.startswith("No feature groups found for feature name:")

    def test_stateful_capability_hook_message_describes_the_first_answer(self) -> None:
        """First answer: the framework is rejected. A later 'supported' answer must not rewrite the message."""
        engine_message = _engine_error(stateful_support_scenario())
        STATEFUL_CALLS.clear()
        resolve_message = _resolve_error(stateful_support_scenario())

        assert engine_message == resolve_message
        assert engine_message.startswith("No feature groups found for feature name:")
        assert "SinglePassFwOne_782" in engine_message

    def test_stateful_capability_hook_is_asked_once_per_pair(self) -> None:
        """The reason the answer cannot flip: the pair is asked exactly once."""
        _engine_error(stateful_support_scenario())

        assert PAIR_CALLS == {("SinglePassStatefulSupportFG_782", "SinglePassFwOne_782"): 1}


class TestRenderingIsPureAndMutatesNothing:
    """Validation and rendering invoke no provider hook and cannot change what they describe."""

    @pytest.mark.parametrize("scenario_name", sorted(COUNTED_SCENARIOS))
    def test_repeated_rendering_calls_no_hook_and_returns_the_same_string(self, scenario_name: str) -> None:
        """After evaluate(), rendering is a projection: no counter moves and the string is stable."""
        scenario = COUNTED_SCENARIOS[scenario_name]()
        accessible_plugins = _accessible(scenario)
        _reset_counters()

        result = IdentifyFeatureGroupClass.evaluate(
            feature=scenario.feature,
            accessible_plugins=accessible_plugins,
            links=scenario.links,
            data_access_collection=scenario.data_access_collection,
        )
        assert result.failure_kind is not None

        snapshot = dict(HOOK_CALLS)
        assert snapshot, "the fixture feature groups must count at least one hook call during evaluate()"

        first = render_resolution_failure(result, scenario.feature)
        for _ in range(3):
            assert render_resolution_failure(result, scenario.feature) == first

        assert first is not None
        assert HOOK_CALLS == snapshot

    @pytest.mark.parametrize("scenario_name", sorted(COUNTED_SCENARIOS))
    def test_rendering_does_not_mutate_the_feature_or_its_options(self, scenario_name: str) -> None:
        """Rendering reads the Feature; it must not write to it."""
        scenario = COUNTED_SCENARIOS[scenario_name]()
        accessible_plugins = _accessible(scenario)

        result = IdentifyFeatureGroupClass.evaluate(
            feature=scenario.feature,
            accessible_plugins=accessible_plugins,
            links=scenario.links,
            data_access_collection=scenario.data_access_collection,
        )
        before = (
            str(scenario.feature.name),
            dict(scenario.feature.options.group),
            dict(scenario.feature.options.context),
        )

        for _ in range(3):
            render_resolution_failure(result, scenario.feature)

        after = (
            str(scenario.feature.name),
            dict(scenario.feature.options.group),
            dict(scenario.feature.options.context),
        )
        assert after == before

    def test_the_engine_attempt_enriches_the_options_exactly_once(self) -> None:
        """The single decision pass stamps one probe key; a speculative re-match would stamp more."""
        scenario = reader_scenario()

        _engine_error(scenario)

        assert _probe_keys(scenario.feature.options) == [f"{OPTION_PROBE_PREFIX_782}0"]

    def test_a_resolve_feature_attempt_enriches_the_options_exactly_once(self) -> None:
        """Same budget across the diagnostic API."""
        scenario = reader_scenario()

        _resolve_error(scenario)

        assert _probe_keys(scenario.feature.options) == [f"{OPTION_PROBE_PREFIX_782}0"]

    def test_the_engine_attempt_leaves_the_options_where_evaluate_left_them(self) -> None:
        """The failure path adds no mutation of its own on top of the decision pass."""
        evaluated = reader_scenario()
        IdentifyFeatureGroupClass.evaluate(
            feature=evaluated.feature,
            accessible_plugins=_accessible(evaluated),
            links=evaluated.links,
            data_access_collection=evaluated.data_access_collection,
        )

        raised = reader_scenario()
        _engine_error(raised)

        assert dict(raised.feature.options.context) == dict(evaluated.feature.options.context)
        assert dict(raised.feature.options.group) == dict(evaluated.feature.options.group)


class TestMalformedHookValuesDegradeLikeARaise:
    """A hook that RETURNS a malformed value degrades exactly like one that raises.

    Memoizing a hook only guards the CALL. The PROJECTION of what it returned (``domain.name``,
    ``cfw.get_class_name()``, a name reaching ``difflib``) must be guarded too, or a provider whose
    annotation lies takes down a call that promises never to fail. Every hook here returns a
    well-typed-looking value that is the wrong type, which no raise-based test can catch.
    """

    def test_resolve_feature_does_not_raise_when_feature_names_supported_returns_non_str(self) -> None:
        """resolve_feature promises never to raise for matching errors; a non-str name must not break that."""
        scenario = bad_catalog_scenario(_build_bad_names_group_782())

        error = _resolve_error(scenario)

        assert error.startswith(f"No feature groups found for feature name: '{BAD_CATALOG_FEATURE_782}'.")
        assert TROUBLESHOOTING_LINE_782 in error, "the malformed name must degrade the catalog, not the message"
        assert str(BAD_NAME_VALUE_782) not in error

    def test_resolve_feature_does_not_raise_when_prefix_returns_non_str(self) -> None:
        """The sibling hole: prefix() feeds the same catalog and needs the same guard."""
        scenario = bad_catalog_scenario(_build_bad_prefix_group_782())

        error = _resolve_error(scenario)

        assert error.startswith(f"No feature groups found for feature name: '{BAD_CATALOG_FEATURE_782}'.")
        assert TROUBLESHOOTING_LINE_782 in error, "the malformed prefix must degrade the catalog, not the message"
        assert str(BAD_PREFIX_VALUE_782) not in error

    def test_engine_raises_its_resolution_error_when_feature_names_supported_returns_non_str(self) -> None:
        """The engine must fail with its own resolution ValueError, not a plugin's TypeError."""
        scenario = bad_catalog_scenario(_build_bad_names_group_782())

        message = _engine_error(scenario)

        assert message.startswith(f"No feature groups found for feature name: '{BAD_CATALOG_FEATURE_782}'.")
        assert TROUBLESHOOTING_LINE_782 in message

    def test_a_hostile_rejection_hook_cannot_inject_a_reason_into_the_eliminations(self) -> None:
        """Two same-named groups override the hook, one with a non-str reason: neither is consulted.

        The old hazard was a sorted() over hook-provided reasons, where two same-named candidates forced
        the comparison onto a malformed (non-str) reason. Rejection reasons are now recorded during the
        first pass only, and these matchers record nothing, so a hook cannot inject a reason at all and
        the sorted() poisoning hazard is structurally gone.
        """
        healthy, malformed = _build_tie_rejection_groups_782()
        feature = Feature(TIE_REJECT_FEATURE_782)
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            healthy: {SinglePassFwOne_782},
            malformed: {SinglePassFwOne_782},
        }

        result = IdentifyFeatureGroupClass.evaluate(feature=feature, accessible_plugins=accessible_plugins, links=None)
        message = render_resolution_failure(result, feature)

        # These matchers name-mismatch and record nothing, so no reason (malformed or not) is ever attributed.
        assert _counts_for("_strict_validation_rejection_reason") == {}
        assert result.eliminations == {}
        assert message is not None
        assert HEALTHY_REJECTION_REASON_782 not in message
        assert str(BAD_REASON_VALUE_782) not in message

    def test_malformed_get_domain_degrades_only_its_own_domain_suffix(self) -> None:
        """get_domain() returning a bare str must cost that candidate its suffix, not crash evaluate()."""
        malformed, healthy = _build_bad_domain_groups_782()
        feature = Feature(BAD_DOMAIN_FEATURE_782)
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            malformed: {SinglePassFwOne_782},
            healthy: {SinglePassFwOne_782},
        }

        result = IdentifyFeatureGroupClass.evaluate(feature=feature, accessible_plugins=accessible_plugins, links=None)

        assert result.failure_kind == "multiple"
        assert render_resolution_failure(result, feature) == (
            f"Multiple feature groups found for feature '{BAD_DOMAIN_FEATURE_782}':\n"
            f"  - {malformed.__name__} ({malformed.__module__})\n"
            f"  - {healthy.__name__} ({healthy.__module__}) [domain: {HEALTHY_DOMAIN_782}]\n"
            f"{TROUBLESHOOTING_LINE_782}"
        )

    def test_malformed_framework_declaration_degrades_only_its_own_candidate(self) -> None:
        """A declaration naming a non-ComputeFramework costs that candidate its names, not the message.

        The whole candidate's names degrade, including the well-formed framework it also declared: the
        guard wraps the projection of one candidate's declaration, matching the pre-refactor behaviour.
        The healthy concrete sibling still contributes its own.
        """
        base, malformed, healthy = _build_bad_declaration_groups_782()
        feature = Feature(BAD_DECLARATION_FEATURE_782)
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            base: {SinglePassFwOne_782},
            malformed: set(),
            healthy: set(),
        }

        result = IdentifyFeatureGroupClass.evaluate(feature=feature, accessible_plugins=accessible_plugins, links=None)

        assert result.failure_kind == "abstract_only"
        assert result.facts.concrete_frameworks == ("SinglePassFwTwo_782",)
        assert render_resolution_failure(result, feature) == (
            f"No feature groups found for feature name: '{BAD_DECLARATION_FEATURE_782}'. "
            "Its concrete implementations require compute framework(s) ['SinglePassFwTwo_782'], "
            "none of which are available or enabled for this run."
        )

    def test_malformed_framework_declaration_alone_falls_back_to_the_bare_abstract_message(self) -> None:
        """With no healthy sibling, no framework name survives, so the bare abstract-only variant renders."""
        base, malformed, _ = _build_bad_declaration_groups_782()
        feature = Feature(BAD_DECLARATION_FEATURE_782)
        accessible_plugins: FeatureGroupEnvironmentMapping = {base: {SinglePassFwOne_782}, malformed: set()}

        result = IdentifyFeatureGroupClass.evaluate(feature=feature, accessible_plugins=accessible_plugins, links=None)

        assert result.facts.concrete_frameworks == ()
        assert render_resolution_failure(result, feature) == (
            f"No feature groups found for feature name: '{BAD_DECLARATION_FEATURE_782}'. "
            "Only abstract feature group base(s) matched, which cannot be instantiated; "
            "no concrete implementation is available or enabled."
        )


class TestTheSecondEvaluationIsGone:
    """Structural: the symbols that made a second evaluation possible no longer exist."""

    def test_engine_failure_message_helper_is_deleted(self) -> None:
        """resolve_feature no longer constructs IdentifyFeatureGroupClass to scrape its ValueError."""
        assert not hasattr(plugin_docs, "_engine_failure_message")

    @pytest.mark.parametrize(
        "attribute",
        [
            "validate",
            "_build_no_feature_group_error",
            "_capability_rejection_message",
            "_abstract_only_message",
            "_strict_validation_rejection_hint",
            "_input_feature_forwarding_hint",
            "get",
        ],
    )
    def test_legacy_raising_path_attribute_is_deleted(self, attribute: str) -> None:
        """The raising wrapper and its winner accessors are gone; failures render from the EvaluationResult."""
        assert not hasattr(IdentifyFeatureGroupClass, attribute)

    def test_split_frameworks_by_capability_is_deleted(self) -> None:
        """The module function re-ran the capability hook over every declared framework."""
        assert not hasattr(identify_feature_group, "split_frameworks_by_capability")
