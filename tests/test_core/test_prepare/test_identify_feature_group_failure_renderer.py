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
from collections.abc import Callable
from typing import ClassVar, Optional

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
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
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
EMPTY_ENV_FEATURE_791 = "renderer_empty_env_791"
FORWARDING_FEATURE_791 = "renderer_forwarding_791"

TROUBLESHOOTING_LINE = (
    "For troubleshooting guide, see: "
    "https://mloda-ai.github.io/mloda/in_depth/troubleshooting/feature-group-resolution-errors/"
)

# Call counters for every provider-overridable hook, keyed "<ClassName>.<hook>". Reset per test.
HOOK_CALLS: dict[str, int] = {}


def _record(class_name: str, hook: str) -> None:
    """Count one call of a provider-overridable hook."""
    key = f"{class_name}.{hook}"
    HOOK_CALLS[key] = HOOK_CALLS.get(key, 0) + 1


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


def scoped_none_scenario() -> Scenario:
    """No match while scoped to a feature group."""
    feature = Feature(SCOPED_NO_MATCH_FEATURE_791, feature_group="RendererKnownNamesFG791")
    return feature, {RendererKnownNamesFG791: {RendererFwOne791}}


def empty_environment_scenario() -> Scenario:
    """No plugins are loaded at all."""
    return Feature(EMPTY_ENV_FEATURE_791), {}


FAILING_SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "multiple": multiple_scenario,
    "capability": capability_scenario,
    "abstract_with_frameworks": abstract_with_frameworks_scenario,
    "abstract_bare": abstract_bare_scenario,
    "ordinary_none": ordinary_none_scenario,
    "scoped_none": scoped_none_scenario,
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
def reset_hook_calls() -> None:
    """Counter state must not leak between tests."""
    HOOK_CALLS.clear()


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
        assert lines[1] == f"Feature group(s) rejected an option value while matching '{TYPO_FEATURE_791}':"
        assert lines[2] == f"  - RendererStrictFG791: {WINDOW_REJECTION_REASON}"
        assert lines[3].startswith("Did you mean one of: [")
        assert f"'{KNOWN_FEATURE_791}'" in lines[3]
        assert lines[4] == "Use resolve_feature(name, options=...) to debug feature resolution."
        assert lines[5] == TROUBLESHOOTING_LINE
        assert len(lines) == 6

    def test_empty_environment_stops_after_plugin_loader_hint(self) -> None:
        """An empty environment renders the PluginLoader hint and nothing else."""
        message = _render(empty_environment_scenario())

        assert message == (
            f"No feature groups found for feature name: '{EMPTY_ENV_FEATURE_791}'."
            "\nNo plugins are loaded. Did you call PluginLoader.all()?"
        )

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
        assert not result.facts.environment_empty
        assert KNOWN_FEATURE_791 in result.facts.known_names
        assert "RendererKnownNamesFG791" in result.facts.known_names
        assert "RendererStrictFG791_" in result.facts.known_names

    def test_environment_empty_captured(self) -> None:
        """An empty accessible_plugins is captured as a fact."""
        result = _evaluate(empty_environment_scenario())

        assert result.failure_kind == "none"
        assert result.facts.environment_empty
        assert result.facts.known_names == ()
