"""Direct target-contract tests for the rewired resolve_feature (issue #722 Stage 3b).

Stage 3b rewires the debug API resolve_feature (mloda/core/api/plugin_docs.py) onto the
same authoritative FeatureGroupResolver as the engine, against a STANDALONE environment
built by build_resolution_environment over all available compute frameworks
(PreFilterPlugins.get_cfw_subclasses()) and the caller's plugin collector.

These tests FAIL until the rewire lands; the engine halves already pass (Stage 3a).
Probe names use the unique probe722g_ prefix so process-global scans in other tests
never trip these fixtures.
"""

from __future__ import annotations

import inspect
from abc import abstractmethod

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.plugin_registry.plugin_registry import PluginRegistry
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.api.plugin_info import ResolvedFeature
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping, PreFilterPlugins
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass


STRICT_FEATURE = "probe722g_strict"
ATTRIBUTION_FEATURE = "probe722g_attribution"
BOOM_FEATURE = "probe722g_boom"

BOOM_ERROR_TEXT = "boom 722g criteria"
STRICT_MODE_TEXT = "Strict mode filtered out all FeatureGroups"


class CfwOne722G(ComputeFramework):
    """First framework of the attribution family; declared by parent and child."""


class CfwTwo722G(ComputeFramework):
    """Second framework of the attribution family; declared by the abstract parent only."""


class CfwStrict722G(ComputeFramework):
    """Framework carrying the strict-mode probes."""


class CfwBoom722G(ComputeFramework):
    """Framework declared by the provider-failure probes."""


class StrictProbe722G(FeatureGroup):
    """Concrete probe that is never registered in the injected empty registry."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwStrict722G}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == STRICT_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class AbstractAttributionParent722G(FeatureGroup):
    """Abstract parent: declares two frameworks and matches the shared attribution name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwOne722G, CfwTwo722G}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == ATTRIBUTION_FEATURE

    @classmethod
    @abstractmethod
    def _probe_hook_722g(cls) -> str:
        """Abstract hook that keeps this probe abstract."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class AttributionChild722G(AbstractAttributionParent722G):
    """Concrete child: narrows the declaration to a strict subset of the parent's."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwOne722G}

    @classmethod
    def _probe_hook_722g(cls) -> str:
        return "concrete 722g"


class BoomCriteriaProbe722G(FeatureGroup):
    """Raises a plain ValueError from matching, gated on its unique shared name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwBoom722G}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        # A plain ValueError (NOT a PropertyValueRejection) is a decision-relevant provider
        # failure; gated on the probe name so process-global scans never trip the raise.
        if str(feature_name) == BOOM_FEATURE:
            raise ValueError(BOOM_ERROR_TEXT)
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class CleanRivalProbe722G(FeatureGroup):
    """Matches the boom feature name cleanly."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwBoom722G}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == BOOM_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


def _strict_collector_with_empty_registry() -> PluginCollector:
    """Enable only the strict probe, strict mode on, EMPTY injected registry (global registry untouched)."""
    collector = PluginCollector.enabled_feature_groups({StrictProbe722G})
    return collector.set_strict_mode("strict").set_registry(PluginRegistry())


# ---------------------------------------------------------------------------
# (a) Environment build failure: strict mode filters everything out
# ---------------------------------------------------------------------------


def test_probe722g_environment_build_failure_returns_error_instead_of_raising() -> None:
    """A failing standalone environment build becomes ResolvedFeature.error, never a raise."""
    collector = _strict_collector_with_empty_registry()

    result = resolve_feature(STRICT_FEATURE, plugin_collector=collector)

    assert isinstance(result, ResolvedFeature)
    assert result.feature_name == STRICT_FEATURE
    assert result.feature_group is None
    assert result.error is not None
    assert STRICT_MODE_TEXT in result.error


# ---------------------------------------------------------------------------
# (b) Per-winner framework attribution, no union across candidates
# ---------------------------------------------------------------------------


def test_probe722g_winner_attribution_is_per_candidate_not_union() -> None:
    """The child winner is credited with ITS OWN supported frameworks only, on both paths."""
    assert inspect.isabstract(AbstractAttributionParent722G), "fixture parent must be abstract"
    assert not inspect.isabstract(AttributionChild722G), "fixture child must be concrete"

    # Engine parity on an equivalent mapping: the abstract parent is rejected, the child wins
    # with exactly its own framework set.
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        AbstractAttributionParent722G: {CfwOne722G, CfwTwo722G},
        AttributionChild722G: {CfwOne722G},
    }
    identifier = IdentifyFeatureGroupClass(
        feature=Feature(ATTRIBUTION_FEATURE),
        accessible_plugins=accessible_plugins,
        links=None,
    )
    resolved, compute_frameworks = identifier.get()
    assert resolved is AttributionChild722G
    assert compute_frameworks == {CfwOne722G}

    # Debug path: same winner, and the parent-only CfwTwo722G is NOT unioned into the winner's
    # supported list.
    collector = PluginCollector.enabled_feature_groups({AbstractAttributionParent722G, AttributionChild722G})
    result = resolve_feature(ATTRIBUTION_FEATURE, plugin_collector=collector)

    assert result.feature_group is AttributionChild722G
    assert result.error is None
    assert result.supported_compute_frameworks == [CfwOne722G.get_class_name()]
    assert CfwTwo722G.get_class_name() not in result.supported_compute_frameworks
    assert result.unsupported_compute_frameworks == []


# ---------------------------------------------------------------------------
# (c) Provider failure is fail-closed on both paths with the same provider message
# ---------------------------------------------------------------------------


def test_probe722g_provider_failure_fails_closed_on_both_paths() -> None:
    """A raising criteria hook fails both paths with the original provider message."""
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        BoomCriteriaProbe722G: {CfwBoom722G},
        CleanRivalProbe722G: {CfwBoom722G},
    }

    # Engine path: fail-closed despite the clean rival.
    with pytest.raises(ValueError, match=BOOM_ERROR_TEXT):
        IdentifyFeatureGroupClass(
            feature=Feature(BOOM_FEATURE),
            accessible_plugins=accessible_plugins,
            links=None,
        )

    # Debug path: same fail-closed verdict carrying the same provider message.
    collector = PluginCollector.enabled_feature_groups({BoomCriteriaProbe722G, CleanRivalProbe722G})
    result = resolve_feature(BOOM_FEATURE, plugin_collector=collector)

    assert result.feature_group is None
    assert result.error is not None
    assert BOOM_ERROR_TEXT in result.error
    assert "BoomCriteriaProbe722G" in result.error


# ---------------------------------------------------------------------------
# (d) Strict-mode collector parity: identical message text on both paths
# ---------------------------------------------------------------------------


def test_probe722g_strict_collector_produces_identical_message_on_both_paths() -> None:
    """The same strict collector yields an engine raise and a debug error with the SAME text."""
    collector = _strict_collector_with_empty_registry()

    with pytest.raises(ValueError) as exc_info:
        PreFilterPlugins(compute_frameworks={CfwStrict722G}, plugin_collector=collector)
    engine_message = str(exc_info.value)
    assert STRICT_MODE_TEXT in engine_message

    result = resolve_feature(STRICT_FEATURE, plugin_collector=collector)

    assert result.feature_group is None
    # Unscoped call: no scope suffix, so the debug error is exactly the environment message.
    assert result.error == engine_message
