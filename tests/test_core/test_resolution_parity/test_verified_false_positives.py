"""Paired engine/debug characterization tests for the three verified resolution false positives (#753).

Each of the three verified user-visible false positives between the engine matcher
(IdentifyFeatureGroupClass) and the debug API (resolve_feature) is pinned by a pair of tests
sharing one probe family: one engine assertion, one resolve_feature assertion. The engine-side
assertions pass against current code. The three resolve_feature-side tests now assert the post-#755
target (resolve_feature delegates to the engine seam), so they FAIL against the current, still-diverging
implementation until the delegation lands and engine and debug agree.

This is deliberately NOT the full 17-way divergence matrix from #722: the presentation-only
divergences are erased wholesale when resolve_feature delegates, so they need no individual pins.
"""

import inspect
from abc import abstractmethod
from typing import Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping, PreFilterPlugins
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.user import PluginCollector


ABSTRACT_ONLY_FEATURE = "probe753_abstract_only"
PARENT_CHILD_FEATURE = "probe753_parent_child"
UNAVAILABLE_FEATURE = "probe753_unavailable"


class CfwAbstract753(ComputeFramework):
    """Framework declared by the abstract-only probe."""


class CfwP753(ComputeFramework):
    """Parent/child family framework declared by both parent and child."""


class CfwQ753(ComputeFramework):
    """Parent/child family framework declared by the parent only."""


class CfwUnavailable753(ComputeFramework):
    """Framework whose backing dependency is never installed."""

    @staticmethod
    def is_available() -> bool:
        return False


class AbstractOnlyProbe753(FeatureGroup):
    """Abstract probe: matches only its own feature name and cannot be instantiated."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAbstract753}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == ABSTRACT_ONLY_FEATURE

    @classmethod
    @abstractmethod
    def _probe_hook_753(cls) -> str:
        """Abstract hook that keeps this probe abstract."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class ParentProbe753(FeatureGroup):
    """Parent probe: declares two frameworks and matches only the shared parent/child name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwP753, CfwQ753}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == PARENT_CHILD_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class ChildProbe753(ParentProbe753):
    """Child probe: narrows the declaration to a single framework."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwP753}


class UnavailableOnlyProbe753(FeatureGroup):
    """Probe declaring ONLY the unavailable framework, gated on its own feature name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwUnavailable753}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == UNAVAILABLE_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


# ---------------------------------------------------------------------------
# False positive #1: a lone abstract base matching a unique name
# ---------------------------------------------------------------------------


def test_engine_rejects_lone_abstract_match() -> None:
    """The engine refuses a lone abstract match with the abstract-only error."""
    assert inspect.isabstract(AbstractOnlyProbe753), "fixture must be abstract"

    feature = Feature(ABSTRACT_ONLY_FEATURE)
    accessible_plugins: FeatureGroupEnvironmentMapping = {AbstractOnlyProbe753: {CfwAbstract753}}

    with pytest.raises(ValueError, match="Only abstract feature group base") as exc_info:
        IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )
    assert "Only abstract feature group base(s) matched" in str(exc_info.value)


def test_resolve_feature_rejects_lone_abstract_match() -> None:
    """resolve_feature refuses a lone abstract base, matching the engine's abstract-only rejection."""
    assert inspect.isabstract(AbstractOnlyProbe753), "fixture must be abstract"

    collector = PluginCollector.enabled_feature_groups({AbstractOnlyProbe753})
    result = resolve_feature(ABSTRACT_ONLY_FEATURE, plugin_collector=collector)

    # #755 target: resolve_feature delegates to the engine seam, which excludes abstract bases from the
    # winners and reports the abstract-only error. The uninstantiable base no longer wins on the debug path.
    assert result.feature_group is None
    assert result.error is not None
    assert "Only abstract feature group base(s) matched" in result.error
    # Abstract-only matches are not concrete candidates: criteria_matched is empty.
    assert result.candidates == []


# ---------------------------------------------------------------------------
# False positive #2: a feature whose ONLY declared framework is unavailable
# ---------------------------------------------------------------------------


def test_engine_unavailable_only_framework_raises_no_feature_groups() -> None:
    """The engine drops the unavailable framework and refuses to resolve the feature."""
    assert CfwUnavailable753.is_available() is False  # premise guard

    # Pin strict mode off: the probe is unregistered, and this test pins the availability drop, not
    # strict-registry filtering. Without this, MLODA_PLUGIN_REGISTRY_STRICT=strict would raise early.
    collector = PluginCollector.enabled_feature_groups({UnavailableOnlyProbe753}).set_strict_mode("off")
    accessible = PreFilterPlugins(
        compute_frameworks={CfwUnavailable753}, plugin_collector=collector
    ).get_accessible_plugins()

    # get_cfw_subclasses drops unavailable frameworks; the probe maps to an EMPTY set.
    assert accessible == {UnavailableOnlyProbe753: set()}

    with pytest.raises(ValueError, match="No feature groups found for feature name: 'probe753_unavailable'"):
        IdentifyFeatureGroupClass(
            feature=Feature(UNAVAILABLE_FEATURE),
            accessible_plugins=accessible,
            links=None,
        )


def test_resolve_feature_unavailable_only_framework_fails_closed() -> None:
    """resolve_feature fails closed for a feature whose only declared framework is not installed."""
    assert CfwUnavailable753.is_available() is False  # premise guard

    collector = PluginCollector.enabled_feature_groups({UnavailableOnlyProbe753})
    result = resolve_feature(UNAVAILABLE_FEATURE, plugin_collector=collector)

    # #755 target: the sole declared framework is unavailable, so the engine seam yields no winner and
    # resolve_feature fails closed instead of reading as runnable.
    assert result.feature_group is None
    assert result.error is not None
    assert "No feature groups found for feature name: 'probe753_unavailable'" in result.error
    assert result.supported_compute_frameworks == []


# ---------------------------------------------------------------------------
# False positive #3: framework capability unioned across candidates, credited to the winner
# ---------------------------------------------------------------------------


def test_engine_parent_child_differing_framework_sets_stays_ambiguous() -> None:
    """The engine keeps the differing-framework-set parent/child pair ambiguous."""
    feature = Feature(PARENT_CHILD_FEATURE)
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        ParentProbe753: {CfwP753, CfwQ753},
        ChildProbe753: {CfwP753},
    }

    with pytest.raises(ValueError, match="Multiple feature groups found") as exc_info:
        IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )
    message = str(exc_info.value)
    assert "ParentProbe753" in message
    assert "ChildProbe753" in message


def test_resolve_feature_parent_child_stays_ambiguous() -> None:
    """resolve_feature keeps the differing-framework-set parent/child pair ambiguous, like the engine."""
    collector = PluginCollector.enabled_feature_groups({ParentProbe753, ChildProbe753})
    result = resolve_feature(PARENT_CHILD_FEATURE, plugin_collector=collector)

    # #755 target: resolve_feature delegates to the engine seam. The parent and child declare differing
    # framework sets ({CfwP753, CfwQ753} vs {CfwP753}), which blocks the seam's subclass collapse, so the
    # pair stays ambiguous. The debug path no longer collapses to the child nor unions CfwQ753 (the
    # parent-only framework) into the winner's attribution. Framework credit is per winner only.
    assert result.feature_group is None
    assert result.error is not None
    assert "Multiple feature groups found" in result.error
    assert "ParentProbe753" in result.error
    assert "ChildProbe753" in result.error
    assert set(result.candidates) == {ParentProbe753, ChildProbe753}
    assert CfwQ753.get_class_name() not in result.supported_compute_frameworks
