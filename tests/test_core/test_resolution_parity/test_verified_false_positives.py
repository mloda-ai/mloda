"""Paired engine/debug characterization tests for the three verified resolution false positives (#753).

Each of the three verified user-visible false positives between the engine matcher
(IdentifyFeatureGroupClass) and the debug API (resolve_feature) is pinned by a pair of tests
sharing one probe family: one engine assertion, one resolve_feature assertion. Every assertion
here PASSES against current code. Assertions marked "PINS CURRENT DIVERGENCE (#755)" pin behavior
that #755 (resolve_feature delegates to the engine seam) will flip to the target contract.

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


def test_resolve_feature_returns_lone_abstract_match() -> None:
    """resolve_feature resolves the same uninstantiable abstract base as the winner."""
    assert inspect.isabstract(AbstractOnlyProbe753), "fixture must be abstract"

    collector = PluginCollector.enabled_feature_groups({AbstractOnlyProbe753})
    result = resolve_feature(ABSTRACT_ONLY_FEATURE, plugin_collector=collector)

    # PINS CURRENT DIVERGENCE (#755): the uninstantiable abstract base wins on the debug path while
    # the engine refuses it; #755 makes resolve_feature reject the abstract winner too.
    assert result.feature_group is AbstractOnlyProbe753
    assert result.error is None
    assert result.candidates == [AbstractOnlyProbe753]


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


def test_resolve_feature_unavailable_only_framework_reads_as_runnable() -> None:
    """resolve_feature reports success for a feature the engine can never run."""
    assert CfwUnavailable753.is_available() is False  # premise guard

    collector = PluginCollector.enabled_feature_groups({UnavailableOnlyProbe753})
    result = resolve_feature(UNAVAILABLE_FEATURE, plugin_collector=collector)

    # PINS CURRENT DIVERGENCE (#755): a feature whose only framework is not installed reads as
    # runnable. The capability split skips unavailable frameworks, so supported and rejected are
    # both empty and the "if not supported and rejected" guard never fires. #755 makes this fail closed.
    assert result.feature_group is UnavailableOnlyProbe753
    assert result.error is None
    assert result.supported_compute_frameworks == []
    assert result.unsupported_compute_frameworks == []


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


def test_resolve_feature_parent_child_unions_framework_attribution() -> None:
    """resolve_feature credits the child winner with the parent-only framework it never declares."""
    collector = PluginCollector.enabled_feature_groups({ParentProbe753, ChildProbe753})
    result = resolve_feature(PARENT_CHILD_FEATURE, plugin_collector=collector)

    # PINS CURRENT DIVERGENCE (#755): _filter_subclasses collapses purely by issubclass so the child
    # wins, and the capability split is unioned across ALL candidates, so CfwQ753 is reported as
    # supported although the winner ChildProbe753 never declares it. #755 attributes per winner only.
    assert result.feature_group is ChildProbe753
    assert result.error is None
    assert set(result.candidates) == {ParentProbe753, ChildProbe753}
    assert CfwP753.get_class_name() in result.supported_compute_frameworks
    assert CfwQ753.get_class_name() in result.supported_compute_frameworks
    assert result.unsupported_compute_frameworks == []
