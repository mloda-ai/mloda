"""Paired engine/diagnostic characterization tests for issue #722 Stage 1.

Each covered divergence between the engine path (IdentifyFeatureGroupClass) and the
diagnostic path (resolve_feature) is pinned by a pair of tests sharing one probe family.
Assertions marked "PINS CURRENT DIVERGENCE" pin behavior that Stage 3 will change;
assertions marked "TARGET CONTRACT" already express the final contract.

Index (divergence -> tests):
- #1  lone abstract match:
      test_engine_rejects_lone_abstract_match / test_resolve_feature_returns_lone_abstract_match
- #3  framework attribution:
      test_engine_parent_child_differing_framework_sets_stays_ambiguous /
      test_resolve_feature_parent_child_unions_framework_attribution
- #4  subclass collapse:
      test_engine_parent_child_differing_framework_sets_stays_ambiguous /
      test_resolve_feature_parent_child_collapses_by_subclass; parity case:
      test_engine_prefers_child_when_framework_sets_are_identical /
      test_resolve_feature_prefers_child_when_framework_sets_are_identical
- #16 raising capability hook:
      test_engine_propagates_capability_hook_runtime_error /
      test_resolve_feature_degrades_capability_hook_error_open
- #17 PropertyValueRejection visibility:
      test_property_value_rejection_is_a_value_error (premise),
      test_engine_swallows_property_value_rejection_silently /
      test_resolve_feature_surfaces_property_value_rejection
"""

import inspect
from abc import abstractmethod
from typing import Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import PropertyValueRejection
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.user import PluginCollector


ABSTRACT_ONLY_FEATURE = "probe722a_abstract_only"
PARENT_CHILD_FEATURE = "probe722a_parent_child"
SAME_FW_FEATURE = "probe722a_same_fw_family"
CAPABILITY_RAISES_FEATURE = "probe722a_capability_raises"
PROPERTY_REJECTION_FEATURE = "probe722a_property_rejection"

CAPABILITY_ERROR_TEXT = "broken capability 722a"
PROPERTY_REJECTION_TEXT = "rejected value 722a: probe option out of range"


class CfwAbstract722A(ComputeFramework):
    """Framework declared by the abstract-only probe."""


class CfwP722A(ComputeFramework):
    """First framework of the parent/child family; declared by parent and child."""


class CfwQ722A(ComputeFramework):
    """Second framework of the parent/child family; declared by the parent only."""


class CfwSame722A(ComputeFramework):
    """Shared framework of the identical-set parity family."""


class CfwCapOne722A(ComputeFramework):
    """First framework declared by the raising-capability probe."""


class CfwCapTwo722A(ComputeFramework):
    """Second framework declared by the raising-capability probe."""


class CfwReject722A(ComputeFramework):
    """Framework declared by the property-rejection probe."""


class AbstractOnlyProbe722A(FeatureGroup):
    """Abstract probe: matches only its own feature name and cannot be instantiated."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwAbstract722A}

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
    def _probe_hook_722a(cls) -> str:
        """Abstract hook that keeps this probe abstract."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class ParentProbe722A(FeatureGroup):
    """Parent probe: declares two frameworks and matches only the shared parent/child name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwP722A, CfwQ722A}

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


class ChildProbe722A(ParentProbe722A):
    """Child probe: narrows the declaration to a single framework."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwP722A}


class SameFwParentProbe722A(FeatureGroup):
    """Parity parent: one framework, matches only the identical-set family name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwSame722A}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == SAME_FW_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class SameFwChildProbe722A(SameFwParentProbe722A):
    """Parity child: inherits the identical framework set from its parent."""


class CapabilityRaisesProbe722A(FeatureGroup):
    """Probe whose capability hook raises, gated on its own feature name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwCapOne722A, CfwCapTwo722A}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == CAPABILITY_RAISES_FEATURE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        # Gated on the probe name so process-global scans in other tests never trip the raise.
        if str(feature_name) == CAPABILITY_RAISES_FEATURE:
            raise RuntimeError(CAPABILITY_ERROR_TEXT)
        return True

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class PropertyRejectionProbe722A(FeatureGroup):
    """Probe whose match hook raises PropertyValueRejection for its own feature name only."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {CfwReject722A}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        # Gated on the probe name so process-global scans in other tests never trip the raise.
        if str(feature_name) == PROPERTY_REJECTION_FEATURE:
            raise PropertyValueRejection(PROPERTY_REJECTION_TEXT)
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


# ---------------------------------------------------------------------------
# Divergence 1: a lone abstract base matching a unique name
# ---------------------------------------------------------------------------


def test_engine_rejects_lone_abstract_match() -> None:
    """The engine refuses a lone abstract match with the abstract-only error."""
    assert inspect.isabstract(AbstractOnlyProbe722A), "fixture must be abstract"

    feature = Feature(ABSTRACT_ONLY_FEATURE)
    accessible_plugins: FeatureGroupEnvironmentMapping = {AbstractOnlyProbe722A: {CfwAbstract722A}}

    # TARGET CONTRACT: an abstract base never wins on the engine path.
    with pytest.raises(ValueError, match="Only abstract feature group base") as exc_info:
        IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )
    assert "Only abstract feature group base(s) matched" in str(exc_info.value)


def test_resolve_feature_returns_lone_abstract_match() -> None:
    """resolve_feature resolves the same lone abstract base cleanly."""
    assert inspect.isabstract(AbstractOnlyProbe722A), "fixture must be abstract"

    collector = PluginCollector.enabled_feature_groups({AbstractOnlyProbe722A})
    result = resolve_feature(ABSTRACT_ONLY_FEATURE, plugin_collector=collector)

    # PINS CURRENT DIVERGENCE (#1): the uninstantiable abstract base wins; target is no abstract winner.
    assert result.feature_group is AbstractOnlyProbe722A
    assert result.error is None
    assert result.candidates == [AbstractOnlyProbe722A]


# ---------------------------------------------------------------------------
# Divergences 3 and 4: parent/child with DIFFERENT declared framework sets
# ---------------------------------------------------------------------------


def test_engine_parent_child_differing_framework_sets_stays_ambiguous() -> None:
    """Engine half of divergences #3/#4: differing framework sets block the subclass collapse."""
    feature = Feature(PARENT_CHILD_FEATURE)
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        ParentProbe722A: {CfwP722A, CfwQ722A},
        ChildProbe722A: {CfwP722A},
    }

    with pytest.raises(ValueError, match="Multiple feature groups found") as exc_info:
        IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )
    message = str(exc_info.value)
    assert "ParentProbe722A" in message
    assert "ChildProbe722A" in message


def test_resolve_feature_parent_child_collapses_by_subclass() -> None:
    """resolve_feature collapses the same parent/child pair the engine calls ambiguous."""
    collector = PluginCollector.enabled_feature_groups({ParentProbe722A, ChildProbe722A})
    result = resolve_feature(PARENT_CHILD_FEATURE, plugin_collector=collector)

    # PINS CURRENT DIVERGENCE (#4): _filter_subclasses collapses purely by issubclass; the engine stays ambiguous.
    assert result.feature_group is ChildProbe722A
    assert result.error is None
    assert set(result.candidates) == {ParentProbe722A, ChildProbe722A}


def test_resolve_feature_parent_child_unions_framework_attribution() -> None:
    """resolve_feature attributes the parent-only framework to the child winner."""
    collector = PluginCollector.enabled_feature_groups({ParentProbe722A, ChildProbe722A})
    result = resolve_feature(PARENT_CHILD_FEATURE, plugin_collector=collector)

    assert result.feature_group is ChildProbe722A
    # PINS CURRENT DIVERGENCE (#3): the capability split is unioned across ALL candidates, so
    # CfwQ722A is reported as supported although the winner ChildProbe722A never declares it.
    assert CfwP722A.get_class_name() in result.supported_compute_frameworks
    assert CfwQ722A.get_class_name() in result.supported_compute_frameworks
    assert result.unsupported_compute_frameworks == []


# ---------------------------------------------------------------------------
# Divergence 4 parity case: parent/child with IDENTICAL framework sets
# ---------------------------------------------------------------------------


def test_engine_prefers_child_when_framework_sets_are_identical() -> None:
    """With identical framework sets, the engine collapses parent/child to the child."""
    feature = Feature(SAME_FW_FEATURE)
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        SameFwParentProbe722A: {CfwSame722A},
        SameFwChildProbe722A: {CfwSame722A},
    }

    identifier = IdentifyFeatureGroupClass(
        feature=feature,
        accessible_plugins=accessible_plugins,
        links=None,
        data_access_collection=None,
    )
    resolved_feature_group, compute_frameworks = identifier.get()
    # TARGET CONTRACT: identical framework sets collapse to the child on both paths.
    assert resolved_feature_group is SameFwChildProbe722A
    assert compute_frameworks == {CfwSame722A}


def test_resolve_feature_prefers_child_when_framework_sets_are_identical() -> None:
    """With identical framework sets, resolve_feature agrees with the engine on the child."""
    collector = PluginCollector.enabled_feature_groups({SameFwParentProbe722A, SameFwChildProbe722A})
    result = resolve_feature(SAME_FW_FEATURE, plugin_collector=collector)

    # TARGET CONTRACT: identical framework sets collapse to the child on both paths.
    assert result.feature_group is SameFwChildProbe722A
    assert result.error is None
    assert result.supported_compute_frameworks == [CfwSame722A.get_class_name()]


# ---------------------------------------------------------------------------
# Divergence 16: a raising supports_compute_framework hook
# ---------------------------------------------------------------------------


def test_engine_propagates_capability_hook_runtime_error() -> None:
    """A raising capability hook aborts the engine with the raw provider error."""
    feature = Feature(CAPABILITY_RAISES_FEATURE)
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        CapabilityRaisesProbe722A: {CfwCapOne722A, CfwCapTwo722A},
    }

    # PINS CURRENT DIVERGENCE (#16): the raw RuntimeError propagates and aborts the run;
    # target is a fail-closed structured provider failure.
    with pytest.raises(RuntimeError, match=CAPABILITY_ERROR_TEXT):
        IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )


def test_resolve_feature_degrades_capability_hook_error_open() -> None:
    """resolve_feature degrades the same raising hook OPEN and reports a clean resolution."""
    collector = PluginCollector.enabled_feature_groups({CapabilityRaisesProbe722A})
    result = resolve_feature(CAPABILITY_RAISES_FEATURE, plugin_collector=collector)

    # PINS CURRENT DIVERGENCE (#16): safe_field degrades the capability split open, so the broken
    # declaration reads as runnable everywhere; target is a fail-closed structured provider failure.
    assert result.feature_group is CapabilityRaisesProbe722A
    assert result.error is None
    assert set(result.supported_compute_frameworks) == {
        CfwCapOne722A.get_class_name(),
        CfwCapTwo722A.get_class_name(),
    }
    assert result.unsupported_compute_frameworks == []


# ---------------------------------------------------------------------------
# Divergence 17: PropertyValueRejection raised by match_feature_group_criteria
# ---------------------------------------------------------------------------


def test_property_value_rejection_is_a_value_error() -> None:
    """Premise: PropertyValueRejection subclasses ValueError, which is why both paths catch it."""
    assert issubclass(PropertyValueRejection, ValueError)


def test_engine_swallows_property_value_rejection_silently() -> None:
    """The engine treats the rejection as a silent non-match; its error omits the rejection text."""
    feature = Feature(PROPERTY_REJECTION_FEATURE)
    accessible_plugins: FeatureGroupEnvironmentMapping = {PropertyRejectionProbe722A: {CfwReject722A}}

    with pytest.raises(ValueError, match="No feature groups found") as exc_info:
        IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )
    message = str(exc_info.value)
    # PINS CURRENT DIVERGENCE (#17): the rejection reason is invisible on the engine path.
    assert PROPERTY_REJECTION_TEXT not in message


def test_resolve_feature_surfaces_property_value_rejection() -> None:
    """resolve_feature records the same rejection and surfaces it in the returned error."""
    collector = PluginCollector.enabled_feature_groups({PropertyRejectionProbe722A})
    result = resolve_feature(PROPERTY_REJECTION_FEATURE, plugin_collector=collector)

    assert result.feature_group is None
    assert result.candidates == []
    assert result.error is not None
    # PINS CURRENT DIVERGENCE (#17): the same rejection is an error string on the diagnostic path.
    assert PROPERTY_REJECTION_TEXT in result.error
    assert "Rejected during matching" in result.error
