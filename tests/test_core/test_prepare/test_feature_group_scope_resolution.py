"""Resolution-scope tests for IdentifyFeatureGroupClass (issues #508, #682).

Two source feature groups (A and B) both match the shared feature name
"subject_token". Requesting it unscoped is ambiguous ("Multiple feature groups
found"). The per-feature scope disambiguates resolution to a single source,
by class identity or by class-name string, without changing feature identity.

Both scope forms match by ancestry (#682): a candidate matches when the scoped
class is in its MRO, or, for the string form, when any class in its MRO (below
the root FeatureGroup base) carries the scoped name. filter_subclasses then
prefers the most specific candidate.

Follows the construction conventions in test_identify_feature_group_error_message.py.
"""

import inspect
from abc import abstractmethod
from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import matches_feature_group_scope
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise
from mloda.provider import BaseInputData, DataCreator, FeatureSet
from mloda.user import PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup


class MockComputeFramework(ComputeFramework):
    """Mock compute framework for testing."""


class SecondMockComputeFramework(ComputeFramework):
    """A second mock compute framework, for rival subclasses of different frameworks."""


class ScopeSourceA(FeatureGroup):
    """Source A: matches the shared "subject_token" plus its own "scoping_value_a"."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"subject_token", "scoping_value_a"}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        name = str(feature_name)
        return name in {"subject_token", "scoping_value_a"}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class ScopeSourceB(FeatureGroup):
    """Source B: matches the shared "subject_token" plus its own "scoping_value_b"."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"subject_token", "scoping_value_b"}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        name = str(feature_name)
        return name in {"subject_token", "scoping_value_b"}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class InaccessibleScopeSource(FeatureGroup):
    """A FeatureGroup that matches "subject_token" but is never added to accessible_plugins."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"subject_token"}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == "subject_token"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class _DupNameBase(FeatureGroup):
    """Base for two feature groups that will share the identical class name."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"subject_token"}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == "subject_token"

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


def _both_sources() -> FeatureGroupEnvironmentMapping:
    return {
        ScopeSourceA: {MockComputeFramework},
        ScopeSourceB: {MockComputeFramework},
    }


def test_unscoped_shared_name_is_ambiguous() -> None:
    """Characterization: unscoped "subject_token" with A and B both accessible is ambiguous."""
    feature = Feature("subject_token")

    with pytest.raises(ValueError, match="Multiple feature groups found"):
        evaluate_or_raise(
            feature=feature,
            accessible_plugins=_both_sources(),
            links=None,
            data_access_collection=None,
        )


def test_scope_class_resolves_uniquely_by_identity() -> None:
    """A class-object scope resolves uniquely to that class (matched by identity)."""
    feature = Feature("subject_token", feature_group=ScopeSourceA)

    identifier = evaluate_or_raise(
        feature=feature,
        accessible_plugins=_both_sources(),
        links=None,
        data_access_collection=None,
    )
    resolved_feature_group, _compute_frameworks = next(iter(identifier.identified.items()))
    assert resolved_feature_group is ScopeSourceA


def test_scope_string_resolves_uniquely_by_class_name() -> None:
    """A class-name string scope resolves uniquely to the matching class."""
    feature = Feature("subject_token", feature_group=ScopeSourceA.get_class_name())

    identifier = evaluate_or_raise(
        feature=feature,
        accessible_plugins=_both_sources(),
        links=None,
        data_access_collection=None,
    )
    resolved_feature_group, _compute_frameworks = next(iter(identifier.identified.items()))
    assert resolved_feature_group is ScopeSourceA


def test_unknown_scope_raises_no_feature_groups_found() -> None:
    """A scope naming no accessible feature group raises 'No feature groups found'."""
    feature = Feature("subject_token", feature_group="CompletelyUnknownScope")

    with pytest.raises(ValueError, match="No feature groups found") as exc_info:
        evaluate_or_raise(
            feature=feature,
            accessible_plugins=_both_sources(),
            links=None,
            data_access_collection=None,
        )
    assert "CompletelyUnknownScope" in str(exc_info.value)


def test_class_scope_pointing_at_inaccessible_group_raises_no_feature_groups_found() -> None:
    """A class-object scope for a FeatureGroup absent from accessible_plugins raises no-match.

    The class matches the feature name but is not registered, so scope filtering
    eliminates every accessible candidate. The error must be 'No feature groups
    found' and name the scoped class so the missing registration is debuggable.
    """
    feature = Feature("subject_token", feature_group=InaccessibleScopeSource)

    with pytest.raises(ValueError, match="No feature groups found") as exc_info:
        evaluate_or_raise(
            feature=feature,
            accessible_plugins=_both_sources(),
            links=None,
            data_access_collection=None,
        )
    assert InaccessibleScopeSource.get_class_name() in str(exc_info.value)


def test_string_scope_name_collision_reports_scope_in_multiple_found() -> None:
    """A string scope that matches two identically-named groups must name the scope.

    Two distinct FeatureGroup classes share the class name 'DupNameSource' and
    both match 'subject_token'. A string scope of 'DupNameSource' therefore stays
    ambiguous. The 'Multiple feature groups found' diagnostics must explicitly
    call out the requested scope (as the no-match branch already does), so the
    string-name collision is debuggable rather than looking like a plain
    unscoped ambiguity.
    """
    dup_a = type("DupNameSource", (_DupNameBase,), {})
    dup_b = type("DupNameSource", (_DupNameBase,), {})
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        dup_a: {MockComputeFramework},
        dup_b: {MockComputeFramework},
    }

    feature = Feature("subject_token", feature_group="DupNameSource")

    with pytest.raises(ValueError, match="Multiple feature groups found") as exc_info:
        evaluate_or_raise(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )
    message = str(exc_info.value)
    assert "Scoped to feature group: 'DupNameSource'" in message


# ---------------------------------------------------------------------------
# issubclass matching for class-object scopes
# ---------------------------------------------------------------------------


class ScopeSourceASub(ScopeSourceA):
    """Subclass of ScopeSourceA; inherits its name matching."""


def test_base_class_scope_resolves_to_accessible_subclass() -> None:
    """A base-class scope matches subclasses of the scoped class.

    Only the subclass ScopeSourceASub is accessible. Scoping to its base
    ScopeSourceA must resolve to the subclass (issubclass matching) instead of
    raising 'No feature groups found'. ScopeSourceB stays filtered out because
    it is unrelated to the scope.
    """
    feature = Feature("subject_token", feature_group=ScopeSourceA)
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        ScopeSourceASub: {MockComputeFramework},
        ScopeSourceB: {MockComputeFramework},
    }

    identifier = evaluate_or_raise(
        feature=feature,
        accessible_plugins=accessible_plugins,
        links=None,
        data_access_collection=None,
    )
    resolved_feature_group, _compute_frameworks = next(iter(identifier.identified.items()))
    assert resolved_feature_group is ScopeSourceASub


def test_base_class_scope_prefers_subclass_when_both_accessible() -> None:
    """With base AND subclass accessible, a base-class scope resolves to the subclass.

    issubclass matching keeps both candidates in the scope filter; the existing
    filter_subclasses preference (same compute-framework set) then drops the
    base in favour of the subclass. Resolving to the base is wrong.
    """
    feature = Feature("subject_token", feature_group=ScopeSourceA)
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        ScopeSourceA: {MockComputeFramework},
        ScopeSourceASub: {MockComputeFramework},
    }

    identifier = evaluate_or_raise(
        feature=feature,
        accessible_plugins=accessible_plugins,
        links=None,
        data_access_collection=None,
    )
    resolved_feature_group, _compute_frameworks = next(iter(identifier.identified.items()))
    assert resolved_feature_group is ScopeSourceASub


# ---------------------------------------------------------------------------
# Capability-rejection error names the scope
# ---------------------------------------------------------------------------


class ScopedCapabilityFw(ComputeFramework):
    """Compute framework rejected by RejectAllScopedSource at match time."""


class RejectAllScopedSource(FeatureGroup):
    """Matches "subject_token" but declares every compute framework unsupported."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {ScopedCapabilityFw}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == "subject_token"

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


def test_capability_rejection_error_names_the_scope() -> None:
    """When every framework of the scoped group is capability-rejected, the error names the scope.

    The scoped group matches the feature name, but supports_compute_framework
    rejects all of its frameworks, so the no-match error takes the capability
    branch. That message must still call out the requested scope, otherwise the
    scoped request looks like a plain capability failure.
    """
    feature = Feature("subject_token", feature_group=RejectAllScopedSource)
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        RejectAllScopedSource: {ScopedCapabilityFw},
    }

    with pytest.raises(ValueError) as exc_info:
        evaluate_or_raise(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )
    message = str(exc_info.value)
    # Prove the capability branch was taken (guards against a fixture bug).
    assert "Unsupported compute framework" in message
    assert "Scoped to feature group: 'RejectAllScopedSource'" in message


# ---------------------------------------------------------------------------
# Ambiguous-error message ordering: scope callout before the trailing URL
# ---------------------------------------------------------------------------


def test_scoped_ambiguity_callout_precedes_troubleshooting_url() -> None:
    """In the scoped 'Multiple feature groups found' error, the URL stays last.

    The scope callout must appear BEFORE the troubleshooting URL, and the
    message's last line must end with the URL so terminals auto-link it.
    Appending the callout after the URL breaks both.
    """
    dup_a = type("DupNameSource", (_DupNameBase,), {})
    dup_b = type("DupNameSource", (_DupNameBase,), {})
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        dup_a: {MockComputeFramework},
        dup_b: {MockComputeFramework},
    }

    feature = Feature("subject_token", feature_group="DupNameSource")

    with pytest.raises(ValueError, match="Multiple feature groups found") as exc_info:
        evaluate_or_raise(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )
    message = str(exc_info.value)
    url = "https://mloda-ai.github.io/mloda/in_depth/troubleshooting/feature-group-resolution-errors/"
    assert "Scoped to feature group: 'DupNameSource'" in message
    assert message.index("Scoped to feature group:") < message.index(url), (
        f"Scope callout must precede the troubleshooting URL, but got: {message}"
    )
    assert message.splitlines()[-1].endswith(url), (
        f"The message's last line must end with the troubleshooting URL, but got: {message}"
    )


# ---------------------------------------------------------------------------
# String-form scopes match by ancestry, like the class-object form (issue #682)
#
# A candidate matches a string scope when any class in its FeatureGroup ancestry
# (its MRO, excluding the root FeatureGroup base) carries that class name. This
# lets a JSON config name an abstract family base and still reach the concrete
# per-framework subclass, instead of hard-coding a compute-framework leaf class.
# ---------------------------------------------------------------------------


def test_base_class_name_string_scope_resolves_to_accessible_subclass() -> None:
    """A base-name STRING scope matches subclasses of the named class.

    The string form now carries the same subclass-preferring semantics as the
    class-object form (see test_base_class_scope_resolves_to_accessible_subclass).
    With only the subclass ScopeSourceASub accessible, the base-name string scope
    'ScopeSourceA' resolves to that subclass.
    """
    feature = Feature("subject_token", feature_group=ScopeSourceA.get_class_name())
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        ScopeSourceASub: {MockComputeFramework},
    }

    identifier = evaluate_or_raise(
        feature=feature,
        accessible_plugins=accessible_plugins,
        links=None,
        data_access_collection=None,
    )
    resolved_feature_group, _compute_frameworks = next(iter(identifier.identified.items()))
    assert resolved_feature_group is ScopeSourceASub


def test_base_class_name_string_scope_prefers_subclass_when_both_accessible() -> None:
    """With base AND subclass accessible, a base-name string scope resolves to the subclass.

    Mirrors test_base_class_scope_prefers_subclass_when_both_accessible for the
    string form: ancestry matching keeps both candidates, then filter_subclasses
    (same compute-framework set) drops the base in favour of the subclass.
    """
    feature = Feature("subject_token", feature_group=ScopeSourceA.get_class_name())
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        ScopeSourceA: {MockComputeFramework},
        ScopeSourceASub: {MockComputeFramework},
    }

    identifier = evaluate_or_raise(
        feature=feature,
        accessible_plugins=accessible_plugins,
        links=None,
        data_access_collection=None,
    )
    resolved_feature_group, _compute_frameworks = next(iter(identifier.identified.items()))
    assert resolved_feature_group is ScopeSourceASub


def test_string_scope_does_not_widen_to_unrelated_groups() -> None:
    """Ancestry matching stays narrow: non-descendants are still filtered out.

    ScopeSourceB also matches 'subject_token' but is unrelated to the scoped
    class, so the base-name string scope 'ScopeSourceA' must exclude it. Widening
    would turn this into 'Multiple feature groups found'.
    """
    feature = Feature("subject_token", feature_group=ScopeSourceA.get_class_name())
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        ScopeSourceASub: {MockComputeFramework},
        ScopeSourceB: {MockComputeFramework},
    }

    identifier = evaluate_or_raise(
        feature=feature,
        accessible_plugins=accessible_plugins,
        links=None,
        data_access_collection=None,
    )
    resolved_feature_group, _compute_frameworks = next(iter(identifier.identified.items()))
    assert resolved_feature_group is ScopeSourceASub


class ScopeAbstractFamilyBase(FeatureGroup):
    """Abstract family base: matches "subject_token" but cannot be instantiated."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"subject_token"}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == "subject_token"

    @classmethod
    @abstractmethod
    def _family_hook(cls) -> str:
        """Abstract hook that makes this base abstract."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class ScopeConcreteFamilyMember(ScopeAbstractFamilyBase):
    """The concrete implementation of the abstract family base."""

    @classmethod
    def _family_hook(cls) -> str:
        return "concrete"


def test_abstract_base_name_string_scope_resolves_to_concrete_subclass() -> None:
    """A string scope naming an ABSTRACT family base resolves to the concrete subclass.

    This is the config use case of issue #682: a JSON config can only carry a
    class-name string, so naming the abstract family base must reach the concrete
    implementation instead of forcing the config to name a framework-specific leaf.
    """
    assert inspect.isabstract(ScopeAbstractFamilyBase), "fixture must be abstract"
    assert not inspect.isabstract(ScopeConcreteFamilyMember), "fixture must be concrete"

    feature = Feature("subject_token", feature_group=ScopeAbstractFamilyBase.get_class_name())
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        ScopeAbstractFamilyBase: {MockComputeFramework},
        ScopeConcreteFamilyMember: {MockComputeFramework},
        ScopeSourceB: {MockComputeFramework},
    }

    identifier = evaluate_or_raise(
        feature=feature,
        accessible_plugins=accessible_plugins,
        links=None,
        data_access_collection=None,
    )
    resolved_feature_group, _compute_frameworks = next(iter(identifier.identified.items()))
    assert resolved_feature_group is ScopeConcreteFamilyMember


def test_string_scope_matching_two_same_named_bases_stays_ambiguous() -> None:
    """Ancestry matching does not disambiguate a class-name collision.

    Two distinct base classes share the name 'DupNameAncestorBase' (different
    "modules"), each with its own concrete subclass. The string scope matches both
    subclasses through their ancestry, so resolution stays ambiguous and must name
    the scope.
    """
    dup_base_a = type("DupNameAncestorBase", (_DupNameBase,), {})
    dup_base_b = type("DupNameAncestorBase", (_DupNameBase,), {})
    sub_a = type("DupNameAncestorSubA", (dup_base_a,), {})
    sub_b = type("DupNameAncestorSubB", (dup_base_b,), {})
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        sub_a: {MockComputeFramework},
        sub_b: {MockComputeFramework},
    }

    feature = Feature("subject_token", feature_group="DupNameAncestorBase")

    with pytest.raises(ValueError, match="Multiple feature groups found") as exc_info:
        evaluate_or_raise(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )
    assert "Scoped to feature group: 'DupNameAncestorBase'" in str(exc_info.value)


def test_base_name_string_scope_with_two_sibling_subclasses_stays_ambiguous() -> None:
    """A base-name string scope whose siblings do not narrow to one candidate still raises.

    Both subclasses match through the base's name with the same compute-framework
    set, and neither is a subclass of the other, so filter_subclasses cannot pick
    a winner. Resolution stays ambiguous.
    """
    sibling_one = type("ScopeSiblingOne", (_DupNameBase,), {})
    sibling_two = type("ScopeSiblingTwo", (_DupNameBase,), {})
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        sibling_one: {MockComputeFramework},
        sibling_two: {MockComputeFramework},
    }

    feature = Feature("subject_token", feature_group=_DupNameBase.get_class_name())

    with pytest.raises(ValueError, match="Multiple feature groups found") as exc_info:
        evaluate_or_raise(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )
    assert "Scoped to feature group: '_DupNameBase'" in str(exc_info.value)


def test_base_name_string_scope_with_differing_framework_siblings_stays_ambiguous() -> None:
    """Rival subclasses on DIFFERENT compute frameworks stay ambiguous under a base-name scope.

    The realistic multi-framework shape: two concrete per-framework siblings of one
    family base (as PandasAggregatedFeatureGroup and PyArrowAggregatedFeatureGroup
    are), both enabled. Their compute-framework sets differ, so filter_subclasses
    takes its early-continue branch and cannot pick a winner, and neither is a
    subclass of the other. Resolution must raise instead of silently choosing one.
    """
    framework_sibling_one = type("ScopeFrameworkSiblingOne", (_DupNameBase,), {})
    framework_sibling_two = type("ScopeFrameworkSiblingTwo", (_DupNameBase,), {})
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        framework_sibling_one: {MockComputeFramework},
        framework_sibling_two: {SecondMockComputeFramework},
    }

    feature = Feature("subject_token", feature_group=_DupNameBase.get_class_name())

    with pytest.raises(ValueError, match="Multiple feature groups found") as exc_info:
        evaluate_or_raise(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )
    message = str(exc_info.value)
    assert "Scoped to feature group: '_DupNameBase'" in message
    assert "ScopeFrameworkSiblingOne" in message
    assert "ScopeFrameworkSiblingTwo" in message


# ---------------------------------------------------------------------------
# The ancestry predicate never treats the root FeatureGroup base as a wildcard
# ---------------------------------------------------------------------------


def test_root_feature_group_base_name_is_not_a_wildcard_scope() -> None:
    """The root base name matches nothing, even though it is in every candidate's MRO.

    Pins the root exclusion in the ancestry walk directly: no public path reaches
    it, because Feature() rejects the root base name before resolution.
    """
    assert matches_feature_group_scope(ScopeSourceA, FeatureGroup.get_class_name()) is False
    assert matches_feature_group_scope(ScopeConcreteFamilyMember, FeatureGroup.get_class_name()) is False


# ---------------------------------------------------------------------------
# The scoped abstract base need not be an accessible plugin at all (#682/#688)
# ---------------------------------------------------------------------------


def test_abstract_base_name_string_scope_resolves_when_base_is_not_accessible() -> None:
    """Characterization: a string scope naming a base that is NOT a key of accessible_plugins resolves.

    PluginCollector.enabled_feature_groups({ConcreteMember}) leaves the abstract
    family base out of accessible_plugins entirely. The name is matched against the
    candidate's MRO, not against the accessible keys, so the concrete member must
    still resolve. A refactor to a lookup over accessible keys would pass every
    other scope test and break exactly this contract.
    """
    assert inspect.isabstract(ScopeAbstractFamilyBase), "fixture must be abstract"

    feature = Feature("subject_token", feature_group=ScopeAbstractFamilyBase.get_class_name())
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        ScopeConcreteFamilyMember: {MockComputeFramework},
        ScopeSourceB: {MockComputeFramework},
    }
    assert ScopeAbstractFamilyBase not in accessible_plugins, "the scoped base must be absent from the mapping"

    identifier = evaluate_or_raise(
        feature=feature,
        accessible_plugins=accessible_plugins,
        links=None,
        data_access_collection=None,
    )
    resolved_feature_group, _compute_frameworks = next(iter(identifier.identified.items()))
    assert resolved_feature_group is ScopeConcreteFamilyMember


# ---------------------------------------------------------------------------
# A compute-framework pin narrows a base-name scope POSITIVELY (#682/#688)
#
# The negative is pinned by test_base_name_string_scope_with_differing_framework_
# siblings_stays_ambiguous. These pin the documented escape hatch from that
# ambiguity: pin compute_frameworks on the Feature (Python only). See
# docs/docs/in_depth/troubleshooting/feature-group-resolution-errors.md.
#
# The framework classes below are new because Feature(compute_framework=...) takes
# a NAME and resolves it across every ComputeFramework subclass in the process;
# "MockComputeFramework" is defined in several test modules, so a pin by that name
# would be resolved non-deterministically under xdist.
# ---------------------------------------------------------------------------


class ScopePinnedPrimaryFramework(ComputeFramework):
    """Uniquely named framework, pinnable by name from a Feature."""


class ScopePinnedSecondaryFramework(ComputeFramework):
    """A rival uniquely named framework, for the second concrete family member."""


class ScopeConcreteFamilyMemberSecondary(ScopeAbstractFamilyBase):
    """A second concrete member of the same family, enabled on the other framework."""

    @classmethod
    def _family_hook(cls) -> str:
        return "concrete-secondary"


def _framework_split_family() -> FeatureGroupEnvironmentMapping:
    """Two concrete siblings of one abstract family base, each on its own framework."""
    return {
        ScopeConcreteFamilyMember: {ScopePinnedPrimaryFramework},
        ScopeConcreteFamilyMemberSecondary: {ScopePinnedSecondaryFramework},
    }


def test_framework_pin_narrows_base_name_scope_to_primary_member() -> None:
    """Characterization: a framework pin resolves the base-name scope ambiguity to that framework's member.

    Premise guard first: unpinned, the two per-framework siblings of the scoped base
    stay ambiguous. Pinning the framework must then select the sibling enabled on it.
    This is the documented escape hatch, and a refactor that stopped applying the
    scope filter and the framework filter together would break it.
    """
    unpinned = Feature("subject_token", feature_group=ScopeAbstractFamilyBase.get_class_name())
    with pytest.raises(ValueError, match="Multiple feature groups found"):
        evaluate_or_raise(
            feature=unpinned,
            accessible_plugins=_framework_split_family(),
            links=None,
            data_access_collection=None,
        )

    feature = Feature(
        "subject_token",
        feature_group=ScopeAbstractFamilyBase.get_class_name(),
        compute_framework=ScopePinnedPrimaryFramework.get_class_name(),
    )

    identifier = evaluate_or_raise(
        feature=feature,
        accessible_plugins=_framework_split_family(),
        links=None,
        data_access_collection=None,
    )
    resolved_feature_group, compute_frameworks = next(iter(identifier.identified.items()))
    assert resolved_feature_group is ScopeConcreteFamilyMember
    assert compute_frameworks == {ScopePinnedPrimaryFramework}


def test_framework_pin_narrows_base_name_scope_to_secondary_member() -> None:
    """Characterization: pinning the OTHER framework selects the other member of the same family.

    The mirror of the primary case, so the pin is proven to select by framework
    rather than to hide a fixed iteration order over accessible_plugins.
    """
    feature = Feature(
        "subject_token",
        feature_group=ScopeAbstractFamilyBase.get_class_name(),
        compute_framework=ScopePinnedSecondaryFramework.get_class_name(),
    )

    identifier = evaluate_or_raise(
        feature=feature,
        accessible_plugins=_framework_split_family(),
        links=None,
        data_access_collection=None,
    )
    resolved_feature_group, compute_frameworks = next(iter(identifier.identified.items()))
    assert resolved_feature_group is ScopeConcreteFamilyMemberSecondary
    assert compute_frameworks == {ScopePinnedSecondaryFramework}


# ---------------------------------------------------------------------------
# End to end through the PYTHON surface with the real shipped family (#682/#688)
#
# The config/JSON equivalent lives in
# tests/test_core/test_api/feature_config/test_feature_config_feature_group_scope.py.
# ---------------------------------------------------------------------------


class ScopePythonAggregationSource(FeatureGroup):
    """Source data for the Python-surface aggregated-family scope test."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"scope_python_sales"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"scope_python_sales": [10, 20, 30, 40]}


def test_end2end_python_feature_abstract_family_base_scope_resolves_to_pandas_subclass() -> None:
    """Characterization: Feature(feature_group="AggregatedFeatureGroup") runs on the Pandas subclass.

    The allowlist enables only the source and the concrete Pandas member, so the
    scoped abstract base is genuinely not a key of accessible_plugins. Naming it
    must still reach the concrete subclass through the MRO walk and compute the
    aggregation, without the caller hard-coding a framework-specific leaf class.
    """
    assert inspect.isabstract(AggregatedFeatureGroup), "the scoped family base must stay abstract"

    feature = Feature("scope_python_sales__sum_aggr", feature_group=AggregatedFeatureGroup.get_class_name())

    results = list(
        mloda.run_all(
            [feature],
            compute_frameworks={PandasDataFrame},
            plugin_collector=PluginCollector.enabled_feature_groups(
                {ScopePythonAggregationSource, PandasAggregatedFeatureGroup}
            ),
        )
    )

    aggregated = [df for df in results if "scope_python_sales__sum_aggr" in df.columns]
    assert len(aggregated) == 1
    assert aggregated[0]["scope_python_sales__sum_aggr"].iloc[0] == 100
