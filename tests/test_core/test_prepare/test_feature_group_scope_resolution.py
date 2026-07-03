"""Resolution-scope tests for IdentifyFeatureGroupClass (issue #508).

Two source feature groups (A and B) both match the shared feature name
"subject_token". Requesting it unscoped is ambiguous ("Multiple feature groups
found"). The new per-feature scope disambiguates resolution to a single source,
by class identity or by class-name string, without changing feature identity.

Follows the construction conventions in test_identify_feature_group_error_message.py.
"""

from typing import Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass


class MockComputeFramework(ComputeFramework):
    """Mock compute framework for testing."""


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
        IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=_both_sources(),
            links=None,
            data_access_collection=None,
        )


def test_scope_class_resolves_uniquely_by_identity() -> None:
    """A class-object scope resolves uniquely to that class (matched by identity)."""
    feature = Feature("subject_token", feature_group=ScopeSourceA)

    identifier = IdentifyFeatureGroupClass(
        feature=feature,
        accessible_plugins=_both_sources(),
        links=None,
        data_access_collection=None,
    )
    resolved_feature_group, _compute_frameworks = identifier.get()
    assert resolved_feature_group is ScopeSourceA


def test_scope_string_resolves_uniquely_by_class_name() -> None:
    """A class-name string scope resolves uniquely to the matching class."""
    feature = Feature("subject_token", feature_group=ScopeSourceA.get_class_name())

    identifier = IdentifyFeatureGroupClass(
        feature=feature,
        accessible_plugins=_both_sources(),
        links=None,
        data_access_collection=None,
    )
    resolved_feature_group, _compute_frameworks = identifier.get()
    assert resolved_feature_group is ScopeSourceA


def test_unknown_scope_raises_no_feature_groups_found() -> None:
    """A scope naming no accessible feature group raises 'No feature groups found'."""
    feature = Feature("subject_token", feature_group="CompletelyUnknownScope")

    with pytest.raises(ValueError, match="No feature groups found") as exc_info:
        IdentifyFeatureGroupClass(
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
        IdentifyFeatureGroupClass(
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
        IdentifyFeatureGroupClass(
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

    identifier = IdentifyFeatureGroupClass(
        feature=feature,
        accessible_plugins=accessible_plugins,
        links=None,
        data_access_collection=None,
    )
    resolved_feature_group, _compute_frameworks = identifier.get()
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

    identifier = IdentifyFeatureGroupClass(
        feature=feature,
        accessible_plugins=accessible_plugins,
        links=None,
        data_access_collection=None,
    )
    resolved_feature_group, _compute_frameworks = identifier.get()
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
        IdentifyFeatureGroupClass(
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
        IdentifyFeatureGroupClass(
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
# String-form scopes match the exact class name only (no subclass matching)
# ---------------------------------------------------------------------------


def test_base_class_name_string_scope_does_not_match_subclass() -> None:
    """Pinning test: a STRING scope matches the exact class name only, never subclasses.

    This pins the INTENDED asymmetry between the two scope forms: a class-object
    scope uses issubclass matching (see
    test_base_class_scope_resolves_to_accessible_subclass), while the string form
    'ScopeSourceA' compares against get_class_name() exactly. With only the
    subclass ScopeSourceASub accessible, the base-name string scope therefore
    matches nothing and raises 'No feature groups found', naming the scope.

    This is a characterization test of current behaviour; it may already pass.
    """
    feature = Feature("subject_token", feature_group=ScopeSourceA.get_class_name())
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        ScopeSourceASub: {MockComputeFramework},
    }

    with pytest.raises(ValueError, match="No feature groups found") as exc_info:
        IdentifyFeatureGroupClass(
            feature=feature,
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )
    assert ScopeSourceA.get_class_name() in str(exc_info.value)
