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
import logging
import sqlite3
from pathlib import Path
from collections.abc import Callable
from abc import abstractmethod
from typing import Any, ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.abstract_plugins.components.input_data.base_input_data import RESERVED_READER_OPTION_KEY
from mloda.core.prepare.identify_feature_group import (
    FeatureResolutionError,
    IdentifyFeatureGroupClass,
    matches_feature_group_scope,
)
from tests.helpers.plugin_stubs import StubFeatureGroup, make_fg
from tests.test_core.test_prepare.identify_seam import evaluate_or_raise
from mloda.provider import BaseInputData, DataCreator, FeatureSet
from mloda.user import Credential, DataAccessCollection, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.input_data.read_db_feature import ReadDBFeature
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader
from mloda_plugins.feature_group.input_data.read_files.parquet import ParquetReader
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup


class MockComputeFramework(ComputeFramework):
    """Mock compute framework for testing."""


class SecondMockComputeFramework(ComputeFramework):
    """A second mock compute framework, for rival subclasses of different frameworks."""


class ScopeSourceA(StubFeatureGroup):
    """Source A: matches the shared "subject_token" plus its own "scoping_value_a"."""

    MATCHED_NAMES: ClassVar[frozenset[str]] = frozenset({"subject_token", "scoping_value_a"})
    SUPPORTED_NAMES: ClassVar[frozenset[str]] = MATCHED_NAMES


class ScopeSourceB(StubFeatureGroup):
    """Source B: matches the shared "subject_token" plus its own "scoping_value_b"."""

    MATCHED_NAMES: ClassVar[frozenset[str]] = frozenset({"subject_token", "scoping_value_b"})
    SUPPORTED_NAMES: ClassVar[frozenset[str]] = MATCHED_NAMES


class ProbedScopeSourceB(ScopeSourceB):
    """Source B variant that counts its criteria probes."""

    MATCHER_CALLS: ClassVar[int] = 0

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        ProbedScopeSourceB.MATCHER_CALLS += 1
        return super().match_feature_group_criteria(feature_name, options, data_access_collection)


InaccessibleScopeSource = make_fg(
    "InaccessibleScopeSource",
    matches="subject_token",
    supported_names="subject_token",
    doc="A FeatureGroup that is never added to accessible_plugins.",
)

_DupNameBase = make_fg(
    "_DupNameBase",
    matches="subject_token",
    supported_names="subject_token",
    doc="Base for two feature groups that will share the identical class name.",
)


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


def test_scope_pin_never_probes_the_out_of_scope_candidate() -> None:
    """The scope gate runs before the criteria probe: the unpinned candidate's matcher is never called."""
    ProbedScopeSourceB.MATCHER_CALLS = 0

    identifier = evaluate_or_raise(
        feature=Feature("subject_token", feature_group=ScopeSourceA),
        accessible_plugins={ScopeSourceA: {MockComputeFramework}, ProbedScopeSourceB: {MockComputeFramework}},
        links=None,
        data_access_collection=None,
    )

    assert next(iter(identifier.identified)) is ScopeSourceA
    assert ProbedScopeSourceB.MATCHER_CALLS == 0


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


class ScopeSourceASubSub(ScopeSourceASub):
    """Grandchild of ScopeSourceA; inherits its name matching."""


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
    assert identifier.specialized_from == ()


def test_base_class_scope_prefers_subclass_when_both_accessible() -> None:
    """With base AND subclass accessible, a base-class scope resolves to the subclass.

    issubclass matching keeps both candidates in the scope filter; the existing
    filter_subclasses preference then drops the base in favour of the subclass.
    Resolving to the base is wrong.
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
    assert identifier.specialized_from == (ScopeSourceA,)


@pytest.mark.parametrize(
    ("accessible_plugins", "expected_winner", "expected_specialized_from"),
    [
        pytest.param(
            {
                ScopeSourceA: {MockComputeFramework},
                ScopeSourceASub: {MockComputeFramework},
                ScopeSourceASubSub: {MockComputeFramework},
            },
            ScopeSourceASubSub,
            (ScopeSourceA, ScopeSourceASub),
            id="grandparent_chain",
        ),
    ],
)
def test_subclass_replaces_parents_and_records_them(
    accessible_plugins: FeatureGroupEnvironmentMapping,
    expected_winner: type[FeatureGroup],
    expected_specialized_from: tuple[type[FeatureGroup], ...],
) -> None:
    """The most specific subclass wins and names every replaced ancestor."""
    identifier = evaluate_or_raise(
        feature=Feature("subject_token", feature_group=ScopeSourceA),
        accessible_plugins=accessible_plugins,
        links=None,
        data_access_collection=None,
    )
    resolved_feature_group, _compute_frameworks = next(iter(identifier.identified.items()))
    assert resolved_feature_group is expected_winner
    assert identifier.specialized_from == expected_specialized_from


def test_no_single_winner_has_empty_specialized_from() -> None:
    """Unrelated rivals on differing framework sets stay ambiguous, so nothing is recorded."""
    feature = Feature("subject_token")
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        ScopeSourceASub: {MockComputeFramework},
        ScopeSourceB: {SecondMockComputeFramework},
    }

    result = IdentifyFeatureGroupClass.evaluate(feature, accessible_plugins, None, None)

    assert result.failure_kind == "multiple"
    assert result.specialized_from == ()


# ---------------------------------------------------------------------------
# Capability-rejection error names the scope
# ---------------------------------------------------------------------------


class ScopedCapabilityFw(ComputeFramework):
    """Compute framework rejected by RejectAllScopedSource at match time."""


class RejectAllScopedSource(StubFeatureGroup):
    """Matches "subject_token" but declares every compute framework unsupported."""

    MATCHED_NAMES: ClassVar[frozenset[str]] = frozenset({"subject_token"})
    FRAMEWORK_RULE: ClassVar[set[type[ComputeFramework]]] = {ScopedCapabilityFw}

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        return False


def test_capability_rejection_error_names_the_scope() -> None:
    """When every framework of the scoped group is capability-rejected, the error names the scope.

    The scoped group matches the feature name, but supports_compute_framework
    rejects all of its frameworks, so the no-match error carries a capability
    near-miss line. That message must still call out the requested scope, otherwise
    the scoped request looks like a plain capability failure.
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
    # Prove the capability near-miss was recorded (guards against a fixture bug).
    assert "Feature group(s) eliminated while matching 'subject_token':" in message
    assert "supports_compute_framework rejected ['ScopedCapabilityFw']" in message
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
    drops the base in favour of the subclass.
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


class ScopeAbstractFamilyBase(StubFeatureGroup):
    """Abstract family base: matches "subject_token" but cannot be instantiated."""

    MATCHED_NAMES: ClassVar[frozenset[str]] = frozenset({"subject_token"})
    SUPPORTED_NAMES: ClassVar[frozenset[str]] = frozenset({"subject_token"})

    @classmethod
    @abstractmethod
    def _family_hook(cls) -> str:
        """Abstract hook that makes this base abstract."""


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
    are), both enabled. The subclass preference drops the concrete base, but the siblings are
    unrelated by inheritance, so resolution must raise instead of silently choosing one.
    """
    framework_sibling_one = type("ScopeFrameworkSiblingOne", (_DupNameBase,), {})
    framework_sibling_two = type("ScopeFrameworkSiblingTwo", (_DupNameBase,), {})
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        _DupNameBase: {MockComputeFramework, SecondMockComputeFramework},
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
    assert "- _DupNameBase (" not in message


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
    def input_data(cls) -> BaseInputData | None:
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
            compute_frameworks=[PandasDataFrame],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {ScopePythonAggregationSource, PandasAggregatedFeatureGroup}
            ),
        )
    )

    aggregated = [df for df in results if "scope_python_sales__sum_aggr" in df.columns]
    assert len(aggregated) == 1
    assert aggregated[0]["scope_python_sales__sum_aggr"].iloc[0] == 100


class ScopePythonAggregationSourceB(FeatureGroup):
    """Second source of the same feature name with different values."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"scope_python_sales"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"scope_python_sales": [1, 2, 3, 4]}


@pytest.mark.parametrize("path", ["name_path", "config_path"])
def test_end2end_one_declared_child_shared_by_two_consumers(path: str) -> None:
    """A shared declared child keeps its scope and is not mutated by either consumer's group options."""
    child = Feature("scope_python_sales", feature_group=ScopePythonAggregationSourceB)
    context = {"in_features": [child]}
    if path == "name_path":
        sum_name, max_name = "scope_python_sales__sum_aggr", "scope_python_sales__max_aggr"
        sum_group: dict[str, Any] = {"g_shared": 1}
        max_group: dict[str, Any] = {"g_shared": 2}
    else:
        sum_name, max_name = "scope_cfg_sum", "scope_cfg_max"
        sum_group = {"g_shared": 1, "aggregation_type": "sum"}
        max_group = {"g_shared": 2, "aggregation_type": "max"}

    results = list(
        mloda.run_all(
            [
                Feature(sum_name, Options(group=sum_group, context=context)),
                Feature(max_name, Options(group=max_group, context=context)),
            ],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {ScopePythonAggregationSource, ScopePythonAggregationSourceB, PandasAggregatedFeatureGroup}
            ),
        )
    )

    summed = [df for df in results if sum_name in df.columns]
    maxed = [df for df in results if max_name in df.columns]
    assert summed[0][sum_name].iloc[0] == 10
    assert maxed[0][max_name].iloc[0] == 4
    assert child.options.get("g_shared") is None


# ---------------------------------------------------------------------------
# A name-owning candidate's marked match abort must not outrank the scope and domain gates
# ---------------------------------------------------------------------------

_ABORT_NAME = "sales__sum_aggr"


class ScopeAbortRival(StubFeatureGroup):
    """Rival that also matches the aggregated name, so a pin or domain can route the feature to it."""

    MATCHED_NAMES: ClassVar[frozenset[str]] = frozenset({_ABORT_NAME})
    SUPPORTED_NAMES: ClassVar[frozenset[str]] = MATCHED_NAMES


class DomainAbortRival(StubFeatureGroup):
    MATCHED_NAMES: ClassVar[frozenset[str]] = frozenset({_ABORT_NAME})
    SUPPORTED_NAMES: ClassVar[frozenset[str]] = MATCHED_NAMES
    DOMAIN_NAME: ClassVar[str | None] = "abort_rival_domain"


def _abort_candidates() -> FeatureGroupEnvironmentMapping:
    return {
        PandasAggregatedFeatureGroup: {PandasDataFrame},
        ScopeAbortRival: {MockComputeFramework},
        DomainAbortRival: {MockComputeFramework},
    }


def _forwarded_max() -> Options:
    options = Options()
    options.inherit_from(Options(group={"aggregation_type": "max"}))
    return options


_CONTRADICTING_OPTIONS = [
    pytest.param(lambda: Options(context={"in_features": ["raw"]}), id="in_features_contradicts_name"),
    pytest.param(lambda: Options(context={"aggregation_type": "max"}), id="declared_option_contradicts_name"),
    pytest.param(_forwarded_max, id="forwarded_option_contradicts_name"),
]


@pytest.mark.parametrize("make_options", _CONTRADICTING_OPTIONS)
def test_pin_to_another_group_skips_the_owning_candidates_abort(make_options: Callable[[], Options]) -> None:
    feature = Feature(_ABORT_NAME, make_options(), feature_group=ScopeAbortRival)

    winner, _frameworks = next(iter(evaluate_or_raise(feature, _abort_candidates()).identified.items()))

    assert winner is ScopeAbortRival


@pytest.mark.parametrize("make_options", _CONTRADICTING_OPTIONS)
def test_domain_gated_owning_candidate_does_not_abort(make_options: Callable[[], Options]) -> None:
    feature = Feature(_ABORT_NAME, make_options(), domain="abort_rival_domain")

    winner, _frameworks = next(iter(evaluate_or_raise(feature, _abort_candidates()).identified.items()))

    assert winner is DomainAbortRival


@pytest.mark.parametrize("make_options", _CONTRADICTING_OPTIONS)
def test_unpinned_contradiction_still_aborts(make_options: Callable[[], Options]) -> None:
    feature = Feature(_ABORT_NAME, make_options())

    with pytest.raises(ValueError):
        evaluate_or_raise(feature, _abort_candidates())


def test_replacement_logs_one_debug_line_naming_feature_winner_and_parents(caplog: pytest.LogCaptureFixture) -> None:
    accessible_plugins: FeatureGroupEnvironmentMapping = {
        ScopeSourceA: {MockComputeFramework},
        ScopeSourceASub: {MockComputeFramework},
    }

    with caplog.at_level(logging.DEBUG, logger="mloda.core.prepare.identify_feature_group"):
        evaluate_or_raise(
            feature=Feature("subject_token", feature_group=ScopeSourceA),
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )

    lines = [r.getMessage() for r in caplog.records if r.name == "mloda.core.prepare.identify_feature_group"]
    assert len(lines) == 1
    assert "subject_token" in lines[0]
    assert "ScopeSourceASub" in lines[0]
    assert "ScopeSourceA" in lines[0].replace("ScopeSourceASub", "")


def test_no_replacement_logs_no_debug_line(caplog: pytest.LogCaptureFixture) -> None:
    accessible_plugins: FeatureGroupEnvironmentMapping = {ScopeSourceASub: {MockComputeFramework}}

    with caplog.at_level(logging.DEBUG, logger="mloda.core.prepare.identify_feature_group"):
        evaluate_or_raise(
            feature=Feature("subject_token", feature_group=ScopeSourceA),
            accessible_plugins=accessible_plugins,
            links=None,
            data_access_collection=None,
        )

    assert [r for r in caplog.records if r.name == "mloda.core.prepare.identify_feature_group"] == []


# ---------------------------------------------------------------------------
# Two reader-backed roots matching one column: ambiguous bare, loadable when scoped
# ---------------------------------------------------------------------------

READER_COL = "scope_reader_shared_col"


class CsvFG(FeatureGroup):
    """Root reading the shared column through CsvReader."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return CsvReader()

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        # Only the unique column, so no other test's file feature can resolve here.
        return str(feature_name) == READER_COL and super().match_feature_group_criteria(
            feature_name, options, data_access_collection
        )

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return CsvReader().load(features)


class ParquetFG(FeatureGroup):
    """Root reading the shared column through ParquetReader."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return ParquetReader()

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        # Only the unique column, so no other test's file feature can resolve here.
        return str(feature_name) == READER_COL and super().match_feature_group_criteria(
            feature_name, options, data_access_collection
        )

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return ParquetReader().load(features)


def _reader_files(tmp_path: Path) -> tuple[str, str]:
    csv_path = tmp_path / "scope_reader_shared.csv"
    csv_path.write_text(f"{READER_COL}\n1\n2\n3\n")
    parquet_path = tmp_path / "scope_reader_shared.parquet"
    pq.write_table(pa.table({READER_COL: [10, 20, 30]}), str(parquet_path))
    return str(csv_path), str(parquet_path)


def _run_reader_roots(feature: Feature, dac: DataAccessCollection) -> list[Any]:
    return list(
        mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            data_access_collection=dac,
            plugin_collector=PluginCollector.enabled_feature_groups({CsvFG, ParquetFG}),
        )
    )


def test_two_reader_backed_roots_are_ambiguous_and_name_their_sources(tmp_path: Path) -> None:
    csv_path, parquet_path = _reader_files(tmp_path)

    with pytest.raises(FeatureResolutionError, match="Multiple feature groups found") as exc_info:
        _run_reader_roots(Feature(READER_COL), DataAccessCollection(files={csv_path, parquet_path}))

    message = str(exc_info.value)
    assert "CsvFG" in message
    assert "ParquetFG" in message
    assert f"CsvReader: {csv_path}" in message
    assert f"ParquetReader: {parquet_path}" in message
    assert "BaseInputData already set" not in message


@pytest.mark.parametrize(
    ("scope", "expected"), [("CsvFG", [1, 2, 3]), ("ParquetFG", [10, 20, 30])], ids=["csv", "parquet"]
)
def test_scoped_reader_backed_root_loads_its_own_file(tmp_path: Path, scope: str, expected: list[int]) -> None:
    csv_path, parquet_path = _reader_files(tmp_path)

    results = _run_reader_roots(
        Feature(READER_COL, feature_group=scope), DataAccessCollection(files={csv_path, parquet_path})
    )

    assert len(results) == 1
    assert results[0].to_pydict() == {READER_COL: expected}


def test_resolved_reader_feature_holds_its_pair_on_input_data_match_not_in_options(tmp_path: Path) -> None:
    csv_path, parquet_path = _reader_files(tmp_path)
    feature = Feature(READER_COL, feature_group="CsvFG")

    mloda.run_all(
        [feature],
        compute_frameworks={PyArrowTable},
        data_access_collection=DataAccessCollection(files={csv_path, parquet_path}),
        plugin_collector=PluginCollector.enabled_feature_groups({CsvFG, ParquetFG}),
        copy_features=False,
    )

    assert feature.input_data_match == (CsvReader, csv_path)
    assert RESERVED_READER_OPTION_KEY not in feature.options.group
    assert RESERVED_READER_OPTION_KEY not in feature.options.context


def test_multiple_message_never_contains_a_credential_secret(tmp_path: Path) -> None:
    csv_path, _ = _reader_files(tmp_path)
    db_path = str(tmp_path / "scope_reader_shared.db")
    conn = sqlite3.connect(db_path)
    conn.execute(f"CREATE TABLE scope_reader_table ({READER_COL} INTEGER)")
    conn.execute("INSERT INTO scope_reader_table VALUES (100)")
    conn.commit()
    conn.close()
    secret = "scope-reader-secret-value"  # nosec B105
    dac = DataAccessCollection(files={csv_path}, credentials=Credential(sqlite=db_path, password=secret))  # nosec B106

    with pytest.raises(FeatureResolutionError, match="Multiple feature groups found") as exc_info:
        mloda.run_all(
            [Feature(READER_COL)],
            compute_frameworks={PyArrowTable},
            data_access_collection=dac,
            plugin_collector=PluginCollector.enabled_feature_groups({CsvFG, ReadDBFeature}),
        )

    message = str(exc_info.value)
    assert "CsvFG" in message
    assert "ReadDBFeature" in message
    assert f"SQLITEReader: {db_path}::scope_reader_table" in message
    assert secret not in message


def test_mistyped_string_scope_suggests_the_intended_group(tmp_path: Path) -> None:
    csv_path, parquet_path = _reader_files(tmp_path)

    with pytest.raises(Exception) as exc_info:
        _run_reader_roots(
            Feature(READER_COL, feature_group="CsvFGG"), DataAccessCollection(files={csv_path, parquet_path})
        )

    message = str(exc_info.value)
    assert "Did you mean" in message
    assert "CsvFG" in message.split("Did you mean", 1)[1]
