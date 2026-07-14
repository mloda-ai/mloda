"""Scope-aware resolution tests for resolve_feature() (issue #693).

The engine's no-match error recommends resolve_feature() as THE debug tool, and that error
even calls out a feature-group scope ("Scoped to feature group: 'X'."). resolve_feature must
therefore be able to express that same scope, so the recommended tool can reproduce the failure
it is recommended for.

The contract pinned here:
  * feature_group= accepts a FeatureGroup subclass object and a class-name string, with the same
    semantics as Feature(..., feature_group=...): the string form matches the named class AND its
    subclasses (MRO walk, matches_feature_group_scope), and the root FeatureGroup base is never a
    wildcard.
  * A scoped no-match / scoped ambiguity reports the engine's scope callout in the error.
  * resolve_feature still NEVER raises, not even for an invalid scope, unlike Feature().
  * A scoped engine failure reproduces through resolve_feature with the same outcome.

Probe feature groups and feature names are prefixed 'Scoped693' / 'scoped_probe_693_' because every
FeatureGroup subclass in the process is globally visible to matching under pytest-xdist.
"""

import inspect
from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.api.plugin_info import ResolvedFeature
from mloda.user import PluginLoader, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


SHARED_FEATURE = "scoped_probe_693_shared"
CLAIMS_ONLY_FEATURE = "scoped_probe_693_claims_only"
POLICY_ONLY_FEATURE = "scoped_probe_693_policy_only"
SIBLING_FEATURE = "scoped_probe_693_sibling"
RAISING_FEATURE = "scoped_probe_693_raising"
RAISING_REASON = "scoped_probe_693 raising source rejects this feature"
UNKNOWN_SCOPE = "Scoped693CompletelyUnknownScope"


def _callout(scope_name: str) -> str:
    """The scope callout the engine's _scope_callout renders, verbatim."""
    return f"Scoped to feature group: '{scope_name}'."


class Scoped693ClaimsReaderBase(FeatureGroup):
    """Claims family base: matches the shared name and the claims-only name.

    Bounded compute_framework_rule so the capability split is deterministic.
    """

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) in {SHARED_FEATURE, CLAIMS_ONLY_FEATURE}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class Scoped693ClaimsReaderPandas(Scoped693ClaimsReaderBase):
    """Concrete member of the claims family; inherits the family's name matching."""


class Scoped693PolicyReader(FeatureGroup):
    """Rival source: matches the shared name and its own policy-only name."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) in {SHARED_FEATURE, POLICY_ONLY_FEATURE}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class Scoped693SiblingBase(FeatureGroup):
    """Base of two rival siblings, both of which sit inside a scope on this base."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == SIBLING_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class Scoped693SiblingOne(Scoped693SiblingBase):
    """First sibling; neither sibling is a subclass of the other, so they cannot be narrowed."""


class Scoped693SiblingTwo(Scoped693SiblingBase):
    """Second sibling; rival of Scoped693SiblingOne inside the same scope."""


class Scoped693RaisingSource(FeatureGroup):
    """Matching this group raises ValueError, which resolve_feature degrades to 'not a match'."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if str(feature_name) == RAISING_FEATURE:
            raise ValueError(RAISING_REASON)
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class Scoped693SafeSource(FeatureGroup):
    """A well-behaved group matching the same name as the raising one, so it can still win."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == RAISING_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


@pytest.fixture(scope="module", autouse=True)
def load_plugins() -> None:
    """Load all plugins before running tests in this module."""
    PluginLoader.all()


class TestScopedResolutionBothForms:
    """A shared name that is ambiguous unscoped resolves to exactly one group when scoped."""

    def test_shared_name_is_ambiguous_unscoped(self) -> None:
        """Premise guard: without a scope, the shared name matches both sources."""
        result = resolve_feature(SHARED_FEATURE)

        assert result.feature_group is None
        assert result.error is not None
        assert "Multiple FeatureGroups match" in result.error

    def test_class_object_scope_resolves_uniquely(self) -> None:
        """A class-object scope on the rival source resolves the shared name to that source."""
        result = resolve_feature(SHARED_FEATURE, feature_group=Scoped693PolicyReader)

        assert result.feature_group is Scoped693PolicyReader
        assert result.error is None
        assert result.candidates == [Scoped693PolicyReader]

    def test_class_name_string_scope_resolves_uniquely(self) -> None:
        """The class-name string form has the same semantics as the class-object form."""
        result = resolve_feature(SHARED_FEATURE, feature_group="Scoped693PolicyReader")

        assert result.feature_group is Scoped693PolicyReader
        assert result.error is None
        assert result.candidates == [Scoped693PolicyReader]

    def test_class_object_base_scope_matches_subclasses_and_prefers_the_subclass(self) -> None:
        """A base-class scope keeps the family (issubclass) and subclass filtering picks the member."""
        result = resolve_feature(SHARED_FEATURE, feature_group=Scoped693ClaimsReaderBase)

        assert result.feature_group is Scoped693ClaimsReaderPandas
        assert result.error is None
        assert Scoped693ClaimsReaderBase in result.candidates
        assert Scoped693ClaimsReaderPandas in result.candidates
        assert Scoped693PolicyReader not in result.candidates

    def test_class_name_string_base_scope_walks_the_mro(self) -> None:
        """The string form matches the named class AND its subclasses, like matches_feature_group_scope."""
        result = resolve_feature(SHARED_FEATURE, feature_group="Scoped693ClaimsReaderBase")

        assert result.feature_group is Scoped693ClaimsReaderPandas
        assert result.error is None
        assert Scoped693PolicyReader not in result.candidates

    def test_scoped_resolution_still_reports_the_capability_split(self) -> None:
        """A scoped resolve keeps the capability-aware fields populated."""
        result = resolve_feature(SHARED_FEATURE, feature_group=Scoped693PolicyReader)

        assert "PandasDataFrame" in result.supported_compute_frameworks
        assert "PandasDataFrame" not in result.unsupported_compute_frameworks

    def test_leaf_class_scope_resolves_to_the_leaf(self) -> None:
        """Scoping to the concrete member itself resolves to it, in both forms."""
        by_class = resolve_feature(SHARED_FEATURE, feature_group=Scoped693ClaimsReaderPandas)
        by_name = resolve_feature(SHARED_FEATURE, feature_group="Scoped693ClaimsReaderPandas")

        assert by_class.feature_group is Scoped693ClaimsReaderPandas
        assert by_name.feature_group is Scoped693ClaimsReaderPandas
        assert by_class.error is None
        assert by_name.error is None


class TestScopedNoMatch:
    """A name that resolves fine unscoped but has no candidate inside the scope."""

    def test_claims_only_name_resolves_unscoped(self) -> None:
        """Premise guard: the claims-only name resolves to the claims family member."""
        result = resolve_feature(CLAIMS_ONLY_FEATURE)

        assert result.feature_group is Scoped693ClaimsReaderPandas
        assert result.error is None

    def test_out_of_scope_class_reports_no_feature_group(self) -> None:
        """Scoping the claims-only name to the rival source leaves no candidate."""
        result = resolve_feature(CLAIMS_ONLY_FEATURE, feature_group=Scoped693PolicyReader)

        assert result.feature_group is None
        assert result.candidates == []
        assert result.error is not None

    def test_out_of_scope_class_error_carries_the_scope_callout(self) -> None:
        """The error carries the engine's verbatim scope callout, so the failure is legible."""
        result = resolve_feature(CLAIMS_ONLY_FEATURE, feature_group=Scoped693PolicyReader)

        assert result.error is not None
        assert _callout("Scoped693PolicyReader") in result.error, (
            f"Scoped no-match must carry the scope callout, got: {result.error}"
        )

    def test_out_of_scope_string_error_carries_the_scope_callout(self) -> None:
        """The string form produces the same scoped no-match error."""
        result = resolve_feature(CLAIMS_ONLY_FEATURE, feature_group="Scoped693PolicyReader")

        assert result.feature_group is None
        assert result.error is not None
        assert _callout("Scoped693PolicyReader") in result.error

    def test_unknown_scope_name_reports_no_feature_group_with_callout(self) -> None:
        """A scope naming no loaded feature group at all is a scoped no-match, not a crash."""
        result = resolve_feature(SHARED_FEATURE, feature_group=UNKNOWN_SCOPE)

        assert isinstance(result, ResolvedFeature)
        assert result.feature_group is None
        assert result.candidates == []
        assert result.error is not None
        assert _callout(UNKNOWN_SCOPE) in result.error


class TestScopedAmbiguity:
    """Two feature groups inside the scope both match: the multiple-match error names the scope."""

    def test_sibling_scope_is_ambiguous_and_names_both_siblings(self) -> None:
        """A base scope covering two rival siblings cannot be narrowed, so it stays ambiguous."""
        result = resolve_feature(SIBLING_FEATURE, feature_group=Scoped693SiblingBase)

        assert result.feature_group is None
        assert result.error is not None
        assert "Multiple FeatureGroups match" in result.error
        assert "Scoped693SiblingOne" in result.error
        assert "Scoped693SiblingTwo" in result.error

    def test_sibling_scope_ambiguity_carries_the_scope_callout(self) -> None:
        """The multiple-match error must ALSO carry the scope callout."""
        result = resolve_feature(SIBLING_FEATURE, feature_group=Scoped693SiblingBase)

        assert result.error is not None
        assert _callout("Scoped693SiblingBase") in result.error, (
            f"Scoped ambiguity must carry the scope callout, got: {result.error}"
        )

    def test_sibling_scope_ambiguity_string_form_carries_the_scope_callout(self) -> None:
        """The string form reaches both siblings through the MRO and stays ambiguous too."""
        result = resolve_feature(SIBLING_FEATURE, feature_group="Scoped693SiblingBase")

        assert result.feature_group is None
        assert result.error is not None
        assert "Multiple FeatureGroups match" in result.error
        assert _callout("Scoped693SiblingBase") in result.error

    def test_scoping_to_one_sibling_resolves_the_ambiguity(self) -> None:
        """Narrowing the scope to a single sibling resolves it, proving the ambiguity is scope-shaped."""
        result = resolve_feature(SIBLING_FEATURE, feature_group=Scoped693SiblingTwo)

        assert result.feature_group is Scoped693SiblingTwo
        assert result.error is None


class TestInvalidScopeNeverRaises:
    """resolve_feature is a debug API: an invalid scope is reported, never raised."""

    def test_root_feature_group_class_scope_is_reported_not_raised(self) -> None:
        """Feature() raises TypeError for the root base; resolve_feature must report it instead."""
        with pytest.raises(TypeError):
            Feature(SHARED_FEATURE, feature_group=FeatureGroup)

        result = resolve_feature(SHARED_FEATURE, feature_group=FeatureGroup)

        assert isinstance(result, ResolvedFeature)
        assert result.feature_group is None
        assert result.error is not None
        assert "FeatureGroup" in result.error

    def test_root_feature_group_name_scope_is_not_a_wildcard(self) -> None:
        """The root base NAME must not silently scope to everything; it is reported as invalid."""
        result = resolve_feature(SHARED_FEATURE, feature_group=FeatureGroup.get_class_name())

        assert result.feature_group is None, "the root base name must never act as a wildcard scope"
        assert result.error is not None

    def test_non_feature_group_type_scope_is_reported_not_raised(self) -> None:
        """A non-FeatureGroup type (int) is an invalid scope, reported in error."""
        resolve_feature_untyped: Any = resolve_feature

        result = resolve_feature_untyped(SHARED_FEATURE, feature_group=int)

        assert isinstance(result, ResolvedFeature)
        assert result.feature_group is None
        assert result.error is not None
        assert "FeatureGroup" in result.error

    def test_non_type_scope_value_is_reported_not_raised(self) -> None:
        """A scope value that is neither a class nor a string (123) is reported, not raised."""
        resolve_feature_untyped: Any = resolve_feature

        result = resolve_feature_untyped(SHARED_FEATURE, feature_group=123)

        assert isinstance(result, ResolvedFeature)
        assert result.feature_name == SHARED_FEATURE
        assert result.feature_group is None
        assert result.error is not None
        assert "FeatureGroup" in result.error


class TestScopedMatchingStaysNonThrowing:
    """A candidate raising ValueError while matching is degraded to 'not a match', scoped or not."""

    def test_raising_candidate_does_not_block_other_candidates(self) -> None:
        """Unscoped, the raising group is degraded and the safe group still resolves."""
        result = resolve_feature(RAISING_FEATURE)

        assert result.feature_group is Scoped693SafeSource
        assert result.error is None

    def test_raising_candidate_does_not_block_a_scoped_sibling(self) -> None:
        """Scoping to the safe group resolves, even though the raising group matches the same name."""
        result = resolve_feature(RAISING_FEATURE, feature_group=Scoped693SafeSource)

        assert result.feature_group is Scoped693SafeSource
        assert result.error is None

    def test_scoping_to_the_raising_candidate_reports_reason_and_callout(self) -> None:
        """Scoped to the raising group, resolution fails with the validation reason AND the scope callout."""
        result = resolve_feature(RAISING_FEATURE, feature_group=Scoped693RaisingSource)

        assert isinstance(result, ResolvedFeature)
        assert result.feature_group is None
        assert result.error is not None
        assert RAISING_REASON in result.error, f"Error must surface the rejection reason, got: {result.error}"
        assert _callout("Scoped693RaisingSource") in result.error, (
            f"Error must carry the scope callout, got: {result.error}"
        )

    def test_out_of_scope_rejection_reason_never_pollutes_a_scoped_result(self) -> None:
        """Scope first: an out-of-scope group is never even asked to match, so it cannot report a reason.

        Scoped to the safe group, the raising group sits outside the scope. Its rejection reason must
        not surface anywhere in the scoped result, which is the invariant the scope-first ordering in
        resolve_feature exists for.
        """
        result = resolve_feature(RAISING_FEATURE, feature_group=Scoped693SafeSource)

        assert result.feature_group is Scoped693SafeSource
        assert RAISING_REASON not in (result.error or ""), (
            f"An out-of-scope group's rejection reason must not appear in a scoped error, got: {result.error}"
        )


class TestScopedEngineParity:
    """The failure the engine reports for a scoped Feature reproduces through resolve_feature (#693)."""

    @staticmethod
    def _collector() -> PluginCollector:
        return PluginCollector.enabled_feature_groups({Scoped693ClaimsReaderPandas, Scoped693PolicyReader})

    def test_engine_raises_for_the_scoped_feature(self) -> None:
        """Premise: mloda raises for a feature whose only source sits outside the requested scope."""
        collector = self._collector()

        with pytest.raises(ValueError) as exc_info:
            list(
                mloda.run_all(
                    [Feature(CLAIMS_ONLY_FEATURE, feature_group=Scoped693PolicyReader)],
                    compute_frameworks={PandasDataFrame},
                    plugin_collector=collector,
                )
            )

        engine_message = str(exc_info.value)
        assert _callout("Scoped693PolicyReader") in engine_message
        assert "No feature groups found" in engine_message

    def test_resolve_feature_reproduces_the_engine_failure(self) -> None:
        """The engine's recommended debug tool reproduces the scoped failure: no group, scope callout."""
        collector = self._collector()

        result = resolve_feature(CLAIMS_ONLY_FEATURE, feature_group=Scoped693PolicyReader, plugin_collector=collector)

        assert result.feature_group is None
        assert result.error is not None
        assert _callout("Scoped693PolicyReader") in result.error

    def test_the_same_feature_resolves_unscoped_under_the_same_collector(self) -> None:
        """Guard: the feature itself is fine; only the scope makes it fail, in engine and debug tool alike."""
        collector = self._collector()

        result = resolve_feature(CLAIMS_ONLY_FEATURE, plugin_collector=collector)

        assert result.feature_group is Scoped693ClaimsReaderPandas
        assert result.error is None


class TestUnscopedBehaviourUnchanged:
    """Scope defaults to None: omitting it and passing None resolve exactly as before."""

    def test_omitted_scope_equals_explicit_none(self) -> None:
        """feature_group=None must be indistinguishable from omitting the argument, on every outcome."""
        for feature_name in (SHARED_FEATURE, CLAIMS_ONLY_FEATURE, SIBLING_FEATURE, RAISING_FEATURE):
            implicit = resolve_feature(feature_name)
            explicit = resolve_feature(feature_name, feature_group=None)

            assert explicit.feature_group is implicit.feature_group, feature_name
            assert explicit.candidates == implicit.candidates, feature_name
            assert explicit.error == implicit.error, feature_name
            assert explicit.supported_compute_frameworks == implicit.supported_compute_frameworks, feature_name

    def test_unscoped_options_threading_still_works(self) -> None:
        """Options keep flowing through matching when no scope is given."""
        result = resolve_feature(CLAIMS_ONLY_FEATURE, options=Options(group={"scoped_probe_693_key": "v"}))

        assert result.feature_group is Scoped693ClaimsReaderPandas
        assert result.error is None


class TestFeatureGroupIsKeywordOnly:
    """The new scope argument is keyword-only, like options and plugin_collector."""

    def test_feature_group_parameter_is_declared_keyword_only(self) -> None:
        """feature_group must sit behind the '*' in the signature, so it can never bind positionally.

        Asserting on the parameter kind, not on a TypeError from a two-positional call: the second
        positional slot already raised before feature_group existed (options was keyword-only), so
        such a call proves nothing about this parameter.
        """
        parameter = inspect.signature(resolve_feature).parameters["feature_group"]

        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"feature_group must be keyword-only, got kind: {parameter.kind}"
        )

    def test_scope_combines_with_options_and_plugin_collector(self) -> None:
        """All three keyword arguments can be combined."""
        collector = PluginCollector.enabled_feature_groups({Scoped693ClaimsReaderPandas, Scoped693PolicyReader})

        result = resolve_feature(
            SHARED_FEATURE,
            options=Options(group={"scoped_probe_693_key": "v"}),
            feature_group="Scoped693PolicyReader",
            plugin_collector=collector,
        )

        assert result.feature_group is Scoped693PolicyReader
        assert result.error is None
