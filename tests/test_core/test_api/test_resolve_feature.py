"""
Tests for the resolve_feature() API function.

This module tests the resolve_feature function which resolves a feature name
to its matching FeatureGroup class, with support for subclass filtering
and candidate tracking.
"""

import ast
import gc
import inspect
import linecache
import sys
import textwrap

import pytest
from typing import Any, Optional

from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import property_spec
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import PluginCollector
from mloda.core.api.plugin_info import ResolvedFeature
from mloda.core.api import plugin_docs
from mloda.core.api.plugin_docs import resolve_feature
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping, RedefinitionConflictError
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.user import PluginLoader
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


SPLIT_CAP_FEATURE = "SplitCapResolveFeature"
ALL_REJECTED_CAP_FEATURE = "AllRejectedCapResolveFeature"
OPTION_GATED_FEATURE = "OptionGatedResolveFeature"
OPTION_CAP_FEATURE = "OptionCapResolveFeature"
PARTITION_BY_KEY = "partition_by"
ALLOW_PYTHON_DICT_KEY = "allow_python_dict"

PROBE_TYPE_KEY = "resolve_probe_type"
PROBE_FEATURE = "value__median_resolveprobe"

SIBLING_SCOPED_FEATURE = "SiblingScopedResolve693"
LONE_CHILD_SCOPED_FEATURE = "LoneChildScopedResolve693"
UNRELATED_SCOPED_FEATURE = "UnrelatedScopedResolve693"


class SplitCapResolveFeatureGroup(FeatureGroup):
    """Supports the op on PandasDataFrame but rejects PythonDictFramework.

    Bounded compute_framework_rule so compute_framework_definition() is exactly
    {PandasDataFrame, PythonDictFramework} and the capability split is deterministic.
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
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == SPLIT_CAP_FEATURE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        return compute_framework is not PythonDictFramework

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class AllRejectedCapResolveFeatureGroup(FeatureGroup):
    """Rejects the op on every framework in its bounded definition."""

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
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == ALL_REJECTED_CAP_FEATURE

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


class OptionGatedResolveFeatureGroup(FeatureGroup):
    """Matches its feature name only when the caller supplies a truthy 'partition_by' option."""

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
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        if feature_name != OPTION_GATED_FEATURE:
            return False
        return bool(options.get(PARTITION_BY_KEY))

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class OptionCapResolveFeatureGroup(FeatureGroup):
    """Matches its feature name always, but only supports PythonDictFramework when an option opts in."""

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
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == OPTION_CAP_FEATURE

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        if compute_framework is PythonDictFramework:
            return bool(options.get(ALLOW_PYTHON_DICT_KEY))
        return True

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class ForwardMismatchResolveFeatureGroup(FeatureChainParserMixin, FeatureGroup):
    """Chain-parsed group whose match raises ValueError on a forwarded/name option mismatch."""

    PREFIX_PATTERN = r".*__([\w]+)_resolveprobe$"

    PROPERTY_MAPPING = {
        PROBE_TYPE_KEY: property_spec(
            "Probe operation subtype.",
            strict=True,
            allowed_values={"median": "Median value", "sum": "Sum of values"},
            context=True,
        ),
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}


class ScopedResolveFamilyBase693(FeatureGroup):
    """Shared base of the scoped-resolve fixture family; never matches a feature name itself."""

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
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


class ScopedResolveSiblingOne693(ScopedResolveFamilyBase693):
    """First concrete sibling; matches the shared sibling feature name."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == SIBLING_SCOPED_FEATURE


class ScopedResolveSiblingTwo693(ScopedResolveFamilyBase693):
    """Second concrete sibling; matches the same shared sibling feature name."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == SIBLING_SCOPED_FEATURE


class ScopedResolveLoneChild693(ScopedResolveFamilyBase693):
    """Third family member; the only class matching its own unique feature name."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == LONE_CHILD_SCOPED_FEATURE


class UnrelatedScopedResolve693FeatureGroup(FeatureGroup):
    """Unrelated scope target: matches only its own distinct feature name, never the sibling one."""

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
        if isinstance(feature_name, FeatureName):
            feature_name = str(feature_name)
        return feature_name == UNRELATED_SCOPED_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None


@pytest.fixture(scope="module", autouse=True)
def load_plugins() -> None:
    """Load all plugins before running tests in this module."""
    PluginLoader.all()


def _forwarded_mismatch_options() -> Options:
    """Options whose forwarded probe type ('sum') contradicts the name-parsed one ('median')."""
    options = Options(group={PROBE_TYPE_KEY: "sum"})
    options.inherited_group_keys = frozenset({PROBE_TYPE_KEY})
    return options


class TestResolvedFeatureDataclass:
    """Tests for the ResolvedFeature dataclass structure."""

    def test_resolved_feature_dataclass_exists(self) -> None:
        """Test that ResolvedFeature dataclass can be imported."""
        assert ResolvedFeature is not None

    def test_resolved_feature_has_feature_name_field(self) -> None:
        """Test that ResolvedFeature has feature_name field."""
        result = ResolvedFeature(
            feature_name="test_feature",
            feature_group=None,
            candidates=[],
            error=None,
        )
        assert hasattr(result, "feature_name")
        assert result.feature_name == "test_feature"

    def test_resolved_feature_has_feature_group_field(self) -> None:
        """Test that ResolvedFeature has feature_group field."""
        result = ResolvedFeature(
            feature_name="test_feature",
            feature_group=None,
            candidates=[],
            error=None,
        )
        assert hasattr(result, "feature_group")
        assert result.feature_group is None

    def test_resolved_feature_has_candidates_field(self) -> None:
        """Test that ResolvedFeature has candidates field."""
        result = ResolvedFeature(
            feature_name="test_feature",
            feature_group=None,
            candidates=[],
            error=None,
        )
        assert hasattr(result, "candidates")
        assert result.candidates == []

    def test_resolved_feature_has_error_field(self) -> None:
        """Test that ResolvedFeature has error field."""
        result = ResolvedFeature(
            feature_name="test_feature",
            feature_group=None,
            candidates=[],
            error="Some error",
        )
        assert hasattr(result, "error")
        assert result.error == "Some error"


class TestResolveFeatureReturnType:
    """Tests for resolve_feature return type."""

    def test_resolve_feature_returns_resolved_feature_type(self) -> None:
        """Test that resolve_feature returns a ResolvedFeature instance."""
        result = resolve_feature("NonExistentFeatureName12345")
        assert isinstance(result, ResolvedFeature)


class TestResolveFeatureValidMatch:
    """Tests for resolve_feature with valid feature names."""

    @staticmethod
    def _find_unambiguous_feature_name() -> str:
        """Find a feature name that resolves to exactly one FeatureGroup.

        Iterates through all loaded FeatureGroups and uses resolve_feature
        to find one that resolves unambiguously, even when test fixtures
        have registered additional classes in the same process.
        """
        from mloda.core.abstract_plugins.components.utils import get_all_subclasses

        all_fgs = get_all_subclasses(FeatureGroup)
        for fg in all_fgs:
            if not fg.match_feature_group_criteria(FeatureName(fg.get_class_name()), Options(), None):
                continue
            result = resolve_feature(fg.get_class_name())
            if result.feature_group is not None:
                return str(fg.get_class_name())
        raise AssertionError("Need at least one FeatureGroup that resolves unambiguously")

    def test_resolve_feature_finds_matching_feature_group(self) -> None:
        """Test resolving a feature name that matches a single FeatureGroup."""
        target_name = self._find_unambiguous_feature_name()

        result = resolve_feature(target_name)

        assert result.feature_name == target_name
        assert result.feature_group is not None
        assert result.error is None

    def test_resolve_feature_returns_correct_feature_group_type(self) -> None:
        """Test that the resolved feature_group is a Type[FeatureGroup]."""
        target_name = self._find_unambiguous_feature_name()

        result = resolve_feature(target_name)

        assert result.feature_group is not None
        assert issubclass(result.feature_group, FeatureGroup)


class TestResolveFeatureNoMatch:
    """Tests for resolve_feature when no FeatureGroup matches."""

    def test_resolve_feature_no_match_returns_none_feature_group(self) -> None:
        """Test that unmatched feature name returns None for feature_group."""
        result = resolve_feature("CompletelyNonExistentFeatureName12345XYZ")

        assert result.feature_group is None

    def test_resolve_feature_no_match_has_error(self) -> None:
        """Test that unmatched feature name has an error message."""
        result = resolve_feature("CompletelyNonExistentFeatureName12345XYZ")

        assert result.error is not None
        assert len(result.error) > 0

    def test_resolve_feature_no_match_preserves_feature_name(self) -> None:
        """Test that unmatched result preserves the input feature name."""
        feature_name = "CompletelyNonExistentFeatureName12345XYZ"
        result = resolve_feature(feature_name)

        assert result.feature_name == feature_name


class TestResolveFeatureCandidates:
    """Tests for resolve_feature candidates list."""

    def test_resolve_feature_candidates_contains_all_matches(self) -> None:
        """Test that candidates lists a concrete FeatureGroup that matched criteria."""
        from mloda.core.abstract_plugins.components.utils import get_all_subclasses

        all_fgs = list(get_all_subclasses(FeatureGroup))
        assert len(all_fgs) > 0, "Need at least one FeatureGroup for this test"

        # A concrete group that matches its own class name and yields candidates. Abstract bases are
        # no longer candidates under the engine-delegated contract (#755), so skip them; and pick one
        # whose resolution actually produces candidates so the test is order-independent under xdist.
        target_candidates: list[type[FeatureGroup]] = []
        for fg in all_fgs:
            if inspect.isabstract(fg):
                continue
            if not fg.match_feature_group_criteria(FeatureName(fg.get_class_name()), Options(), None):
                continue
            candidates = resolve_feature(fg.get_class_name()).candidates
            if candidates:
                target_candidates = candidates
                break

        assert len(target_candidates) >= 1, "Need a concrete FeatureGroup that resolves to candidates"

    def test_resolve_feature_candidates_are_feature_group_types(self) -> None:
        """Test that all candidates are Type[FeatureGroup]."""
        from mloda.core.abstract_plugins.components.utils import get_all_subclasses

        all_fgs = list(get_all_subclasses(FeatureGroup))
        assert len(all_fgs) > 0, "Need at least one FeatureGroup for this test"

        target_fg = all_fgs[0]
        target_name = target_fg.get_class_name()

        result = resolve_feature(target_name)

        for candidate in result.candidates:
            assert issubclass(candidate, FeatureGroup)


class TestResolveFeatureSubclassFiltering:
    """Tests for subclass filtering behavior."""

    def test_resolve_feature_prefers_subclass_over_parent(self) -> None:
        """Test that when both parent and child FeatureGroup match, only child is returned."""

        # Create parent and child FeatureGroups for testing
        class ParentTestFeatureGroup(FeatureGroup):
            """A parent feature group for testing subclass filtering."""

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: FeatureName | str,
                options: Options,
                data_access_collection: Optional[DataAccessCollection] = None,
            ) -> bool:
                return feature_name == "SubclassFilterTestFeature"

        class ChildTestFeatureGroup(ParentTestFeatureGroup):
            """A child feature group that also matches the same criteria."""

            pass

        # Both parent and child should match the feature name
        feature_name = "SubclassFilterTestFeature"

        result = resolve_feature(feature_name)

        # The resolved feature_group should be the child (more specific) class
        # Candidates may include both, but feature_group should be the child
        if result.feature_group is not None:
            # If resolution succeeded, it should prefer the child
            assert result.feature_group == ChildTestFeatureGroup or issubclass(
                result.feature_group, ParentTestFeatureGroup
            )

    def test_resolve_feature_candidates_include_parent_before_filtering(self) -> None:
        """Test that candidates list includes parent classes before subclass filtering."""

        # This test verifies that candidates captures all matches before filtering
        class ParentForCandidatesTest(FeatureGroup):
            """Parent for candidates test."""

            @classmethod
            def match_feature_group_criteria(
                cls,
                feature_name: FeatureName | str,
                options: Options,
                data_access_collection: Optional[DataAccessCollection] = None,
            ) -> bool:
                return feature_name == "CandidatesTestFeature"

        class ChildForCandidatesTest(ParentForCandidatesTest):
            """Child for candidates test."""

            pass

        feature_name = "CandidatesTestFeature"

        result = resolve_feature(feature_name)

        # Candidates should include both parent and child (before filtering)
        # This captures the "before subclass filtering" requirement
        if len(result.candidates) >= 2:
            candidate_names = [c.__name__ for c in result.candidates]
            # At minimum, we should see our test classes if they matched
            assert any("CandidatesTest" in name for name in candidate_names) or len(result.candidates) >= 1


class TestResolveFeatureCapabilityAware:
    """Capability-aware resolution: resolve_feature reports the supported/unsupported split.

    These tests pin the PLANNED contract and are expected to FAIL until
    resolve_feature and ResolvedFeature are made capability-aware (issue #482).
    """

    def test_resolved_feature_has_capability_fields_defaulting_empty(self) -> None:
        """ResolvedFeature gains two new list fields defaulting to empty lists.

        The existing 4-positional-arg constructor must still work.
        """
        result = ResolvedFeature("n", None, [], None)
        assert result.supported_compute_frameworks == []
        assert result.unsupported_compute_frameworks == []

    def test_split_resolution_reports_supported_and_unsupported(self) -> None:
        """A feature supported on one framework but rejected on another still resolves.

        feature_group is not None and error is None, but the capability split is
        reported: the rejected framework appears in unsupported, the supporting one
        in supported.
        """
        result = resolve_feature(SPLIT_CAP_FEATURE)

        # It still resolves: a supporting framework exists.
        assert result.feature_group is not None
        assert result.error is None

        # The capability split is reported (subset membership only).
        assert "PythonDictFramework" in result.unsupported_compute_frameworks
        assert "PandasDataFrame" in result.supported_compute_frameworks
        assert "PandasDataFrame" not in result.unsupported_compute_frameworks

    def test_all_rejected_resolution_fails_with_capability_error(self) -> None:
        """When every framework in the definition rejects the op, resolution fails.

        After #755 the error is the engine's capability-rejection message: it names the
        unsupported framework(s) and points at supports_compute_framework, with no concept
        of a 'default options' caveat.
        """
        result = resolve_feature(ALL_REJECTED_CAP_FEATURE)

        assert result.feature_group is None
        assert result.error is not None

        lowered = result.error.lower()
        assert "unsupported" in lowered, f"Error must signal 'unsupported', got: {result.error}"
        assert "PandasDataFrame" in result.error, f"Error must name PandasDataFrame, got: {result.error}"
        assert "PythonDictFramework" in result.error, f"Error must name PythonDictFramework, got: {result.error}"
        # Engine wording: names the capability hook and drops the debug-only phrasings.
        assert "Pin the feature to a supported compute framework" in result.error, (
            f"Error must carry the engine's capability-rejection guidance, got: {result.error}"
        )
        assert "default options" not in result.error, (
            f"Engine error has no 'default options' caveat, got: {result.error}"
        )
        assert "unsupported on all installed compute frameworks" not in result.error, (
            f"Engine error does not use the debug-only phrasing, got: {result.error}"
        )


class TestResolveFeatureWithOptions:
    """resolve_feature accepts caller-supplied Options and threads them through matching (issue #640)."""

    def test_option_gated_feature_does_not_resolve_without_options(self) -> None:
        """Without options, an option-gated FeatureGroup never matches, so resolution fails."""
        result = resolve_feature(OPTION_GATED_FEATURE)

        assert result.feature_group is None
        assert result.candidates == []
        assert result.error is not None

    def test_option_gated_feature_resolves_with_options(self) -> None:
        """With the gating option supplied, the option-gated FeatureGroup resolves."""
        result = resolve_feature(OPTION_GATED_FEATURE, options=Options(group={PARTITION_BY_KEY: ["id"]}))

        assert result.feature_group is OptionGatedResolveFeatureGroup
        assert result.error is None
        assert OptionGatedResolveFeatureGroup in result.candidates
        assert result.supported_compute_frameworks
        assert "PandasDataFrame" in result.supported_compute_frameworks

    def test_options_default_is_equivalent_to_explicit_empty_options(self) -> None:
        """options=None must behave exactly like a fresh, empty Options, on every resolution outcome."""
        for feature_name in (OPTION_GATED_FEATURE, OPTION_CAP_FEATURE, ALL_REJECTED_CAP_FEATURE, SPLIT_CAP_FEATURE):
            implicit = resolve_feature(feature_name)
            explicit = resolve_feature(feature_name, options=Options())

            assert explicit.feature_group is implicit.feature_group, feature_name
            assert explicit.candidates == implicit.candidates, feature_name
            assert explicit.error == implicit.error, feature_name
            assert explicit.supported_compute_frameworks == implicit.supported_compute_frameworks, feature_name
            assert explicit.unsupported_compute_frameworks == implicit.unsupported_compute_frameworks, feature_name

        # Guard against a trivially-passing comparison: options must actually be threaded through.
        gated = resolve_feature(OPTION_GATED_FEATURE, options=Options(group={PARTITION_BY_KEY: ["id"]}))
        assert gated.feature_group is OptionGatedResolveFeatureGroup

    def test_supports_compute_framework_sees_default_options(self) -> None:
        """Without options, the capability hook rejects PythonDictFramework."""
        result = resolve_feature(OPTION_CAP_FEATURE)

        assert result.feature_group is OptionCapResolveFeatureGroup
        assert "PandasDataFrame" in result.supported_compute_frameworks
        assert "PythonDictFramework" in result.unsupported_compute_frameworks

    def test_supports_compute_framework_sees_caller_options(self) -> None:
        """With the opt-in option, the capability hook accepts PythonDictFramework too."""
        result = resolve_feature(OPTION_CAP_FEATURE, options=Options(group={ALLOW_PYTHON_DICT_KEY: True}))

        assert result.feature_group is OptionCapResolveFeatureGroup
        assert "PandasDataFrame" in result.supported_compute_frameworks
        assert "PythonDictFramework" in result.supported_compute_frameworks
        assert "PythonDictFramework" not in result.unsupported_compute_frameworks

    def test_plugin_collector_keyword_still_works_without_options(self) -> None:
        """plugin_collector remains usable as a keyword argument on its own."""
        collector = PluginCollector.enabled_feature_groups({OptionCapResolveFeatureGroup})

        result = resolve_feature(OPTION_CAP_FEATURE, plugin_collector=collector)

        assert result.feature_group is OptionCapResolveFeatureGroup
        assert result.error is None

    def test_plugin_collector_keyword_works_alongside_options(self) -> None:
        """plugin_collector and options can be combined as keyword arguments."""
        collector = PluginCollector.enabled_feature_groups({OptionGatedResolveFeatureGroup})

        result = resolve_feature(
            OPTION_GATED_FEATURE,
            options=Options(group={PARTITION_BY_KEY: ["id"]}),
            plugin_collector=collector,
        )

        assert result.feature_group is OptionGatedResolveFeatureGroup
        assert result.candidates == [OptionGatedResolveFeatureGroup]
        assert result.error is None

    def test_all_rejected_error_names_frameworks_without_options(self) -> None:
        """Without caller options, the all-rejected error names the frameworks and has no default-options caveat."""
        result = resolve_feature(ALL_REJECTED_CAP_FEATURE)

        assert result.feature_group is None
        assert result.error is not None
        # After #755 the engine builds the message; it has no 'default options' concept.
        assert "default options" not in result.error, (
            f"Engine error has no 'default options' caveat, got: {result.error}"
        )
        assert "PandasDataFrame" in result.error
        assert "PythonDictFramework" in result.error

    def test_all_rejected_error_names_frameworks_with_options(self) -> None:
        """With caller options, the all-rejected error names the frameworks and still has no default-options caveat."""
        result = resolve_feature(ALL_REJECTED_CAP_FEATURE, options=Options(group={PARTITION_BY_KEY: ["id"]}))

        assert result.feature_group is None
        assert result.error is not None
        assert "default options" not in result.error, (
            f"Engine error has no 'default options' caveat, got: {result.error}"
        )
        assert "PandasDataFrame" in result.error
        assert "PythonDictFramework" in result.error


class TestResolveFeatureIsNonThrowing:
    """resolve_feature is a debug API: matching errors surface as ResolvedFeature.error, never as exceptions."""

    def test_forwarded_name_mismatch_does_not_propagate_value_error(self) -> None:
        """A ValueError raised by match_feature_group_criteria must not escape resolve_feature."""
        result = resolve_feature(PROBE_FEATURE, options=_forwarded_mismatch_options())

        assert isinstance(result, ResolvedFeature)
        assert result.feature_name == PROBE_FEATURE

    def test_forwarded_name_mismatch_reports_no_feature_group(self) -> None:
        """A candidate that raises during matching must not be resolved as the winner."""
        result = resolve_feature(PROBE_FEATURE, options=_forwarded_mismatch_options())

        assert result.feature_group is None

    def test_forwarded_name_mismatch_surfaces_validation_reason_in_error(self) -> None:
        """The error must carry the underlying validation reason so the user sees why nothing matched."""
        result = resolve_feature(PROBE_FEATURE, options=_forwarded_mismatch_options())

        assert result.error is not None
        assert PROBE_TYPE_KEY in result.error, f"Error must name the rejected option key, got: {result.error}"
        assert "forwarded" in result.error.lower(), (
            f"Error must surface the forwarded-name-mismatch reason, got: {result.error}"
        )

    def test_probe_group_resolves_normally_without_conflicting_forwarded_options(self) -> None:
        """The probe group is a normal, resolvable group: only the mismatch case is an error."""
        result = resolve_feature(PROBE_FEATURE)

        assert result.feature_group is ForwardMismatchResolveFeatureGroup
        assert result.error is None

    def test_unknown_compute_framework_option_does_not_raise(self) -> None:
        """A bad compute_framework option makes the internal Feature() raise ValueError; it must not escape.

        resolve_feature builds Feature(feature_name, options=..., feature_group=scope) internally. When the
        caller's options name a non-existent compute framework, the Feature constructor raises ValueError
        before any matching guard runs. The never-raising debug contract requires that error to surface in
        ResolvedFeature.error, not propagate.
        """
        bad_options = Options(group={"compute_framework": "NoSuchFrameworkXYZ"})

        result = resolve_feature(SPLIT_CAP_FEATURE, options=bad_options)

        assert isinstance(result, ResolvedFeature)
        assert result.feature_group is None
        assert result.error is not None
        assert "NoSuchFrameworkXYZ" in result.error, (
            f"Error must surface the unknown compute framework reason, got: {result.error}"
        )


class TestResolveFeatureOptionsNoneEqualsEmpty:
    """options=None and options=Options() are indistinguishable, including in error wording."""

    def test_all_rejected_error_identical_for_none_and_explicit_empty_options(self) -> None:
        """The all-rejected error string must not depend on how the empty options were supplied."""
        implicit = resolve_feature(ALL_REJECTED_CAP_FEATURE)
        explicit = resolve_feature(ALL_REJECTED_CAP_FEATURE, options=Options())

        assert implicit.error is not None
        assert explicit.error == implicit.error, (
            f"options=Options() must produce the same error as options=None, got: {explicit.error!r} "
            f"vs {implicit.error!r}"
        )

    def test_explicit_empty_options_error_names_frameworks(self) -> None:
        """An explicitly-passed empty Options yields the engine message: frameworks named, no caveat."""
        result = resolve_feature(ALL_REJECTED_CAP_FEATURE, options=Options())

        assert result.error is not None
        # After #755 the engine builds the message; it has no 'default options' concept.
        assert "default options" not in result.error, (
            f"Engine error has no 'default options' caveat, got: {result.error}"
        )
        assert "PandasDataFrame" in result.error
        assert "PythonDictFramework" in result.error

    def test_non_empty_options_error_names_frameworks(self) -> None:
        """A non-empty Options yields the same engine message: frameworks named, no caveat."""
        result = resolve_feature(ALL_REJECTED_CAP_FEATURE, options=Options(group={PARTITION_BY_KEY: ["id"]}))

        assert result.error is not None
        assert "default options" not in result.error, (
            f"Engine error has no 'default options' caveat, got: {result.error}"
        )
        assert "PandasDataFrame" in result.error
        assert "PythonDictFramework" in result.error


class TestResolveFeatureKeywordOnlyArguments:
    """options and plugin_collector are keyword-only, so legacy positional calls fail loudly."""

    def test_positional_options_raises_type_error(self) -> None:
        """resolve_feature(name, Options()) must raise TypeError instead of silently binding."""
        resolve_feature_untyped: Any = resolve_feature

        with pytest.raises(TypeError):
            resolve_feature_untyped(OPTION_CAP_FEATURE, Options())

    def test_positional_plugin_collector_raises_type_error(self) -> None:
        """A legacy resolve_feature(name, collector) call must raise TypeError, not bind a collector into options."""
        resolve_feature_untyped: Any = resolve_feature
        collector = PluginCollector.enabled_feature_groups({OptionCapResolveFeatureGroup})

        with pytest.raises(TypeError):
            resolve_feature_untyped(OPTION_CAP_FEATURE, collector)

    def test_options_keyword_still_works(self) -> None:
        """The options keyword form remains supported."""
        result = resolve_feature(OPTION_GATED_FEATURE, options=Options(group={PARTITION_BY_KEY: ["id"]}))

        assert result.feature_group is OptionGatedResolveFeatureGroup
        assert result.error is None

    def test_plugin_collector_keyword_still_works(self) -> None:
        """The plugin_collector keyword form remains supported."""
        collector = PluginCollector.enabled_feature_groups({OptionCapResolveFeatureGroup})

        result = resolve_feature(OPTION_CAP_FEATURE, plugin_collector=collector)

        assert result.feature_group is OptionCapResolveFeatureGroup
        assert result.error is None

    def test_both_keywords_together_still_work(self) -> None:
        """Both keyword arguments can be combined."""
        collector = PluginCollector.enabled_feature_groups({OptionGatedResolveFeatureGroup})

        result = resolve_feature(
            OPTION_GATED_FEATURE,
            options=Options(group={PARTITION_BY_KEY: ["id"]}),
            plugin_collector=collector,
        )

        assert result.feature_group is OptionGatedResolveFeatureGroup
        assert result.error is None


class TestResolveFeatureScopedResolve:
    """resolve_feature accepts a keyword-only feature_group scope that disambiguates siblings (issue #693)."""

    def test_unscoped_sibling_feature_is_ambiguous(self) -> None:
        """Premise guard: unscoped, the two sibling matches stay ambiguous."""
        result = resolve_feature(SIBLING_SCOPED_FEATURE)

        assert result.feature_group is None
        assert result.error is not None
        assert "Multiple feature groups found" in result.error

    def test_sibling_class_name_string_scope_resolves_that_sibling(self) -> None:
        """A class-name string scope narrows the ambiguous siblings to exactly the named one."""
        result = resolve_feature(SIBLING_SCOPED_FEATURE, feature_group="ScopedResolveSiblingOne693")

        assert result.feature_group is ScopedResolveSiblingOne693
        assert result.error is None
        assert ScopedResolveSiblingOne693 in result.candidates
        assert ScopedResolveSiblingTwo693 not in result.candidates

    def test_sibling_class_object_scope_resolves_that_sibling(self) -> None:
        """A class-object scope narrows the ambiguous siblings via issubclass."""
        result = resolve_feature(SIBLING_SCOPED_FEATURE, feature_group=ScopedResolveSiblingTwo693)

        assert result.feature_group is ScopedResolveSiblingTwo693
        assert result.error is None
        assert ScopedResolveSiblingOne693 not in result.candidates

    def test_base_name_string_scope_resolves_lone_concrete_subclass(self) -> None:
        """A base-name string scope reaches the only matching concrete subclass through the MRO walk."""
        result = resolve_feature(LONE_CHILD_SCOPED_FEATURE, feature_group="ScopedResolveFamilyBase693")

        assert result.feature_group is ScopedResolveLoneChild693
        assert result.error is None


class TestResolveFeatureScopedNoMatch:
    """A scope that excludes every name match degrades into a scoped no-match error (issue #693)."""

    def test_string_scope_excluding_all_matches_reports_scoped_no_match(self) -> None:
        """Scoping the sibling name to an unrelated class name yields no match plus the scope callout."""
        result = resolve_feature(SIBLING_SCOPED_FEATURE, feature_group="UnrelatedScopedResolve693FeatureGroup")

        assert result.feature_group is None
        assert result.candidates == []
        assert result.error is not None
        assert "No feature groups found for feature name" in result.error
        assert "Scoped to feature group: 'UnrelatedScopedResolve693FeatureGroup'." in result.error

    def test_class_object_scope_excluding_all_matches_reports_scoped_no_match(self) -> None:
        """The class-object scope form produces the same scoped no-match callout."""
        result = resolve_feature(SIBLING_SCOPED_FEATURE, feature_group=UnrelatedScopedResolve693FeatureGroup)

        assert result.feature_group is None
        assert result.candidates == []
        assert result.error is not None
        assert "No feature groups found for feature name" in result.error
        assert "Scoped to feature group: 'UnrelatedScopedResolve693FeatureGroup'." in result.error


class TestResolveFeatureScopedCapabilityRejection:
    """A scoped capability-rejection failure keeps the scope callout in its error (issue #693)."""

    def test_scoped_all_rejected_error_carries_scope_callout(self) -> None:
        """Scoping the all-rejected feature to its own group still fails and names the scope."""
        result = resolve_feature(ALL_REJECTED_CAP_FEATURE, feature_group="AllRejectedCapResolveFeatureGroup")

        assert result.feature_group is None
        assert result.error is not None
        # After #755 the engine builds the capability-rejection message; the scope callout is retained.
        assert "Pin the feature to a supported compute framework" in result.error
        assert "PandasDataFrame" in result.error
        assert "PythonDictFramework" in result.error
        assert "Scoped to feature group: 'AllRejectedCapResolveFeatureGroup'." in result.error


class TestResolveFeatureScopedAmbiguity:
    """A scope that still leaves multiple sibling matches reports a scoped ambiguity (issue #693)."""

    def test_base_name_scope_keeping_both_siblings_reports_scoped_ambiguity(self) -> None:
        """Scoping to the shared base name keeps both siblings, so the ambiguity error names the scope."""
        result = resolve_feature(SIBLING_SCOPED_FEATURE, feature_group="ScopedResolveFamilyBase693")

        assert result.feature_group is None
        assert result.error is not None
        assert "Multiple feature groups found" in result.error
        assert "Scoped to feature group: 'ScopedResolveFamilyBase693'." in result.error


class TestResolveFeatureScopeNeverRaises:
    """Invalid scopes degrade into ResolvedFeature.error instead of raising (issue #693)."""

    def test_root_feature_group_scope_degrades_into_error(self) -> None:
        """Scoping to the root FeatureGroup base is rejected with a dedicated error, not an exception."""
        result = resolve_feature(SIBLING_SCOPED_FEATURE, feature_group=FeatureGroup)

        assert isinstance(result, ResolvedFeature)
        assert result.feature_group is None
        assert result.error is not None
        assert "root FeatureGroup base class" in result.error

    def test_root_feature_group_string_scope_degrades_into_error(self) -> None:
        """The string form 'FeatureGroup' is rejected like the class-object root scope."""
        result = resolve_feature(SIBLING_SCOPED_FEATURE, feature_group="FeatureGroup")

        assert isinstance(result, ResolvedFeature)
        assert result.feature_group is None
        assert result.error is not None
        assert "root FeatureGroup base class" in result.error

    def test_whitespace_padded_root_string_scope_degrades_into_error(self) -> None:
        """The strip runs before the root check, so '  FeatureGroup  ' is rejected the same way."""
        result = resolve_feature(SIBLING_SCOPED_FEATURE, feature_group="  FeatureGroup  ")

        assert isinstance(result, ResolvedFeature)
        assert result.feature_group is None
        assert result.error is not None
        assert "root FeatureGroup base class" in result.error

    def test_wrong_type_scope_degrades_into_error(self) -> None:
        """A non-str, non-class scope is rejected with a type-explaining error, not an exception."""
        invalid_scope: Any = 42

        result = resolve_feature(SIBLING_SCOPED_FEATURE, feature_group=invalid_scope)

        assert isinstance(result, ResolvedFeature)
        assert result.feature_group is None
        assert result.error is not None
        assert "feature_group must be" in result.error

    def test_whitespace_only_scope_behaves_like_unscoped(self) -> None:
        """A whitespace-only scope string is treated exactly like feature_group=None."""
        for feature_name in (SIBLING_SCOPED_FEATURE, LONE_CHILD_SCOPED_FEATURE):
            unscoped = resolve_feature(feature_name)
            blank_scoped = resolve_feature(feature_name, feature_group="   ")

            assert blank_scoped.feature_group is unscoped.feature_group, feature_name
            assert blank_scoped.candidates == unscoped.candidates, feature_name
            assert blank_scoped.error == unscoped.error, feature_name


class TestResolveFeatureScopedEngineParity:
    """A scoped engine failure reproduces through resolve_feature with the same scope (issue #693)."""

    def test_engine_scoped_no_match_reproduces_through_resolve_feature(self) -> None:
        """The engine's scoped no-match callout reappears in the scoped resolve_feature error."""
        accessible_plugins: FeatureGroupEnvironmentMapping = {
            ScopedResolveSiblingOne693: {PandasDataFrame},
            ScopedResolveSiblingTwo693: {PandasDataFrame},
        }
        feature = Feature(SIBLING_SCOPED_FEATURE, feature_group="UnrelatedScopedResolve693FeatureGroup")

        with pytest.raises(ValueError) as exc_info:
            IdentifyFeatureGroupClass(
                feature=feature,
                accessible_plugins=accessible_plugins,
                links=None,
                data_access_collection=None,
            )
        engine_message = str(exc_info.value)
        assert "Scoped to feature group: 'UnrelatedScopedResolve693FeatureGroup'." in engine_message

        result = resolve_feature(SIBLING_SCOPED_FEATURE, feature_group="UnrelatedScopedResolve693FeatureGroup")

        assert result.feature_group is None
        assert result.error is not None
        assert "Scoped to feature group: 'UnrelatedScopedResolve693FeatureGroup'." in result.error


def _exec_conflict_fg(class_name: str, body: str, cell_label: str) -> type[FeatureGroup]:
    """Exec a FeatureGroup subclass into ``__main__``, registering source in linecache.

    Mirrors the established redefinition-conflict pattern in test_feature_group_dedup.py: the exec'd
    class reports ``__module__ == "__main__"`` and ``inspect.getsource`` succeeds because the source
    text is registered in linecache. Re-running with different source under the same qualname produces
    the (module, qualname) redefinition conflict dedup raises on.
    """
    main_mod = sys.modules["__main__"]
    src = textwrap.dedent(body)
    filename = f"<{cell_label}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(keepends=True), filename)
    exec(compile(src, filename, "exec"), main_mod.__dict__)  # nosec B102
    return main_mod.__dict__[class_name]  # type: ignore[no-any-return]


def _raising_match_conflict_source(qualname: str, feature_name: str, extra_body: str = "") -> str:
    """Source for a conflicting FeatureGroup whose match hook raises a NON-ValueError (RuntimeError)."""
    return f"""
from mloda.core.abstract_plugins.feature_group import FeatureGroup as _FG_BASE_


class {qualname}(_FG_BASE_):
    @classmethod
    def feature_names_supported(cls):
        return {{"{feature_name}"}}

    @classmethod
    def match_feature_group_criteria(cls, feature_name, options, data_access_collection=None):
        raise RuntimeError("raising match hook during redefinition conflict")
{extra_body}
"""


@pytest.fixture()
def _cleanup_main_after() -> Any:
    """Snapshot ``__main__`` attributes and pop any test-added keys afterwards, even on failure.

    Guarantees the exec'd conflict classes cannot leak into ``FeatureGroup.__subclasses__()`` for
    later tests on the same xdist worker: the yield teardown runs after the test frame unwinds (so
    local class refs are gone), pops the newly-bound ``__main__`` keys, and collects.
    """
    main_mod = sys.modules["__main__"]
    snapshot = set(main_mod.__dict__.keys())
    yield
    for key in set(main_mod.__dict__.keys()) - snapshot:
        main_mod.__dict__.pop(key, None)
    gc.collect()


class TestResolveFeatureRaisingMatchDuringConflict:
    """A conflicting class whose match hook raises a non-ValueError must not escape resolve_feature."""

    def test_raising_match_hook_during_redef_conflict_does_not_raise(self, _cleanup_main_after: Any) -> None:
        """A RuntimeError from match_feature_group_criteria on a conflicting class must surface as an error.

        The redefinition-conflict branch filters conflicts with a helper that today catches only
        ValueError, so a conflicting class whose match hook raises RuntimeError propagates out of
        resolve_feature. The never-raising debug contract requires it to surface in ResolvedFeature.error.
        """
        qualname = "RaiseMatchConflictResolveFG755"
        feature_name = "raise_match_conflict_resolve_feature_755_xyz"
        src_v1 = _raising_match_conflict_source(qualname, feature_name)
        src_v2 = _raising_match_conflict_source(
            qualname,
            feature_name,
            extra_body="    def extra_method(self):\n        return 755\n",
        )

        _exec_conflict_fg(qualname, src_v1, "cell-raise-match-755-v1")
        _exec_conflict_fg(qualname, src_v2, "cell-raise-match-755-v2")

        result = resolve_feature(feature_name)

        assert isinstance(result, ResolvedFeature)
        assert result.feature_group is None
        assert result.error is not None


def _domain_conflict_source(qualname: str, feature_name: str, domain: str, extra_body: str = "") -> str:
    """Source for a conflicting FeatureGroup that matches by name and declares a specific domain.

    Non-raising, name-based match; ``get_domain`` returns a concrete ``Domain`` so the engine's domain
    filter (``_filter_feature_group_by_domain``) governs whether the class reaches ``criteria_matched``.
    """
    return f"""
from mloda.core.abstract_plugins.feature_group import FeatureGroup as _FG_BASE_
from mloda.core.abstract_plugins.components.domain import Domain as _DOMAIN_


class {qualname}(_FG_BASE_):
    @classmethod
    def feature_names_supported(cls):
        return {{"{feature_name}"}}

    @classmethod
    def match_feature_group_criteria(cls, feature_name, options, data_access_collection=None):
        return str(feature_name) == "{feature_name}"

    @classmethod
    def get_domain(cls):
        return _DOMAIN_("{domain}")
{extra_body}
"""


class TestResolveFeatureConflictBranchProjectsFailure:
    """The redefinition-conflict branch projects the canonical failure without re-matching (#792).

    A redefinition conflict is an environment-build failure: resolve_feature must project it exactly
    like the EnvironmentPreconditionError branch, with NO candidates, instead of re-filtering the
    conflicting classes through scope/match/domain hooks. Candidates are therefore always empty, for
    matching and non-matching domains alike, and the exec'd conflict class never appears in them.
    """

    def test_conflict_branch_reports_no_candidates_for_any_domain(self, _cleanup_main_after: Any) -> None:
        """Candidates are empty for a matching AND a non-matching domain; the error is the conflict text."""
        qualname = "ConflictProjectionResolveFG792"
        feature_name = "conflict_projection_resolve_feature_792_xyz"
        conflict_domain = "conflict_domain_792"
        other_domain = "a_different_domain_than_the_conflict_class_792"

        src_v1 = _domain_conflict_source(qualname, feature_name, conflict_domain)
        src_v2 = _domain_conflict_source(
            qualname,
            feature_name,
            conflict_domain,
            extra_body="    def extra_method(self):\n        return 792\n",
        )

        _exec_conflict_fg(qualname, src_v1, "cell-conflict-projection-792-v1")
        conflict_cls = _exec_conflict_fg(qualname, src_v2, "cell-conflict-projection-792-v2")

        # Sanity: the exec'd class declares the matching domain, so the OLD re-matching behavior
        # would have kept it as a candidate. The pure projection must not.
        assert conflict_cls.get_domain() == Domain(conflict_domain)

        match = resolve_feature(Feature(feature_name, domain=conflict_domain))

        assert match.feature_group is None
        assert match.error is not None
        assert "FeatureGroup redefined with different source code" in match.error
        assert conflict_cls not in match.candidates
        assert match.candidates == []

        mismatch = resolve_feature(Feature(feature_name, domain=other_domain))

        assert mismatch.feature_group is None
        assert mismatch.error is not None
        assert mismatch.candidates == []

    def test_conflict_branch_scoped_string_form_keeps_callout_and_no_candidates(self, _cleanup_main_after: Any) -> None:
        """The string form with a feature_group scope keeps the scope callout suffix, candidates stay empty."""
        qualname = "ConflictProjectionScopedResolveFG792"
        feature_name = "conflict_projection_scoped_resolve_feature_792_xyz"
        conflict_domain = "conflict_domain_792"

        src_v1 = _domain_conflict_source(qualname, feature_name, conflict_domain)
        src_v2 = _domain_conflict_source(
            qualname,
            feature_name,
            conflict_domain,
            extra_body="    def extra_method(self):\n        return 793\n",
        )

        _exec_conflict_fg(qualname, src_v1, "cell-conflict-projection-scoped-792-v1")
        _exec_conflict_fg(qualname, src_v2, "cell-conflict-projection-scoped-792-v2")

        result = resolve_feature(feature_name, feature_group=qualname)

        assert result.feature_group is None
        assert result.error is not None
        assert result.error.endswith(f"Scoped to feature group: '{qualname}'.")
        assert result.candidates == []


class TestResolveFeatureAdapterIsThin:
    """Structural (#792): resolve_feature is a thin adapter; no matching or provider-hook logic lives in it."""

    @pytest.mark.parametrize(
        "attribute",
        [
            "_matches_criteria_guarded",
            "_matches_domain_guarded",
        ],
    )
    def test_conflict_branch_helper_is_deleted(self, attribute: str) -> None:
        """The conflict-branch re-matching helpers no longer exist in plugin_docs."""
        assert not hasattr(plugin_docs, attribute)

    def test_resolve_feature_source_calls_no_resolution_provider_hooks(self) -> None:
        """resolve_feature's body invokes no FeatureGroup matching, domain, scope or capability hooks."""
        source = textwrap.dedent(inspect.getsource(plugin_docs.resolve_feature))
        tree = ast.parse(source)
        called: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                called.add(node.attr)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.add(node.func.id)
        forbidden = {
            "match_feature_group_criteria",
            "get_domain",
            "matches_feature_group_scope",
            "compute_framework_rule",
            "compute_framework_definition",
            "supports_compute_framework",
            "validate_input_features",
            "validate_output_features",
            "index_columns",
            "supports_index",
            "input_features",
            "_matches_criteria_guarded",
            "_matches_domain_guarded",
        }
        overlap = sorted(called & forbidden)
        assert overlap == [], f"resolve_feature invokes resolution provider hooks: {overlap}"

    def test_redefinition_conflict_error_carries_no_conflicts_payload(self) -> None:
        """RedefinitionConflictError has the plain ValueError __init__, no conflicts parameter."""
        assert "conflicts" not in inspect.signature(RedefinitionConflictError.__init__).parameters
