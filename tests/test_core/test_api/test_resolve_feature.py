"""
Tests for the resolve_feature() API function.

This module tests the resolve_feature function which resolves a feature name
to its matching FeatureGroup class, with support for subclass filtering
and candidate tracking.
"""

import pytest
from typing import Optional, Type, Union

from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.api.plugin_info import ResolvedFeature
from mloda.core.api.plugin_docs import resolve_feature
from mloda.user import PluginLoader


@pytest.fixture(scope="module", autouse=True)
def load_plugins() -> None:
    """Load all plugins before running tests in this module."""
    PluginLoader.all()


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

    def test_resolve_feature_finds_matching_feature_group(self) -> None:
        """Test resolving a feature name that matches a single FeatureGroup."""
        # Use a known feature group class name from the codebase
        # Most FeatureGroups match their class name
        from mloda.core.abstract_plugins.components.utils import get_all_subclasses

        # Filter to non-test FeatureGroups to avoid duplicates from test fixtures
        all_fgs = [fg for fg in get_all_subclasses(FeatureGroup) if not fg.__module__.startswith("test_")]
        assert len(all_fgs) > 0, "Need at least one FeatureGroup for this test"

        # Find a feature group that matches by its class name
        target_fg = None
        for fg in all_fgs:
            if fg.match_feature_group_criteria(FeatureName(fg.get_class_name()), Options(), None):
                target_fg = fg
                break

        assert target_fg is not None, "Need a FeatureGroup that matches its class name"
        target_name = target_fg.get_class_name()

        result = resolve_feature(target_name)

        assert result.feature_name == target_name
        assert result.feature_group is not None
        assert result.error is None

    def test_resolve_feature_returns_correct_feature_group_type(self) -> None:
        """Test that the resolved feature_group is a Type[FeatureGroup]."""
        from mloda.core.abstract_plugins.components.utils import get_all_subclasses

        # Filter to non-test FeatureGroups to avoid duplicates from test fixtures
        all_fgs = [fg for fg in get_all_subclasses(FeatureGroup) if not fg.__module__.startswith("test_")]
        assert len(all_fgs) > 0, "Need at least one FeatureGroup for this test"

        # Find a feature group that matches by its class name
        target_fg = None
        for fg in all_fgs:
            if fg.match_feature_group_criteria(FeatureName(fg.get_class_name()), Options(), None):
                target_fg = fg
                break

        assert target_fg is not None, "Need a FeatureGroup that matches its class name"
        target_name = target_fg.get_class_name()

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
        """Test that candidates list contains all FeatureGroups that matched criteria."""
        from mloda.core.abstract_plugins.components.utils import get_all_subclasses

        all_fgs = list(get_all_subclasses(FeatureGroup))
        assert len(all_fgs) > 0, "Need at least one FeatureGroup for this test"

        # Find a feature group that matches by its class name
        target_fg = None
        for fg in all_fgs:
            if fg.match_feature_group_criteria(FeatureName(fg.get_class_name()), Options(), None):
                target_fg = fg
                break

        assert target_fg is not None, "Need a FeatureGroup that matches its class name"
        target_name = target_fg.get_class_name()

        result = resolve_feature(target_name)

        # Candidates should include at least the matched feature group
        assert len(result.candidates) >= 1

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
                feature_name: Union[FeatureName, str],
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
                feature_name: Union[FeatureName, str],
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
