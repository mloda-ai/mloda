"""
Test Polymorphic Link Matching

Tests that Links defined with base classes match when actual
feature groups are subclasses of those base classes.

The current implementation uses exact class name matching:
    Link.matches() compares get_class_name() strings

Expected behavior after implementation:
    Link(BaseClass, ...) should match ConcreteSubclass (polymorphic)
    Link(ConcreteClass, ...) should NOT match BaseClass (reverse fails)
"""

from typing import Any, Optional, Set, cast

from mloda.provider import FeatureGroup
from mloda.core.prepare.graph.graph import Graph
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import Link, JoinSpec
from mloda.user import Options
from mloda.core.prepare.resolve_links import ResolveLinks


# ============================================================================
# Test Feature Groups - Base Classes
# ============================================================================
class BaseGroupA(FeatureGroup):
    """Base class A for testing polymorphic links."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[Set[Any]]:
        return None


class BaseGroupB(FeatureGroup):
    """Base class B for testing polymorphic links."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[Set[Any]]:
        return None


# ============================================================================
# Test Feature Groups - Concrete Subclasses
# ============================================================================
class ConcreteGroupA(BaseGroupA):
    """Concrete implementation of BaseGroupA."""

    pass


class ConcreteGroupB(BaseGroupB):
    """Concrete implementation of BaseGroupB."""

    pass


# ============================================================================
# Test Feature Groups - Unrelated
# ============================================================================
class UnrelatedGroup(FeatureGroup):
    """Unrelated group for negative tests."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[Set[Any]]:
        return None


# ============================================================================
# Test Feature Groups - Multi-level Hierarchy for Distance Testing
# ============================================================================
class GrandparentGroup(FeatureGroup):
    """Three-level hierarchy root."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[Set[Any]]:
        return None


class ParentGroupA(GrandparentGroup):
    """Middle level - branch A."""

    pass


class ParentGroupB(GrandparentGroup):
    """Middle level - branch B (sibling of ParentGroupA)."""

    pass


class ChildGroupA(ParentGroupA):
    """Leaf level - under ParentGroupA."""

    pass


# ============================================================================
# Unit Tests for Link.matches() Polymorphic Behavior
# ============================================================================
class TestLinkMatchesPolymorphic:
    """Unit tests for Link.matches() polymorphic behavior."""

    def test_exact_match(self) -> None:
        """Test that exact class match still works."""
        # Link defined with ConcreteGroupA and ConcreteGroupB
        link = Link.inner(
            JoinSpec(ConcreteGroupA, Index(("id",))),
            JoinSpec(ConcreteGroupB, Index(("id",))),
        )

        # Should match the exact same classes
        assert link.matches(ConcreteGroupA, ConcreteGroupB) is True

    def test_polymorphic_match_subclass(self) -> None:
        """Test that link with base class matches concrete subclass."""
        # Link defined with BASE classes
        link = Link.inner(
            JoinSpec(BaseGroupA, Index(("id",))),
            JoinSpec(BaseGroupB, Index(("id",))),
        )

        # Should match when we provide CONCRETE subclasses
        # This is the KEY polymorphic behavior: base type accepts subtype
        assert link.matches(ConcreteGroupA, ConcreteGroupB) is True

    def test_polymorphic_match_base_class_itself(self) -> None:
        """Test that link with base class matches the base class itself."""
        # Link defined with base classes
        link = Link.inner(
            JoinSpec(BaseGroupA, Index(("id",))),
            JoinSpec(BaseGroupB, Index(("id",))),
        )

        # Should match the base classes themselves (exact match is also valid)
        assert link.matches(BaseGroupA, BaseGroupB) is True

    def test_no_match_wrong_hierarchy(self) -> None:
        """Test that link does NOT match unrelated classes."""
        # Link defined with BaseGroupA and BaseGroupB
        link = Link.inner(
            JoinSpec(BaseGroupA, Index(("id",))),
            JoinSpec(BaseGroupB, Index(("id",))),
        )

        # Should NOT match when we swap sides with unrelated class
        assert link.matches(UnrelatedGroup, ConcreteGroupB) is False
        assert link.matches(ConcreteGroupA, UnrelatedGroup) is False

    def test_no_match_concrete_to_base(self) -> None:
        """Test that link with concrete class does NOT match base class."""
        # Link defined with CONCRETE classes
        link = Link.inner(
            JoinSpec(ConcreteGroupA, Index(("id",))),
            JoinSpec(ConcreteGroupB, Index(("id",))),
        )

        # Should NOT match when we provide BASE classes
        # This is reverse polymorphism and should FAIL
        # (can't assign base type to subtype variable)
        assert link.matches(BaseGroupA, BaseGroupB) is False

    def test_partial_match_fails_left_side(self) -> None:
        """Test that partial match (left side matches, right doesn't) fails."""
        # Link defined with base classes
        link = Link.inner(
            JoinSpec(BaseGroupA, Index(("id",))),
            JoinSpec(BaseGroupB, Index(("id",))),
        )

        # Left side matches (ConcreteGroupA is subclass of BaseGroupA)
        # Right side does NOT match (UnrelatedGroup is not subclass of BaseGroupB)
        assert link.matches(ConcreteGroupA, UnrelatedGroup) is False

    def test_partial_match_fails_right_side(self) -> None:
        """Test that partial match (right side matches, left doesn't) fails."""
        # Link defined with base classes
        link = Link.inner(
            JoinSpec(BaseGroupA, Index(("id",))),
            JoinSpec(BaseGroupB, Index(("id",))),
        )

        # Left side does NOT match (UnrelatedGroup is not subclass of BaseGroupA)
        # Right side matches (ConcreteGroupB is subclass of BaseGroupB)
        assert link.matches(UnrelatedGroup, ConcreteGroupB) is False

    def test_polymorphic_match_mixed(self) -> None:
        """Test polymorphic match with one exact match and one subclass match."""
        # Link defined with one base and one concrete
        link = Link.inner(
            JoinSpec(BaseGroupA, Index(("id",))),
            JoinSpec(ConcreteGroupB, Index(("id",))),
        )

        # Left: subclass of BaseGroupA (polymorphic match)
        # Right: exact match with ConcreteGroupB
        assert link.matches(ConcreteGroupA, ConcreteGroupB) is True

    def test_different_join_types_work_same(self) -> None:
        """Test that polymorphic matching works for all join types."""
        # Test LEFT join
        left_link = Link.left(
            JoinSpec(BaseGroupA, Index(("id",))),
            JoinSpec(BaseGroupB, Index(("id",))),
        )
        assert left_link.matches(ConcreteGroupA, ConcreteGroupB) is True

        # Test RIGHT join
        right_link = Link.right(
            JoinSpec(BaseGroupA, Index(("id",))),
            JoinSpec(BaseGroupB, Index(("id",))),
        )
        assert right_link.matches(ConcreteGroupA, ConcreteGroupB) is True

        # Test OUTER join
        outer_link = Link.outer(
            JoinSpec(BaseGroupA, Index(("id",))),
            JoinSpec(BaseGroupB, Index(("id",))),
        )
        assert outer_link.matches(ConcreteGroupA, ConcreteGroupB) is True

        # Test APPEND
        append_link = Link.append(
            JoinSpec(BaseGroupA, Index(("id",))),
            JoinSpec(BaseGroupB, Index(("id",))),
        )
        assert append_link.matches(ConcreteGroupA, ConcreteGroupB) is True

        # Test UNION
        union_link = Link.union(
            JoinSpec(BaseGroupA, Index(("id",))),
            JoinSpec(BaseGroupB, Index(("id",))),
        )
        assert union_link.matches(ConcreteGroupA, ConcreteGroupB) is True


# ============================================================================
# Unit Tests for ResolveLinks._inheritance_distance()
# ============================================================================
class TestInheritanceDistance:
    """Unit tests for calculating MRO-based inheritance distance."""

    def test_distance_to_self_is_zero(self) -> None:
        """Test that distance from a class to itself is 0."""
        resolver = ResolveLinks(graph=cast(Graph, None), links=None)

        # Distance from GrandparentGroup to itself should be 0
        distance = resolver._inheritance_distance(GrandparentGroup, GrandparentGroup)
        assert distance == 0

    def test_distance_to_direct_parent(self) -> None:
        """Test that distance to immediate parent is 1."""
        resolver = ResolveLinks(graph=cast(Graph, None), links=None)

        # ParentGroupA -> GrandparentGroup is 1 step
        distance = resolver._inheritance_distance(ParentGroupA, GrandparentGroup)
        assert distance == 1

    def test_distance_to_grandparent(self) -> None:
        """Test that distance to grandparent is 2."""
        resolver = ResolveLinks(graph=cast(Graph, None), links=None)

        # ChildGroupA -> ParentGroupA -> GrandparentGroup is 2 steps
        distance = resolver._inheritance_distance(ChildGroupA, GrandparentGroup)
        assert distance == 2

    def test_distance_to_unrelated_class(self) -> None:
        """Test that distance to unrelated class is large sentinel value."""
        resolver = ResolveLinks(graph=cast(Graph, None), links=None)

        # ParentGroupA is not related to UnrelatedGroup
        distance = resolver._inheritance_distance(ParentGroupA, UnrelatedGroup)
        assert distance == 9999


# ============================================================================
# Unit Tests for ResolveLinks._select_most_specific_links()
# ============================================================================
class TestSelectMostSpecificLinks:
    """Unit tests for selecting the most specific (closest) polymorphic links."""

    def test_most_specific_link_wins(self) -> None:
        """Test that the most specific (closest) link is selected over broader links."""
        # Create links at different hierarchy levels
        grandparent_link = Link.inner(
            JoinSpec(GrandparentGroup, Index(("id",))),
            JoinSpec(GrandparentGroup, Index(("id",))),
        )
        parent_link = Link.inner(
            JoinSpec(ParentGroupA, Index(("id",))),
            JoinSpec(ParentGroupA, Index(("id",))),
        )

        links = [grandparent_link, parent_link]
        resolver = ResolveLinks(graph=cast(Graph, None), links=None)

        # Query with ChildGroupA on both sides
        # ParentGroupA is closer (distance 1) than GrandparentGroup (distance 2)
        result = resolver._select_most_specific_links(links, ChildGroupA, ChildGroupA)

        assert len(result) == 1
        assert result[0] == parent_link

    def test_sibling_matching_behavior(self) -> None:
        """Test that sibling classes (different branches) should NOT match.

        Per documentation: Link(GrandparentGroup, GrandparentGroup) should NOT match
        (ParentGroupA, ParentGroupB) because they are siblings from different branches.

        Even though both have distance 1 to GrandparentGroup, the link expects the
        SAME class on both sides, not different siblings.
        """
        # Link defined with common ancestor
        grandparent_link = Link.inner(
            JoinSpec(GrandparentGroup, Index(("id",))),
            JoinSpec(GrandparentGroup, Index(("id",))),
        )

        links = [grandparent_link]
        resolver = ResolveLinks(graph=cast(Graph, None), links=None)

        # Query with siblings ParentGroupA and ParentGroupB
        # These are different classes from different branches
        # Link expects (GrandparentGroup, GrandparentGroup) - same class on both sides
        # Therefore, siblings should NOT match
        result = resolver._select_most_specific_links(links, ParentGroupA, ParentGroupB)

        # Expected behavior: siblings should NOT match
        assert len(result) == 0  # Siblings should NOT match

    def test_same_class_both_sides_matches(self) -> None:
        """Test that querying with same class on both sides matches correctly."""
        # Link with common ancestor
        grandparent_link = Link.inner(
            JoinSpec(GrandparentGroup, Index(("id",))),
            JoinSpec(GrandparentGroup, Index(("id",))),
        )

        links = [grandparent_link]
        resolver = ResolveLinks(graph=cast(Graph, None), links=None)

        # Query with ParentGroupA on both sides
        # Both have distance 1, so it's balanced
        result = resolver._select_most_specific_links(links, ParentGroupA, ParentGroupA)

        assert len(result) == 1
        assert result[0] == grandparent_link

    def test_unbalanced_inheritance_rejected(self) -> None:
        """Test that links with unbalanced inheritance distances are rejected."""
        # Link at grandparent level
        grandparent_link = Link.inner(
            JoinSpec(GrandparentGroup, Index(("id",))),
            JoinSpec(GrandparentGroup, Index(("id",))),
        )

        links = [grandparent_link]
        resolver = ResolveLinks(graph=cast(Graph, None), links=None)

        # Query with ParentGroupA (distance 1) and ChildGroupA (distance 2)
        # Distances don't match (1 != 2), so should be rejected
        result = resolver._select_most_specific_links(links, ParentGroupA, ChildGroupA)

        assert len(result) == 0


# ============================================================================
# Unit Tests for ResolveLinks._find_matching_links()
# ============================================================================
class TestFindMatchingLinks:
    """Unit tests for two-pass link matching (exact first, then polymorphic)."""

    def test_exact_match_priority(self) -> None:
        """Test that exact matches take priority over polymorphic matches."""
        # Create links at different levels
        parent_link = Link.inner(
            JoinSpec(ParentGroupA, Index(("id",))),
            JoinSpec(ParentGroupA, Index(("id",))),
        )
        child_link = Link.inner(
            JoinSpec(ChildGroupA, Index(("id",))),
            JoinSpec(ChildGroupA, Index(("id",))),
        )

        links = {parent_link, child_link}
        resolver = ResolveLinks(graph=cast(Graph, None), links=links)

        # Query with ChildGroupA - should return ONLY exact match, not polymorphic
        result = resolver._find_matching_links(ChildGroupA, ChildGroupA)

        assert len(result) == 1
        assert result[0] == child_link

    def test_polymorphic_fallback(self) -> None:
        """Test that polymorphic matching works when no exact match exists."""
        # Only parent-level link
        parent_link = Link.inner(
            JoinSpec(ParentGroupA, Index(("id",))),
            JoinSpec(ParentGroupA, Index(("id",))),
        )

        links = {parent_link}
        resolver = ResolveLinks(graph=cast(Graph, None), links=links)

        # Query with ChildGroupA - should fall back to polymorphic match
        result = resolver._find_matching_links(ChildGroupA, ChildGroupA)

        assert len(result) == 1
        assert result[0] == parent_link
