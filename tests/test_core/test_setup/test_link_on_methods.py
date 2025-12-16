"""
Test Link._on Factory Methods

Tests for the Link._on suffix factory methods that accept feature groups directly
and derive JoinSpecs automatically from index_columns().

Usage:
    # Instead of verbose JoinSpec construction:
    Link.inner(
        JoinSpec(UserFG, UserFG.index_columns()[0]),
        JoinSpec(OrderFG, OrderFG.index_columns()[0])
    )

    # Use the convenient _on methods:
    Link.inner_on(UserFG, OrderFG)

    # With multi-index selection:
    Link.inner_on(UserFG, OrderFG, left_index=1, right_index=0)

    # Self-joins with aliases:
    Link.inner_on(UserFG, UserFG, self_left_alias={"side": "manager"}, self_right_alias={"side": "employee"})

See Also:
    - GitHub Issue #133: JoinSpec Convenience Function
    - Design Decision: /memory-bank/design-decisions/joinspec-convenience-function.md
"""

from typing import Any, List, Optional, Set

import pytest

from mloda import FeatureGroup
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import JoinSpec, JoinType, Link
from mloda import Options


# ============================================================================
# Mock Feature Groups for Testing
# ============================================================================
class MockFGWithSingleIndex(FeatureGroup):
    """Mock feature group with a single index column."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[Set[Any]]:
        return None

    @classmethod
    def index_columns(cls) -> Optional[List[Index]]:
        return [Index(("id",))]


class MockFGWithMultipleIndexes(FeatureGroup):
    """Mock feature group with multiple index options."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[Set[Any]]:
        return None

    @classmethod
    def index_columns(cls) -> Optional[List[Index]]:
        return [
            Index(("id",)),
            Index(("user_id", "timestamp")),
            Index(("order_id",)),
        ]


class MockFGWithNoIndex(FeatureGroup):
    """Mock feature group that returns None for index_columns (default behavior)."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[Set[Any]]:
        return None

    # Uses default implementation which returns None


class MockFGWithEmptyIndex(FeatureGroup):
    """Mock feature group that returns empty list for index_columns."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[Set[Any]]:
        return None

    @classmethod
    def index_columns(cls) -> Optional[List[Index]]:
        return []


class AnotherMockFGWithSingleIndex(FeatureGroup):
    """Another mock feature group with a single index - for distinct feature group testing."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[Set[Any]]:
        return None

    @classmethod
    def index_columns(cls) -> Optional[List[Index]]:
        return [Index(("user_id",))]


# ============================================================================
# Test Link.inner_on Method
# ============================================================================
class TestLinkInnerOn:
    """Test Link.inner_on factory method with automatic index derivation."""

    def test_inner_on_creates_link_with_default_indexes(self) -> None:
        """Test Link.inner_on creates an INNER join using first index from each feature group."""
        link = Link.inner_on(MockFGWithSingleIndex, AnotherMockFGWithSingleIndex)

        # Should create an INNER join
        assert link.jointype == JoinType.INNER

        # Should use first index from each feature group
        assert link.left_feature_group is MockFGWithSingleIndex
        assert link.right_feature_group is AnotherMockFGWithSingleIndex
        assert link.left_index == Index(("id",))
        assert link.right_index == Index(("user_id",))

    def test_inner_on_with_index_selection(self) -> None:
        """Test Link.inner_on with explicit left_index and right_index parameters."""
        link = Link.inner_on(
            MockFGWithMultipleIndexes,
            MockFGWithMultipleIndexes,
            left_index=1,  # Select second index: Index(("user_id", "timestamp"))
            right_index=2,  # Select third index: Index(("order_id",))
        )

        assert link.jointype == JoinType.INNER
        assert link.left_index == Index(("user_id", "timestamp"))
        assert link.right_index == Index(("order_id",))

    def test_inner_on_with_aliases(self) -> None:
        """Test Link.inner_on supports self_left_alias and self_right_alias for self-joins."""
        left_alias = {"side": "manager"}
        right_alias = {"side": "employee"}

        link = Link.inner_on(
            MockFGWithSingleIndex,
            MockFGWithSingleIndex,
            self_left_alias=left_alias,
            self_right_alias=right_alias,
        )

        assert link.jointype == JoinType.INNER
        assert link.left_feature_group is MockFGWithSingleIndex
        assert link.right_feature_group is MockFGWithSingleIndex
        assert link.self_left_alias == left_alias
        assert link.self_right_alias == right_alias

    def test_inner_on_raises_value_error_on_none_index_columns(self) -> None:
        """Test Link.inner_on raises ValueError when feature group returns None for index_columns."""
        with pytest.raises(ValueError, match="[Ii]ndex.*None|No.*index|index_columns.*None"):
            Link.inner_on(MockFGWithNoIndex, MockFGWithSingleIndex)

    def test_inner_on_raises_value_error_on_empty_index_columns(self) -> None:
        """Test Link.inner_on raises ValueError when feature group returns empty list."""
        with pytest.raises(ValueError, match="[Ii]ndex.*empty|No.*index|empty.*list"):
            Link.inner_on(MockFGWithEmptyIndex, MockFGWithSingleIndex)

    def test_inner_on_raises_index_error_on_invalid_left_index(self) -> None:
        """Test Link.inner_on raises IndexError when left_index is out of range."""
        with pytest.raises(IndexError, match="[Ii]ndex.*range|out of.*range"):
            Link.inner_on(
                MockFGWithSingleIndex,
                MockFGWithSingleIndex,
                left_index=5,  # Out of range (only has 1 index)
            )

    def test_inner_on_raises_index_error_on_invalid_right_index(self) -> None:
        """Test Link.inner_on raises IndexError when right_index is out of range."""
        with pytest.raises(IndexError, match="[Ii]ndex.*range|out of.*range"):
            Link.inner_on(
                MockFGWithSingleIndex,
                MockFGWithSingleIndex,
                right_index=10,  # Out of range
            )

    def test_inner_on_equivalent_to_manual_construction(self) -> None:
        """Test Link.inner_on produces same result as manual JoinSpec construction."""
        link_on = Link.inner_on(MockFGWithSingleIndex, AnotherMockFGWithSingleIndex)

        # Manual construction for comparison
        left_spec = JoinSpec(MockFGWithSingleIndex, Index(("id",)))
        right_spec = JoinSpec(AnotherMockFGWithSingleIndex, Index(("user_id",)))
        link_manual = Link.inner(left_spec, right_spec)

        # Should be equivalent
        assert link_on.jointype == link_manual.jointype
        assert link_on.left_feature_group is link_manual.left_feature_group
        assert link_on.right_feature_group is link_manual.right_feature_group
        assert link_on.left_index == link_manual.left_index
        assert link_on.right_index == link_manual.right_index


# ============================================================================
# Test All _on Methods Exist
# ============================================================================
class TestAllOnMethodsExist:
    """Test that all six _on factory methods are implemented and work correctly."""

    def test_left_on_method_exists(self) -> None:
        """Test Link.left_on creates a LEFT join."""
        link = Link.left_on(MockFGWithSingleIndex, AnotherMockFGWithSingleIndex)

        assert link.jointype == JoinType.LEFT
        assert link.left_feature_group is MockFGWithSingleIndex
        assert link.right_feature_group is AnotherMockFGWithSingleIndex

    def test_right_on_method_exists(self) -> None:
        """Test Link.right_on creates a RIGHT join."""
        link = Link.right_on(MockFGWithSingleIndex, AnotherMockFGWithSingleIndex)

        assert link.jointype == JoinType.RIGHT
        assert link.left_feature_group is MockFGWithSingleIndex
        assert link.right_feature_group is AnotherMockFGWithSingleIndex

    def test_outer_on_method_exists(self) -> None:
        """Test Link.outer_on creates an OUTER join."""
        link = Link.outer_on(MockFGWithSingleIndex, AnotherMockFGWithSingleIndex)

        assert link.jointype == JoinType.OUTER
        assert link.left_feature_group is MockFGWithSingleIndex
        assert link.right_feature_group is AnotherMockFGWithSingleIndex

    def test_append_on_method_exists(self) -> None:
        """Test Link.append_on creates an APPEND join."""
        link = Link.append_on(MockFGWithSingleIndex, AnotherMockFGWithSingleIndex)

        assert link.jointype == JoinType.APPEND
        assert link.left_feature_group is MockFGWithSingleIndex
        assert link.right_feature_group is AnotherMockFGWithSingleIndex

    def test_union_on_method_exists(self) -> None:
        """Test Link.union_on creates a UNION join."""
        link = Link.union_on(MockFGWithSingleIndex, AnotherMockFGWithSingleIndex)

        assert link.jointype == JoinType.UNION
        assert link.left_feature_group is MockFGWithSingleIndex
        assert link.right_feature_group is AnotherMockFGWithSingleIndex


# ============================================================================
# Test Edge Cases and Validation
# ============================================================================
class TestOnMethodsEdgeCases:
    """Test edge cases and validation for _on methods."""

    def test_inner_on_with_default_index_zero(self) -> None:
        """Test that omitting left_index/right_index uses index position 0."""
        link = Link.inner_on(MockFGWithMultipleIndexes, MockFGWithMultipleIndexes)

        # Should default to first index (position 0)
        assert link.left_index == Index(("id",))
        assert link.right_index == Index(("id",))

    def test_inner_on_with_multi_column_index(self) -> None:
        """Test Link.inner_on works with multi-column indexes."""
        link = Link.inner_on(
            MockFGWithMultipleIndexes,
            MockFGWithMultipleIndexes,
            left_index=1,  # Multi-column index
            right_index=1,
        )

        assert link.left_index == Index(("user_id", "timestamp"))
        assert link.right_index == Index(("user_id", "timestamp"))
        assert link.left_index.is_multi_index() is True
        assert link.right_index.is_multi_index() is True

    def test_inner_on_with_mixed_index_selection(self) -> None:
        """Test Link.inner_on with different index positions for left and right."""
        link = Link.inner_on(
            MockFGWithMultipleIndexes,
            MockFGWithMultipleIndexes,
            left_index=0,  # Single-column index
            right_index=1,  # Multi-column index
        )

        assert link.left_index == Index(("id",))
        assert link.right_index == Index(("user_id", "timestamp"))

    def test_inner_on_preserves_feature_group_identity(self) -> None:
        """Test that Link.inner_on preserves feature group class references (not instances)."""
        link = Link.inner_on(MockFGWithSingleIndex, AnotherMockFGWithSingleIndex)

        # Should reference the class, not an instance
        assert link.left_feature_group is MockFGWithSingleIndex
        assert link.right_feature_group is AnotherMockFGWithSingleIndex
        assert isinstance(link.left_feature_group, type)
        assert isinstance(link.right_feature_group, type)

    def test_inner_on_negative_index_raises_error(self) -> None:
        """Test Link.inner_on raises error for negative index values."""
        with pytest.raises((ValueError, IndexError), match="[Ii]ndex|[Nn]egative"):
            Link.inner_on(
                MockFGWithMultipleIndexes,
                MockFGWithMultipleIndexes,
                left_index=-1,  # Negative index not allowed
            )
