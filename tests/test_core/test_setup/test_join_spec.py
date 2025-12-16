"""
Test JoinSpec Frozen Dataclass

Tests for the JoinSpec dataclass that will replace anonymous tuples in the Link class.
JoinSpec is a self-documenting frozen dataclass that holds feature_group and index.

These tests MUST fail initially because JoinSpec does not exist yet.
They will pass once JoinSpec is implemented as a frozen dataclass.
"""

from dataclasses import FrozenInstanceError
from typing import Any, Optional, Set

import pytest

from mloda import FeatureGroup
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import JoinSpec  # This import will fail
from mloda import Options


# ============================================================================
# Mock Feature Group for Testing
# ============================================================================
class MockFeatureGroup(FeatureGroup):
    """Minimal feature group for testing JoinSpec."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[Set[Any]]:
        return None


class AnotherMockFeatureGroup(FeatureGroup):
    """Another feature group for testing equality and hashing."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[Set[Any]]:
        return None


# ============================================================================
# Test JoinSpec Instantiation
# ============================================================================
class TestJoinSpecInstantiation:
    """Test that JoinSpec can be instantiated correctly."""

    def test_create_join_spec_with_feature_group_and_index(self) -> None:
        """Test JoinSpec can be created with feature_group and index parameters."""
        idx = Index(("id",))

        # Should be able to instantiate JoinSpec
        spec = JoinSpec(feature_group=MockFeatureGroup, index=idx)

        # Verify instance was created
        assert spec is not None
        assert isinstance(spec, JoinSpec)

    def test_join_spec_has_feature_group_attribute(self) -> None:
        """Test that JoinSpec has accessible feature_group attribute."""
        idx = Index(("id",))
        spec = JoinSpec(feature_group=MockFeatureGroup, index=idx)

        # Should be able to access feature_group by name
        assert spec.feature_group is MockFeatureGroup

    def test_join_spec_has_index_attribute(self) -> None:
        """Test that JoinSpec has accessible index attribute."""
        idx = Index(("id",))
        spec = JoinSpec(feature_group=MockFeatureGroup, index=idx)

        # Should be able to access index by name
        assert spec.index is idx
        assert spec.index == idx

    def test_join_spec_with_multi_index(self) -> None:
        """Test JoinSpec works with multi-column indexes."""
        idx = Index(("user_id", "timestamp"))
        spec = JoinSpec(feature_group=MockFeatureGroup, index=idx)

        assert spec.feature_group is MockFeatureGroup
        assert spec.index == idx
        assert spec.index.is_multi_index() is True


# ============================================================================
# Test JoinSpec Immutability (frozen=True)
# ============================================================================
class TestJoinSpecImmutability:
    """Test that JoinSpec is frozen and immutable."""

    def test_cannot_modify_feature_group_after_creation(self) -> None:
        """Test that feature_group cannot be reassigned after instantiation."""
        idx = Index(("id",))
        spec = JoinSpec(feature_group=MockFeatureGroup, index=idx)

        # Attempting to modify feature_group should raise FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            spec.feature_group = AnotherMockFeatureGroup

    def test_cannot_modify_index_after_creation(self) -> None:
        """Test that index cannot be reassigned after instantiation."""
        idx = Index(("id",))
        spec = JoinSpec(feature_group=MockFeatureGroup, index=idx)

        # Attempting to modify index should raise FrozenInstanceError
        new_idx = Index(("other_id",))
        with pytest.raises(FrozenInstanceError):
            spec.index = new_idx

    def test_cannot_add_new_attributes(self) -> None:
        """Test that new attributes cannot be added to frozen dataclass."""
        idx = Index(("id",))
        spec = JoinSpec(feature_group=MockFeatureGroup, index=idx)

        # Attempting to add new attribute should raise FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            spec.new_attribute = "value"


# ============================================================================
# Test JoinSpec Equality
# ============================================================================
class TestJoinSpecEquality:
    """Test that JoinSpec supports equality comparison."""

    def test_equal_join_specs_are_equal(self) -> None:
        """Test that two JoinSpecs with same feature_group and index are equal."""
        idx = Index(("id",))
        spec1 = JoinSpec(feature_group=MockFeatureGroup, index=idx)
        spec2 = JoinSpec(feature_group=MockFeatureGroup, index=idx)

        # Should be equal (same feature_group and index)
        assert spec1 == spec2

    def test_different_feature_groups_not_equal(self) -> None:
        """Test that JoinSpecs with different feature_groups are not equal."""
        idx = Index(("id",))
        spec1 = JoinSpec(feature_group=MockFeatureGroup, index=idx)
        spec2 = JoinSpec(feature_group=AnotherMockFeatureGroup, index=idx)

        # Should NOT be equal (different feature_groups)
        assert spec1 != spec2

    def test_different_indexes_not_equal(self) -> None:
        """Test that JoinSpecs with different indexes are not equal."""
        idx1 = Index(("id",))
        idx2 = Index(("other_id",))
        spec1 = JoinSpec(feature_group=MockFeatureGroup, index=idx1)
        spec2 = JoinSpec(feature_group=MockFeatureGroup, index=idx2)

        # Should NOT be equal (different indexes)
        assert spec1 != spec2

    def test_equality_with_different_index_objects_same_value(self) -> None:
        """Test that JoinSpecs are equal when indexes have same value but different objects."""
        idx1 = Index(("id",))
        idx2 = Index(("id",))  # Different object, same value
        spec1 = JoinSpec(feature_group=MockFeatureGroup, index=idx1)
        spec2 = JoinSpec(feature_group=MockFeatureGroup, index=idx2)

        # Should be equal (indexes have same value)
        assert spec1 == spec2


# ============================================================================
# Test JoinSpec Hashability
# ============================================================================
class TestJoinSpecHashability:
    """Test that JoinSpec is hashable and can be used in sets/dicts."""

    def test_join_spec_is_hashable(self) -> None:
        """Test that JoinSpec can be hashed."""
        idx = Index(("id",))
        spec = JoinSpec(feature_group=MockFeatureGroup, index=idx)

        # Should be able to call hash() without error
        hash_value = hash(spec)
        assert isinstance(hash_value, int)

    def test_join_spec_can_be_added_to_set(self) -> None:
        """Test that JoinSpec instances can be added to a set."""
        idx = Index(("id",))
        spec1 = JoinSpec(feature_group=MockFeatureGroup, index=idx)
        spec2 = JoinSpec(feature_group=AnotherMockFeatureGroup, index=idx)

        # Should be able to create a set with JoinSpec instances
        spec_set = {spec1, spec2}
        assert len(spec_set) == 2
        assert spec1 in spec_set
        assert spec2 in spec_set

    def test_equal_join_specs_have_same_hash(self) -> None:
        """Test that equal JoinSpecs have the same hash value."""
        idx = Index(("id",))
        spec1 = JoinSpec(feature_group=MockFeatureGroup, index=idx)
        spec2 = JoinSpec(feature_group=MockFeatureGroup, index=idx)

        # Equal objects should have equal hashes
        assert spec1 == spec2
        assert hash(spec1) == hash(spec2)

    def test_join_spec_can_be_dict_key(self) -> None:
        """Test that JoinSpec can be used as dictionary key."""
        idx = Index(("id",))
        spec = JoinSpec(feature_group=MockFeatureGroup, index=idx)

        # Should be able to use JoinSpec as dict key
        mapping = {spec: "some_value"}
        assert mapping[spec] == "some_value"

    def test_duplicate_join_specs_deduplicated_in_set(self) -> None:
        """Test that duplicate JoinSpecs are deduplicated in a set."""
        idx = Index(("id",))
        spec1 = JoinSpec(feature_group=MockFeatureGroup, index=idx)
        spec2 = JoinSpec(feature_group=MockFeatureGroup, index=idx)

        # Adding duplicates to set should result in only one entry
        spec_set = {spec1, spec2}
        assert len(spec_set) == 1


# ============================================================================
# Test Link Class with JoinSpec
# ============================================================================
class TestLinkWithJoinSpec:
    """Test that Link class can use JoinSpec objects instead of tuples.

    These tests will FAIL initially because Link.__init__ still expects
    Tuple[Type[Any], Index] for left and right parameters, not JoinSpec objects.

    The tests will pass once Link is updated to accept JoinSpec objects.
    """

    def test_link_instantiation_with_join_spec_objects(self) -> None:
        """Test Link can be instantiated with JoinSpec objects for left and right."""
        from mloda.user import Link, JoinType

        left_idx = Index(("user_id",))
        right_idx = Index(("user_id",))

        left_spec = JoinSpec(feature_group=MockFeatureGroup, index=left_idx)
        right_spec = JoinSpec(feature_group=AnotherMockFeatureGroup, index=right_idx)

        # Should accept JoinSpec objects directly (will fail with current tuple-based signature)
        link = Link(
            jointype=JoinType.INNER,
            left=left_spec,
            right=right_spec,
        )

        # Verify link was created
        assert link is not None
        assert isinstance(link, Link)

    def test_link_left_feature_group_from_join_spec(self) -> None:
        """Test Link.left_feature_group returns the feature_group from left JoinSpec."""
        from mloda.user import Link, JoinType

        left_idx = Index(("user_id",))
        right_idx = Index(("order_id",))

        left_spec = JoinSpec(feature_group=MockFeatureGroup, index=left_idx)
        right_spec = JoinSpec(feature_group=AnotherMockFeatureGroup, index=right_idx)

        link = Link(
            jointype=JoinType.INNER,
            left=left_spec,
            right=right_spec,
        )

        # Should extract feature_group from left JoinSpec
        assert link.left_feature_group is MockFeatureGroup

    def test_link_right_feature_group_from_join_spec(self) -> None:
        """Test Link.right_feature_group returns the feature_group from right JoinSpec."""
        from mloda.user import Link, JoinType

        left_idx = Index(("user_id",))
        right_idx = Index(("order_id",))

        left_spec = JoinSpec(feature_group=MockFeatureGroup, index=left_idx)
        right_spec = JoinSpec(feature_group=AnotherMockFeatureGroup, index=right_idx)

        link = Link(
            jointype=JoinType.INNER,
            left=left_spec,
            right=right_spec,
        )

        # Should extract feature_group from right JoinSpec
        assert link.right_feature_group is AnotherMockFeatureGroup

    def test_link_left_index_from_join_spec(self) -> None:
        """Test Link.left_index returns the index from left JoinSpec."""
        from mloda.user import Link, JoinType

        left_idx = Index(("user_id",))
        right_idx = Index(("order_id",))

        left_spec = JoinSpec(feature_group=MockFeatureGroup, index=left_idx)
        right_spec = JoinSpec(feature_group=AnotherMockFeatureGroup, index=right_idx)

        link = Link(
            jointype=JoinType.INNER,
            left=left_spec,
            right=right_spec,
        )

        # Should extract index from left JoinSpec
        assert link.left_index is left_idx
        assert link.left_index == left_idx

    def test_link_right_index_from_join_spec(self) -> None:
        """Test Link.right_index returns the index from right JoinSpec."""
        from mloda.user import Link, JoinType

        left_idx = Index(("user_id",))
        right_idx = Index(("order_id",))

        left_spec = JoinSpec(feature_group=MockFeatureGroup, index=left_idx)
        right_spec = JoinSpec(feature_group=AnotherMockFeatureGroup, index=right_idx)

        link = Link(
            jointype=JoinType.INNER,
            left=left_spec,
            right=right_spec,
        )

        # Should extract index from right JoinSpec
        assert link.right_index is right_idx
        assert link.right_index == right_idx

    def test_link_factory_method_inner_with_join_spec(self) -> None:
        """Test Link.inner factory method works with JoinSpec objects."""
        from mloda.user import Link, JoinType

        left_idx = Index(("id",))
        right_idx = Index(("id",))

        left_spec = JoinSpec(feature_group=MockFeatureGroup, index=left_idx)
        right_spec = JoinSpec(feature_group=AnotherMockFeatureGroup, index=right_idx)

        # Factory method should accept JoinSpec objects
        link = Link.inner(
            left=left_spec,
            right=right_spec,
        )

        assert link.jointype == JoinType.INNER
        assert link.left_feature_group is MockFeatureGroup
        assert link.right_feature_group is AnotherMockFeatureGroup

    def test_link_factory_method_left_with_join_spec(self) -> None:
        """Test Link.left factory method works with JoinSpec objects."""
        from mloda.user import Link, JoinType

        left_idx = Index(("id",))
        right_idx = Index(("id",))

        left_spec = JoinSpec(feature_group=MockFeatureGroup, index=left_idx)
        right_spec = JoinSpec(feature_group=AnotherMockFeatureGroup, index=right_idx)

        # Factory method should accept JoinSpec objects
        link = Link.left(
            left=left_spec,
            right=right_spec,
        )

        assert link.jointype == JoinType.LEFT
        assert link.left_feature_group is MockFeatureGroup
        assert link.right_feature_group is AnotherMockFeatureGroup

    def test_link_factory_method_right_with_join_spec(self) -> None:
        """Test Link.right factory method works with JoinSpec objects."""
        from mloda.user import Link, JoinType

        left_idx = Index(("id",))
        right_idx = Index(("id",))

        left_spec = JoinSpec(feature_group=MockFeatureGroup, index=left_idx)
        right_spec = JoinSpec(feature_group=AnotherMockFeatureGroup, index=right_idx)

        # Factory method should accept JoinSpec objects
        link = Link.right(
            left=left_spec,
            right=right_spec,
        )

        assert link.jointype == JoinType.RIGHT
        assert link.left_feature_group is MockFeatureGroup
        assert link.right_feature_group is AnotherMockFeatureGroup

    def test_link_factory_method_outer_with_join_spec(self) -> None:
        """Test Link.outer factory method works with JoinSpec objects."""
        from mloda.user import Link, JoinType

        left_idx = Index(("id",))
        right_idx = Index(("id",))

        left_spec = JoinSpec(feature_group=MockFeatureGroup, index=left_idx)
        right_spec = JoinSpec(feature_group=AnotherMockFeatureGroup, index=right_idx)

        # Factory method should accept JoinSpec objects
        link = Link.outer(
            left=left_spec,
            right=right_spec,
        )

        assert link.jointype == JoinType.OUTER
        assert link.left_feature_group is MockFeatureGroup
        assert link.right_feature_group is AnotherMockFeatureGroup

    def test_link_factory_method_append_with_join_spec(self) -> None:
        """Test Link.append factory method works with JoinSpec objects."""
        from mloda.user import Link, JoinType

        left_idx = Index(("id",))
        right_idx = Index(("id",))

        left_spec = JoinSpec(feature_group=MockFeatureGroup, index=left_idx)
        right_spec = JoinSpec(feature_group=AnotherMockFeatureGroup, index=right_idx)

        # Factory method should accept JoinSpec objects
        link = Link.append(
            left=left_spec,
            right=right_spec,
        )

        assert link.jointype == JoinType.APPEND
        assert link.left_feature_group is MockFeatureGroup
        assert link.right_feature_group is AnotherMockFeatureGroup

    def test_link_factory_method_union_with_join_spec(self) -> None:
        """Test Link.union factory method works with JoinSpec objects."""
        from mloda.user import Link, JoinType

        left_idx = Index(("id",))
        right_idx = Index(("id",))

        left_spec = JoinSpec(feature_group=MockFeatureGroup, index=left_idx)
        right_spec = JoinSpec(feature_group=AnotherMockFeatureGroup, index=right_idx)

        # Factory method should accept JoinSpec objects
        link = Link.union(
            left=left_spec,
            right=right_spec,
        )

        assert link.jointype == JoinType.UNION
        assert link.left_feature_group is MockFeatureGroup
        assert link.right_feature_group is AnotherMockFeatureGroup

    def test_link_with_join_spec_and_pointers(self) -> None:
        """Test Link works with JoinSpec objects and pointer arguments."""
        from mloda.user import Link, JoinType

        left_idx = Index(("id",))
        right_idx = Index(("id",))

        left_spec = JoinSpec(feature_group=MockFeatureGroup, index=left_idx)
        right_spec = JoinSpec(feature_group=MockFeatureGroup, index=right_idx)

        self_left_alias = {"side": "manager"}
        self_right_alias = {"side": "employee"}

        # Should accept JoinSpec objects with alias arguments
        link = Link(
            jointype=JoinType.INNER,
            left=left_spec,
            right=right_spec,
            self_left_alias=self_left_alias,
            self_right_alias=self_right_alias,
        )

        assert link.self_left_alias == self_left_alias
        assert link.self_right_alias == self_right_alias

    def test_link_with_multi_index_join_spec(self) -> None:
        """Test Link works with JoinSpec objects containing multi-column indexes."""
        from mloda.user import Link, JoinType

        left_idx = Index(("user_id", "timestamp"))
        right_idx = Index(("user_id", "timestamp"))

        left_spec = JoinSpec(feature_group=MockFeatureGroup, index=left_idx)
        right_spec = JoinSpec(feature_group=AnotherMockFeatureGroup, index=right_idx)

        # Should accept JoinSpec objects with multi-column indexes
        link = Link(
            jointype=JoinType.INNER,
            left=left_spec,
            right=right_spec,
        )

        assert link.left_index.is_multi_index() is True
        assert link.right_index.is_multi_index() is True
        assert link.left_index == left_idx
        assert link.right_index == right_idx


# ============================================================================
# Test JoinSpec Auto-Convert Convenience
# ============================================================================
class TestJoinSpecAutoConvert:
    """Test that JoinSpec auto-converts string and tuple inputs to Index objects.

    These tests will FAIL initially because JoinSpec currently requires an Index
    object and does not perform automatic conversion from strings or tuples.

    The tests will pass once JoinSpec.__init__ or __post_init__ is updated to:
    - Accept str, Tuple[str, ...], or Index for the index parameter
    - Automatically convert str -> Index((str,))
    - Automatically convert Tuple[str, ...] -> Index(tuple)
    - Keep Index objects as-is (backwards compatibility)
    """

    def test_string_input_creates_single_column_index(self) -> None:
        """Test JoinSpec accepts a string and creates Index with single column.

        Given: JoinSpec(MockFeatureGroup, "id")
        Expected: JoinSpec with index = Index(("id",))
        """
        # This will fail because JoinSpec expects Index, not str
        spec = JoinSpec(feature_group=MockFeatureGroup, index="id")

        # Should auto-convert to Index(("id",))
        assert isinstance(spec.index, Index)
        assert spec.index == Index(("id",))
        assert spec.index.index == ("id",)

    def test_single_element_tuple_creates_single_column_index(self) -> None:
        """Test JoinSpec accepts a single-element tuple and creates Index.

        Given: JoinSpec(MockFeatureGroup, ("id",))
        Expected: JoinSpec with index = Index(("id",))
        """
        # This will fail because JoinSpec expects Index, not tuple
        spec = JoinSpec(feature_group=MockFeatureGroup, index=("id",))

        # Should auto-convert to Index(("id",))
        assert isinstance(spec.index, Index)
        assert spec.index == Index(("id",))
        assert spec.index.index == ("id",)

    def test_multi_element_tuple_creates_multi_column_index(self) -> None:
        """Test JoinSpec accepts a multi-element tuple and creates multi-column Index.

        Given: JoinSpec(MockFeatureGroup, ("a", "b"))
        Expected: JoinSpec with index = Index(("a", "b"))
        """
        # This will fail because JoinSpec expects Index, not tuple
        spec = JoinSpec(feature_group=MockFeatureGroup, index=("a", "b"))

        # Should auto-convert to Index(("a", "b"))
        assert isinstance(spec.index, Index)
        assert spec.index == Index(("a", "b"))
        assert spec.index.index == ("a", "b")
        assert spec.index.is_multi_index() is True

    def test_explicit_index_still_works(self) -> None:
        """Test JoinSpec still accepts explicit Index objects (backwards compatibility).

        Given: JoinSpec(MockFeatureGroup, Index(("id",)))
        Expected: JoinSpec with index = Index(("id",)) (unchanged)
        """
        idx = Index(("id",))
        spec = JoinSpec(feature_group=MockFeatureGroup, index=idx)

        # Should keep the same Index object
        assert spec.index is idx
        assert isinstance(spec.index, Index)
        assert spec.index == Index(("id",))

    def test_string_and_tuple_produce_equivalent_results(self) -> None:
        """Test that string input and single-element tuple input produce equal JoinSpecs.

        Given: JoinSpec(G, "id") and JoinSpec(G, ("id",))
        Expected: Both should have equal index values
        """
        # These will fail because JoinSpec expects Index, not str/tuple
        spec_from_string = JoinSpec(feature_group=MockFeatureGroup, index="id")
        spec_from_tuple = JoinSpec(feature_group=MockFeatureGroup, index=("id",))

        # Both should produce the same index
        assert spec_from_string.index == spec_from_tuple.index
        assert spec_from_string.index == Index(("id",))
        assert spec_from_tuple.index == Index(("id",))

    def test_all_input_forms_produce_index_type(self) -> None:
        """Test that all input forms (str, tuple, Index) result in Index type.

        Given: JoinSpec with string, tuple, or Index
        Expected: spec.index is always an Index instance
        """
        # These will fail for str and tuple inputs
        spec_from_string = JoinSpec(feature_group=MockFeatureGroup, index="id")
        spec_from_tuple = JoinSpec(feature_group=MockFeatureGroup, index=("id",))
        spec_from_index = JoinSpec(feature_group=MockFeatureGroup, index=Index(("id",)))

        # All should have Index type
        assert isinstance(spec_from_string.index, Index)
        assert isinstance(spec_from_tuple.index, Index)
        assert isinstance(spec_from_index.index, Index)

    def test_multi_column_string_tuple_equivalence(self) -> None:
        """Test that multi-column tuple and explicit Index produce equal results.

        Given: JoinSpec(G, ("a", "b")) and JoinSpec(G, Index(("a", "b")))
        Expected: Both should have equal index values
        """
        # Tuple input will fail
        spec_from_tuple = JoinSpec(feature_group=MockFeatureGroup, index=("a", "b"))
        spec_from_index = JoinSpec(feature_group=MockFeatureGroup, index=Index(("a", "b")))

        # Both should produce the same index
        assert spec_from_tuple.index == spec_from_index.index
        assert spec_from_tuple.index == Index(("a", "b"))
        assert spec_from_index.index == Index(("a", "b"))

    def test_empty_tuple_raises_error(self) -> None:
        """Test that empty tuple raises appropriate error.

        Given: JoinSpec(MockFeatureGroup, ())
        Expected: Should raise ValueError (index cannot be empty)

        This test will FAIL initially because:
        1. The dataclass accepts empty tuple without validation
        2. Once auto-convert is implemented, it should validate and reject empty indexes
        """
        # Empty tuple should not be allowed - will fail because no validation exists yet
        with pytest.raises(ValueError, match="[Ii]ndex.*empty|[Ee]mpty.*index|at least one"):
            JoinSpec(feature_group=MockFeatureGroup, index=())

    def test_empty_string_raises_error(self) -> None:
        """Test that empty string raises appropriate error.

        Given: JoinSpec(MockFeatureGroup, "")
        Expected: Should raise ValueError (index cannot be empty)

        This test will FAIL initially because:
        1. The dataclass accepts empty string without validation
        2. Once auto-convert is implemented, it should validate and reject empty indexes
        """
        # Empty string should not be allowed - will fail because no validation exists yet
        with pytest.raises(ValueError, match="[Ii]ndex.*empty|[Ee]mpty.*index|at least one"):
            JoinSpec(feature_group=MockFeatureGroup, index="")
