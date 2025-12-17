import pytest
from typing import Set

from mloda.user import Link, JoinSpec, JoinType
from mloda.provider import LinkValidator


# Mock feature group classes for testing
class MockFeatureGroupA:
    """Mock feature group A for testing."""

    @classmethod
    def get_class_name(cls) -> str:
        return "MockFeatureGroupA"


class MockFeatureGroupB:
    """Mock feature group B for testing."""

    @classmethod
    def get_class_name(cls) -> str:
        return "MockFeatureGroupB"


class MockFeatureGroupC:
    """Mock feature group C for testing."""

    @classmethod
    def get_class_name(cls) -> str:
        return "MockFeatureGroupC"


class TestValidateIndexNotEmpty:
    """Test the validate_index_not_empty static method."""

    def test_valid_string_index_passes(self) -> None:
        """Non-empty string index should not raise an error."""
        # Act & Assert - should not raise
        LinkValidator.validate_index_not_empty(index="user_id", context="string index")

    def test_valid_tuple_index_passes(self) -> None:
        """Non-empty tuple index should not raise an error."""
        # Act & Assert - should not raise
        LinkValidator.validate_index_not_empty(index=("col1", "col2"), context="tuple index")

    def test_empty_string_raises_value_error(self) -> None:
        """Empty string index should raise ValueError."""
        # Act & Assert
        with pytest.raises(ValueError):
            LinkValidator.validate_index_not_empty(index="", context="string index")

    def test_empty_tuple_raises_value_error(self) -> None:
        """Empty tuple index should raise ValueError."""
        # Act & Assert
        with pytest.raises(ValueError):
            LinkValidator.validate_index_not_empty(index=(), context="tuple index")

    def test_error_message_includes_context(self) -> None:
        """Error message should include the provided context string."""
        context = "custom_context"

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            LinkValidator.validate_index_not_empty(index="", context=context)

        # Verify error message includes context
        assert context in str(exc_info.value)


class TestValidateJoinType:
    """Test the validate_join_type static method."""

    def test_valid_join_types_pass(self) -> None:
        """All valid JoinType enum values should pass validation."""
        # Test each valid join type
        for join_type in JoinType:
            # Act & Assert - should not raise
            LinkValidator.validate_join_type(jointype=join_type)

    def test_invalid_join_type_raises(self) -> None:
        """Invalid join type (not in JoinType enum) should raise ValueError."""
        # Create a mock invalid join type by using a string that's not in the enum
        # We need to simulate what would happen if someone passed an invalid type
        # Since Link.__init__ converts strings to JoinType, we'll test with a non-enum value

        # Create a mock object that looks like a JoinType but isn't in the enum
        class FakeJoinType:
            value = "fake_join"

        fake_join = FakeJoinType()

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            LinkValidator.validate_join_type(jointype=fake_join)

        # Verify error message mentions join type
        assert "join" in str(exc_info.value).lower() or "type" in str(exc_info.value).lower()


class TestValidateNoDoubleJoins:
    """Test the validate_no_double_joins static method."""

    def test_no_conflicts_passes(self) -> None:
        """Non-conflicting links should pass validation."""
        # Create links that don't conflict
        link1 = Link.inner(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        link2 = Link.inner(left=JoinSpec(MockFeatureGroupB, "id"), right=JoinSpec(MockFeatureGroupC, "id"))
        links = {link1, link2}

        # Act & Assert - should not raise
        LinkValidator.validate_no_double_joins(links=links)

    def test_double_join_raises_value_error(self) -> None:
        """Links with A->B and B->A should raise ValueError."""
        # Create conflicting bidirectional links
        link1 = Link.inner(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        link2 = Link.inner(left=JoinSpec(MockFeatureGroupB, "id"), right=JoinSpec(MockFeatureGroupA, "id"))
        links = {link1, link2}

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            LinkValidator.validate_no_double_joins(links=links)

        # Verify error message mentions the conflict
        error_msg = str(exc_info.value).lower()
        assert "join" in error_msg or "conflict" in error_msg or "different" in error_msg

    def test_append_union_exempt_from_double_join(self) -> None:
        """APPEND and UNION join types should be allowed in both directions."""
        # Test APPEND
        link1 = Link.append(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        link2 = Link.append(left=JoinSpec(MockFeatureGroupB, "id"), right=JoinSpec(MockFeatureGroupA, "id"))
        links_append = {link1, link2}

        # Act & Assert - should not raise for APPEND
        LinkValidator.validate_no_double_joins(links=links_append)

        # Test UNION
        link3 = Link.union(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        link4 = Link.union(left=JoinSpec(MockFeatureGroupB, "id"), right=JoinSpec(MockFeatureGroupA, "id"))
        links_union = {link3, link4}

        # Act & Assert - should not raise for UNION
        LinkValidator.validate_no_double_joins(links=links_union)


class TestValidateNoConflictingJoinTypes:
    """Test the validate_no_conflicting_join_types static method."""

    def test_same_join_types_passes(self) -> None:
        """Same join type for same feature group pairs should pass."""
        # Create two identical links (same join type, same groups)
        link1 = Link.inner(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        link2 = Link.inner(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        links = {link1, link2}

        # Act & Assert - should not raise
        LinkValidator.validate_no_conflicting_join_types(links=links)

    def test_different_join_types_raises(self) -> None:
        """Different join types for same feature group pairs should raise ValueError."""
        # Create links with same groups but different join types
        link1 = Link.inner(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        link2 = Link.left(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        links = {link1, link2}

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            LinkValidator.validate_no_conflicting_join_types(links=links)

        # Verify error message mentions different join types
        error_msg = str(exc_info.value).lower()
        assert "different" in error_msg or "join type" in error_msg or "conflict" in error_msg


class TestValidateRightJoinConstraints:
    """Test the validate_right_join_constraints static method."""

    def test_single_right_join_passes(self) -> None:
        """Single right join should pass validation."""
        # Create a single right join
        link = Link.right(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        links = {link}

        # Act & Assert - should not raise
        LinkValidator.validate_right_join_constraints(links=links)

    def test_multiple_right_joins_same_left_raises(self) -> None:
        """Multiple right joins with same left feature group should raise ValueError."""
        # Create multiple right joins with same left side
        link1 = Link.right(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        link2 = Link.right(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupC, "id"))
        links = {link1, link2}

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            LinkValidator.validate_right_join_constraints(links=links)

        # Verify error message mentions right join constraints
        error_msg = str(exc_info.value).lower()
        assert "right" in error_msg or "multiple" in error_msg

    def test_right_join_switching_sides_raises(self) -> None:
        """Right joins that switch from left to right side should raise ValueError."""
        # Create right joins where the same feature group appears on different sides
        link1 = Link.right(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        # Second link has A on the right side instead of left
        link2 = Link.right(left=JoinSpec(MockFeatureGroupC, "id"), right=JoinSpec(MockFeatureGroupA, "id"))
        links = {link1, link2}

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            LinkValidator.validate_right_join_constraints(links=links)

        # Verify error message mentions the constraint violation
        error_msg = str(exc_info.value).lower()
        assert "right" in error_msg


class TestValidateLinks:
    """Test the validate_links class method that orchestrates all validations."""

    def test_none_links_passes(self) -> None:
        """None links parameter should pass without raising."""
        # Act & Assert - should not raise
        LinkValidator.validate_links(links=None)

    def test_empty_set_passes(self) -> None:
        """Empty set of links should pass validation."""
        empty_links: Set[Link] = set()

        # Act & Assert - should not raise
        LinkValidator.validate_links(links=empty_links)

    def test_valid_links_passes(self) -> None:
        """Valid links that pass all validations should not raise."""
        # Create a valid set of links
        link1 = Link.inner(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        link2 = Link.left(left=JoinSpec(MockFeatureGroupB, "id"), right=JoinSpec(MockFeatureGroupC, "id"))
        links = {link1, link2}

        # Act & Assert - should not raise
        LinkValidator.validate_links(links=links)

    def test_calls_all_validators(self) -> None:
        """validate_links should call all individual validation methods."""
        # Create links that would fail different validations to ensure all are called

        # Test 1: Invalid join type should be caught
        link_invalid_join = Link.inner(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        # Manually corrupt the join type to test validation
        link_invalid_join.jointype = "invalid_type"  # type: ignore[assignment]
        links_invalid = {link_invalid_join}

        with pytest.raises(ValueError):
            LinkValidator.validate_links(links=links_invalid)

        # Test 2: Double join should be caught
        link1 = Link.inner(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        link2 = Link.inner(left=JoinSpec(MockFeatureGroupB, "id"), right=JoinSpec(MockFeatureGroupA, "id"))
        links_double = {link1, link2}

        with pytest.raises(ValueError):
            LinkValidator.validate_links(links=links_double)

        # Test 3: Conflicting join types should be caught
        link3 = Link.inner(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        link4 = Link.left(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        links_conflicting = {link3, link4}

        with pytest.raises(ValueError):
            LinkValidator.validate_links(links=links_conflicting)

        # Test 4: Right join constraints should be caught
        link5 = Link.right(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        link6 = Link.right(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupC, "id"))
        links_right = {link5, link6}

        with pytest.raises(ValueError):
            LinkValidator.validate_links(links=links_right)
