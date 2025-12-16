import pytest
from collections import OrderedDict
from typing import Any, Set, Type
from uuid import UUID, uuid4

from mloda.user import Link, JoinSpec, JoinType
from mloda import ComputeFramework
from mloda.core.prepare.validators.resolve_link_validator import ResolveLinkValidator


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


# Mock compute framework for testing
class MockComputeFramework(ComputeFramework):
    """Mock compute framework for testing."""

    pass


class TestValidateDataConsistency:
    """Test the validate_data_consistency method (from line 207 in resolve_links.py)."""

    def test_same_length_passes(self) -> None:
        """Same length data and data_ordered should not raise."""
        # Arrange
        link1 = Link.inner(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        link2 = Link.inner(left=JoinSpec(MockFeatureGroupB, "id"), right=JoinSpec(MockFeatureGroupC, "id"))

        link_fw_trekker1 = (link1, MockComputeFramework, MockComputeFramework)
        link_fw_trekker2 = (link2, MockComputeFramework, MockComputeFramework)

        data = {link_fw_trekker1: {uuid4()}, link_fw_trekker2: {uuid4()}}

        data_ordered: OrderedDict[Any, Set[UUID]] = OrderedDict()
        data_ordered[link_fw_trekker1] = {uuid4()}
        data_ordered[link_fw_trekker2] = {uuid4()}

        # Act & Assert - should not raise
        ResolveLinkValidator.validate_data_consistency(data=data, data_ordered=data_ordered)

    def test_different_length_raises(self) -> None:
        """Different lengths should raise ValueError."""
        # Arrange
        link1 = Link.inner(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        link2 = Link.inner(left=JoinSpec(MockFeatureGroupB, "id"), right=JoinSpec(MockFeatureGroupC, "id"))

        link_fw_trekker1 = (link1, MockComputeFramework, MockComputeFramework)
        link_fw_trekker2 = (link2, MockComputeFramework, MockComputeFramework)

        # data has 2 items
        data = {link_fw_trekker1: {uuid4()}, link_fw_trekker2: {uuid4()}}

        # data_ordered has only 1 item - inconsistent!
        data_ordered: OrderedDict[Any, Set[UUID]] = OrderedDict()
        data_ordered[link_fw_trekker1] = {uuid4()}

        # Act & Assert
        with pytest.raises(ValueError):
            ResolveLinkValidator.validate_data_consistency(data=data, data_ordered=data_ordered)


class TestValidateNoConflictingJoinTypes:
    """Test the validate_no_conflicting_join_types method (from lines 256-264 in resolve_links.py)."""

    def test_no_conflicts_passes(self) -> None:
        """No conflicting join types should not raise."""
        # Arrange - different feature groups
        link1 = Link.inner(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        link2 = Link.left(left=JoinSpec(MockFeatureGroupB, "id"), right=JoinSpec(MockFeatureGroupC, "id"))

        link_fw_trekker1 = (link1, MockComputeFramework, MockComputeFramework)
        link_fw_trekker2 = (link2, MockComputeFramework, MockComputeFramework)

        data = {link_fw_trekker1: {uuid4()}, link_fw_trekker2: {uuid4()}}

        # Act & Assert - should not raise (different feature groups)
        ResolveLinkValidator.validate_no_conflicting_join_types(data=data)

    def test_same_feature_groups_different_join_raises(self) -> None:
        """Same feature groups with different join types should raise Exception."""
        # Arrange - same feature groups, different join types
        link1 = Link.inner(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        link2 = Link.left(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))

        link_fw_trekker1 = (link1, MockComputeFramework, MockComputeFramework)
        link_fw_trekker2 = (link2, MockComputeFramework, MockComputeFramework)

        data = {link_fw_trekker1: {uuid4()}, link_fw_trekker2: {uuid4()}}

        # Act & Assert - should raise due to conflicting join types
        with pytest.raises(Exception):  # Original code raises Exception, not ValueError
            ResolveLinkValidator.validate_no_conflicting_join_types(data=data)

    def test_different_feature_groups_passes(self) -> None:
        """Different feature groups should not raise even with different join types."""
        # Arrange - completely different feature group pairs
        link1 = Link.inner(left=JoinSpec(MockFeatureGroupA, "id"), right=JoinSpec(MockFeatureGroupB, "id"))
        link2 = Link.left(left=JoinSpec(MockFeatureGroupB, "id"), right=JoinSpec(MockFeatureGroupC, "id"))

        link_fw_trekker1 = (link1, MockComputeFramework, MockComputeFramework)
        link_fw_trekker2 = (link2, MockComputeFramework, MockComputeFramework)

        data = {link_fw_trekker1: {uuid4()}, link_fw_trekker2: {uuid4()}}

        # Act & Assert - should not raise
        ResolveLinkValidator.validate_no_conflicting_join_types(data=data)
