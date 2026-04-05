"""Tests for circular dependency resolution in LinkTrekker.

Exercises both branches of the adjust_order logic inside
drop_dependency_in_case_of_circular_dependencies: the case where
the outbound link has >= dependants, and where the inbound link has more.
"""

from unittest.mock import MagicMock
from uuid import UUID, uuid4

from mloda.core.abstract_plugins.components.link import Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.prepare.resolve_links import LinkTrekker


class MockCfw(ComputeFramework):
    pass


def _make_link(link_uuid: UUID) -> MagicMock:
    link = MagicMock(spec=Link)
    link.uuid = link_uuid
    return link


class TestDropCircularDependencies:
    """Tests for drop_dependency_in_case_of_circular_dependencies."""

    def test_drops_link_with_more_or_equal_dependants(self) -> None:
        """When found_out has >= dependants than found_in, found_out is dropped."""
        link_uuid_a = uuid4()
        link_uuid_b = uuid4()
        feature_1 = uuid4()
        feature_2 = uuid4()
        feature_3 = uuid4()

        link_a = _make_link(link_uuid_a)
        link_b = _make_link(link_uuid_b)

        lt = LinkTrekker()

        # link_a has 2 dependant features, link_b has 1
        lt.data[(link_a, MockCfw, MockCfw)] = {feature_1, feature_2}
        lt.data[(link_b, MockCfw, MockCfw)] = {feature_3}

        # Circular: a depends on b and b depends on a
        lt.order[link_uuid_a] = {link_uuid_b}
        lt.order[link_uuid_b] = {link_uuid_a}

        lt.drop_dependency_in_case_of_circular_dependencies()

        # link_a has >= dependants, so link_a is dropped from link_b's order
        assert link_uuid_a not in lt.order[link_uuid_b]

    def test_drops_link_with_fewer_dependants(self) -> None:
        """When found_in has more dependants than found_out, found_in is dropped."""
        link_uuid_a = uuid4()
        link_uuid_b = uuid4()
        feature_1 = uuid4()
        feature_2 = uuid4()
        feature_3 = uuid4()

        link_a = _make_link(link_uuid_a)
        link_b = _make_link(link_uuid_b)

        lt = LinkTrekker()

        # link_a has 1 dependant feature, link_b has 2
        lt.data[(link_a, MockCfw, MockCfw)] = {feature_1}
        lt.data[(link_b, MockCfw, MockCfw)] = {feature_2, feature_3}

        # Circular: a depends on b and b depends on a
        lt.order[link_uuid_a] = {link_uuid_b}
        lt.order[link_uuid_b] = {link_uuid_a}

        lt.drop_dependency_in_case_of_circular_dependencies()

        # link_b has more dependants (found_in side when k_out=a, k_in=b),
        # so found_in is dropped from found_out's order
        # The function drops k_to_drop from the order of k_not_to_drop
        # When found_out < found_in: k_to_drop = found_in, k_not_to_drop = found_out
        # So link_b is dropped from link_a's order
        assert link_uuid_b not in lt.order[link_uuid_a]

    def test_equal_dependant_counts_takes_ge_branch(self) -> None:
        """When both links have the same number of dependants, the >= branch fires."""
        link_uuid_a = uuid4()
        link_uuid_b = uuid4()
        feature_1 = uuid4()
        feature_2 = uuid4()

        link_a = _make_link(link_uuid_a)
        link_b = _make_link(link_uuid_b)

        lt = LinkTrekker()

        # Both have exactly 1 dependant
        lt.data[(link_a, MockCfw, MockCfw)] = {feature_1}
        lt.data[(link_b, MockCfw, MockCfw)] = {feature_2}

        lt.order[link_uuid_a] = {link_uuid_b}
        lt.order[link_uuid_b] = {link_uuid_a}

        lt.drop_dependency_in_case_of_circular_dependencies()

        # Equal counts: >= branch fires, found_out is dropped
        assert link_uuid_a not in lt.order[link_uuid_b]

    def test_no_circular_dependency_is_noop(self) -> None:
        """When there is no circular dependency, nothing is modified."""
        link_uuid_a = uuid4()
        link_uuid_b = uuid4()

        lt = LinkTrekker()

        # a depends on b, but b does NOT depend on a
        lt.order[link_uuid_a] = {link_uuid_b}
        lt.order[link_uuid_b] = set()

        lt.drop_dependency_in_case_of_circular_dependencies()

        assert lt.order[link_uuid_a] == {link_uuid_b}
        assert lt.order[link_uuid_b] == set()
