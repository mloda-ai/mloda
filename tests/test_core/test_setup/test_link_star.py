"""
Test Link.star Factory Method

Tests for the Link.star star-join builder that joins every feature group to a
shared row-index column, replacing hand-rolled loops of the shape:

    links = set()
    idx = Index((ROW_INDEX_COLUMN,))
    base_group = feature_groups[0]
    for other_group in feature_groups[1:]:
        links.add(Link.inner(JoinSpec(base_group, idx), JoinSpec(other_group, idx)))

Usage:
    # Star-join three feature groups on a shared row-index column:
    Link.star(HubFG, SpokeAFG, SpokeBFG, index_column="row_id")

The first feature group is the hub; every remaining group is joined to it with an
INNER join on the shared index. Returns a set of Links.

See Also:
    - GitHub Issue #571: Star-join builder Link.star
"""

from typing import Any, Optional

import pytest

from mloda.provider import FeatureGroup
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import JoinSpec, JoinType, Link
from mloda.user import Options


# ============================================================================
# Mock Feature Groups for Testing
# ============================================================================
class HubFG(FeatureGroup):
    """Hub feature group for star-join testing."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[set[Any]]:
        return None


class SpokeAFG(FeatureGroup):
    """First spoke feature group for star-join testing."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[set[Any]]:
        return None


class SpokeBFG(FeatureGroup):
    """Second spoke feature group for star-join testing."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[set[Any]]:
        return None


ROW_INDEX = Index(("row_id",))


# ============================================================================
# Test Link.star Method
# ============================================================================
class TestLinkStar:
    """Test Link.star star-join builder."""

    def test_star_default_three_groups_returns_two_links(self) -> None:
        """Test Link.star with three groups returns two INNER links from the hub."""
        links = Link.star(HubFG, SpokeAFG, SpokeBFG, index_column="row_id")

        assert isinstance(links, set)
        assert len(links) == 2

        # Every link is an INNER join from the hub to a spoke, on the shared index.
        for link in links:
            assert link.jointype == JoinType.INNER
            assert link.left_feature_group is HubFG
            assert link.left_index == ROW_INDEX
            assert link.right_index == ROW_INDEX

        spokes = {link.right_feature_group for link in links}
        assert spokes == {SpokeAFG, SpokeBFG}

    def test_star_single_spoke_returns_one_link(self) -> None:
        """Test Link.star with two groups returns a single hub-to-spoke INNER link."""
        links = Link.star(HubFG, SpokeAFG, index_column="row_id")

        assert isinstance(links, set)
        assert len(links) == 1

        link = next(iter(links))
        assert link.jointype == JoinType.INNER
        assert link.left_feature_group is HubFG
        assert link.right_feature_group is SpokeAFG
        assert link.left_index == ROW_INDEX
        assert link.right_index == ROW_INDEX

    def test_star_accepts_str_tuple_and_index_equivalently(self) -> None:
        """Test index_column accepts str, tuple, or Index and produces equivalent links."""
        links_str = Link.star(HubFG, SpokeAFG, index_column="row_id")
        links_tuple = Link.star(HubFG, SpokeAFG, index_column=("row_id",))
        links_index = Link.star(HubFG, SpokeAFG, index_column=Index(("row_id",)))

        assert links_str == links_tuple
        assert links_tuple == links_index

    def test_star_dedupes_duplicate_spokes(self) -> None:
        """Test Link.star dedupes a repeated spoke into a single link (set semantics)."""
        links = Link.star(HubFG, SpokeAFG, SpokeAFG, index_column="row_id")

        assert len(links) == 1
        link = next(iter(links))
        assert link.left_feature_group is HubFG
        assert link.right_feature_group is SpokeAFG

    def test_star_equivalent_to_manual_inner_construction(self) -> None:
        """Test Link.star produces a link equal to manual Link.inner construction."""
        links = Link.star(HubFG, SpokeAFG, index_column="row_id")

        manual = Link.inner(
            JoinSpec(HubFG, Index(("row_id",))),
            JoinSpec(SpokeAFG, Index(("row_id",))),
        )

        assert next(iter(links)) == manual

    def test_star_with_multi_column_tuple_index(self) -> None:
        """Test Link.star with a multi-column tuple index yields a multi-column index on both sides."""
        links = Link.star(HubFG, SpokeAFG, index_column=("row_id", "region"))

        assert len(links) == 1
        link = next(iter(links))
        multi_index = Index(("row_id", "region"))
        assert link.left_index == multi_index
        assert link.right_index == multi_index
        assert link.left_index.is_multi_index() is True

    def test_star_raises_value_error_on_empty_index_column(self) -> None:
        """Test Link.star raises ValueError on an empty index column string."""
        with pytest.raises(ValueError, match="empty"):
            Link.star(HubFG, SpokeAFG, index_column="")

    def test_star_raises_value_error_with_single_group(self) -> None:
        """Test Link.star raises ValueError when only a hub is provided (no spokes)."""
        with pytest.raises(ValueError):
            Link.star(HubFG, index_column="row_id")

    def test_star_raises_value_error_with_no_groups(self) -> None:
        """Test Link.star raises ValueError when no feature groups are provided."""
        with pytest.raises(ValueError):
            Link.star(index_column="row_id")


# ============================================================================
# Test Link.star jointype Parameter
# ============================================================================
class TestLinkStarJoinType:
    """Test the keyword-only jointype parameter on Link.star."""

    def test_star_left_yields_left_links(self) -> None:
        """Test jointype=JoinType.LEFT yields LEFT links: hub left, spoke right, shared index."""
        links = Link.star(HubFG, SpokeAFG, index_column="row_id", jointype=JoinType.LEFT)

        assert len(links) == 1
        link = next(iter(links))
        assert link.jointype == JoinType.LEFT
        assert link.left_feature_group is HubFG
        assert link.right_feature_group is SpokeAFG
        assert link.left_index == ROW_INDEX
        assert link.right_index == ROW_INDEX

    def test_star_left_string_form_yields_left_links(self) -> None:
        """Test the string form jointype="left" yields LEFT links."""
        links = Link.star(HubFG, SpokeAFG, index_column="row_id", jointype="left")

        assert len(links) == 1
        link = next(iter(links))
        assert link.jointype == JoinType.LEFT

    def test_star_outer_yields_outer_links(self) -> None:
        """Test jointype=JoinType.OUTER yields OUTER links."""
        links = Link.star(HubFG, SpokeAFG, index_column="row_id", jointype=JoinType.OUTER)

        assert len(links) == 1
        link = next(iter(links))
        assert link.jointype == JoinType.OUTER
        assert link.left_feature_group is HubFG
        assert link.right_feature_group is SpokeAFG

    def test_star_string_and_enum_forms_are_equivalent(self) -> None:
        """Test the string and enum forms of jointype produce equal link sets."""
        links_str = Link.star(HubFG, SpokeAFG, index_column="row_id", jointype="left")
        links_enum = Link.star(HubFG, SpokeAFG, index_column="row_id", jointype=JoinType.LEFT)

        assert links_str == links_enum

    def test_star_multi_spoke_left_returns_two_left_links(self) -> None:
        """Test jointype=JoinType.LEFT with two spokes returns two LEFT links anchored on the hub."""
        links = Link.star(HubFG, SpokeAFG, SpokeBFG, index_column="row_id", jointype=JoinType.LEFT)

        assert len(links) == 2
        for link in links:
            assert link.jointype == JoinType.LEFT
            assert link.left_feature_group is HubFG

        spokes = {link.right_feature_group for link in links}
        assert spokes == {SpokeAFG, SpokeBFG}

    def test_star_right_jointype_raises_value_error(self) -> None:
        """Test jointype=JoinType.RIGHT is rejected: a star join anchors on the hub."""
        with pytest.raises(ValueError):
            Link.star(HubFG, SpokeAFG, index_column="row_id", jointype=JoinType.RIGHT)

    def test_star_append_jointype_raises_value_error(self) -> None:
        """Test jointype=JoinType.APPEND is rejected for star joins."""
        with pytest.raises(ValueError):
            Link.star(HubFG, SpokeAFG, index_column="row_id", jointype=JoinType.APPEND)

    def test_star_union_jointype_raises_value_error(self) -> None:
        """Test jointype=JoinType.UNION is rejected for star joins."""
        with pytest.raises(ValueError):
            Link.star(HubFG, SpokeAFG, index_column="row_id", jointype=JoinType.UNION)

    def test_star_asof_jointype_raises_value_error(self) -> None:
        """Test jointype=JoinType.ASOF is rejected for star joins."""
        with pytest.raises(ValueError):
            Link.star(HubFG, SpokeAFG, index_column="row_id", jointype=JoinType.ASOF)

    def test_star_invalid_jointype_string_raises_value_error(self) -> None:
        """Test an unknown jointype string is rejected (JoinType("banana") raises ValueError)."""
        with pytest.raises(ValueError):
            Link.star(HubFG, SpokeAFG, index_column="row_id", jointype="banana")

    def test_star_right_string_form_raises_value_error(self) -> None:
        """Test the string form "right" is rejected: it converts to JoinType.RIGHT then fails the allowed set."""
        with pytest.raises(ValueError):
            Link.star(HubFG, SpokeAFG, index_column="row_id", jointype="right")

    def test_star_rejection_message_names_allowed_set(self) -> None:
        """Test the rejection message names the allowed set of join types."""
        with pytest.raises(ValueError, match="only inner, left, or outer"):
            Link.star(HubFG, SpokeAFG, index_column="row_id", jointype=JoinType.RIGHT)

    def test_star_hub_as_spoke_raises_value_error(self) -> None:
        """Test the hub class cannot also be a spoke: a star cannot self-join the hub.

        Self-joining a class requires left/right discriminators to disambiguate the two
        same-class nodes, which Link.star does not support, so this must be rejected.
        """
        with pytest.raises(ValueError, match="hub"):
            Link.star(HubFG, HubFG, index_column="row_id")
