"""
Tests that Link equality/hash incorporate the left/right discriminators.

Two links that differ only by their discriminators identify different join nodes
(e.g. the same FeatureGroup class reading two different CSV files) and must not
collapse into one another in a ``set``/``dict``.

See Also:
    - GitHub Issue #1235: Link equality/hash ignore discriminators
"""

from typing import Any

from mloda.provider import FeatureGroup
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import JoinSpec, Link
from mloda.user import Options


# ============================================================================
# Mock Feature Group for Testing (mirrors test_link_on_methods.py pattern)
# ============================================================================
class DiscriminatorFG(FeatureGroup):
    """Mock feature group with a single index column 'id'."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> set[Any] | None:
        return None

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("id",))]


# ============================================================================
# eq / hash incorporate discriminators
# ============================================================================
class TestLinkDiscriminatorEqHash:
    def test_different_right_discriminator_not_equal(self) -> None:
        """Two links identical except for right_discriminator must NOT be equal."""
        link_b = Link.inner(
            JoinSpec(DiscriminatorFG, "id"),
            JoinSpec(DiscriminatorFG, "id"),
            left_discriminator={"r": "a.csv"},
            right_discriminator={"r": "b.csv"},
        )
        link_c = Link.inner(
            JoinSpec(DiscriminatorFG, "id"),
            JoinSpec(DiscriminatorFG, "id"),
            left_discriminator={"r": "a.csv"},
            right_discriminator={"r": "c.csv"},
        )
        assert link_b != link_c

    def test_different_left_discriminator_not_equal(self) -> None:
        """Two links identical except for left_discriminator must NOT be equal."""
        link_a = Link.inner(
            JoinSpec(DiscriminatorFG, "id"),
            JoinSpec(DiscriminatorFG, "id"),
            left_discriminator={"r": "a.csv"},
            right_discriminator={"r": "z.csv"},
        )
        link_b = Link.inner(
            JoinSpec(DiscriminatorFG, "id"),
            JoinSpec(DiscriminatorFG, "id"),
            left_discriminator={"r": "b.csv"},
            right_discriminator={"r": "z.csv"},
        )
        assert link_a != link_b

    def test_discriminator_differing_links_both_survive_in_set(self) -> None:
        """Links differing only by discriminator must both persist in a set."""
        link_b = Link.inner(
            JoinSpec(DiscriminatorFG, "id"),
            JoinSpec(DiscriminatorFG, "id"),
            left_discriminator={"r": "a.csv"},
            right_discriminator={"r": "b.csv"},
        )
        link_c = Link.inner(
            JoinSpec(DiscriminatorFG, "id"),
            JoinSpec(DiscriminatorFG, "id"),
            left_discriminator={"r": "a.csv"},
            right_discriminator={"r": "c.csv"},
        )
        assert len({link_b, link_c}) == 2

    def test_identical_discriminators_equal_with_equal_hash(self) -> None:
        """Two truly-identical links remain equal and hash equal (eq/hash contract)."""
        link_1 = Link.inner(
            JoinSpec(DiscriminatorFG, "id"),
            JoinSpec(DiscriminatorFG, "id"),
            left_discriminator={"r": "a.csv"},
            right_discriminator={"r": "b.csv"},
        )
        link_2 = Link.inner(
            JoinSpec(DiscriminatorFG, "id"),
            JoinSpec(DiscriminatorFG, "id"),
            left_discriminator={"r": "a.csv"},
            right_discriminator={"r": "b.csv"},
        )
        assert link_1 == link_2
        assert hash(link_1) == hash(link_2)
        assert len({link_1, link_2}) == 1

    def test_discriminator_hash_is_order_independent(self) -> None:
        """A multi-key discriminator hashes equal regardless of insertion order."""
        link_1 = Link.inner(
            JoinSpec(DiscriminatorFG, "id"),
            JoinSpec(DiscriminatorFG, "id"),
            left_discriminator={"reader": "a.csv", "region": "us"},
        )
        link_2 = Link.inner(
            JoinSpec(DiscriminatorFG, "id"),
            JoinSpec(DiscriminatorFG, "id"),
            left_discriminator={"region": "us", "reader": "a.csv"},
        )
        assert link_1 == link_2
        assert hash(link_1) == hash(link_2)

    def test_discriminator_vs_none_not_equal(self) -> None:
        """A link with a discriminator is not equal to one without it."""
        link_with = Link.inner(
            JoinSpec(DiscriminatorFG, "id"),
            JoinSpec(DiscriminatorFG, "id"),
            left_discriminator={"r": "a.csv"},
        )
        link_without = Link.inner(
            JoinSpec(DiscriminatorFG, "id"),
            JoinSpec(DiscriminatorFG, "id"),
        )
        assert link_with != link_without
        assert len({link_with, link_without}) == 2

    def test_list_valued_discriminator_is_hashable(self) -> None:
        """A discriminator holding a list value (a valid Options value, e.g. file_paths) must not
        raise on hash(); Link.__hash__ has to normalize it instead of hashing it directly."""
        link_1 = Link.inner(
            JoinSpec(DiscriminatorFG, "id"),
            JoinSpec(DiscriminatorFG, "id"),
            left_discriminator={"file_paths": ["a.csv", "b.csv"]},
        )
        link_2 = Link.inner(
            JoinSpec(DiscriminatorFG, "id"),
            JoinSpec(DiscriminatorFG, "id"),
            left_discriminator={"file_paths": ["a.csv", "b.csv"]},
        )
        link_3 = Link.inner(
            JoinSpec(DiscriminatorFG, "id"),
            JoinSpec(DiscriminatorFG, "id"),
            left_discriminator={"file_paths": ["a.csv", "c.csv"]},
        )
        assert link_1 == link_2
        assert hash(link_1) == hash(link_2)
        assert link_1 != link_3
        assert len({link_1, link_2, link_3}) == 2
