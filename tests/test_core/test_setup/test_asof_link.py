"""
Tests for the JoinType.ASOF (point-in-time / as-of join) feature.

These tests target the new ASOF join API:
    - JoinType.ASOF enum member
    - AsOfJoinConfig frozen dataclass
    - Link.asof / Link.asof_on factory methods
    - eq/hash incorporating asof_config

The implementation does not exist yet, so these tests are expected to FAIL
(at import / attribute level or assertion level).
"""

from typing import Any, Optional

import pytest

from mloda.provider import FeatureGroup
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import JoinSpec, JoinType, Link
from mloda.user import Options


# ============================================================================
# Mock Feature Groups for Testing (mirror test_link_on_methods.py pattern)
# ============================================================================
class AsofFGa(FeatureGroup):
    """Mock left feature group with a single by-key index column 'k'."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[set[Any]]:
        return None

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("k",))]


class AsofFGb(FeatureGroup):
    """Mock right feature group with a single by-key index column 'k'."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[set[Any]]:
        return None

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("k",))]


# ============================================================================
# JoinType.ASOF enum member
# ============================================================================
class TestJoinTypeAsof:
    def test_asof_member_exists(self) -> None:
        """JoinType.ASOF must exist and have value 'asof'."""
        assert JoinType.ASOF.value == "asof"


# ============================================================================
# AsOfJoinConfig dataclass
# ============================================================================
class TestAsOfJoinConfig:
    def test_construction_defaults(self) -> None:
        """AsOfJoinConfig defaults: direction='backward', tolerance=None, allow_exact_matches=True."""
        from mloda.core.abstract_plugins.components.link import AsOfJoinConfig

        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t")
        assert cfg.left_time_column == "t"
        assert cfg.right_time_column == "t"
        assert cfg.direction == "backward"
        assert cfg.tolerance is None
        assert cfg.allow_exact_matches is True

    def test_construction_explicit(self) -> None:
        """AsOfJoinConfig accepts explicit field values."""
        from mloda.core.abstract_plugins.components.link import AsOfJoinConfig

        cfg = AsOfJoinConfig(
            left_time_column="lt",
            right_time_column="rt",
            direction="forward",
            tolerance=5.0,
            allow_exact_matches=False,
        )
        assert cfg.left_time_column == "lt"
        assert cfg.right_time_column == "rt"
        assert cfg.direction == "forward"
        assert cfg.tolerance == 5.0
        assert cfg.allow_exact_matches is False

    def test_invalid_direction_raises(self) -> None:
        """An invalid direction must raise ValueError in __post_init__."""
        from mloda.core.abstract_plugins.components.link import AsOfJoinConfig

        with pytest.raises(ValueError):
            AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="sideways")  # type: ignore[arg-type]  # intentionally invalid to exercise the runtime guard

    def test_is_frozen(self) -> None:
        """AsOfJoinConfig is frozen (immutable)."""
        from dataclasses import FrozenInstanceError

        from mloda.core.abstract_plugins.components.link import AsOfJoinConfig

        cfg = AsOfJoinConfig(left_time_column="t", right_time_column="t")
        with pytest.raises(FrozenInstanceError):
            cfg.direction = "forward"  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        """Two AsOfJoinConfig instances can be placed in a set (hashable)."""
        from mloda.core.abstract_plugins.components.link import AsOfJoinConfig

        cfg1 = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="backward")
        cfg2 = AsOfJoinConfig(left_time_column="t", right_time_column="t", direction="forward")
        cfg_set = {cfg1, cfg2}
        assert len(cfg_set) == 2


# ============================================================================
# Link.asof factory
# ============================================================================
class TestLinkAsof:
    def test_asof_sets_jointype_and_config(self) -> None:
        """Link.asof builds a Link with jointype ASOF and a populated asof_config."""
        link = Link.asof(
            JoinSpec(AsofFGa, "k"),
            JoinSpec(AsofFGb, "k"),
            left_time_column="t",
            right_time_column="t",
        )
        assert link.jointype == JoinType.ASOF
        assert link.left_index == Index(("k",))
        assert link.right_index == Index(("k",))
        assert link.asof_config is not None
        assert link.asof_config.left_time_column == "t"
        assert link.asof_config.right_time_column == "t"
        assert link.asof_config.direction == "backward"

    def test_asof_passes_through_options(self) -> None:
        """Link.asof forwards direction/tolerance/allow_exact_matches into asof_config."""
        link = Link.asof(
            JoinSpec(AsofFGa, "k"),
            JoinSpec(AsofFGb, "k"),
            left_time_column="lt",
            right_time_column="rt",
            direction="nearest",
            tolerance=2.5,
            allow_exact_matches=False,
        )
        assert link.asof_config is not None
        assert link.asof_config.direction == "nearest"
        assert link.asof_config.tolerance == 2.5
        assert link.asof_config.allow_exact_matches is False
        assert link.asof_config.left_time_column == "lt"
        assert link.asof_config.right_time_column == "rt"


# ============================================================================
# Link.asof_on factory (derives by-key index from index_columns())
# ============================================================================
class TestLinkAsofOn:
    def test_asof_on_derives_index(self) -> None:
        """Link.asof_on derives the by-key Index from each feature group's index_columns()."""
        link = Link.asof_on(
            AsofFGa,
            AsofFGb,
            left_time_column="t",
            right_time_column="t",
        )
        assert link.jointype == JoinType.ASOF
        assert link.left_feature_group is AsofFGa
        assert link.right_feature_group is AsofFGb
        assert link.left_index == Index(("k",))
        assert link.right_index == Index(("k",))
        assert link.asof_config is not None
        assert link.asof_config.left_time_column == "t"
        assert link.asof_config.right_time_column == "t"


# ============================================================================
# eq / hash incorporate asof_config
# ============================================================================
class TestAsofLinkEqHash:
    def test_different_direction_not_equal(self) -> None:
        """Two ASOF links identical except direction must NOT be equal."""
        link_backward = Link.asof(
            JoinSpec(AsofFGa, "k"),
            JoinSpec(AsofFGb, "k"),
            left_time_column="t",
            right_time_column="t",
            direction="backward",
        )
        link_forward = Link.asof(
            JoinSpec(AsofFGa, "k"),
            JoinSpec(AsofFGb, "k"),
            left_time_column="t",
            right_time_column="t",
            direction="forward",
        )
        assert link_backward != link_forward

    def test_different_direction_both_survive_in_set(self) -> None:
        """Both ASOF links (differing direction) must survive in a set of size 2."""
        link_backward = Link.asof(
            JoinSpec(AsofFGa, "k"),
            JoinSpec(AsofFGb, "k"),
            left_time_column="t",
            right_time_column="t",
            direction="backward",
        )
        link_forward = Link.asof(
            JoinSpec(AsofFGa, "k"),
            JoinSpec(AsofFGb, "k"),
            left_time_column="t",
            right_time_column="t",
            direction="forward",
        )
        link_set = {link_backward, link_forward}
        assert len(link_set) == 2
