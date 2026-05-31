from typing import Optional

from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import AsOfJoinConfig, JoinSpec, JoinType, Link
from mloda.provider import FeatureGroup


class _MergeTestFeatureGroup(FeatureGroup):
    """Placeholder feature group for building merge-engine unit-test Links.

    merge() reads only jointype / indexes / asof_config off the Link, never the
    feature groups, so one shared placeholder class is a sufficient carrier.
    """


def make_merge_link(
    jointype: JoinType,
    left_index: Index,
    right_index: Index,
    asof_config: Optional[AsOfJoinConfig] = None,
) -> Link:
    """Build a minimal Link carrying jointype + indexes (+ optional asof_config) for unit tests."""
    return Link(
        jointype,
        JoinSpec(_MergeTestFeatureGroup, left_index),
        JoinSpec(_MergeTestFeatureGroup, right_index),
        asof_config=asof_config,
    )
