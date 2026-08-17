"""Guards a JoinStep's Link against a feature group class that pickle cannot round-trip,
which otherwise fails deep inside a multiprocessing worker with an opaque
PicklingError instead of being rejected clearly at plan time. See issue #1117.
"""

import pickle  # nosec
from collections.abc import Iterable
from typing import Any

from mloda.core.abstract_plugins.feature_group import format_feature_group_class
from mloda.core.core.step.join_step import JoinStep

_UNPICKLABLE_ERRORS = (pickle.PicklingError, AttributeError, TypeError)


def _is_picklable(feature_group: type[Any]) -> bool:
    try:
        pickle.dumps(feature_group)
    except _UNPICKLABLE_ERRORS:
        return False
    return True


def _unpicklable_link_error(feature_group: type[Any], step: JoinStep) -> str:
    return (
        f"Link {step.link} references {format_feature_group_class(feature_group)}, which cannot be "
        "pickled for multiprocessing. Feature groups created inside a local function or via a "
        "dynamic type(...) factory are not picklable; define the feature group at module level."
    )


def raise_on_unpicklable_join_link(steps: Iterable[Any]) -> None:
    """Raise ValueError if any JoinStep in steps carries a Link whose feature group is unpicklable."""
    for step in steps:
        if not isinstance(step, JoinStep):
            continue

        if not _is_picklable(step.link.left_feature_group):
            raise ValueError(_unpicklable_link_error(step.link.left_feature_group, step))

        if not _is_picklable(step.link.right_feature_group):
            raise ValueError(_unpicklable_link_error(step.link.right_feature_group, step))
