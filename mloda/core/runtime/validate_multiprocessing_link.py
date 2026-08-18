"""Guards JoinStep, FeatureGroupStep, and TransformFrameworkStep against classes and values
pickle cannot round-trip, which otherwise fail deep inside a multiprocessing worker with an
opaque PicklingError instead of being rejected clearly at plan time. This only proves
resolvability in the current process: a class resolvable here but not inside a freshly spawned
worker (e.g. one under `if __name__ == "__main__":`) can still fail there.
"""

import pickle  # nosec
from collections.abc import Iterable
from typing import Any

from mloda.core.core.step.feature_group_step import FeatureGroupStep
from mloda.core.core.step.join_step import JoinStep
from mloda.core.core.step.transform_frame_work_step import TransformFrameworkStep

_UNPICKLABLE_ERRORS = (pickle.PicklingError, AttributeError, TypeError)


def _is_picklable(value: Any) -> bool:
    try:
        pickle.dumps(value)
    except _UNPICKLABLE_ERRORS:
        return False
    return True


def _unpicklable_link_error(feature_group: type[Any], step: JoinStep) -> str:
    return (
        f"Link {step.link} references {feature_group.__name__} "
        f"({feature_group.__module__}.{feature_group.__qualname__}), which pickle cannot resolve back by "
        "that path, so multiprocessing cannot send this join to a worker process. This happens when a "
        "feature group class is created inside a function or by a dynamic type(...) factory instead of "
        "being defined at module level.\n"
        "Resolution: define the feature group class at module level, or run without "
        "ParallelizationMode.MULTIPROCESSING."
    )


def _unpicklable_link_generic_error(step: JoinStep) -> str:
    return (
        f"Link {step.link} cannot be pickled for multiprocessing, though both its feature group classes "
        "are picklable on their own. The left_discriminator, right_discriminator, or asof_config likely "
        "holds a value pickle cannot resolve (e.g. a locally defined class or a lambda).\n"
        "Resolution: use only picklable values in the link's discriminators and asof configuration, or "
        "run without ParallelizationMode.MULTIPROCESSING."
    )


def _unpicklable_feature_group_step_error(feature_group: type[Any], step: FeatureGroupStep) -> str:
    return (
        f"FeatureGroupStep for {feature_group.__name__} "
        f"({feature_group.__module__}.{feature_group.__qualname__}) references a feature group class that pickle "
        "cannot resolve back by that path, so multiprocessing cannot send this step to a worker process. This happens "
        "when a feature group class is created inside a function or by a dynamic type(...) factory instead of being "
        "defined at module level.\n"
        "Resolution: define the feature group class at module level, or run without "
        "ParallelizationMode.MULTIPROCESSING."
    )


def _unpicklable_transform_step_error(feature_group: type[Any], step: TransformFrameworkStep) -> str:
    return (
        f"TransformFrameworkStep references {feature_group.__name__} "
        f"({feature_group.__module__}.{feature_group.__qualname__}), which pickle cannot resolve back by "
        "that path, so multiprocessing cannot send this transformation to a worker process. This happens when a "
        "feature group class is created inside a function or by a dynamic type(...) factory instead of "
        "being defined at module level.\n"
        "Resolution: define the feature group class at module level, or run without "
        "ParallelizationMode.MULTIPROCESSING."
    )


def raise_on_unpicklable_multiprocessing_steps(steps: Iterable[Any]) -> None:
    """Raise ValueError if any step in steps carries an unpicklable FeatureGroup, Link, or transformation."""
    for step in steps:
        if isinstance(step, JoinStep):
            if _is_picklable(step.link):
                continue

            if not _is_picklable(step.link.left_feature_group):
                raise ValueError(_unpicklable_link_error(step.link.left_feature_group, step))

            if not _is_picklable(step.link.right_feature_group):
                raise ValueError(_unpicklable_link_error(step.link.right_feature_group, step))

            raise ValueError(_unpicklable_link_generic_error(step))

        if isinstance(step, FeatureGroupStep):
            if not _is_picklable(step.feature_group):
                raise ValueError(_unpicklable_feature_group_step_error(step.feature_group, step))

        if isinstance(step, TransformFrameworkStep):
            if not _is_picklable(step.from_feature_group):
                raise ValueError(_unpicklable_transform_step_error(step.from_feature_group, step))
            if not _is_picklable(step.to_feature_group):
                raise ValueError(_unpicklable_transform_step_error(step.to_feature_group, step))


raise_on_unpicklable_join_link = raise_on_unpicklable_multiprocessing_steps

