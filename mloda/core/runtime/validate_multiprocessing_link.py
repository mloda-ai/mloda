"""Guards a JoinStep's Link, and a FeatureGroupStep's or TransformFrameworkStep's feature group
classes, against a value pickle cannot round-trip, which otherwise fails deep inside a
multiprocessing worker with an opaque PicklingError instead of being rejected clearly at plan
time. This only proves resolvability in the current process: a class resolvable here but not
inside a freshly spawned worker (e.g. one under `if __name__ == "__main__":`) can still fail there.
"""

import pickle  # nosec
from collections.abc import Iterable
from typing import Any

from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
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


def raise_on_unpicklable_join_link(steps: Iterable[Any]) -> None:
    """Raise ValueError if any JoinStep in steps carries a Link that multiprocessing cannot pickle."""
    for step in steps:
        if not isinstance(step, JoinStep):
            continue

        if _is_picklable(step.link):
            continue

        if not _is_picklable(step.link.left_feature_group):
            raise ValueError(_unpicklable_link_error(step.link.left_feature_group, step))

        if not _is_picklable(step.link.right_feature_group):
            raise ValueError(_unpicklable_link_error(step.link.right_feature_group, step))

        raise ValueError(_unpicklable_link_generic_error(step))


def _unpicklable_step_feature_group_error(feature_group: type[Any], step: Any) -> str:
    return (
        f"{type(step).__name__} (uuid={step.uuid}) references {feature_group.__name__} "
        f"({feature_group.__module__}.{feature_group.__qualname__}), which pickle cannot resolve back by "
        "that path, so multiprocessing cannot send this step to a worker process. This happens when a "
        "feature group class is created inside a function, by a dynamic type(...) factory, or by "
        "DynamicFeatureGroupCreator, instead of being defined at module level.\n"
        "Resolution: define the feature group class at module level, or run without "
        "ParallelizationMode.MULTIPROCESSING."
    )


def _unpicklable_step_generic_error(step: Any) -> str:
    return (
        f"{type(step).__name__} (uuid={step.uuid}) cannot be pickled for multiprocessing, though its "
        "feature group class(es) are picklable on their own. Some other value the step carries (e.g. a "
        "Feature's Options) likely holds a value pickle cannot resolve, such as a locally defined class "
        "or a lambda.\n"
        "Resolution: use only picklable values on this step, or run without "
        "ParallelizationMode.MULTIPROCESSING."
    )


def _step_feature_groups(step: Any) -> tuple[type[Any], ...]:
    if isinstance(step, FeatureGroupStep):
        return (step.feature_group,)
    return (step.from_feature_group, step.to_feature_group)


def raise_on_unpicklable_step_feature_group(steps: Iterable[Any]) -> None:
    """Raise ValueError if a FeatureGroupStep or TransformFrameworkStep in steps, or anything it
    carries, is something multiprocessing cannot pickle."""
    for step in steps:
        if not isinstance(step, (FeatureGroupStep, TransformFrameworkStep)):
            continue

        if ParallelizationMode.MULTIPROCESSING not in step.get_parallelization_mode():
            continue

        if _is_picklable(step):
            continue

        for feature_group in _step_feature_groups(step):
            if not _is_picklable(feature_group):
                raise ValueError(_unpicklable_step_feature_group_error(feature_group, step))

        raise ValueError(_unpicklable_step_generic_error(step))
