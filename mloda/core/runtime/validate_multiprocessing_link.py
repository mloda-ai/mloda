"""Guards a Link, a feature group class, a child_bootstrap callable, or an extender against a value
pickle cannot round-trip, which otherwise fails deep inside a multiprocessing worker with an opaque
PicklingError instead of being rejected clearly at plan time. Also guards against a compute framework
that supports MULTIPROCESSING but resolved a live connection from the DataAccessCollection: such a
TFS destination running in a spawned worker is never handed a connection at all, since the worker-side
connection bind only runs on the sync and threading execute paths. This only proves resolvability in
the current process: a value resolvable here but not inside a freshly spawned worker can still fail
there.
"""

import pickle  # nosec
from collections.abc import Callable, Iterable
from typing import Any

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.parallelization_modes import ParallelizationMode
from mloda.core.abstract_plugins.function_extender import Extender
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

        if ParallelizationMode.MULTIPROCESSING not in step.get_parallelization_mode():
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


def _unpicklable_child_bootstrap_error(child_bootstrap: Callable[[], None]) -> str:
    return (
        f"child_bootstrap ({child_bootstrap!r}) cannot be pickled for multiprocessing, so mloda cannot send it to "
        "a spawned worker process. This happens when the callable is a lambda, a closure over a local variable or "
        "an unpicklable object, or an instance of a class defined inside a function instead of at module level.\n"
        "Resolution: use a plain, picklable, no-argument callable defined at module level, or run without "
        "ParallelizationMode.MULTIPROCESSING."
    )


def raise_on_unpicklable_child_bootstrap(child_bootstrap: Callable[[], None] | None) -> None:
    """Raise ValueError if child_bootstrap is not None and multiprocessing cannot pickle it."""
    if child_bootstrap is None:
        return

    if _is_picklable(child_bootstrap):
        return

    raise ValueError(_unpicklable_child_bootstrap_error(child_bootstrap))


def _unpicklable_extender_error(extender: Extender) -> str:
    return (
        f"Extender {extender!r} cannot be pickled for multiprocessing, so mloda cannot send it to a spawned "
        "worker process. This happens when an extender instance holds unpicklable state (e.g. a "
        "threading.Lock, an open connection, or a client handle) set eagerly in __init__.\n"
        "Resolution: rebuild such state lazily (e.g. inside __call__ or __setstate__) instead of storing it "
        "eagerly in __init__, or run without ParallelizationMode.MULTIPROCESSING."
    )


def raise_on_unpicklable_extender(function_extender: set[Extender] | None) -> None:
    """Raise ValueError if function_extender is not None/empty and any extender in it cannot be pickled."""
    if not function_extender:
        return

    for extender in function_extender:
        if not _is_picklable(extender):
            raise ValueError(_unpicklable_extender_error(extender))


def _multiprocessing_connection_conflict_error(cfw_class: type[ComputeFramework]) -> str:
    return (
        f"{cfw_class.__name__} supports ParallelizationMode.MULTIPROCESSING and a connection was "
        "resolved for it from the DataAccessCollection, but a TFS destination that runs in a spawned "
        "worker is never handed a connection (the worker-side connection bind only runs on the sync "
        "and threading execute paths).\n"
        "Resolution: the framework author excludes ParallelizationMode.MULTIPROCESSING from "
        f"{cfw_class.__name__}.supported_parallelization_modes(), or the caller runs without "
        "ParallelizationMode.MULTIPROCESSING, or the caller omits that connection from the "
        "DataAccessCollection."
    )


def raise_on_multiprocessing_connection_conflict(tfs_connection_map: dict[type[ComputeFramework], Any]) -> None:
    """Raise ValueError if a cfw class in tfs_connection_map supports MULTIPROCESSING.

    Every key is already a TFS destination class for which Engine resolved a real connection, so the
    map alone carries the full condition; no Step objects need inspecting.
    """
    for cfw_class in tfs_connection_map:
        if ParallelizationMode.MULTIPROCESSING in cfw_class.supported_parallelization_modes():
            raise ValueError(_multiprocessing_connection_conflict_error(cfw_class))
