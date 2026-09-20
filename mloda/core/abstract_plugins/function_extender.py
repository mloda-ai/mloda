from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any
import functools
import inspect
import logging

from mloda.core.abstract_plugins.components.utils import contained_raise_reason

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.components.feature_set import FeatureSet
    from mloda.core.abstract_plugins.components.options import Options


logger = logging.getLogger(__name__)


class ExtenderHook(Enum):
    FEATURE_GROUP_CALCULATE_FEATURE = "feature_group_calculate_feature"
    VALIDATE_INPUT_FEATURE = "validate_input_feature"
    VALIDATE_OUTPUT_FEATURE = "validate_output_feature"
    FEATURE_GROUP_MATCHED = "feature_group_matched"
    INPUT_DATA_LOAD = "input_data_load"
    JOIN = "join"


class Extender(ABC):
    """
    - Automated Metadata harvester connector
    - Messaging Integration ( email )
    - Automation Tools
    - data lineage mapping
    - Impact Analysis
    - Audit Trail
    - Monitoring alerts
    - metadata capture
    - Event logging
    - metrics on feature calculation
    - visibility / observability
    - Performance
    """

    @property
    def priority(self) -> int:
        """Lower priority runs first. Default is 100."""
        if hasattr(self, "_priority"):
            return self._priority
        return 100

    @priority.setter
    def priority(self, value: int) -> None:
        self._priority = value

    @property
    def raise_on_error(self) -> bool:
        """Whether a failure in this extender breaks the calculation.

        True (default) means a failure in this extender propagates and breaks the
        calculation; False means the failure is logged as a warning and the wrapped
        function is called instead. Ignored when never_fall_back is True.
        """
        return getattr(self, "_raise_on_error", True)

    @raise_on_error.setter
    def raise_on_error(self, value: bool) -> None:
        self._raise_on_error = value

    # For gates: True makes a failure always propagate, never falling back to the wrapped call.
    never_fall_back: bool = False

    @abstractmethod
    def wraps(self) -> set[ExtenderHook]:
        pass

    @abstractmethod
    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        pass

    def close(self) -> None:
        """Called once per worker-side copy on graceful MULTIPROCESSING worker exit, to flush a
        buffering sink. Never called in SYNC or THREADING mode. Exceptions raised here are caught
        and logged, never propagated. On the parent-death watchdog's ``os._exit(0)`` path this is
        best-effort and racy: it may not run at all, since that path skips Python cleanup. It must
        return within the run's ``graceful_shutdown_timeout`` (default 2.0s), a budget shared across
        every extender closing in that worker, or the worker may be terminated mid-close."""

    def on_run_complete(self, run_id: str | None) -> None:
        """Called once per run in the PARENT on the caller's own extender objects, after all workers
        were joined, in every mode (close() is MULTIPROCESSING worker only). Fires when execution raised,
        not when the run failed before execution started (setup or validation errors) nor when joining
        the workers raised. A session re-run fires again with the same run_id. raise_on_error and
        never_fall_back do not apply: an Exception raised here is always logged, never propagated."""

    @staticmethod
    def feature_group_name(func: Any) -> str:
        """Resolve the owning feature group class name of the hooked callable.

        Returns the string sentinel "unknown" (never None) when the owner cannot be resolved.
        """

        def owner_name(candidate: Any) -> str:
            owner = candidate.__self__
            if isinstance(owner, type):
                return str(owner.__name__)
            return str(owner.__class__.__name__)

        if hasattr(func, "__self__"):
            return owner_name(func)
        unwrapped = inspect.unwrap(func)
        if hasattr(unwrapped, "__self__"):
            return owner_name(unwrapped)
        qualname = getattr(unwrapped, "__qualname__", "")
        parts = qualname.split(".")
        if len(parts) >= 2 and parts[-2] != "<locals>":
            return str(parts[-2])
        return "unknown"

    @staticmethod
    def feature_set(args: tuple[Any, ...]) -> "FeatureSet | None":
        """Return the first FeatureSet in the hook args, else None."""
        from mloda.core.abstract_plugins.components.feature_set import FeatureSet

        for arg in args:
            if isinstance(arg, FeatureSet):
                return arg
        return None

    @staticmethod
    def feature_name(args: tuple[Any, ...]) -> str | None:
        """Return the name of one feature from the FeatureSet in the hook args, else None."""
        feature_set = Extender.feature_set(args)
        if feature_set is None or feature_set.name_of_one_feature is None:
            return None
        return str(feature_set.name_of_one_feature)

    @staticmethod
    def feature_options(args: tuple[Any, ...]) -> "Options | None":
        """Return the Options of the FeatureSet in the hook args, else None."""
        feature_set = Extender.feature_set(args)
        if feature_set is None:
            return None
        return feature_set.options


def get_function_extender(function_extender: set[Extender], hook: ExtenderHook) -> Extender | None:
    """Select the Extender(s) wrapping hook: None, the sole match, or a priority-sorted CompositeExtender."""
    matching_extenders = [ext for ext in function_extender if hook in ext.wraps()]
    if len(matching_extenders) == 0:
        return None
    if len(matching_extenders) == 1:
        return matching_extenders[0]
    sorted_extenders = sorted(matching_extenders, key=lambda e: e.priority)
    return CompositeExtender(sorted_extenders, hook)


class CompositeExtender(Extender):
    """Chains multiple Extenders together, running them in priority order.

    Constructed internally by get_function_extender(); not meant to be subclassed or instantiated directly.
    """

    def __init__(self, extenders: list[Extender], function_type: ExtenderHook | None = None):
        self.extenders = sorted(extenders, key=lambda e: e.priority)
        self.function_type = function_type

    def wraps(self) -> set[ExtenderHook]:
        if self.function_type:
            return {self.function_type}
        result = set()
        for extender in self.extenders:
            result.update(extender.wraps())
        return result

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        def make_wrapper(ext: Extender, inner_func: Any) -> Any:
            @functools.wraps(inner_func)
            def wrapper(*a: Any, **kw: Any) -> Any:
                return _invoke_extender(ext, inner_func, *a, **kw)

            return wrapper

        wrapped_func = func
        for extender in reversed(self.extenders):
            wrapped_func = make_wrapper(extender, wrapped_func)
        return wrapped_func(*args, **kwargs)


def _invoke_extender(ext: Extender, inner_func: Any, *args: Any, **kwargs: Any) -> Any:
    """Invoke an extender around inner_func, scoping any warning-only fallback to the
    extender's OWN code so inner-function failures propagate and inner never re-runs."""
    # Breaking (default) or never_fall_back: call directly, everything propagates.
    if ext.raise_on_error or ext.never_fall_back:
        return ext.__call__(inner_func, *args, **kwargs)

    # Warning-only: guard ONLY the extender's own code. Wrap inner_func so we can tell
    # whether a raised exception came from inner_func (must propagate, never swallow,
    # never re-run) versus the extender's own instrumentation (log + fall back).
    sentinel = object()
    state: dict[str, Any] = {"result": sentinel, "inner_raised": False}

    @functools.wraps(inner_func)
    def guarded_inner(*a: Any, **kw: Any) -> Any:
        try:
            result = inner_func(*a, **kw)
        except BaseException:
            state["inner_raised"] = True
            raise
        state["result"] = result
        return result

    try:
        return ext.__call__(guarded_inner, *args, **kwargs)
    except Exception as e:
        if state["inner_raised"]:
            # The failure came from the wrapped function / downstream chain, not this
            # extender. Propagate unchanged; do not swallow, do not re-run.
            raise
        name = ext.name if hasattr(ext, "name") else ""
        # Text, not exc_info=True: the record would pin the traceback, and its frames hold the extender and
        # the wrapped callable.
        logger.warning("%s %s %s", ext.__class__.__name__, name, contained_raise_reason(e))
        if state["result"] is not sentinel:
            # Inner already ran successfully; the extender failed afterwards.
            # Return the already-computed result; do NOT re-run inner.
            return state["result"]
        # Extender failed before delegating; run inner exactly once as the fallback.
        return inner_func(*args, **kwargs)
