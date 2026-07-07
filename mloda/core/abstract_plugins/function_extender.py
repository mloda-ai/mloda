from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional
import functools
import inspect
import logging

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.components.feature_set import FeatureSet
    from mloda.core.abstract_plugins.components.options import Options


logger = logging.getLogger(__name__)


class ExtenderHook(Enum):
    FEATURE_GROUP_CALCULATE_FEATURE = "feature_group_calculate_feature"
    VALIDATE_INPUT_FEATURE = "validate_input_feature"
    VALIDATE_OUTPUT_FEATURE = "validate_output_feature"


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
        function is called instead.
        """
        return getattr(self, "_raise_on_error", True)

    @raise_on_error.setter
    def raise_on_error(self, value: bool) -> None:
        self._raise_on_error = value

    @abstractmethod
    def wraps(self) -> set[ExtenderHook]:
        pass

    @abstractmethod
    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        pass

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


class _CompositeExtender(Extender):
    """Internal class that chains multiple Extenders in priority order."""

    def __init__(self, extenders: list[Extender], function_type: Optional[ExtenderHook] = None):
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
    # Breaking (default): call directly, everything propagates.
    if ext.raise_on_error:
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
        logger.warning(f"{ext.__class__.__name__} {name} {type(e).__name__}: {e}", exc_info=True)
        if state["result"] is not sentinel:
            # Inner already ran successfully; the extender failed afterwards.
            # Return the already-computed result; do NOT re-run inner.
            return state["result"]
        # Extender failed before delegating; run inner exactly once as the fallback.
        return inner_func(*args, **kwargs)
