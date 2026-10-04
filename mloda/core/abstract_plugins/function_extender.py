from abc import ABC, abstractmethod
from enum import Enum
from collections.abc import Iterable
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


class GateBypassError(RuntimeError):
    """Raised when an extender skips the wrapped call while a never_fall_back gate sits inside it."""


_SEALED_ATTRS = frozenset(
    {"never_fall_back", "raise_on_run_complete", "raise_on_error", "priority", "_raise_on_error", "_priority"}
)


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

    Once __call__ invokes the wrapped call, its own return value is discarded in favor of
    the wrapped result (mutating shared args in place still works). If it calls the wrapped
    function more than once, only the first successful call's result is used.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _SEALED_ATTRS and self.__dict__.get("_sealed", False):
            raise AttributeError(f"{type(self).__name__}.{name} cannot be changed after first use")
        object.__setattr__(self, name, value)

    def _seal(self) -> None:
        object.__setattr__(self, "_sealed", True)

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

    # For gates: True makes an Exception from on_run_complete fail a run that otherwise succeeded.
    raise_on_run_complete: bool = False

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
        every extender closing in that worker, or the worker may be terminated mid-close.
        ``CloseContext.current()`` returns the run fields, the close reason, and ``remaining()``.
        ``remaining()`` is approximate: it can overstate by the time between the parent sending
        STOP and this worker starting to close. Capture it and pass it to threads, since a thread
        started inside close() does not inherit it; close order across extenders is unspecified."""

    def on_run_complete(self, run_id: str | None) -> None:
        """Called once per run in the PARENT on the caller's own extender objects, after all workers
        were joined, in every mode (close() is MULTIPROCESSING worker only). Fires once per run(),
        run_all(), stream_run() or stream_all() call that got as far as setting up execution (a stream
        does on its first iteration; one closed early fires after its workers are joined). Fires when
        setup (e.g. the MULTIPROCESSING picklability preflight) or execution raised, so it is not a
        success signal. Does not fire for prepare, explain, a never-iterated stream, a failure while
        planning before setup, or when finalizing raised (collecting artifacts, joining or terminating
        the workers). A session re-run fires again with the same run_id. raise_on_error and
        never_fall_back do not apply; an Exception here is logged, unless raise_on_run_complete is
        True and the run succeeded (a stream only when exhausted), then the first such one is
        re-raised after every extender was notified.
        The worker copy is pickled once per run at setup (a stream's first iteration), so for runs
        executed one after another it reflects the parent's state after the previous run's
        on_run_complete; its changes never flow back to the parent."""

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


def extender_sort_key(extender: Extender) -> tuple[int, int, str, str]:
    """Deterministic order: gates first, then priority, then class module and qualified name."""
    return (
        0 if extender.never_fall_back else 1,
        extender.priority,
        type(extender).__module__,
        type(extender).__qualname__,
    )


def build_hook_extenders(function_extender: Iterable[Extender]) -> dict[ExtenderHook, Extender]:
    """Map each hook to its sole extender or a sorted CompositeExtender, reading wraps() once per extender."""
    grouped: dict[ExtenderHook, list[Extender]] = {}
    extenders = sorted(function_extender, key=extender_sort_key)
    for extender in extenders:
        hooks = extender.wraps()
        if not all(isinstance(hook, ExtenderHook) for hook in hooks):
            raise TypeError(f"{type(extender).__name__}.wraps() must return ExtenderHook members, got {hooks!r}")
        for hook in dict.fromkeys(hooks):
            grouped.setdefault(hook, []).append(extender)
    built = {hook: exts[0] if len(exts) == 1 else CompositeExtender(exts, hook) for hook, exts in grouped.items()}
    for extender in [*extenders, *built.values()]:
        extender._seal()
    return built


def get_function_extender(function_extender: Iterable[Extender], hook: ExtenderHook) -> Extender | None:
    """Select the Extender(s) wrapping hook: None, the sole match, or a sorted CompositeExtender."""
    return build_hook_extenders(function_extender).get(hook)


class CompositeExtender(Extender):
    """Chains multiple Extenders together, running them in priority order.

    Constructed internally by build_hook_extenders(); not meant to be subclassed or instantiated directly.
    """

    def __init__(self, extenders: list[Extender], function_type: ExtenderHook | None = None):
        self.extenders = sorted(extenders, key=extender_sort_key)
        self.function_type = function_type
        self.never_fall_back = any(child.never_fall_back for child in extenders)

    def wraps(self) -> set[ExtenderHook]:
        if self.function_type:
            return {self.function_type}
        result = set()
        for extender in self.extenders:
            result.update(extender.wraps())
        return result

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        def make_wrapper(ext: Extender, inner_func: Any, gate_inside: bool) -> Any:
            @functools.wraps(inner_func)
            def wrapper(*a: Any, **kw: Any) -> Any:
                return _invoke_extender(ext, inner_func, *a, gate_inside=gate_inside, **kw)

            return wrapper

        wrapped_func = func
        for index in range(len(self.extenders) - 1, -1, -1):
            gate_inside = any(e.never_fall_back for e in self.extenders[index + 1 :])
            wrapped_func = make_wrapper(self.extenders[index], wrapped_func, gate_inside)
        return wrapped_func(*args, **kwargs)


def _invoke_extender(ext: Extender, inner_func: Any, *args: Any, gate_inside: bool = False, **kwargs: Any) -> Any:
    """Invoke an extender around inner_func, scoping any warning-only fallback to the
    extender's OWN code so inner-function failures propagate and inner never re-runs."""
    # Guard inner_func so its result wins over ext.__call__'s return, and (warning-only
    # branch) so an inner exception can be told apart from the extender's own failure.
    sentinel = object()
    state: dict[str, Any] = {"result": sentinel, "inner_raised": False, "inner_exc": None, "called": False}

    @functools.wraps(inner_func)
    def guarded_inner(*a: Any, **kw: Any) -> Any:
        state["called"] = True
        try:
            result = inner_func(*a, **kw)
        except BaseException as exc:
            state["inner_raised"] = True
            state["inner_exc"] = exc
            raise
        if state["result"] is sentinel:
            state["result"] = result
        return result

    def _settle(ext_return: Any) -> Any:
        inner_exc = state["inner_exc"]
        state["inner_exc"] = None
        if state["result"] is not sentinel:
            return state["result"]
        if inner_exc is not None:
            try:
                raise inner_exc
            finally:
                del inner_exc
        if gate_inside and not state["called"]:
            raise GateBypassError(
                f"{type(ext).__name__} {getattr(ext, 'name', '')} did not call the wrapped function while a gate is inside it"
            )
        return ext_return

    # Breaking (default) or never_fall_back: call directly, everything propagates.
    if ext.raise_on_error or ext.never_fall_back:
        try:
            return _settle(ext.__call__(guarded_inner, *args, **kwargs))
        finally:
            state["inner_exc"] = None

    # Warning-only: guard ONLY the extender's own code.
    try:
        return _settle(ext.__call__(guarded_inner, *args, **kwargs))
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
    finally:
        state["inner_exc"] = None
