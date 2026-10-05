from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal
import functools
import inspect
import logging

from mloda.core.abstract_plugins.components.utils import contained_raise_reason

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.components.feature_set import FeatureSet
    from mloda.core.abstract_plugins.components.options import Options
    from mloda.core.abstract_plugins.plan_context import PlanContext
    from mloda.core.abstract_plugins.run_context import RunContext
    from mloda.core.api.plan_info import PlanStep


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LifecycleOutcome:
    """How a plan or run ended, passed to on_plan_complete and on_run_complete."""

    status: Literal["succeeded", "failed", "cancelled"]
    error_type: str | None = None


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

    # Opt-in: True makes an Exception from on_run_complete fail a run that otherwise succeeded.
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

    def on_plan_start(self, plan: "PlanContext") -> None:
        """Called once in the PARENT when planning begins (prepare, explain, diagnose, run_all, stream_all).
        An Exception raised here is logged, never propagated."""

    def on_plan_complete(self, plan: "PlanContext", outcome: "LifecycleOutcome") -> None:
        """Called once in the PARENT when planning ends, whether it succeeded or failed.
        An Exception raised here is logged, never propagated."""

    def on_run_start(self, run: "RunContext", plan: "PlanContext", steps: "tuple[PlanStep, ...]") -> None:
        """Called once in the PARENT per run, before any setup or compute, in sorted extender order.
        An Exception propagates and refuses the run when raise_on_error or never_fall_back is True,
        else it is logged and the run proceeds. Extenders after a refusing one get on_run_complete only."""

    def on_run_complete(self, run: "RunContext", outcome: "LifecycleOutcome") -> None:
        """Called once per run() or stream, in the PARENT on the caller's own extender objects, after the
        workers were joined and the runner exited, in every mode (close() is MULTIPROCESSING worker only).
        Fires with a failed outcome when setup, execution, finalizing or a run_start refusal raised, and
        with cancelled when a stream is closed early or never iterated. An Exception raised here is
        logged, unless raise_on_run_complete is True and the outcome succeeded, then the first such one
        is re-raised after every extender was called. The worker copy is pickled once per run at setup,
        so its changes never flow back to the parent."""

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


def call_contained_hook(extenders: Iterable[Extender], hook: str, *args: Any) -> None:
    """Call a lifecycle hook on every extender in sorted order, logging exceptions and continuing."""
    for extender in sorted(extenders, key=extender_sort_key):
        try:
            getattr(extender, hook)(*args)
        except Exception as e:
            logger.error("Extender %s.%s() %s", extender.__class__.__name__, hook, contained_raise_reason(e))


def call_run_complete_hook(extenders: Iterable[Extender], run: "RunContext", outcome: LifecycleOutcome) -> None:
    """Call on_run_complete on every extender; re-raise the first raise_on_run_complete failure of a successful run."""
    failure: Exception | None = None
    for extender in sorted(extenders, key=extender_sort_key):
        try:
            extender.on_run_complete(run, outcome)
        except Exception as e:
            if extender.raise_on_run_complete and outcome.status == "succeeded" and failure is None:
                failure = e
            else:
                logger.error("Extender %s.on_run_complete() %s", extender.__class__.__name__, contained_raise_reason(e))
    if failure is not None:
        raise failure


def call_run_start_hook(extenders: Iterable[Extender], run: "RunContext", plan: "PlanContext", steps: Any) -> None:
    """Call on_run_start in sorted order; a breaking or never_fall_back extender's exception refuses the run."""
    for extender in sorted(extenders, key=extender_sort_key):
        try:
            extender.on_run_start(run, plan, steps)
        except Exception as e:
            if extender.raise_on_error or extender.never_fall_back:
                raise
            logger.warning("Extender %s.on_run_start() %s", extender.__class__.__name__, contained_raise_reason(e))


def reject_old_run_complete_signature(extenders: Iterable[Extender]) -> None:
    """Raise TypeError for an extender whose on_run_complete still takes the old single run_id parameter."""
    for extender in extenders:
        parameters = inspect.signature(extender.on_run_complete).parameters.values()
        positional = [p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if len(positional) < 2 and not any(p.kind is p.VAR_POSITIONAL for p in parameters):
            raise TypeError(
                f"{type(extender).__name__}.on_run_complete must accept (run, outcome); "
                "the single run_id parameter was replaced."
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

    extenders: tuple[Extender, ...]

    def __init__(self, extenders: Iterable[Extender], function_type: ExtenderHook | None = None):
        extenders = list(extenders)
        self.extenders = tuple(sorted(extenders, key=extender_sort_key))
        self.function_type = function_type
        self.never_fall_back = any(child.never_fall_back for child in extenders)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("extenders", "function_type") and self.__dict__.get("_sealed"):
            raise AttributeError(f"CompositeExtender.{name} is read-only once sealed")
        super().__setattr__(name, value)

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
