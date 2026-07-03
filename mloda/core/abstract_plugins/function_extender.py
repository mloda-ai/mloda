from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional
import functools
import inspect
import logging

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.components.feature_set import FeatureSet
    from mloda.core.abstract_plugins.components.options import Options


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
                try:
                    return ext.__call__(inner_func, *a, **kw)
                except Exception as e:
                    logging.error(f"{ext.__class__.__name__} {ext.name if hasattr(ext, 'name') else ''} {str(e)}")
                    return inner_func(*a, **kw)

            return wrapper

        wrapped_func = func
        for extender in reversed(self.extenders):
            wrapped_func = make_wrapper(extender, wrapped_func)
        return wrapped_func(*args, **kwargs)
