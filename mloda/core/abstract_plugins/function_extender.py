from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional, Set
import logging


class ExtenderHook(Enum):
    FEATURE_GROUP_CALCULATE_FEATURE = "feature_group_calculate_feature"
    VALIDATE_INPUT_FEATURE = "validate_input_feature"
    VALIDATE_OUTPUT_FEATURE = "validate_output_feature"


class Extender(ABC):
    """
    - Automated Metadata harvestor connector
    - Messaging Integration ( email )
    - Automation Tools
    - data lineage mapping
    - Impact Analysis
    - Audit Trail
    - Monitoring alerts
    - metadata capture
    - Event logging
    - metrics on feature calculation
    - visibility / observibility
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
    def wraps(self) -> Set[ExtenderHook]:
        pass

    @abstractmethod
    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        pass


class _CompositeExtender(Extender):
    """Internal class that chains multiple Extenders in priority order."""

    def __init__(self, extenders: List[Extender], function_type: Optional[ExtenderHook] = None):
        self.extenders = sorted(extenders, key=lambda e: e.priority)
        self.function_type = function_type

    def wraps(self) -> Set[ExtenderHook]:
        if self.function_type:
            return {self.function_type}
        result = set()
        for extender in self.extenders:
            result.update(extender.wraps())
        return result

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        def make_wrapper(ext: Extender, inner_func: Any) -> Any:
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
