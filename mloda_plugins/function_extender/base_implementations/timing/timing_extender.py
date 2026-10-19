import logging
import time
from typing import Any

from mloda.steward import Extender, ExtenderHook, HookContext

logger = logging.getLogger(__name__)


class TimingExtender(Extender):
    """Logs how long the wrapped feature-group calculation took."""

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        context = HookContext.current()
        name = (
            context.feature_group_class
            if context is not None and context.feature_group_class
            else "unknown feature group"
        )
        logger.info(f"Time taken for {name}: {elapsed}s")
        return result
