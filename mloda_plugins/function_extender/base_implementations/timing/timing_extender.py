import logging
import time
from typing import Any

from mloda.steward import Extender, ExtenderHook

logger = logging.getLogger(__name__)


class TimingExtender(Extender):
    """Logs how long the wrapped feature-group calculation took."""

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Time taken: {time.time() - start}")
        return result
