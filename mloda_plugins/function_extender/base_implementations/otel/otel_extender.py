from typing import Any, Set
from mloda.steward import Extender, ExtenderHook


import logging

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
except ImportError:
    trace = None  # type: ignore[assignment]


class OtelExtender(Extender):
    def __init__(self) -> None:
        if trace is None:
            return

        self.wrapped = {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def wraps(self) -> Set[ExtenderHook]:
        return self.wrapped

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        logger.warning("OtelExtender")
        result = func(*args, **kwargs)
        return result
