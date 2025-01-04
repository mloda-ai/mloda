from typing import Any, Set
from mloda_core.abstract_plugins.function_extender import WrapperFunctionEnum, WrapperFunctionExtender


import logging

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
except ImportError:
    trace = None  # type: ignore


class OtelExtender(WrapperFunctionExtender):
    def __init__(self) -> None:
        if trace is None:
            return

        self.wrapped = {WrapperFunctionEnum.FEATURE_GROUP_CALCULATE_FEATURE}

    def wraps(self) -> Set[WrapperFunctionEnum]:
        return self.wrapped

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        logger.warning("OtelExtender")
        result = func(*args, **kwargs)
        return result
