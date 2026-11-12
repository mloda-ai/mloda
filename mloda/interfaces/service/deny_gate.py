"""Extender that refuses catalog-denied names at source matching, whether or not a source has the column."""

from collections.abc import Callable
from typing import Any

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext


class DeniedNameGate(Extender):
    """Refuses source matching for a denied name; its message names nothing."""

    never_fall_back = True

    def __init__(self, is_denied: Callable[[str], bool]) -> None:
        self._is_denied = is_denied

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_MATCH}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        if context is None or any(self._is_denied(name) for name in context.feature_names):
            raise PermissionError("denied by the service policy")
        return func(*args, **kwargs)
