"""Extender that refuses source groups' matching of denied names, whether or not a source has the column."""

from collections.abc import Callable
from typing import Any

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.abstract_plugins.hook_context import HookContext


def _deny_every_name(name: str) -> bool:
    return True


class DeniedNameGate(Extender):
    """Gate on INPUT_DATA_MATCH: refuses a source group's match for any name is_denied accepts.

    Default refuses every name, and always when no hook context is active; the message does not name the feature.
    """

    never_fall_back = True

    def __init__(self, is_denied: Callable[[str], bool] = _deny_every_name) -> None:
        self._is_denied = is_denied

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.INPUT_DATA_MATCH}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        if context is None or any(self._is_denied(name) for name in context.feature_names):
            raise PermissionError("denied by policy")
        return func(*args, **kwargs)
