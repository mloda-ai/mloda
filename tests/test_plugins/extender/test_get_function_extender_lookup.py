"""Pins the module-level get_function_extender lookup, reusable without a ComputeFramework instance."""

from typing import Any

from mloda.core.abstract_plugins.function_extender import (
    Extender,
    ExtenderHook,
    CompositeExtender,
    get_function_extender,
)


class _DummyExtender(Extender):
    """Minimal Extender wrapping a single configurable hook."""

    def __init__(self, hook: ExtenderHook, priority: int = 100) -> None:
        self.priority = priority
        self._hook = hook

    def wraps(self) -> set[ExtenderHook]:
        return {self._hook}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


class TestGetFunctionExtenderLookup:
    """Free-function get_function_extender(function_extender, hook) mirrors ComputeFramework's instance method."""

    def test_no_match_returns_none(self) -> None:
        result = get_function_extender(set(), ExtenderHook.JOIN)

        assert result is None

    def test_no_match_among_non_matching_extenders_returns_none(self) -> None:
        extenders: set[Extender] = {_DummyExtender(ExtenderHook.INPUT_DATA_LOAD)}

        result = get_function_extender(extenders, ExtenderHook.JOIN)

        assert result is None

    def test_single_match_returns_that_extender(self) -> None:
        only = _DummyExtender(ExtenderHook.JOIN)

        result = get_function_extender({only}, ExtenderHook.JOIN)

        assert result is only

    def test_multiple_matches_returns_composite_sorted_by_priority(self) -> None:
        low = _DummyExtender(ExtenderHook.JOIN, priority=10)
        high = _DummyExtender(ExtenderHook.JOIN, priority=50)
        mid = _DummyExtender(ExtenderHook.JOIN, priority=30)

        result = get_function_extender({high, low, mid}, ExtenderHook.JOIN)

        assert isinstance(result, CompositeExtender)
        assert result.extenders == [low, mid, high]
