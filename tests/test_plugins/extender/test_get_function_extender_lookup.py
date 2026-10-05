"""Pins the module-level get_function_extender lookup, reusable without a ComputeFramework instance."""

from typing import Any

import pytest

from mloda.core.abstract_plugins.function_extender import (
    Extender,
    ExtenderHook,
    CompositeExtender,
    build_hook_extenders,
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


class _TieAlpha(_DummyExtender):
    """Module-level extender class, sorts before _TieBeta on (module, qualname)."""

    def __init__(self) -> None:
        super().__init__(ExtenderHook.JOIN)


class _TieBeta(_DummyExtender):
    """Module-level extender class, sorts after _TieAlpha on (module, qualname)."""

    def __init__(self) -> None:
        super().__init__(ExtenderHook.JOIN)


class _WrapsCountingExtender(Extender):
    """Counts wraps() calls and wraps a fixed hook set."""

    def __init__(self, hooks: set[ExtenderHook]) -> None:
        self._hooks = hooks
        self.wraps_calls = 0

    def wraps(self) -> set[ExtenderHook]:
        self.wraps_calls += 1
        return set(self._hooks)

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


class _StringWrapsExtender(Extender):
    """wraps() wrongly returns a string instead of a collection of hooks."""

    def wraps(self) -> Any:
        return "join"

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


class _DuplicateWrapsExtender(Extender):
    """wraps() lists the same hook twice."""

    def wraps(self) -> Any:
        return [ExtenderHook.JOIN, ExtenderHook.JOIN]

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


class TestDeterministicTieOrder:
    """Equal-priority extenders order by (module, qualname), independent of input order."""

    @pytest.mark.parametrize("alpha_first", [True, False])
    def test_equal_priority_extenders_are_ordered_by_module_and_qualname(self, alpha_first: bool) -> None:
        alpha, beta = _TieAlpha(), _TieBeta()
        ordered = [alpha, beta] if alpha_first else [beta, alpha]

        result = get_function_extender(ordered, ExtenderHook.JOIN)

        assert isinstance(result, CompositeExtender)
        assert result.extenders == (alpha, beta)


class TestBuildHookExtenders:
    """build_hook_extenders selects each hook's extender once, calling wraps() exactly once per extender."""

    def test_wraps_called_once_per_extender_and_one_entry_per_wrapped_hook(self) -> None:
        first = _WrapsCountingExtender({ExtenderHook.JOIN, ExtenderHook.INPUT_DATA_LOAD})
        second = _WrapsCountingExtender({ExtenderHook.JOIN})

        table = build_hook_extenders([first, second])

        assert first.wraps_calls == 1
        assert second.wraps_calls == 1
        assert set(table) == {ExtenderHook.JOIN, ExtenderHook.INPUT_DATA_LOAD}
        assert table[ExtenderHook.INPUT_DATA_LOAD] is first
        assert isinstance(table[ExtenderHook.JOIN], CompositeExtender)

    def test_string_wraps_raises_type_error_naming_the_extender(self) -> None:
        with pytest.raises(TypeError, match="_StringWrapsExtender"):
            build_hook_extenders([_StringWrapsExtender()])

    def test_duplicate_hook_in_wraps_selects_the_extender_once(self) -> None:
        extender = _DuplicateWrapsExtender()

        assert build_hook_extenders([extender])[ExtenderHook.JOIN] is extender


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
        assert result.extenders == (low, mid, high)
