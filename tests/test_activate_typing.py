"""Regression tests: activate() returns a non-Optional with-block value."""

from mloda.core.abstract_plugins.close_context import CloseContext
from mloda.core.abstract_plugins.hook_context import HookContext


def _hook_context() -> HookContext:
    return HookContext(hook="pre", feature_group_class=None, feature_group_version=None, compute_framework_name=None)


def _close_context() -> CloseContext:
    return CloseContext(deadline=0.0, reason="stop")


def test_hook_context_activate_returns_context() -> None:
    context = _hook_context()
    with context.activate() as active:
        assert active is context


def test_hook_context_activate_restores_previous() -> None:
    outer = _hook_context()
    inner = _hook_context()
    with outer.activate():
        with inner.activate() as active:
            assert active is inner
        assert HookContext.current() is outer
    assert HookContext.current() is None


def test_close_context_activate_returns_context() -> None:
    context = _close_context()
    with context.activate() as active:
        assert active is context


def test_close_context_activate_restores_previous() -> None:
    outer = _close_context()
    inner = _close_context()
    with outer.activate():
        with inner.activate() as active:
            assert active is inner
        assert CloseContext.current() is outer
    assert CloseContext.current() is None
