"""Unit tests for raise_on_unpicklable_child_bootstrap. A lambda, or a closure over an
unpicklable object, is genuinely unpicklable (pickle cannot resolve it back by module/qualname),
which is what a child_bootstrap callable queued to a multiprocessing worker would otherwise fail
deep inside pickle for."""

from __future__ import annotations

import threading
from collections.abc import Callable

import pytest

from mloda.core.runtime.validate_multiprocessing_link import raise_on_unpicklable_child_bootstrap


class _Unpicklable:
    """An object pickle cannot resolve: captured by a closure below, so the closure itself
    becomes unpicklable too (threading.Lock is never picklable)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()


def _make_closure_over_unpicklable() -> Callable[[], None]:
    unpicklable = _Unpicklable()

    def _bootstrap() -> None:
        unpicklable.lock.acquire()

    return _bootstrap


class _ModuleLevelBootstrap:
    """Picklable no-argument callable defined at module level."""

    def __call__(self) -> None:
        pass


def test_a_lambda_child_bootstrap_is_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_child_bootstrap(lambda: None)

    message = str(excinfo.value)
    assert "child_bootstrap" in message, f"the offending parameter must be named; got: {message}"
    assert "cannot be pickled for multiprocessing" in message, f"the problem must be named; got: {message}"


def test_a_closure_over_an_unpicklable_object_is_rejected() -> None:
    bootstrap = _make_closure_over_unpicklable()

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_child_bootstrap(bootstrap)

    message = str(excinfo.value)
    assert "child_bootstrap" in message, f"the offending parameter must be named; got: {message}"


def test_none_child_bootstrap_does_not_raise() -> None:
    raise_on_unpicklable_child_bootstrap(None)


def test_a_picklable_module_level_callable_does_not_raise() -> None:
    raise_on_unpicklable_child_bootstrap(_ModuleLevelBootstrap())
