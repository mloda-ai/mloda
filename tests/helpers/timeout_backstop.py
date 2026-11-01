"""Re-arm pytest-timeout's signal alarm when its timeout is swallowed in a gc callback or finalizer."""

import signal
import sys
from collections.abc import Callable, Generator
from typing import Any

import pytest

_previous_hook = pytest.StashKey[Callable[[Any], object]]()


@pytest.hookimpl(wrapper=True, optionalhook=True)
def pytest_timeout_set_timer(item: pytest.Item, settings: Any) -> Generator[None, object, object]:
    armed = yield
    # The signal method armed ITIMER_REAL; the thread method and its off-main-thread fallback leave it at 0.
    if settings.method == "signal" and signal.getitimer(signal.ITIMER_REAL)[0]:
        previous = sys.unraisablehook

        def rearm(unraisable: Any) -> None:
            if isinstance(unraisable.exc_value, pytest.fail.Exception) and "from pytest-timeout" in str(
                unraisable.exc_value
            ):
                signal.setitimer(signal.ITIMER_REAL, settings.timeout)
            previous(unraisable)

        item.stash[_previous_hook] = previous
        sys.unraisablehook = rearm
    return armed


@pytest.hookimpl(wrapper=True, optionalhook=True)
def pytest_timeout_cancel_timer(item: pytest.Item) -> Generator[None, object, object]:
    cancelled = yield
    if _previous_hook in item.stash:
        sys.unraisablehook = item.stash[_previous_hook]
        del item.stash[_previous_hook]
    return cancelled
