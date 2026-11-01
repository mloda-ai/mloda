"""Repeat pytest-timeout's signal alarm so a timeout swallowed in a gc callback fires again."""

import signal
from collections.abc import Generator
from typing import Any

import pytest


@pytest.hookimpl(wrapper=True, optionalhook=True)
def pytest_timeout_set_timer(item: pytest.Item, settings: Any) -> Generator[None, object, object]:
    armed = yield
    remaining, _ = signal.getitimer(signal.ITIMER_REAL)
    if remaining:  # the signal method armed it; the thread fallback leaves it at 0
        signal.setitimer(signal.ITIMER_REAL, remaining, settings.timeout)
    return armed
