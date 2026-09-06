"""Unit tests for raise_on_unpicklable_extender. An Extender class is picklable at the class
level, but an instance can still hold unpicklable state (e.g. a threading.Lock), which would
otherwise fail deep inside pickle when the extender is sent to the multiprocessing manager
process."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook
from mloda.core.runtime.validate_multiprocessing_link import raise_on_unpicklable_extender


class ValidateExtenderPicklable(Extender):
    """Ordinary module-level Extender with no instance state: picklable."""

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


class ValidateExtenderUnpicklableInstance(Extender):
    """Module-level Extender class whose instance holds a threading.Lock: never picklable."""

    def __init__(self) -> None:
        self.lock = threading.Lock()

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)


def test_none_function_extender_does_not_raise() -> None:
    raise_on_unpicklable_extender(None)


def test_empty_function_extender_set_does_not_raise() -> None:
    raise_on_unpicklable_extender(set())


def test_a_picklable_extender_does_not_raise() -> None:
    raise_on_unpicklable_extender({ValidateExtenderPicklable()})


def test_an_extender_with_unpicklable_instance_state_is_rejected() -> None:
    extender = ValidateExtenderUnpicklableInstance()

    with pytest.raises(ValueError) as excinfo:
        raise_on_unpicklable_extender({extender})

    message = str(excinfo.value)
    assert repr(extender) in message, f"the offending extender must be named; got: {message}"
    assert "cannot be pickled for multiprocessing" in message, f"the problem must be named; got: {message}"
    assert "Resolution" in message, f"a resolution must be offered; got: {message}"
