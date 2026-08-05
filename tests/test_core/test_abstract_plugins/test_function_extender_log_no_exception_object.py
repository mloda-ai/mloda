"""A contained extender failure is logged as text, never with exc_info.

``exc_info`` puts the traceback on the record, whose frames hold the extender and the wrapped callable.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pytest

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook, _invoke_extender

EXTENDER_LOGGER_NAME = Extender.__module__

EXTENDER_MESSAGE = "extender instrumentation is broken"


class ExtenderExploded(RuntimeError):
    """Raised by the extender's own code, so the failure is contained and the fallback runs."""


class BrokenExtender(Extender):
    """Fails before delegating, which is the branch that logs and falls back to the wrapped callable."""

    name = "broken_probe"

    def __init__(self) -> None:
        self.raise_on_error = False

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        raise ExtenderExploded(EXTENDER_MESSAGE)


def _inner(value: int) -> int:
    return value * 2


def _exception_args(record: logging.LogRecord) -> list[BaseException]:
    """Exception objects the record retains through its args, tuple form and dict form alike."""
    args = record.args
    if args is None:
        return []
    values = list(args.values()) if isinstance(args, Mapping) else list(args)
    return [value for value in values if isinstance(value, BaseException)]


def _extender_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == EXTENDER_LOGGER_NAME]


def _run_broken_extender(caplog: pytest.LogCaptureFixture) -> Any:
    with caplog.at_level(logging.WARNING, logger=EXTENDER_LOGGER_NAME):
        return _invoke_extender(BrokenExtender(), _inner, 21)


class TestExtenderFailureLogsNoExceptionObject:
    """The contained extender failure must reach the log as text only."""

    def test_record_retains_no_exception_object(self, caplog: pytest.LogCaptureFixture) -> None:
        assert _run_broken_extender(caplog) == 42, "the fallback must still run the wrapped callable"

        assert _extender_records(caplog), "the contained failure must report at WARNING"
        for record in _extender_records(caplog):
            assert _exception_args(record) == [], f"record pins an exception object: {record.getMessage()}"

    def test_record_retains_no_traceback_via_exc_info(self, caplog: pytest.LogCaptureFixture) -> None:
        _run_broken_extender(caplog)

        assert _extender_records(caplog), "the contained failure must report at WARNING"
        for record in _extender_records(caplog):
            assert record.exc_info is None, f"record pins a traceback via exc_info: {record.getMessage()}"

    def test_message_still_names_extender_and_exception(self, caplog: pytest.LogCaptureFixture) -> None:
        _run_broken_extender(caplog)

        messages = [record.getMessage() for record in _extender_records(caplog)]
        assert len(messages) == 1, f"exactly one WARNING record reports the failure, got: {messages}"
        assert "BrokenExtender" in messages[0], f"the extender class must stay in the message: {messages[0]}"
        assert "broken_probe" in messages[0], f"the extender name must stay in the message: {messages[0]}"
        assert "ExtenderExploded" in messages[0], f"the exception type must stay in the message: {messages[0]}"
        assert EXTENDER_MESSAGE in messages[0], f"the reason must stay readable: {messages[0]}"
